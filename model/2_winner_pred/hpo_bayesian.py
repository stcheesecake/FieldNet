#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import random
import warnings
import time
import sys
from datetime import datetime
from contextlib import suppress

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# HPO CONFIG (START, END, STEP)
# =========================================================
N_FUTURE_ACTIONS = 10
N_FOLDS = 3
SEED = 42
N_ROUNDS = 100

# [START, END, STEP] ranges
N_PREV_ACTIONS_R = [50, 300, 50]  # 슬라이딩 윈도우 크기 (최근 몇 개의 액션을 볼 것인가)
SIGMA_R = [1, 10, 2]  # 시간 변화 계수의 변동성 가중치
SIGMA_SCALE = 0.01

ITER_R = [500, 1500, 250]  # 학습 반복 횟수
LR_R = [1, 5, 1]  # Learning Rate
LR_SCALE = 0.001

# ---------------------------------------------------------
DATA_PATH = "../../data/"
TRAIN_PATH = f"{DATA_PATH}train.csv"
OUT_DIR    = "../../../FiledNet_pkl_temp/hpo_results/bayesian"
USE_GPU = torch.cuda.is_available()


# =========================================================
# MODEL: Bayesian Time-Varying Logistic Regression
# =========================================================
class BayesianTimeVaryingModel(nn.Module):
    def __init__(self, input_dim, sigma):
        super().__init__()
        # 가우시안 사전 분포(Prior)를 따르는 가중치 설정
        self.weights = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        self.sigma = sigma

    def forward(self, x, t):
        # 논문의 핵심: 진행률(t)에 따라 가중치의 영향력이 비선형적으로 변함
        # 시간 t는 0~1 사이의 값
        time_factor = torch.cos(t * math.pi / 2) * self.sigma
        logit = torch.mv(x, self.weights) + self.bias + time_factor.mean()
        return torch.sigmoid(logit)


# =========================================================
# PREPROCESSING: Action-Level Sliding Window
# =========================================================
def prepare_data(df, n_prev):
    df = df.sort_values(['game_id', 'period_id', 'time_seconds']).reset_index(drop=True)

    # 1. 레이블 생성 (N_FUTURE_ACTIONS 내 득점 여부)
    # 실제 데이터에선 type_result 기반 골 코드를 찾아야 함 (여기선 예시로 생성)
    # y = label_score_in_next_k_actions(...) 로직이 적용되었다고 가정

    # 2. 피처 생성 (액션 단위 슬라이딩 윈도우)
    # 현재 액션 정보
    curr_feats = df[['start_x', 'end_x', 'dx', 'dy']].values

    # 윈도우 통계량 (Rolling) - 데이터 뭉개짐 없이 '기세' 추출
    # 최근 n_prev개의 액션 동안 얼마나 전진(dx)했는지 합산
    momentum = df.groupby('game_id')['dx'].rolling(window=n_prev, min_periods=1).sum().values

    # 시간 정규화 (0~1)
    # 축구 경기 약 95분(5700초) 기준
    t_norm = (df['time_seconds'] + (df['period_id'] - 1) * 2700) / 5700.0
    t_norm = np.clip(t_norm, 0, 1)

    X = np.column_stack([curr_feats, momentum, t_norm])
    y = df['home_score'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0).values  # 임시 레이블
    groups = df['game_id'].values

    return X.astype(np.float32), y.astype(np.float32), groups


# =========================================================
# CV EVALUATION (One-line Progress 적용)
# =========================================================
def run_cv_eval(X_np, y_np, groups, params, outer_pbar, t_idx, best_str_func):
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(y_np))
    device = torch.device("cuda" if USE_GPU else "cpu")

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_np, y_np, groups)):
        # ✅ 한 줄 출력을 위해 postfix_str 사용
        outer_pbar.set_postfix_str(f"Fold {fold + 1}/{N_FOLDS}.. | {best_str_func()}", refresh=True)

        X_tr = torch.from_numpy(X_np[tr_idx]).to(device)
        y_tr = torch.from_numpy(y_np[tr_idx]).to(device)
        X_va = torch.from_numpy(X_np[va_idx]).to(device)

        # 모델 입력 차원: 마지막 컬럼(t) 제외
        model = BayesianTimeVaryingModel(X_np.shape[1] - 1, params['sigma']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
        criterion = nn.BCELoss()

        # 학습 (ADVI를 모사한 배치 학습)
        model.train()
        for i in range(params['iters']):
            optimizer.zero_grad()
            outputs = model(X_tr[:, :-1], X_tr[:, -1])
            loss = criterion(outputs, y_tr)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_va[:, :-1], X_va[:, -1]).cpu().numpy()
            oof_pred[va_idx] = preds

    brier = brier_score_loss(y_np, oof_pred)
    return {"brier": brier}


# =========================================================
# MAIN HPO
# =========================================================
def main():
    set_all_seeds(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 데이터 로드
    df = pd.read_csv(TRAIN_PATH)

    # 그리드 구축
    prev_vals = build_values(N_PREV_ACTIONS_R, as_int=True)
    sigma_vals = build_values(SIGMA_R, scale=SIGMA_SCALE)
    iter_vals = build_values(ITER_R, as_int=True)
    lr_vals = build_values(LR_R, scale=LR_SCALE)

    best = {"trial": -1, "brier": float('inf'), "params": None}

    def get_best_msg():
        if best["trial"] == -1: return "best#-"
        return f"best#{best['trial']} brier={best['brier']:.5f}"

    pbar = tqdm(range(1, N_ROUNDS + 1), desc="HPO", dynamic_ncols=True)

    for t_idx in pbar:
        # 무작위 샘플링
        params = {
            "n_prev": random.choice(prev_vals),
            "sigma": random.choice(sigma_vals),
            "iters": random.choice(iter_vals),
            "lr": random.choice(lr_vals)
        }

        # 1. 라운드별 윈도우 크기에 맞춘 전처리
        X_np, y_np, groups = prepare_data(df, params['n_prev'])

        # 2. CV 실행
        metrics = run_cv_eval(X_np, y_np, groups, params, pbar, t_idx, get_best_msg)

        # 3. 결과 기록
        if metrics['brier'] < best['brier']:
            best.update({"trial": t_idx, "brier": metrics['brier'], "params": params})

        pbar.set_postfix_str(get_best_msg(), refresh=True)

    print(f"\n[최종 결과] {best}")


def build_values(rng3, scale=1.0, as_int=False):
    s, e, st = rng3
    vals = np.arange(s, e + st, st)
    if scale != 1.0: vals = vals * scale
    if as_int: vals = np.round(vals).astype(int)
    return list(vals)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()