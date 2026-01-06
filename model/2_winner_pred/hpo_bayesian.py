#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from scipy.stats import truncnorm, norm

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
TRAIN_CSV = "../../data/train.csv"
MAP_JSON  = "../../data/preprocess_maps.json"

OUT_DIR = "../../../FiledNet_pkl_temp/results/results_bayes_winprob"
N_FOLDS = 3
SEED = 42

# time grid (논문은 1~90분 = 90개 모델)
MAX_MINUTE = 90

# Gibbs sampler
N_ITER = 2500
BURN   = 1200
THIN   = 5

# priors
BETA_VAR = 50.0     # beta ~ N(0, BETA_VAR * I)
DELTA_VAR = 200.0**2  # delta_j ~ N(0, DELTA_VAR) (논문도 큰 값으로 둠)

EPS = 1e-6

# =========================================================
# Utilities
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def minute_from_period_time(period_id: int, time_seconds: float) -> int:
    """
    train.csv에서:
    - period_id: 1(전),2(후)
    - time_seconds: 각 period 시작 후 경과초 (0~3600+)
    논문처럼 stoppage를 45/90분에 몰아넣는 방식으로 매핑.
    """
    m_in_half = int(time_seconds // 60) + 1
    m_in_half = min(45, max(1, m_in_half))
    if period_id == 1:
        return m_in_half
    else:
        return 45 + m_in_half

def safe_truncnorm_rvs(mean: float, sd: float, low: float, high: float) -> float:
    """
    low/high가 무한대여도 truncnorm이 처리 가능.
    low >= high면 fallback으로 clip(mean)을 반환.
    """
    if not np.isfinite(low):
        a = -np.inf
    else:
        a = (low - mean) / sd
    if not np.isfinite(high):
        b = np.inf
    else:
        b = (high - mean) / sd

    if (np.isfinite(a) and np.isfinite(b) and a >= b) or (low >= high):
        return float(np.clip(mean, low + EPS, high - EPS)) if np.isfinite(low) and np.isfinite(high) else float(mean)

    return float(truncnorm.rvs(a=a, b=b, loc=mean, scale=sd))

def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: List[int]) -> float:
    """
    Multiclass Brier score: mean_i sum_c (p_ic - 1[y_i==c])^2
    y_true: shape (n,)
    proba: shape (n, C)
    classes: list of class labels aligned with proba columns
    """
    n = len(y_true)
    Y = np.zeros_like(proba)
    for j, c in enumerate(classes):
        Y[:, j] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

# =========================================================
# Event mapping (논문 8개 이벤트 카테고리로 압축)
# =========================================================
def decode_type_result(df: pd.DataFrame, inv_map: Dict[int, str]) -> pd.DataFrame:
    s = df["type_result"].map(inv_map)
    # type_name, result_name 분리 (맨 마지막 "__" 기준)
    tmp = s.str.rsplit("__", n=1, expand=True)
    df["type_name"] = tmp[0]
    df["result_name"] = tmp[1].fillna("NA")
    return df

def build_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    논문 이벤트:
    Goal, Shot-on, Shot-off, Red card, Yellow card, Corner, Cross, Foul
    -> 홈/원정 각각 카운트가 필요하니, 각 row를 H/A로 credit해서 0/1 플래그 생성
    """
    # side: action 수행 팀이 홈팀인가?
    df["is_home_action"] = (df["team_id"] == df["home_team_id"]).astype(np.int8)

    # 기본 side(H/A) one-hot
    df["H_side"] = df["is_home_action"]
    df["A_side"] = (1 - df["is_home_action"]).astype(np.int8)

    tn = df["type_name"].astype(str)
    rn = df["result_name"].astype(str)

    # ---- Goal: Goal 자체 + Shot/FreeKick/Penalty goal + Own Goal ----
    is_goal = (tn == "Goal") | ((tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) & (rn == "Goal"))
    is_own_goal = (tn == "Own Goal")

    # ---- Shots (on/off) ----
    is_shot_on = (tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) & (rn.isin(["On Target", "Keeper Rush-Out"]))
    is_shot_off = (
        (tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) &
        (rn.isin(["Off Target", "Blocked", "Low Quality Shot"]))
    ) | (tn.isin(["Goal Miss", "Goal Post"]))

    # ---- Cards ----
    is_red = tn.isin(["Foul", "Handball_Foul"]) & rn.isin(["Direct_Red_Card", "Second_Yellow_Card"])
    is_yellow = tn.isin(["Foul", "Handball_Foul"]) & (rn == "Yellow_Card")

    # ---- Corner / Cross / Foul ----
    is_corner = (tn == "Pass_Corner")
    is_cross  = (tn == "Cross")
    is_foul   = tn.isin(["Foul", "Handball_Foul", "Foul_Throw"])

    # credit side:
    # - 일반 이벤트: action team side로 credit
    # - own goal: 반대편에 goal credit (상대에게 득점이니까)
    H = df["H_side"].values.astype(np.int8)
    A = df["A_side"].values.astype(np.int8)

    # init columns
    for ev in ["goal", "shot_on", "shot_off", "red", "yellow", "corner", "cross", "foul"]:
        df[f"H_{ev}"] = 0
        df[f"A_{ev}"] = 0

    # helper to set
    def set_side(ev_mask: np.ndarray, colH: str, colA: str, flip: bool = False):
        if not flip:
            df.loc[ev_mask, colH] = H[ev_mask]
            df.loc[ev_mask, colA] = A[ev_mask]
        else:
            # flip: own goal
            df.loc[ev_mask, colH] = A[ev_mask]
            df.loc[ev_mask, colA] = H[ev_mask]

    set_side(is_goal.values,     "H_goal",     "A_goal",     flip=False)
    set_side(is_own_goal.values, "H_goal",     "A_goal",     flip=True)

    set_side(is_shot_on.values,  "H_shot_on",  "A_shot_on",  flip=False)
    set_side(is_shot_off.values, "H_shot_off", "A_shot_off", flip=False)
    set_side(is_red.values,      "H_red",      "A_red",      flip=False)
    set_side(is_yellow.values,   "H_yellow",   "A_yellow",   flip=False)
    set_side(is_corner.values,   "H_corner",   "A_corner",   flip=False)
    set_side(is_cross.values,    "H_cross",    "A_cross",    flip=False)
    set_side(is_foul.values,     "H_foul",     "A_foul",     flip=False)

    return df

# =========================================================
# Feature builders
# =========================================================
EVENT_COLS = [
    "H_goal","A_goal",
    "H_shot_on","A_shot_on",
    "H_shot_off","A_shot_off",
    "H_red","A_red",
    "H_yellow","A_yellow",
    "H_corner","A_corner",
    "H_cross","A_cross",
    "H_foul","A_foul",
]

DIFF_COLS = [
    "goal_diff",
    "shot_on_diff",
    "shot_off_diff",
    "red_diff",
    "yellow_diff",
    "corner_diff",
    "cross_diff",
    "foul_diff",
]

def add_time_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df["match_minute"] = [
        minute_from_period_time(p, t) for p, t in zip(df["period_id"].values, df["time_seconds"].values)
    ]
    df = df.sort_values(["game_id","period_id","time_seconds","team_id"], kind="mergesort").reset_index(drop=True)
    df["action_idx"] = df.groupby("game_id").cumcount()
    return df

def add_cumulative_counts(df: pd.DataFrame) -> pd.DataFrame:
    # cumulative per action (game-wise)
    for c in EVENT_COLS:
        df[c] = df.groupby("game_id")[c].cumsum().astype(np.int16)

    # diff features (H - A)
    df["goal_diff"]     = (df["H_goal"]     - df["A_goal"]).astype(np.int16)
    df["shot_on_diff"]  = (df["H_shot_on"]  - df["A_shot_on"]).astype(np.int16)
    df["shot_off_diff"] = (df["H_shot_off"] - df["A_shot_off"]).astype(np.int16)
    df["red_diff"]      = (df["H_red"]      - df["A_red"]).astype(np.int16)
    df["yellow_diff"]   = (df["H_yellow"]   - df["A_yellow"]).astype(np.int16)
    df["corner_diff"]   = (df["H_corner"]   - df["A_corner"]).astype(np.int16)
    df["cross_diff"]    = (df["H_cross"]    - df["A_cross"]).astype(np.int16)
    df["foul_diff"]     = (df["H_foul"]     - df["A_foul"]).astype(np.int16)

    return df

def build_minute_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 경기(game_id)마다 1~90분의 '그 시점까지 누적 이벤트 카운트'를 가진 패널 생성.
    - 어떤 분에 action이 없으면 이전 분 누적값을 carry-forward(ffill)
    """
    keep_cols = ["game_id","match_minute"] + DIFF_COLS
    # 각 (game, minute)에서 마지막 action의 누적값
    last_per_min = df.groupby(["game_id","match_minute"], as_index=False).tail(1)[keep_cols].copy()

    panels = []
    for gid, gdf in last_per_min.groupby("game_id"):
        gdf = gdf.set_index("match_minute").sort_index()
        gdf = gdf.reindex(range(1, MAX_MINUTE+1))
        gdf["game_id"] = gid
        gdf[DIFF_COLS] = gdf[DIFF_COLS].ffill().fillna(0)
        gdf = gdf.reset_index().rename(columns={"index":"match_minute"})
        panels.append(gdf)

    panel = pd.concat(panels, ignore_index=True)
    return panel

def build_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    홈팀 관점 결과 y in {-1,0,1}
    train.csv의 home_score/away_score는 (샘플 확인 결과) 경기 전체에서 상수=최종스코어로 보임.
    """
    ms = df.groupby("game_id")[["home_score","away_score"]].first().copy()
    diff = ms["home_score"] - ms["away_score"]
    y = np.sign(diff).astype(int)  # -1/0/1
    ms["y"] = y
    return ms.reset_index()

# =========================================================
# Bayesian Ordered Probit (Gibbs)
# =========================================================
@dataclass
class OrderedProbitParams:
    beta_mean: np.ndarray
    delta1_mean: float
    delta2_mean: float

def fit_ordered_probit_gibbs(X: np.ndarray, y: np.ndarray,
                             n_iter=N_ITER, burn=BURN, thin=THIN,
                             beta_var=BETA_VAR, delta_var=DELTA_VAR,
                             seed=SEED) -> OrderedProbitParams:
    """
    Standard ordered probit Gibbs:
    latent z | beta, delta ~ TruncNormal
    beta | z ~ Normal (conjugate)
    delta_j | z ~ TruncNormal(prior) with ordering constraints

    sigma^2 fixed to 1.
    """
    rng = np.random.default_rng(seed)

    n, p = X.shape
    # prior precision
    prior_prec = np.eye(p) / beta_var

    beta = np.zeros(p, dtype=float)
    delta1, delta2 = -0.5, 0.5

    z = X @ beta

    # store
    betas = []
    d1s = []
    d2s = []

    XtX = X.T @ X

    for it in range(n_iter):
        mu = X @ beta

        # 1) sample latent z_i
        z_new = np.empty(n, dtype=float)
        for i in range(n):
            if y[i] == -1:
                low, high = -np.inf, delta1
            elif y[i] == 0:
                low, high = delta1, delta2
            else:
                low, high = delta2, np.inf
            z_new[i] = safe_truncnorm_rvs(mu[i], 1.0, low, high)
        z = z_new

        # 2) sample beta | z  -> Normal
        post_prec = XtX + prior_prec
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ z)
        beta = rng.multivariate_normal(mean=post_mean, cov=post_cov)

        # 3) sample deltas with constraints (and N(0, delta_var) prior)
        sd_delta = np.sqrt(delta_var)

        # bounds from z samples
        z_m1 = z[y == -1]
        z_0  = z[y == 0]
        z_p1 = z[y == 1]

        # delta1: max(z|-1) < delta1 < min(z|0) and < delta2
        low1 = np.max(z_m1) if len(z_m1) else -np.inf
        up1a = np.min(z_0)  if len(z_0)  else delta2 - 0.1
        high1 = min(up1a, delta2 - EPS)

        delta1 = safe_truncnorm_rvs(0.0, sd_delta, low1, high1)

        # delta2: max(z|0) < delta2 < min(z|+1) and > delta1
        low2a = np.max(z_0) if len(z_0) else delta1 + 0.1
        low2 = max(low2a, delta1 + EPS)
        high2 = np.min(z_p1) if len(z_p1) else np.inf

        delta2 = safe_truncnorm_rvs(0.0, sd_delta, low2, high2)

        # store after burn/thin
        if it >= burn and ((it - burn) % thin == 0):
            betas.append(beta.copy())
            d1s.append(delta1)
            d2s.append(delta2)

    beta_mean = np.mean(np.stack(betas, axis=0), axis=0)
    return OrderedProbitParams(beta_mean=beta_mean,
                               delta1_mean=float(np.mean(d1s)),
                               delta2_mean=float(np.mean(d2s)))

def predict_proba_ordered_probit(X: np.ndarray, params: OrderedProbitParams) -> np.ndarray:
    """
    Returns proba for classes [-1, 0, 1] in this order: [Loss, Draw, Win] (home perspective).
    """
    mu = X @ params.beta_mean
    d1, d2 = params.delta1_mean, params.delta2_mean
    p_loss = norm.cdf(d1 - mu)
    p_draw = norm.cdf(d2 - mu) - norm.cdf(d1 - mu)
    p_win  = 1.0 - norm.cdf(d2 - mu)
    proba = np.vstack([p_loss, p_draw, p_win]).T
    # numeric safety
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba

# =========================================================
# Main CV pipeline
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    # 1) load
    df = pd.read_csv(TRAIN_CSV)
    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)
    inv_type = {v: k for k, v in maps["type_result"].items()}

    # 2) decode + time + event flags + cumulative
    df = decode_type_result(df, inv_type)
    df = add_time_and_sort(df)
    df = build_event_flags(df)
    df = add_cumulative_counts(df)

    # 3) build per-minute panel (one row per match per minute)
    minute_panel = build_minute_panel(df)
    labels = build_match_labels(df)  # game_id, y

    panel = minute_panel.merge(labels[["game_id","y"]], on="game_id", how="left")

    # Features for model (diff only + intercept)
    panel["intercept"] = 1.0
    FEATS = ["intercept"] + DIFF_COLS

    # GroupKFold by game_id
    games = labels["game_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    metrics_rows = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(games, groups=games), start=1):
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
        ensure_dir(fold_dir)

        tr_games = set(games[tr_idx])
        va_games = set(games[va_idx])

        # ---- Train 90 minute-models (t=1..90) ----
        minute_models: Dict[int, OrderedProbitParams] = {}

        for t in tqdm(range(1, MAX_MINUTE+1), desc=f"Fold {fold} | Fit minute models"):
            tr_t = panel[(panel["match_minute"] == t) & (panel["game_id"].isin(tr_games))].copy()
            X = tr_t[FEATS].values.astype(float)
            y = tr_t["y"].values.astype(int)

            params = fit_ordered_probit_gibbs(X, y, seed=SEED + 1000*fold + t)
            minute_models[t] = params

        # save minute models
        with open(os.path.join(fold_dir, "minute_models.pkl"), "wb") as f:
            pickle.dump(minute_models, f)

        # ---- Validation: action-level probabilities ----
        va_actions = df[df["game_id"].isin(va_games)].copy()
        va_actions = va_actions.reset_index(drop=True)  # ✅ 이 줄 추가
        va_actions["intercept"] = 1.0

        # action feature matrix uses current cumulative diffs (already computed per action)
        X_action = va_actions[["intercept"] + DIFF_COLS].values.astype(float)

        # per-action predict: pick params by its minute
        proba_list = []
        for t, idxs in va_actions.groupby("match_minute").groups.items():
            params = minute_models.get(int(t), None)
            if params is None:
                # should not happen, but safe
                params = minute_models[MAX_MINUTE]
            X_part = X_action[np.array(list(idxs))]
            proba_part = predict_proba_ordered_probit(X_part, params)
            proba_list.append((idxs, proba_part))

        # stitch back
        proba = np.zeros((len(va_actions), 3), dtype=float)
        for idxs, pp in proba_list:
            proba[np.array(list(idxs)), :] = pp

        # y_true per action = match y
        y_map = labels.set_index("game_id")["y"].to_dict()
        y_true = va_actions["game_id"].map(y_map).values.astype(int)

        # logloss (classes order [-1,0,1] -> columns [loss,draw,win])
        # sklearn expects labels [0..C-1], so remap
        class_order = [-1, 0, 1]
        y_remap = np.array([class_order.index(v) for v in y_true], dtype=int)
        ll = log_loss(y_remap, proba, labels=[0,1,2])

        brier = multiclass_brier(y_true, proba, classes=class_order)

        metrics_rows.append({
            "fold": fold,
            "val_games": len(va_games),
            "val_actions": len(va_actions),
            "logloss_action": ll,
            "brier_action": brier
        })

        # save per-action probs
        out = va_actions[[
            "game_id","period_id","time_seconds","match_minute",
            "home_team_id","away_team_id","home_team_name","away_team_name",
            "team_id","type_result","type_name","result_name"
        ]].copy()
        out["p_home_loss"] = proba[:, 0]
        out["p_draw"]      = proba[:, 1]
        out["p_home_win"]  = proba[:, 2]
        out["p_away_win"]  = out["p_home_loss"]  # away win = home loss

        out.to_csv(os.path.join(fold_dir, "val_action_probs.csv"), index=False)

        print(f"[Fold {fold}] action-logloss={ll:.6f} | action-brier={brier:.6f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "cv_metrics.csv"), index=False)
    print("\nSaved:", os.path.join(OUT_DIR, "cv_metrics.csv"))


if __name__ == "__main__":
    main()
