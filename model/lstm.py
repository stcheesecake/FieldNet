#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from contextlib import contextmanager

@contextmanager
def autocast_cuda(enabled: bool):
    if not enabled:
        yield
        return
    try:
        # ✅ 최신 (권장)
        with torch.amp.autocast(device_type="cuda"):
            yield
    except Exception:
        # ✅ 구버전 fallback
        with torch.cuda.amp.autocast():
            yield

# =========================
# CONFIG
# =========================
HORIZON  = 10    # ✅ 다음 180 actions 내 득점
LOOKBACK = 20    # ✅ 입력 시퀀스 길이(최근 60 actions)

N_FOLDS = 5
SEED = 42

DATA_PATH  = "../data/"
TRAIN_PATH = f"{DATA_PATH}train.csv"
MAP_PATH   = f"{DATA_PATH}preprocess_maps.json"

TEMP_DIR = "../temp"
MODEL_OUT = f"SEED{SEED}_H{HORIZON}_L{LOOKBACK}_lstm.pkl"

LEAKAGE_COLS = ["home_score", "away_score"]

# pitch
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0
GOAL_X      = 105.0
GOAL_Y      = 34.0
GOAL_HALF_W = 3.66
MID_X = PITCH_X_MAX / 2.0

# training
BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3
EMB_DIM = 16
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.1

# 불균형이면 켜서 속도/밸런스 조절
NEG_SUBSAMPLE = 1.0  # 1.0=사용 안함, 0.2면 음성 20%만 사용

# DataLoader 최적화
AUTO_NUM_WORKERS = True
NUM_WORKERS = 0  # AUTO_NUM_WORKERS=False일 때만 사용
USE_AMP = True   # cuda면 mixed precision

# =========================
# REPRO
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =========================
# UTIL
# =========================
def match_time_seconds(period_id, time_seconds):
    return int(float(time_seconds) + max(int(period_id) - 1, 0) * 45 * 60)

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def flip_x_if_needed(x, attack_right: bool):
    return float(x) if attack_right else float(PITCH_X_MAX - float(x))

def goal_angle(x, y):
    x = float(x); y = float(y)
    dx = GOAL_X - x
    if dx <= 1e-4:
        return 0.0
    y1 = GOAL_Y - GOAL_HALF_W
    y2 = GOAL_Y + GOAL_HALF_W
    a1 = math.atan2(y1 - y, dx)
    a2 = math.atan2(y2 - y, dx)
    return float(abs(a2 - a1))

def goal_distance(x, y):
    x = float(x); y = float(y)
    return float(math.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2))

def build_code_maps_from_json(map_path: str):
    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    if "type_result" not in maps:
        raise KeyError(f"Missing key 'type_result' in {map_path}. keys={list(maps.keys())}")

    str_to_code = maps["type_result"]
    code_to_str = {int(v): str(k) for k, v in str_to_code.items()}

    goal_codes = set()
    own_goal_codes = set()
    shot_codes = set()

    for c, s in code_to_str.items():
        type_name = str(s).split("__", 1)[0].strip()
        if type_name in ["Shot", "Shot_Freekick", "Shot_Corner", "Penalty Kick"]:
            shot_codes.add(c)
        if s.endswith("__Goal") or type_name == "Goal":
            goal_codes.add(c)
        if type_name in ["Own Goal", "Own_Goal"]:
            own_goal_codes.add(c)

    return maps, code_to_str, goal_codes, own_goal_codes, shot_codes

def get_teams_by_game(df: pd.DataFrame):
    if ("home_team_id" in df.columns) and ("away_team_id" in df.columns):
        g = df.groupby("game_id")[["home_team_id", "away_team_id"]].first()
        return {int(idx): (int(r.home_team_id), int(r.away_team_id)) for idx, r in g.iterrows()}

    tmp = df.groupby("game_id")["team_id"].unique().to_dict()
    out = {}
    for gid, arr in tmp.items():
        arr = list(arr)
        if len(arr) >= 2:
            out[int(gid)] = (int(arr[0]), int(arr[1]))
    return out

def estimate_attack_direction(df: pd.DataFrame, shot_codes: set):
    key_cols = ["game_id", "period_id", "team_id"]
    attack = {}

    df_shot = df[df["type_result"].isin(shot_codes)]
    if len(df_shot) > 0:
        m = df_shot.groupby(key_cols)["end_x"].mean()
        for k, v in m.items():
            attack[k] = bool(v > MID_X)

    m2 = df.groupby(key_cols)["end_x"].mean()
    for k, v in m2.items():
        if k not in attack:
            attack[k] = bool(v > MID_X)
    return attack

def build_goal_action_indices(df_g: pd.DataFrame, goal_codes: set, own_goal_codes: set, teams_by_game: dict):
    gid = int(df_g["game_id"].iloc[0])
    home, away = teams_by_game.get(gid, (None, None))

    goal_idx_by_team = {}
    idxs = np.where(df_g["type_result"].isin(goal_codes).values)[0]

    for i in idxs:
        actor = int(df_g.iloc[i]["team_id"])
        tr = int(df_g.iloc[i]["type_result"])

        if tr in own_goal_codes and (home is not None and away is not None):
            scorer = away if actor == home else home
        else:
            scorer = actor
        goal_idx_by_team.setdefault(scorer, []).append(int(i))

    for tid in list(goal_idx_by_team.keys()):
        goal_idx_by_team[tid] = sorted(goal_idx_by_team[tid])
    return goal_idx_by_team

def label_score_in_next_k_actions(goal_indices_sorted, cur_idx: int, k: int):
    if not goal_indices_sorted:
        return 0
    j = np.searchsorted(goal_indices_sorted, cur_idx, side="right")
    if j >= len(goal_indices_sorted):
        return 0
    return 1 if goal_indices_sorted[j] <= (cur_idx + k) else 0

# =========================
# STEP FEATS (1 action -> (cat, num))
# =========================
def action_to_step(action, team_pov: int, attack_map: dict):
    """
    한 action -> (cat_id, numeric_features[])
    cat_id: type_result(+1)  (0은 PAD)
    numeric: [sx,sy,ex,ey,dx,dy,dist,angle,dt,forward,is_team]
    """
    gid = int(action["game_id"])
    period_id = int(action["period_id"])
    attack_right = attack_map.get((gid, period_id, team_pov), True)

    sx0 = safe_float(action.get("start_x", 0.0))
    sy0 = safe_float(action.get("start_y", 0.0))
    ex0 = safe_float(action.get("end_x", 0.0))
    ey0 = safe_float(action.get("end_y", 0.0))

    sx = flip_x_if_needed(sx0, attack_right)
    ex = flip_x_if_needed(ex0, attack_right)
    sy = sy0
    ey = ey0

    dx = ex - sx
    dy = ey - sy

    dist = goal_distance(ex, ey)
    ang  = goal_angle(ex, ey)

    cat = int(action["type_result"]) + 1  # PAD=0
    is_team = 1.0 if int(action["team_id"]) == int(team_pov) else 0.0
    dt = safe_float(action.get("dt", 0.0))
    forward = dx

    num = np.array([sx, sy, ex, ey, dx, dy, dist, ang, dt, forward, is_team], dtype=np.float32)
    return cat, num

# =========================
# BUILD INDEX TABLE (labels + global feats)
# =========================
def build_index_table(train: pd.DataFrame,
                      teams_by_game: dict,
                      attack_map: dict,
                      goal_codes: set,
                      own_goal_codes: set):
    """
    모든 샘플의 (gid, team, idx, y, global_feats...) 인덱스 테이블 만들기
    동시에 game별 action list(정렬) 저장
    """
    actions_by_game = {}
    index_rows = []

    game_ids = sorted(train["game_id"].unique().tolist())
    for gid in tqdm(game_ids, desc="Prepare games"):
        gid = int(gid)
        if gid not in teams_by_game:
            continue

        A, B = teams_by_game[gid]
        g = train[train["game_id"] == gid].sort_values(["t"]).reset_index(drop=True)

        # dt 계산
        if len(g) > 0:
            t_arr = g["t"].astype(int).values
            dt = np.zeros(len(g), dtype=np.float32)
            dt[1:] = np.maximum(0, t_arr[1:] - t_arr[:-1]).astype(np.float32)
            g["dt"] = dt

        goal_idx_by_team = build_goal_action_indices(g, goal_codes, own_goal_codes, teams_by_game)
        gA_set = set(goal_idx_by_team.get(A, []))
        gB_set = set(goal_idx_by_team.get(B, []))

        scoreA = 0
        scoreB = 0

        actions_list = g.to_dict(orient="records")
        actions_by_game[gid] = actions_list

        for i in range(len(actions_list)):
            # action i가 골이면 점수 반영
            if i in gA_set:
                scoreA += 1
            if i in gB_set:
                scoreB += 1

            yA = label_score_in_next_k_actions(goal_idx_by_team.get(A, []), i, HORIZON)
            yB = label_score_in_next_k_actions(goal_idx_by_team.get(B, []), i, HORIZON)

            cur_t = int(actions_list[i]["t"])
            minute = cur_t / 60.0

            poss_changed = 0
            if i - 1 >= 0:
                poss_changed = 1 if int(actions_list[i]["team_id"]) != int(actions_list[i-1]["team_id"]) else 0

            if yA == 0 and NEG_SUBSAMPLE < 1.0 and random.random() > NEG_SUBSAMPLE:
                pass
            else:
                index_rows.append([gid, A, i, int(yA), float(minute), int(scoreA - scoreB), int(poss_changed)])

            if yB == 0 and NEG_SUBSAMPLE < 1.0 and random.random() > NEG_SUBSAMPLE:
                pass
            else:
                index_rows.append([gid, B, i, int(yB), float(minute), int(scoreB - scoreA), int(poss_changed)])

    index_df = pd.DataFrame(index_rows, columns=[
        "game_id", "team_id", "action_idx", "y", "minute", "score_diff", "possession_changed"
    ])
    return actions_by_game, index_df

# =========================
# PRECOMPUTE CACHE (핵심 가속)
# =========================
def precompute_seq_cache(actions_by_game: dict, teams_by_game: dict, attack_map: dict):
    """
    (gid, team) -> (cat_full[T], num_full[T,11]) 를 한 번만 계산해서 캐시
    """
    cache = {}
    for gid in tqdm(sorted(actions_by_game.keys()), desc="Precompute sequences"):
        gid = int(gid)
        if gid not in teams_by_game:
            continue
        A, B = teams_by_game[gid]
        actions = actions_by_game[gid]
        T = len(actions)
        if T == 0:
            continue

        for team in (A, B):
            cat_full = np.empty((T,), dtype=np.int64)
            num_full = np.empty((T, 11), dtype=np.float32)
            for i, a in enumerate(actions):
                c, n = action_to_step(a, team, attack_map)
                cat_full[i] = c
                num_full[i] = n
            cache[(gid, int(team))] = (cat_full, num_full)

    return cache

# =========================
# DATASET (캐시 사용)
# =========================
class SeqGoalDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, seq_cache: dict):
        df = index_df.reset_index(drop=True)

        self.gid  = df["game_id"].values.astype(np.int64)
        self.team = df["team_id"].values.astype(np.int64)
        self.aidx = df["action_idx"].values.astype(np.int64)

        # global feats + y
        self.gfeat = df[["minute", "score_diff", "possession_changed"]].values.astype(np.float32)
        self.y = df["y"].values.astype(np.float32)

        self.seq_cache = seq_cache

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        gid = int(self.gid[idx])
        team = int(self.team[idx])
        i = int(self.aidx[idx])

        cat_full, num_full = self.seq_cache[(gid, team)]

        start = i - LOOKBACK + 1
        if start < 0:
            start = 0

        seq_cat = cat_full[start:i+1]
        seq_num = num_full[start:i+1]

        pad_len = LOOKBACK - len(seq_cat)

        cat_seq = np.zeros((LOOKBACK,), dtype=np.int64)
        num_seq = np.zeros((LOOKBACK, 11), dtype=np.float32)

        cat_seq[pad_len:] = seq_cat
        num_seq[pad_len:] = seq_num

        gfeat = self.gfeat[idx]
        y = self.y[idx]
        return cat_seq, num_seq, gfeat, y, gid

def collate_fn(batch):
    cat = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.long)          # (B,L)
    num = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)       # (B,L,11)
    g   = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)       # (B,3)
    y   = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32), dtype=torch.float32)  # (B,)
    gid = np.array([b[4] for b in batch], dtype=np.int64)
    return cat, num, g, y, gid

# =========================
# MODEL
# =========================
class LSTMGoalModel(nn.Module):
    def __init__(self, n_cat: int, emb_dim=16, hidden=64, num_layers=1, dropout=0.1, num_dim=11, gdim=3):
        super().__init__()
        self.emb = nn.Embedding(n_cat, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim + num_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + gdim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cat_seq, num_seq, gfeat):
        e = self.emb(cat_seq)                    # (B,L,emb)
        x = torch.cat([e, num_seq], dim=-1)      # (B,L,emb+num)
        _, (h, _) = self.lstm(x)                 # h: (num_layers,B,hidden)
        last = h[-1]                             # (B,hidden)
        z = torch.cat([last, gfeat], dim=-1)     # (B,hidden+gdim)
        logit = self.head(z).squeeze(1)          # (B,)
        return logit

# =========================
# METRICS
# =========================
def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)

@torch.no_grad()
def eval_model(model, loader, device, amp_enabled=False):
    model.eval()
    ys, ps = [], []
    for cat, num, g, y, _ in loader:
        cat = cat.to(device, non_blocking=True)
        num = num.to(device, non_blocking=True)
        g   = g.to(device, non_blocking=True)

        with autocast_cuda(amp_enabled):
            logit = model(cat, num, g)
            prob = torch.sigmoid(logit)

        ps.append(prob.detach().cpu().numpy())
        ys.append(y.numpy())

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)

    auc = roc_auc_score(y_all, p_all) if len(np.unique(y_all)) > 1 else float("nan")
    ll  = log_loss(y_all, np.clip(p_all, 1e-6, 1-1e-6))
    bs  = brier_score_loss(y_all, p_all)
    ece = ece_score(y_all, p_all, n_bins=10)
    return auc, ll, bs, ece, y_all, p_all

# =========================
# MAIN
# =========================
def main():
    os.makedirs(TEMP_DIR, exist_ok=True)

    # cudnn 최적화
    torch.backends.cudnn.benchmark = True

    maps, code_to_str, goal_codes, own_goal_codes, shot_codes = build_code_maps_from_json(MAP_PATH)

    train = pd.read_csv(TRAIN_PATH).copy()
    train["t"] = [match_time_seconds(p, s) for p, s in zip(train["period_id"], train["time_seconds"])]

    # dx/dy 없으면 생성
    if "dx" not in train.columns:
        train["dx"] = train["end_x"] - train["start_x"]
    if "dy" not in train.columns:
        train["dy"] = train["end_y"] - train["start_y"]

    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns], errors="ignore")

    teams_by_game = get_teams_by_game(train)
    attack_map = estimate_attack_direction(train, shot_codes)

    actions_by_game, index_df = build_index_table(train, teams_by_game, attack_map, goal_codes, own_goal_codes)

    # ✅ (핵심) 시퀀스 피처를 게임/팀별로 미리 계산해서 캐시
    seq_cache = precompute_seq_cache(actions_by_game, teams_by_game, attack_map)

    # categorical size (type_result + 1(PAD))
    max_code = int(max(maps["type_result"].values()))
    n_cat = (max_code + 1) + 1  # codes 0..max_code, shift +1 => 1..max_code+1, PAD=0

    X_groups = index_df["game_id"].values
    y_all = index_df["y"].values.astype(int)
    gkf = GroupKFold(n_splits=N_FOLDS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = (device == "cuda" and USE_AMP)
    print(f"[Device] {device} / torch.cuda.is_available()={torch.cuda.is_available()} / AMP={amp_enabled}")

    # DataLoader workers
    if AUTO_NUM_WORKERS:
        cpu = os.cpu_count() or 4
        nw = max(2, min(8, cpu // 2))
    else:
        nw = int(NUM_WORKERS)

    # Windows/환경에 따라 불안정하면 nw=0으로 두고 시작
    persistent = True if nw > 0 else False
    prefetch_factor = 2 if nw > 0 else None

    oof_pred = np.zeros(len(index_df), dtype=float)
    fold_reports = []
    fold_states = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(index_df, y_all, X_groups)):
        tr_df = index_df.iloc[tr_idx].reset_index(drop=True)
        va_df = index_df.iloc[va_idx].reset_index(drop=True)

        # pos_weight (불균형 보정) = neg/pos
        pos = tr_df["y"].sum()
        neg = len(tr_df) - pos
        pos_weight = float(neg / max(pos, 1))
        print(f"[Fold {fold}] train n={len(tr_df)} pos_rate={pos/len(tr_df):.6f} pos_weight={pos_weight:.2f}")

        train_ds = SeqGoalDataset(tr_df, seq_cache)
        valid_ds = SeqGoalDataset(va_df, seq_cache)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=nw,
            pin_memory=(device == "cuda"),
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=nw,
            pin_memory=(device == "cuda"),
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn
        )

        model = LSTMGoalModel(
            n_cat=n_cat,
            emb_dim=EMB_DIM,
            hidden=HIDDEN,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        try:
            scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)  # ✅ 어떤 버전은 device=
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)  # ✅ 구버전 fallback

        best_ll = 1e9
        best_state = None

        for ep in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            n = 0

            for cat, num, g, y, _ in tqdm(train_loader, desc=f"Fold{fold} Ep{ep}", leave=False):
                cat = cat.to(device, non_blocking=True)
                num = num.to(device, non_blocking=True)
                g   = g.to(device, non_blocking=True)
                y   = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with autocast_cuda(amp_enabled):
                    logit = model(cat, num, g)
                    loss = crit(logit, y)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.item()) * len(y)
                n += len(y)

            auc, ll, bs, ece, _, _ = eval_model(model, valid_loader, device, amp_enabled=amp_enabled)
            print(f"[Fold {fold}][Ep {ep}] train_loss={total_loss/max(n,1):.6f}  val_ll={ll:.6f}  val_bs={bs:.6f}  val_auc={auc:.5f}  val_ece={ece:.6f}")

            if ll < best_ll:
                best_ll = ll
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # best로 복원 후 OOF
        model.load_state_dict(best_state)
        auc, ll, bs, ece, yv, pv = eval_model(model, valid_loader, device, amp_enabled=amp_enabled)

        oof_pred[va_idx] = pv

        fold_reports.append({
            "fold": fold,
            "n_val": len(va_idx),
            "pos_rate_val": float(va_df["y"].mean()),
            "auc": float(auc),
            "logloss": float(ll),
            "brier": float(bs),
            "ece": float(ece),
        })
        fold_states.append(best_state)

    # ===== OOF metrics =====
    y_true = index_df["y"].values.astype(int)
    p = oof_pred

    auc_oof = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    ll_oof = log_loss(y_true, np.clip(p, 1e-6, 1-1e-6))
    bs_oof = brier_score_loss(y_true, p)
    ece_oof = ece_score(y_true, p, n_bins=10)

    # baseline(상수 확률 p0)
    p0 = float(y_true.mean())
    ll_base = log_loss(y_true, np.full_like(p, p0, dtype=float))
    bs_base = brier_score_loss(y_true, np.full_like(p, p0, dtype=float))
    nll = 1.0 - (ll_oof / ll_base) if ll_base > 0 else float("nan")
    nbs = 1.0 - (bs_oof / bs_base) if bs_base > 0 else float("nan")

    print(f"[OOF] AUC={auc_oof:.5f}  LogLoss={ll_oof:.5f}  Brier={bs_oof:.5f}  NBS={nbs:.5f}  ECE={ece_oof:.5f}")

    # 저장(아티팩트)
    artifact = {
        "config": {
            "HORIZON": HORIZON,
            "LOOKBACK": LOOKBACK,
            "EMB_DIM": EMB_DIM,
            "HIDDEN": HIDDEN,
            "NUM_LAYERS": NUM_LAYERS,
            "DROPOUT": DROPOUT,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "NEG_SUBSAMPLE": NEG_SUBSAMPLE,
            "NUM_WORKERS": nw,
            "AMP": bool(amp_enabled),
        },
        "maps": maps,
        "code_to_str": code_to_str,
        "goal_codes": sorted(list(goal_codes)),
        "own_goal_codes": sorted(list(own_goal_codes)),
        "shot_codes": sorted(list(shot_codes)),
        "n_cat": n_cat,
        "fold_states": fold_states,
        "oof_pred": oof_pred,
        "index_df_head": index_df.head(50),
        "fold_reports": fold_reports,
        "oof": {
            "auc": float(auc_oof),
            "logloss": float(ll_oof),
            "brier": float(bs_oof),
            "ece": float(ece_oof),
            "logloss_base": float(ll_base),
            "brier_base": float(bs_base),
            "NLL": float(nll),
            "NBS": float(nbs),
        }
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)

if __name__ == "__main__":
    main()
