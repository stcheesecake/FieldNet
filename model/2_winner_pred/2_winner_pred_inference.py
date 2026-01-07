#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import truncnorm, norm

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATA_CSV = "../../data/data.csv"
MAP_JSON = "../../data/preprocess_maps.json"

OUT_DIR = "../../../FiledNet_pkl_temp/results/results_bayes_winprob"
N_FOLDS = 5
SEED = 42

# ✅ Option B: 전/후반 "하프 내 분(minute_in_half)" 그대로 사용
# 안전장치(너무 긴 outlier 방지). 필요하면 90까지 올려도 됨.
CAP_MINUTES_PER_HALF = 70

# Gibbs sampler
N_ITER = 2500
BURN   = 1200
THIN   = 5

# priors
BETA_VAR  = 50.0
DELTA_VAR = 200.0**2
EPS = 1e-6

# backoff settings
MIN_TRAIN_SAMPLES = 15     # 이 분의 학습샘플 수가 너무 적으면 이전 분 모델 복사
LATE_MIN_START = 50        # "막판" 백오프 로그 집계 기준

# save options
SAVE_DRAW_PROB = True     # True면 draw_pred도 CSV에 추가 저장

# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def minute_in_half_from_time(time_seconds: float) -> int:
    m = int(time_seconds // 60) + 1
    return max(1, m)

def safe_truncnorm_rvs(mean: float, sd: float, low: float, high: float) -> float:
    if not np.isfinite(low):
        a = -np.inf
    else:
        a = (low - mean) / sd
    if not np.isfinite(high):
        b = np.inf
    else:
        b = (high - mean) / sd

    if (np.isfinite(a) and np.isfinite(b) and a >= b) or (low >= high):
        if np.isfinite(low) and np.isfinite(high):
            return float(np.clip(mean, low + EPS, high - EPS))
        return float(mean)

    return float(truncnorm.rvs(a=a, b=b, loc=mean, scale=sd))

def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: List[int]) -> float:
    Y = np.zeros_like(proba)
    for j, c in enumerate(classes):
        Y[:, j] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

def nbs_from_brier(y_true: np.ndarray, proba: np.ndarray, classes: List[int]) -> float:
    brier = multiclass_brier(y_true, proba, classes)
    ps = np.array([(y_true == c).mean() for c in classes], dtype=float)
    brier_clim = 1.0 - float(np.sum(ps ** 2))
    if brier_clim <= 0:
        return 0.0
    return float(1.0 - (brier / brier_clim))


# =========================================================
# Event mapping
# =========================================================
def decode_type_result(df: pd.DataFrame, inv_map: Dict[int, str]) -> pd.DataFrame:
    s = df["type_result"].map(inv_map)
    tmp = s.str.rsplit("__", n=1, expand=True)
    df["type_name"] = tmp[0]
    df["result_name"] = tmp[1].fillna("NA")
    return df

def build_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_home_action"] = (df["team_id"] == df["home_team_id"]).astype(np.int8)
    df["H_side"] = df["is_home_action"]
    df["A_side"] = (1 - df["is_home_action"]).astype(np.int8)

    tn = df["type_name"].astype(str)
    rn = df["result_name"].astype(str)

    is_goal = (tn == "Goal") | ((tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) & (rn == "Goal"))
    is_own_goal = (tn == "Own Goal")

    is_shot_on = (tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) & (rn.isin(["On Target", "Keeper Rush-Out"]))
    is_shot_off = (
        (tn.isin(["Shot", "Shot_Freekick", "Penalty Kick"])) &
        (rn.isin(["Off Target", "Blocked", "Low Quality Shot"]))
    ) | (tn.isin(["Goal Miss", "Goal Post"]))

    is_red = tn.isin(["Foul", "Handball_Foul"]) & rn.isin(["Direct_Red_Card", "Second_Yellow_Card"])
    is_yellow = tn.isin(["Foul", "Handball_Foul"]) & (rn == "Yellow_Card")

    is_corner = (tn == "Pass_Corner")
    is_cross  = (tn == "Cross")
    is_foul   = tn.isin(["Foul", "Handball_Foul", "Foul_Throw"])

    H = df["H_side"].values.astype(np.int8)
    A = df["A_side"].values.astype(np.int8)

    for ev in ["goal", "shot_on", "shot_off", "red", "yellow", "corner", "cross", "foul"]:
        df[f"H_{ev}"] = 0
        df[f"A_{ev}"] = 0

    def set_side(ev_mask: np.ndarray, colH: str, colA: str, flip: bool = False):
        if not flip:
            df.loc[ev_mask, colH] = H[ev_mask]
            df.loc[ev_mask, colA] = A[ev_mask]
        else:
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
    df["minute_in_half"] = [minute_in_half_from_time(t) for t in df["time_seconds"].values]
    df = df.sort_values(["game_id","period_id","time_seconds","team_id"], kind="mergesort").reset_index(drop=True)
    df["action_idx"] = df.groupby("game_id").cumcount()
    return df

def add_cumulative_counts(df: pd.DataFrame) -> pd.DataFrame:
    for c in EVENT_COLS:
        df[c] = df.groupby("game_id")[c].cumsum().astype(np.int16)

    df["goal_diff"]     = (df["H_goal"]     - df["A_goal"]).astype(np.int16)
    df["shot_on_diff"]  = (df["H_shot_on"]  - df["A_shot_on"]).astype(np.int16)
    df["shot_off_diff"] = (df["H_shot_off"] - df["A_shot_off"]).astype(np.int16)
    df["red_diff"]      = (df["H_red"]      - df["A_red"]).astype(np.int16)
    df["yellow_diff"]   = (df["H_yellow"]   - df["A_yellow"]).astype(np.int16)
    df["corner_diff"]   = (df["H_corner"]   - df["A_corner"]).astype(np.int16)
    df["cross_diff"]    = (df["H_cross"]    - df["A_cross"]).astype(np.int16)
    df["foul_diff"]     = (df["H_foul"]     - df["A_foul"]).astype(np.int16)

    return df

def build_game_period_max(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    ✅ 핵심: (game_id, period_id)별 '실제 존재하는' 최대 minute_in_half
    cap으로 상한 적용.
    """
    gp = df.groupby(["game_id","period_id"])["minute_in_half"].max().reset_index()
    gp["max_minute_in_half"] = gp["minute_in_half"].astype(int).clip(upper=cap)
    gp = gp.drop(columns=["minute_in_half"])
    return gp

def build_minute_panel(df: pd.DataFrame, gp_max: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ '경기 끝난 뒤 분'은 아예 만들지 않음.
    각 (game, period)은 1..max_minute_in_half까지만 reindex + ffill.
    """
    keep_cols = ["game_id","period_id","minute_in_half"] + DIFF_COLS

    last_per_min = (
        df.groupby(["game_id","period_id","minute_in_half"], as_index=False)
          .tail(1)[keep_cols]
          .copy()
    )

    # max join
    last_per_min = last_per_min.merge(gp_max, on=["game_id","period_id"], how="left")

    panels = []
    for (gid, pid), gdf in last_per_min.groupby(["game_id","period_id"]):
        max_m = int(gdf["max_minute_in_half"].iloc[0])
        gdf = gdf.drop(columns=["max_minute_in_half"])

        gdf = gdf.set_index("minute_in_half").sort_index()
        gdf = gdf.reindex(range(1, max_m + 1))  # ✅ 경기 실제 마지막 분까지만
        gdf["game_id"] = gid
        gdf["period_id"] = int(pid)
        gdf[DIFF_COLS] = gdf[DIFF_COLS].ffill().fillna(0)
        gdf = gdf.reset_index().rename(columns={"index": "minute_in_half"})
        panels.append(gdf)

    return pd.concat(panels, ignore_index=True)

def build_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    ms = df.groupby("game_id")[["home_score","away_score"]].first().copy()
    diff = ms["home_score"] - ms["away_score"]
    ms["y"] = np.sign(diff).astype(int)  # -1/0/1 (home perspective)
    return ms.reset_index()

def build_match_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("game_id")[["home_team_name","away_team_name"]].first().reset_index()


# =========================================================
# Bayesian Ordered Probit (Gibbs)
# =========================================================
@dataclass
class OrderedProbitParams:
    beta_mean: np.ndarray
    delta1_mean: float
    delta2_mean: float

def fit_ordered_probit_gibbs(
    X: np.ndarray, y: np.ndarray,
    n_iter=N_ITER, burn=BURN, thin=THIN,
    beta_var=BETA_VAR, delta_var=DELTA_VAR,
    seed=SEED
) -> OrderedProbitParams:
    rng = np.random.default_rng(seed)

    n, p = X.shape
    prior_prec = np.eye(p) / beta_var

    beta = np.zeros(p, dtype=float)
    delta1, delta2 = -0.5, 0.5

    betas, d1s, d2s = [], [], []
    XtX = X.T @ X

    for it in range(n_iter):
        mu = X @ beta

        z = np.empty(n, dtype=float)
        for i in range(n):
            if y[i] == -1:
                low, high = -np.inf, delta1
            elif y[i] == 0:
                low, high = delta1, delta2
            else:
                low, high = delta2, np.inf
            z[i] = safe_truncnorm_rvs(mu[i], 1.0, low, high)

        post_prec = XtX + prior_prec
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ z)
        beta = rng.multivariate_normal(mean=post_mean, cov=post_cov)

        sd_delta = np.sqrt(delta_var)
        z_m1 = z[y == -1]
        z_0  = z[y == 0]
        z_p1 = z[y == 1]

        low1 = np.max(z_m1) if len(z_m1) else -np.inf
        up1a = np.min(z_0)  if len(z_0)  else delta2 - 0.1
        high1 = min(up1a, delta2 - EPS)
        delta1 = safe_truncnorm_rvs(0.0, sd_delta, low1, high1)

        low2a = np.max(z_0) if len(z_0) else delta1 + 0.1
        low2 = max(low2a, delta1 + EPS)
        high2 = np.min(z_p1) if len(z_p1) else np.inf
        delta2 = safe_truncnorm_rvs(0.0, sd_delta, low2, high2)

        if it >= burn and ((it - burn) % thin == 0):
            betas.append(beta.copy())
            d1s.append(delta1)
            d2s.append(delta2)

    beta_mean = np.mean(np.stack(betas, axis=0), axis=0)
    return OrderedProbitParams(
        beta_mean=beta_mean,
        delta1_mean=float(np.mean(d1s)),
        delta2_mean=float(np.mean(d2s))
    )

def predict_proba_ordered_probit(X: np.ndarray, params: OrderedProbitParams) -> np.ndarray:
    """
    proba columns: [home_loss, draw, home_win] == classes [-1,0,1]
    """
    mu = X @ params.beta_mean
    d1, d2 = params.delta1_mean, params.delta2_mean

    p_loss = norm.cdf(d1 - mu)
    p_draw = norm.cdf(d2 - mu) - norm.cdf(d1 - mu)
    p_win  = 1.0 - norm.cdf(d2 - mu)

    proba = np.vstack([p_loss, p_draw, p_win]).T
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


# =========================================================
# Main: 5-fold GroupKFold OOF (Option B improved)
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(DATA_CSV)

    with open(MAP_JSON, "r", encoding="utf-8") as f:
        maps = json.load(f)
    inv_type = {v: k for k, v in maps["type_result"].items()}

    # preprocess
    df = decode_type_result(df, inv_type)
    df = add_time_and_sort(df)
    df = build_event_flags(df)
    df = add_cumulative_counts(df)

    # ✅ 경기/하프별 실제 마지막 분 (cap 적용)
    gp_max = build_game_period_max(df, CAP_MINUTES_PER_HALF)

    # ✅ minute panel: 경기 끝난 뒤 분은 아예 생성 안 함
    minute_panel = build_minute_panel(df, gp_max)

    labels = build_match_labels(df)
    meta = build_match_meta(df)

    panel = minute_panel.merge(labels[["game_id","y"]], on="game_id", how="left")
    panel["intercept"] = 1.0
    FEATS = ["intercept"] + DIFF_COLS

    games = labels["game_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    classes = [-1, 0, 1]
    metrics_rows = []
    oof_rows = []

    all_y_true = []
    all_proba = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(games, groups=games), start=1):
        tr_games = set(games[tr_idx])
        va_games = set(games[va_idx])

        print(f"\n[Fold {fold}] train_games={len(tr_games)} | val_games={len(va_games)}")

        # ✅ fold별 학습 범위(훈련 경기에서 실제 존재하는 최대 minute_in_half)
        df_tr = df[df["game_id"].isin(tr_games)]
        max_train_by_period = df_tr.groupby("period_id")["minute_in_half"].max().to_dict()
        max_train_by_period = {
            1: int(min(max_train_by_period.get(1, 45), CAP_MINUTES_PER_HALF)),
            2: int(min(max_train_by_period.get(2, 45), CAP_MINUTES_PER_HALF)),
        }
        print(f"[Fold {fold}] max_train_minutes_per_half={max_train_by_period}")

        models: Dict[Tuple[int, int], OrderedProbitParams] = {}

        backoff_minutes = {1: [], 2: []}
        backoff_late = {1: [], 2: []}

        for pid in [1, 2]:
            max_m = max_train_by_period.get(pid, 45)

            for m in tqdm(range(1, max_m + 1), desc=f"Fold {fold} | Fit models (period={pid})"):
                tr_t = panel[
                    (panel["game_id"].isin(tr_games)) &
                    (panel["period_id"] == pid) &
                    (panel["minute_in_half"] == m)
                ]
                # 샘플 부족이면 이전 분 모델 복사(backoff)
                if len(tr_t) < MIN_TRAIN_SAMPLES and (pid, m - 1) in models:
                    models[(pid, m)] = models[(pid, m - 1)]
                    backoff_minutes[pid].append(m)
                    if m >= LATE_MIN_START:
                        backoff_late[pid].append(m)
                    continue

                X = tr_t[FEATS].values.astype(float)
                y = tr_t["y"].values.astype(int)

                models[(pid, m)] = fit_ordered_probit_gibbs(
                    X, y,
                    seed=SEED + 1000*fold + pid*100 + m
                )

        # backoff logs
        for pid in [1, 2]:
            bo = backoff_minutes[pid]
            bo_late = backoff_late[pid]
            print(f"[Fold {fold}] period={pid} backoff={len(bo)} minutes"
                  f" | late_backoff(>= {LATE_MIN_START})={len(bo_late)}"
                  f"{' | mins=' + str(bo_late[:20]) + ('...' if len(bo_late)>20 else '') if len(bo_late)>0 else ''}")

        # ---- Validation prediction on minute_panel (이미 '실제 마지막 분까지만' 포함) ----
        va_panel = minute_panel[minute_panel["game_id"].isin(va_games)].copy().reset_index(drop=True)
        va_panel["intercept"] = 1.0
        X_min = va_panel[["intercept"] + DIFF_COLS].values.astype(float)

        proba = np.zeros((len(va_panel), 3), dtype=float)

        for (pid, m), sub in va_panel.groupby(["period_id","minute_in_half"], sort=False):
            pid = int(pid)
            m = int(m)
            idxs = sub.index.to_numpy()

            max_m = max_train_by_period.get(pid, 45)
            use_m = m if m <= max_m else max_m  # 훈련에 없는 분이면 마지막 분 모델로

            # 혹시 어떤 이유로 models가 비어있을 때 대비
            if (pid, use_m) not in models:
                # 최소한 (pid,1)이 있길 기대, 없으면 그냥 스킵 방지용
                use_m = 1
            params = models[(pid, use_m)]
            proba[idxs, :] = predict_proba_ordered_probit(X_min[idxs], params)

        y_map = labels.set_index("game_id")["y"].to_dict()
        y_true = va_panel["game_id"].map(y_map).values.astype(int)

        brier = multiclass_brier(y_true, proba, classes)
        nbs = nbs_from_brier(y_true, proba, classes)
        y_remap = np.array([classes.index(v) for v in y_true], dtype=int)
        auc = roc_auc_score(y_remap, proba, multi_class="ovr", average="macro")

        metrics_rows.append({
            "fold": fold,
            "val_games": len(va_games),
            "val_rows_minutes": len(va_panel),
            "brier": brier,
            "nbs": nbs,
            "auc_ovr_macro": auc,
            "backoff_p1": len(backoff_minutes[1]),
            "backoff_p2": len(backoff_minutes[2]),
            "late_backoff_p1": len(backoff_late[1]),
            "late_backoff_p2": len(backoff_late[2]),
        })

        print(f"[Fold {fold}] Brier={brier:.6f} | NBS={nbs:.6f} | AUC(ovr,macro)={auc:.6f}")

        all_y_true.append(y_true)
        all_proba.append(proba)

        # ---- OOF output (요청 컬럼 그대로) ----
        out = va_panel[["game_id","period_id","minute_in_half"]].copy()
        out = out.merge(meta, on="game_id", how="left")

        out["time"] = (out["minute_in_half"].astype(int) - 1) * 60.0

        out["home_pred"] = proba[:, 2]  # P(home win)
        out["away_pred"] = proba[:, 0]  # P(away win) = P(home loss)
        if SAVE_DRAW_PROB:
            out["draw_pred"] = proba[:, 1]

        keep_cols = [
            "game_id","period_id","time",
            "home_pred","away_pred",
            "home_team_name","away_team_name"
        ]
        if SAVE_DRAW_PROB:
            keep_cols.insert(5, "draw_pred")  # home_pred, away_pred 사이/옆이든 취향대로

        out = out[keep_cols]
        oof_rows.append(out)

    # ---- Save outputs ----
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(OUT_DIR, "cv_metrics_minutes_ext.csv")
    metrics_df.to_csv(metrics_path, index=False)

    oof_all = pd.concat(oof_rows, ignore_index=True)
    oof_path = os.path.join(OUT_DIR, "oof_all_minutes_ext.csv")
    oof_all.to_csv(oof_path, index=False)

    # ---- Overall (exact) metrics ----
    y_all = np.concatenate(all_y_true, axis=0)
    proba_all = np.concatenate(all_proba, axis=0)

    overall_brier = multiclass_brier(y_all, proba_all, classes)
    overall_nbs = nbs_from_brier(y_all, proba_all, classes)
    y_all_remap = np.array([classes.index(v) for v in y_all], dtype=int)
    overall_auc = roc_auc_score(y_all_remap, proba_all, multi_class="ovr", average="macro")

    print("\n====================")
    print("[Overall (exact OOF)]")
    print(f"Brier={overall_brier:.6f} | NBS={overall_nbs:.6f} | AUC(ovr,macro)={overall_auc:.6f}")
    print("====================")
    print("Saved:", oof_path)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
