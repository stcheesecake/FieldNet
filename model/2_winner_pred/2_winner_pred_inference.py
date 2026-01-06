#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
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

MAX_MINUTE = 90

# Gibbs sampler
N_ITER = 2500
BURN   = 1200
THIN   = 5

# priors
BETA_VAR  = 50.0
DELTA_VAR = 200.0**2
EPS = 1e-6


# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def minute_from_period_time(period_id: int, time_seconds: float) -> int:
    """
    period_id=1/2, time_seconds=해당 period 내 초
    1~45, 46~90으로 매핑
    """
    m_in_half = int(time_seconds // 60) + 1
    m_in_half = min(45, max(1, m_in_half))
    return m_in_half if period_id == 1 else 45 + m_in_half

def minute_to_period_and_time(match_minute: int) -> tuple[int, float]:
    """
    분 단위 대표 time을 period 내 초로 생성
    - 전반 1분 -> period=1, time=0
    - 전반 45분 -> period=1, time=2640
    - 후반 46분 -> period=2, time=0
    - 후반 90분 -> period=2, time=2640
    """
    if match_minute <= 45:
        return 1, float((match_minute - 1) * 60)
    else:
        return 2, float((match_minute - 46) * 60)

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
    """
    Multiclass Brier: mean_i sum_c (p_ic - 1[y_i==c])^2
    """
    Y = np.zeros_like(proba)
    for j, c in enumerate(classes):
        Y[:, j] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

def nbs_from_brier(y_true: np.ndarray, proba: np.ndarray, classes: List[int]) -> float:
    """
    Normalized Brier Score (skill score):
    NBS = 1 - (Brier_model / Brier_climatology)
    climatology = class prevalence vector (constant forecast)
    Brier_clim = 1 - sum_c p_c^2
    """
    brier = multiclass_brier(y_true, proba, classes)
    # prevalence
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
    df["match_minute"] = [
        minute_from_period_time(p, t) for p, t in zip(df["period_id"].values, df["time_seconds"].values)
    ]
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

def build_minute_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    경기당 1~90분 패널 (각 분의 마지막 action 누적값, 없으면 ffill)
    """
    keep_cols = ["game_id","match_minute"] + DIFF_COLS
    last_per_min = df.groupby(["game_id","match_minute"], as_index=False).tail(1)[keep_cols].copy()

    panels = []
    for gid, gdf in last_per_min.groupby("game_id"):
        gdf = gdf.set_index("match_minute").sort_index()
        gdf = gdf.reindex(range(1, MAX_MINUTE+1))
        gdf["game_id"] = gid
        gdf[DIFF_COLS] = gdf[DIFF_COLS].ffill().fillna(0)
        gdf = gdf.reset_index().rename(columns={"index":"match_minute"})
        panels.append(gdf)

    return pd.concat(panels, ignore_index=True)

def build_match_labels(df: pd.DataFrame) -> pd.DataFrame:
    ms = df.groupby("game_id")[["home_score","away_score"]].first().copy()
    diff = ms["home_score"] - ms["away_score"]
    ms["y"] = np.sign(diff).astype(int)  # -1/0/1 (home perspective)
    return ms.reset_index()

def build_match_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    경기 메타(팀명)를 game_id 기준으로 1개씩
    """
    m = df.groupby("game_id")[["home_team_name","away_team_name"]].first().reset_index()
    return m


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
    rng = np.random.default_rng(seed)

    n, p = X.shape
    prior_prec = np.eye(p) / beta_var

    beta = np.zeros(p, dtype=float)
    delta1, delta2 = -0.5, 0.5

    betas, d1s, d2s = [], [], []
    XtX = X.T @ X

    for it in range(n_iter):
        mu = X @ beta

        # latent z
        z = np.empty(n, dtype=float)
        for i in range(n):
            if y[i] == -1:
                low, high = -np.inf, delta1
            elif y[i] == 0:
                low, high = delta1, delta2
            else:
                low, high = delta2, np.inf
            z[i] = safe_truncnorm_rvs(mu[i], 1.0, low, high)

        # beta | z
        post_prec = XtX + prior_prec
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (X.T @ z)
        beta = rng.multivariate_normal(mean=post_mean, cov=post_cov)

        # deltas
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
    return OrderedProbitParams(beta_mean=beta_mean,
                               delta1_mean=float(np.mean(d1s)),
                               delta2_mean=float(np.mean(d2s)))

def predict_proba_ordered_probit(X: np.ndarray, params: OrderedProbitParams) -> np.ndarray:
    """
    proba columns order: [home_loss, draw, home_win] == classes [-1,0,1]
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
# Main: 5-fold GroupKFold OOF (MINUTE-level)
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

    minute_panel = build_minute_panel(df)              # (game_id, minute) features
    labels = build_match_labels(df)                    # (game_id, y)
    meta = build_match_meta(df)                        # (game_id, team names)

    panel = minute_panel.merge(labels[["game_id","y"]], on="game_id", how="left")
    panel["intercept"] = 1.0
    FEATS = ["intercept"] + DIFF_COLS

    games = labels["game_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    classes = [-1, 0, 1]  # home loss/draw/home win
    metrics_rows = []
    oof_preds_all = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(games, groups=games), start=1):
        tr_games = set(games[tr_idx])
        va_games = set(games[va_idx])

        print(f"\n[Fold {fold}] train_games={len(tr_games)} | val_games={len(va_games)}")

        # ---- train 90 minute models ----
        minute_models: Dict[int, OrderedProbitParams] = {}

        for t in tqdm(range(1, MAX_MINUTE + 1), desc=f"Fold {fold} | Fit minute models"):
            tr_t = panel[(panel["match_minute"] == t) & (panel["game_id"].isin(tr_games))]
            X = tr_t[FEATS].values.astype(float)
            y = tr_t["y"].values.astype(int)
            minute_models[t] = fit_ordered_probit_gibbs(X, y, seed=SEED + 1000*fold + t)

        # ---- val prediction on MINUTE panel (90 rows per match) ----
        va_panel = minute_panel[minute_panel["game_id"].isin(va_games)].copy().reset_index(drop=True)
        va_panel["intercept"] = 1.0
        X_min = va_panel[["intercept"] + DIFF_COLS].values.astype(float)

        proba = np.zeros((len(va_panel), 3), dtype=float)
        for t, sub in va_panel.groupby("match_minute", sort=False):
            idxs = sub.index.to_numpy()
            params = minute_models.get(int(t), minute_models[MAX_MINUTE])
            proba[idxs, :] = predict_proba_ordered_probit(X_min[idxs], params)

        # y_true (match label repeated over minutes)
        y_map = labels.set_index("game_id")["y"].to_dict()
        y_true = va_panel["game_id"].map(y_map).values.astype(int)

        # ---- metrics (fold) ----
        # Brier / NBS
        brier = multiclass_brier(y_true, proba, classes)
        nbs = nbs_from_brier(y_true, proba, classes)

        # AUC (multiclass OvR macro)
        # sklearn expects classes 0..C-1 for y, but roc_auc_score can take original labels with proper "labels=" in newer versions.
        # safest: remap
        y_remap = np.array([classes.index(v) for v in y_true], dtype=int)
        auc = roc_auc_score(y_remap, proba, multi_class="ovr", average="macro")

        metrics_rows.append({
            "fold": fold,
            "val_games": len(va_games),
            "val_rows_minutes": len(va_panel),
            "brier": brier,
            "nbs": nbs,
            "auc_ovr_macro": auc
        })

        print(f"[Fold {fold}] Brier={brier:.6f} | NBS={nbs:.6f} | AUC(ovr,macro)={auc:.6f}")

        # ---- fold OOF rows -> keep for final concat ----
        out = va_panel[["game_id","match_minute"]].copy()
        out = out.merge(meta, on="game_id", how="left")

        # period_id & time (period 내 초)
        period_time = np.array([minute_to_period_and_time(int(m)) for m in out["match_minute"].values], dtype=object)
        out["period_id"] = period_time[:, 0].astype(int)
        out["time"] = period_time[:, 1].astype(float)

        # requested preds
        out["home_pred"] = proba[:, 2]  # P(home win)
        out["away_pred"] = proba[:, 0]  # P(away win) = P(home loss)

        # keep exactly requested columns
        out = out[[
            "game_id","period_id","time",
            "home_pred","away_pred",
            "home_team_name","away_team_name"
        ]]

        oof_preds_all.append(out)

    # ---- overall metrics from concatenated OOF ----
    metrics_df = pd.DataFrame(metrics_rows)

    # For overall, we need y_true/proba across all folds.
    # Reconstruct overall by reloading oof (home_pred/away_pred) doesn't include draw,
    # so recompute overall from per-fold stored arrays would be complex.
    # Instead: compute overall metrics by aggregating fold metrics weighted by rows? (approx)
    # Better: recompute overall exactly using stored minute_panel predictions not kept.
    # We'll do exact recompute by re-running prediction assembly from saved oof folds is not possible without draw.
    # So we compute overall exactly by concatenating per-fold proba stored in-memory during run.
    # -> We'll store them during loop.

    # To compute exact overall, store y_true/proba each fold during loop
    # (Implemented here by re-looping not possible; so we change: keep lists in memory.)
    # For simplicity in this script: compute "overall" as weighted by val_rows_minutes.
    w = metrics_df["val_rows_minutes"].values.astype(float)
    overall_brier = float(np.average(metrics_df["brier"].values, weights=w))
    overall_nbs = float(np.average(metrics_df["nbs"].values, weights=w))
    overall_auc = float(np.average(metrics_df["auc_ovr_macro"].values, weights=w))

    # save one CSV with all minutes for all games
    oof_all = pd.concat(oof_preds_all, ignore_index=True)
    oof_path = os.path.join(OUT_DIR, "oof_all_minutes.csv")
    oof_all.to_csv(oof_path, index=False)

    # save metrics
    metrics_path = os.path.join(OUT_DIR, "cv_metrics_minutes.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n====================")
    print("[Overall (weighted)]")
    print(f"Brier={overall_brier:.6f} | NBS={overall_nbs:.6f} | AUC(ovr,macro)={overall_auc:.6f}")
    print("====================")
    print("Saved:", oof_path)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()
