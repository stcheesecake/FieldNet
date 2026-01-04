#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import math
import random
import warnings
from contextlib import suppress
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


# =========================================================
# HPO CONFIG (YOU EDIT HERE)
# =========================================================
# =========================================================
# HPO CONFIG (YOU EDIT HERE)  ✅ 10분 내 1 trial 목표
# =========================================================
N_FUTURE_ACTIONS = 10
N_FOLDS = 3
SEED = 42

N_ROUNDS = 30   # trial 자체는 늘려도 됨 (1 trial 시간 제한이 핵심)

# ---- [START, END, STEP] ranges ----
# ✅ 핵심: n_prev가 커질수록 원핫 피처 수가 폭증 → 1~6로 강제 제한
N_PREV_ACTIONS_R = [3, 10, 1]

# ✅ depth는 GPU에서도 비용 직격탄 → 3~8 정도까지만
MAX_DEPTH_R      = [3, 8, 1]

# ✅ min_child_weight는 너무 넓게 잡으면 탐색만 낭비 → 20~120, step 20
MIN_CHILD_W_R    = [20, 120, 20]

# ✅ 시간 상한의 핵심: num_boost_round 제한 + early_stop도 상한 제한
#   - 3fold 기준: 300~1200 (step 150) 정도면 대부분 10분 내로 수렴
NUM_BOOST_R      = [300, 1500, 100]
EARLY_STOP_R     = [50, 120, 10]

# ✅ eta 너무 작으면 오래 돎 → 0.03~0.10만
ETA_R            = [3, 10, 1]
ETA_SCALE        = 0.01

# ✅ subsample/colsample 너무 촘촘하면 비효율 → 0.6~1.0, step 0.1
SUBSAMPLE_R      = [60, 100, 10]
SUBSAMPLE_SCALE  = 0.01

COLSAMPLE_R      = [60, 100, 10]
COLSAMPLE_SCALE  = 0.01

# ✅ L2는 보통 0~3.0이면 충분 (0~200은 탐색 낭비 + 경우에 따라 느려짐)
REG_LAMBDA_R     = [0, 300, 50]
REG_LAMBDA_SCALE = 0.01

# ✅ gamma/alpha는 넓게 주면 효과 대비 낭비 → 0~0.5
GAMMA_R          = [0, 50, 10]
GAMMA_SCALE      = 0.01

REG_ALPHA_R      = [0, 50, 10]
REG_ALPHA_SCALE  = 0.01

# ✅ scale_pos_weight는 학습시간엔 영향 적지만 캘리브레이션 깨고 탐색 낭비 큼 → 고정
SCALE_POS_W_R    = [1, 1, 0]

# ✅ GPU hist에서 max_bin 커지면 느려짐 → 128/256만
MAX_BIN_R        = [128, 256, 128]
# ---------------------------------------------------------

DATA_PATH  = "../../data/"
TRAIN_PATH = f"{DATA_PATH}train.csv"
MAP_PATH   = f"{DATA_PATH}preprocess_maps.json"

TEMP_DIR   = "../../../FiledNet_pkl_temp/xgboost_hpo_pkl"
OUT_DIR    = "hpo_results"

# 누수 가능 컬럼 제거
LEAKAGE_COLS = ["home_score", "away_score"]

# pitch
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0
GOAL_X      = 105.0
GOAL_Y      = 34.0
GOAL_HALF_W = 3.66
MID_X       = PITCH_X_MAX / 2.0

# GPU 사용 (GPU 없으면 False로)
USE_GPU = True


# =========================================================
# UTIL
# =========================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def match_time_seconds(period_id, time_seconds):
    return int(float(time_seconds) + max(int(period_id) - 1, 0) * 45 * 60)


def type_name_from_type_result_str(s: str) -> str:
    return str(s).split("__", 1)[0].strip()


def result_name_from_type_result_str(s: str) -> str:
    parts = str(s).split("__", 1)
    return parts[1].strip() if len(parts) == 2 else "NA"


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def build_code_maps_from_json(map_path: str):
    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    if "type_result" not in maps:
        raise KeyError(f"Missing key 'type_result' in {map_path}. keys={list(maps.keys())}")

    str_to_code = maps["type_result"]
    code_to_str = {int(v): str(k) for k, v in str_to_code.items()}
    code_to_type = {c: type_name_from_type_result_str(s) for c, s in code_to_str.items()}
    code_to_result = {c: result_name_from_type_result_str(s) for c, s in code_to_str.items()}

    goal_codes, own_goal_codes, shot_codes = set(), set(), set()
    for c, s in code_to_str.items():
        tn = code_to_type[c]
        if tn in ["Shot", "Shot_Freekick", "Shot_Corner", "Penalty Kick"]:
            shot_codes.add(c)
        if s.endswith("__Goal") or tn == "Goal":
            goal_codes.add(c)
        if tn in ["Own Goal", "Own_Goal"]:
            own_goal_codes.add(c)

    return maps, code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes, shot_codes


def get_teams_by_game(df: pd.DataFrame):
    if ("home_team_id" in df.columns) and ("away_team_id" in df.columns):
        g = df.groupby("game_id")[["home_team_id", "away_team_id"]].first()
        teams_by_game = {}
        for row in g.itertuples():
            gid = int(row.Index)
            teams_by_game[gid] = (int(row.home_team_id), int(row.away_team_id))
        return teams_by_game

    tmp = df.groupby("game_id")["team_id"].unique().to_dict()
    teams_by_game = {}
    for gid, arr in tmp.items():
        arr = list(arr)
        if len(arr) >= 2:
            teams_by_game[int(gid)] = (int(arr[0]), int(arr[1]))
    return teams_by_game


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


def flip_x_if_needed(x, attack_right: bool):
    return float(x) if attack_right else float(PITCH_X_MAX - float(x))


def goal_angle(x, y):
    x = float(x); y = float(y)
    dx = GOAL_X - x
    if dx <= 0.0001:
        return 0.0
    y1 = GOAL_Y - GOAL_HALF_W
    y2 = GOAL_Y + GOAL_HALF_W
    a1 = math.atan2(y1 - y, dx)
    a2 = math.atan2(y2 - y, dx)
    return float(abs(a2 - a1))


def goal_distance(x, y):
    x = float(x); y = float(y)
    return float(math.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2))


# =========================================================
# LABEL
# =========================================================
def build_goal_action_indices(df_g: pd.DataFrame, goal_codes: set, own_goal_codes: set, teams_by_game: dict):
    gid = int(df_g["game_id"].iloc[0])
    home, away = teams_by_game.get(gid, (None, None))

    goal_idx_by_team = {}
    gmask = df_g["type_result"].isin(goal_codes)
    idxs = np.where(gmask.values)[0]

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
    if goal_indices_sorted is None or len(goal_indices_sorted) == 0:
        return 0
    j = np.searchsorted(goal_indices_sorted, cur_idx, side="right")
    if j >= len(goal_indices_sorted):
        return 0
    return 1 if goal_indices_sorted[j] <= (cur_idx + k) else 0


# =========================================================
# FEATURE
# =========================================================
def action_row_to_basic(a, code_to_type, code_to_result):
    tr = int(a["type_result"])
    return {
        "type_name": code_to_type.get(tr, "UNK"),
        "result_name": code_to_result.get(tr, "NA"),
        "team_id": int(a["team_id"]),
        "period_id": int(a["period_id"]),
        "t": int(a["t"]),
        "start_x": safe_float(a.get("start_x", 0.0)),
        "start_y": safe_float(a.get("start_y", 0.0)),
        "end_x":   safe_float(a.get("end_x", 0.0)),
        "end_y":   safe_float(a.get("end_y", 0.0)),
        "dx":      safe_float(a.get("dx", None), default=safe_float(a.get("end_x", 0.0)) - safe_float(a.get("start_x", 0.0))),
        "dy":      safe_float(a.get("dy", None), default=safe_float(a.get("end_y", 0.0)) - safe_float(a.get("start_y", 0.0))),
    }


def build_state_features(actions_list, idx: int, team_pov: int, attack_map: dict,
                        code_to_type, code_to_result, score_diff: int,
                        n_prev_actions: int):
    feats = {}
    cur = actions_list[idx]
    gid = int(cur["game_id"])
    period_id = int(cur["period_id"])
    attack_right = attack_map.get((gid, period_id, team_pov), True)

    for k in range(n_prev_actions):
        j = idx - k
        prefix = f"a{k}_"
        if j < 0:
            feats.update({
                prefix + "exists": 0,
                prefix + "is_team": 0,
                prefix + "type": "PAD",
                prefix + "result": "PAD",
                prefix + "sx": 0.0,
                prefix + "sy": 0.0,
                prefix + "ex": 0.0,
                prefix + "ey": 0.0,
                prefix + "dx": 0.0,
                prefix + "dy": 0.0,
                prefix + "dist_ex": 0.0,
                prefix + "angle_ex": 0.0,
                prefix + "dt": 0.0,
                prefix + "forward_ex": 0.0,
            })
            continue

        a = action_row_to_basic(actions_list[j], code_to_type, code_to_result)
        sx = flip_x_if_needed(a["start_x"], attack_right)
        ex = flip_x_if_needed(a["end_x"], attack_right)
        sy = a["start_y"]
        ey = a["end_y"]

        feats[prefix + "exists"] = 1
        feats[prefix + "is_team"] = 1 if a["team_id"] == team_pov else 0
        feats[prefix + "type"] = a["type_name"]
        feats[prefix + "result"] = a["result_name"]

        feats[prefix + "sx"] = float(sx)
        feats[prefix + "sy"] = float(sy)
        feats[prefix + "ex"] = float(ex)
        feats[prefix + "ey"] = float(ey)

        dx = ex - sx
        dy = ey - sy
        feats[prefix + "dx"] = float(dx)
        feats[prefix + "dy"] = float(dy)

        feats[prefix + "dist_ex"] = goal_distance(ex, ey)
        feats[prefix + "angle_ex"] = goal_angle(ex, ey)
        feats[prefix + "forward_ex"] = float(dx)

        if j - 1 >= 0:
            t_prev = int(actions_list[j - 1]["t"])
            feats[prefix + "dt"] = float(max(0, a["t"] - t_prev))
        else:
            feats[prefix + "dt"] = 0.0

    feats["score_diff"] = int(score_diff)
    if idx - 1 >= 0:
        feats["possession_changed"] = 1 if int(actions_list[idx]["team_id"]) != int(actions_list[idx - 1]["team_id"]) else 0
    else:
        feats["possession_changed"] = 0

    return feats


def make_event_based_samples(df: pd.DataFrame,
                             teams_by_game: dict,
                             attack_map: dict,
                             code_to_type: dict,
                             code_to_result: dict,
                             goal_codes: set,
                             own_goal_codes: set,
                             k_future_actions: int,
                             n_prev_actions: int):
    rows = []
    game_ids = sorted(df["game_id"].unique().tolist())

    # ✅ 내부 tqdm 금지 (HPO tqdm 1줄만 보이게)
    for gid in game_ids:
        gid = int(gid)
        if gid not in teams_by_game:
            continue
        A, B = teams_by_game[gid]

        g = df[df["game_id"] == gid].sort_values(["t"]).reset_index(drop=True)
        goal_idx_by_team = build_goal_action_indices(g, goal_codes, own_goal_codes, teams_by_game)

        scoreA, scoreB = 0, 0
        gA_set = set(goal_idx_by_team.get(A, []))
        gB_set = set(goal_idx_by_team.get(B, []))

        actions_list = g.to_dict(orient="records")

        for i in range(len(actions_list)):
            if i in gA_set:
                scoreA += 1
            if i in gB_set:
                scoreB += 1

            yA = label_score_in_next_k_actions(goal_idx_by_team.get(A, []), i, k_future_actions)
            yB = label_score_in_next_k_actions(goal_idx_by_team.get(B, []), i, k_future_actions)

            cur_t = int(actions_list[i]["t"])
            minute = cur_t / 60.0

            fA = build_state_features(actions_list, i, A, attack_map, code_to_type, code_to_result,
                                      scoreA - scoreB, n_prev_actions=n_prev_actions)
            rows.append({
                "game_id": gid,
                "team_id": A,
                "t": cur_t,
                "minute": minute,
                "action_idx": i,
                "y": int(yA),
                **fA
            })

            fB = build_state_features(actions_list, i, B, attack_map, code_to_type, code_to_result,
                                      scoreB - scoreA, n_prev_actions=n_prev_actions)
            rows.append({
                "game_id": gid,
                "team_id": B,
                "t": cur_t,
                "minute": minute,
                "action_idx": i,
                "y": int(yB),
                **fB
            })

    return pd.DataFrame(rows)


# =========================================================
# METRICS
# =========================================================
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


# =========================================================
# CACHE (per n_prev_actions)
# =========================================================
def file_signature(path: str):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}


def cache_paths(k_future_actions: int, n_prev_actions: int):
    os.makedirs(TEMP_DIR, exist_ok=True)
    tag = f"k{k_future_actions}_prev{n_prev_actions}"
    samples_pkl = os.path.join(TEMP_DIR, f"samples_{tag}.pkl")
    meta_json   = os.path.join(TEMP_DIR, f"samples_{tag}.meta.json")
    return samples_pkl, meta_json


def load_samples_cache(k_future_actions: int, n_prev_actions: int):
    samples_pkl, meta_json = cache_paths(k_future_actions, n_prev_actions)
    if not (os.path.exists(samples_pkl) and os.path.exists(meta_json)):
        return None

    with suppress(Exception):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sig_now = file_signature(TRAIN_PATH)
        if meta.get("N_FUTURE_ACTIONS") != k_future_actions:
            return None
        if meta.get("N_PREV_ACTIONS") != n_prev_actions:
            return None

        ms = meta.get("TRAIN_SIG", {})
        if (ms.get("size") != sig_now["size"]) or (ms.get("mtime") != sig_now["mtime"]):
            return None

        return pd.read_pickle(samples_pkl)

    return None


def save_samples_cache(samples: pd.DataFrame, k_future_actions: int, n_prev_actions: int):
    samples_pkl, meta_json = cache_paths(k_future_actions, n_prev_actions)
    samples.to_pickle(samples_pkl)
    meta = {
        "N_FUTURE_ACTIONS": k_future_actions,
        "N_PREV_ACTIONS": n_prev_actions,
        "TRAIN_SIG": file_signature(TRAIN_PATH),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================================================
# RANGE → VALUES / SAMPLING
# =========================================================
def build_values(rng3, scale=1.0, as_int=False):
    """
    rng3: [start, end, step] (inclusive)
    scale: multiply AFTER generating
    """
    s, e, st = rng3
    if st == 0:
        vals = [s]
    else:
        vals = []
        v = s
        # inclusive
        while v <= e + 1e-12:
            vals.append(v)
            v += st

    if scale != 1.0:
        vals = [round(float(v) * scale, 10) for v in vals]

    if as_int:
        vals = [int(round(v)) for v in vals]

    # uniq
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


def sample_from(vals, rs: np.random.RandomState):
    return vals[int(rs.randint(0, len(vals)))]


def compact_params_str(p: dict):
    # postfix가 너무 길면 보기 힘들어서 핵심만 짧게
    return (
        f"prev={p['n_prev']},eta={p['eta']},md={p['max_depth']},"
        f"ss={p['subsample']},cs={p['colsample_bytree']},"
        f"mcw={p['min_child_weight']},l2={p['reg_lambda']},"
        f"nb={p['num_boost_round']},es={p['early_stop']}"
    )


# =========================================================
# CV EVAL (Brier 기준)
# =========================================================
def run_cv_eval(samples: pd.DataFrame, params: dict):
    # features
    base_drop = ["y", "game_id"]
    feature_df = samples.drop(columns=[c for c in base_drop if c in samples.columns], errors="ignore").copy()
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, dummy_na=False)

    X = feature_df
    y = samples["y"].astype(int).values
    groups = samples["game_id"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(samples), dtype=float)

    # XGB params
    train_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=params["eta"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],

        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        scale_pos_weight=params["scale_pos_weight"],
        max_bin=params["max_bin"],

        tree_method="hist",
        seed=SEED,
    )

    if USE_GPU:
        train_params["device"] = "cuda"
    else:
        train_params["device"] = "cpu"

    num_boost_round = params["num_boost_round"]
    early_stop = params["early_stop"]

    # fold loop (no tqdm)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=X.columns.tolist())
        dva = xgb.DMatrix(Xva, label=yva, feature_names=X.columns.tolist())

        bst = xgb.train(
            params=train_params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dva, "valid")],
            early_stopping_rounds=early_stop,
            verbose_eval=False
        )

        best_iter = getattr(bst, "best_iteration", None)
        if best_iter is not None and best_iter >= 0:
            pred = bst.predict(dva, iteration_range=(0, best_iter + 1))
        else:
            pred = bst.predict(dva)

        oof_pred[va_idx] = pred

    # OOF metrics
    pred_all = oof_pred
    y_all = y

    bs_oof = brier_score_loss(y_all, pred_all)
    auc_oof = roc_auc_score(y_all, pred_all) if len(np.unique(y_all)) > 1 else float("nan")
    ll_oof  = log_loss(y_all, np.clip(pred_all, 1e-6, 1-1e-6))

    p0_all = float(np.mean(y_all))
    bs_base_all = brier_score_loss(y_all, np.full_like(pred_all, p0_all))
    nbs_oof = 1.0 - (bs_oof / bs_base_all) if bs_base_all > 0 else float("nan")

    ece_oof = ece_score(y_all, pred_all, n_bins=10)

    return {
        "brier": float(bs_oof),
        "nbs": float(nbs_oof),
        "auc": float(auc_oof),
        "logloss": float(ll_oof),
        "ece": float(ece_oof),
        "pos_rate": float(p0_all),
        "n_samples": int(len(samples)),
        "n_features": int(X.shape[1]),
    }


# =========================================================
# MAIN (HPO)
# =========================================================
def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)
    set_all_seeds(SEED)

    # xgboost global verbosity off
    with suppress(Exception):
        xgb.set_config(verbosity=0)

    # maps
    maps, code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes, shot_codes = build_code_maps_from_json(MAP_PATH)

    # load train once
    train = pd.read_csv(TRAIN_PATH).copy()
    train["t"] = [match_time_seconds(p, s) for p, s in zip(train["period_id"], train["time_seconds"])]

    if "dx" not in train.columns:
        train["dx"] = train["end_x"] - train["start_x"]
    if "dy" not in train.columns:
        train["dy"] = train["end_y"] - train["start_y"]

    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns], errors="ignore")

    teams_by_game = get_teams_by_game(train)
    attack_map = estimate_attack_direction(train, shot_codes)

    # build value grids (discrete)
    prev_vals   = build_values(N_PREV_ACTIONS_R, as_int=True)
    eta_vals    = build_values(ETA_R, scale=ETA_SCALE, as_int=False)
    md_vals     = build_values(MAX_DEPTH_R, as_int=True)
    ss_vals     = build_values(SUBSAMPLE_R, scale=SUBSAMPLE_SCALE, as_int=False)
    cs_vals     = build_values(COLSAMPLE_R, scale=COLSAMPLE_SCALE, as_int=False)
    l2_vals     = build_values(REG_LAMBDA_R, scale=REG_LAMBDA_SCALE, as_int=False)
    mcw_vals    = build_values(MIN_CHILD_W_R, as_int=True)

    nb_vals     = build_values(NUM_BOOST_R, as_int=True)
    es_vals     = build_values(EARLY_STOP_R, as_int=True)

    gamma_vals  = build_values(GAMMA_R, scale=GAMMA_SCALE, as_int=False)
    alpha_vals  = build_values(REG_ALPHA_R, scale=REG_ALPHA_SCALE, as_int=False)
    spw_vals    = build_values(SCALE_POS_W_R, as_int=False)
    maxbin_vals = build_values(MAX_BIN_R, as_int=True)

    rs = np.random.RandomState(SEED)

    best = {
        "trial": -1,
        "brier": float("inf"),
        "nbs": float("-inf"),
        "params": None,
        "metrics": None,
    }

    run_tag = datetime.now().strftime("%y%m%d_%H%M%S")
    csv_path = os.path.join(OUT_DIR, f"{run_tag}_hpo.csv")

    # 혹시 같은 이름 파일이 이미 있으면 삭제(원하면 주석)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    csv_header_written = False


    trials = []



    pbar = tqdm(range(1, N_ROUNDS + 1), total=N_ROUNDS, desc="HPO", dynamic_ncols=True)
    for t_idx in pbar:
        # sample one combination
        params = {
            "n_prev":           sample_from(prev_vals, rs),
            "eta":              sample_from(eta_vals, rs),
            "max_depth":        sample_from(md_vals, rs),
            "subsample":        sample_from(ss_vals, rs),
            "colsample_bytree": sample_from(cs_vals, rs),
            "reg_lambda":       sample_from(l2_vals, rs),
            "min_child_weight": sample_from(mcw_vals, rs),
            "num_boost_round":  sample_from(nb_vals, rs),
            "early_stop":       sample_from(es_vals, rs),

            "gamma":            sample_from(gamma_vals, rs),
            "reg_alpha":        sample_from(alpha_vals, rs),
            "scale_pos_weight": sample_from(spw_vals, rs),
            "max_bin":          sample_from(maxbin_vals, rs),
        }

        # samples (cached per prev)
        samples = load_samples_cache(N_FUTURE_ACTIONS, params["n_prev"])
        if samples is None:
            samples = make_event_based_samples(
                train,
                teams_by_game=teams_by_game,
                attack_map=attack_map,
                code_to_type=code_to_type,
                code_to_result=code_to_result,
                goal_codes=goal_codes,
                own_goal_codes=own_goal_codes,
                k_future_actions=N_FUTURE_ACTIONS,
                n_prev_actions=params["n_prev"],
            )
            save_samples_cache(samples, N_FUTURE_ACTIONS, params["n_prev"])

        # CV eval
        metrics = run_cv_eval(samples, params)

        rec = {"trial": t_idx, **params, **metrics}
        trials.append(rec)
        # ✅ trial 끝날 때마다 csv에 즉시 append
        df_one = pd.DataFrame([rec])
        df_one.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            mode="a",
            header=(not csv_header_written)
        )
        csv_header_written = True

        # update best (min brier)
        if metrics["brier"] < best["brier"]:
            best.update({
                "trial": t_idx,
                "brier": metrics["brier"],
                "nbs": metrics["nbs"],
                "params": dict(params),
                "metrics": dict(metrics),
            })

        # tqdm postfix (only this line)
        pbar.set_postfix_str(
            f"best#{best['trial']} brier={best['brier']:.5f} nbs={best['nbs']:.5f} | {compact_params_str(best['params'])}"
        )

    out_best = {
        "best_trial": best["trial"],
        "best_brier": best["brier"],
        "best_nbs": best["nbs"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "config": {
            "N_FUTURE_ACTIONS": N_FUTURE_ACTIONS,
            "N_FOLDS": N_FOLDS,
            "SEED": SEED,
            "N_ROUNDS": N_ROUNDS,
        }
    }
    with open(os.path.join(OUT_DIR, "hpo_best.json"), "w", encoding="utf-8") as f:
        json.dump(out_best, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
