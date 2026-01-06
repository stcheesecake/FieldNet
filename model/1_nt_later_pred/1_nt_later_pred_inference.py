#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import random
import warnings
from contextlib import suppress
from datetime import datetime
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

from catboost import CatBoostClassifier, Pool


# =========================================================
# ✅ CONFIG (HPO처럼 여기만 바꾸면 됨)
# =========================================================
N_FUTURE_ACTIONS = 10
N_FOLDS = 5
SEED = 42
USE_GPU = True  # True면 GPU, False면 CPU

# 같은 폴더(스크립트 경로)에 둔다고 했으니 기본은 "현재 파일 기준"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH  = os.path.normpath(os.path.join(THIS_DIR, "../../data"))
TRAIN_PATH = os.path.join(DATA_PATH, "data.csv")
MAP_PATH   = os.path.join(DATA_PATH, "preprocess_maps.json")

# ✅ hpo_best.json은 "추론모델이랑 같은 경로"라고 했으니
HPO_BEST_PATH = os.path.join(THIS_DIR, "hpo_best.json")

# 캐시/출력
TEMP_DIR = os.path.normpath(os.path.join(THIS_DIR, "../../../FiledNet_pkl_temp/catboost_hpo_pkl"))
OUT_DIR  = os.path.normpath(os.path.join(THIS_DIR, "../../../FiledNet_pkl_temp/infer_results/catboost"))

LEAKAGE_COLS = ["home_score", "away_score"]

# pitch
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0
GOAL_X      = 105.0
GOAL_Y      = 34.0
GOAL_HALF_W = 3.66
MID_X       = PITCH_X_MAX / 2.0


# =========================================================
# UTIL
# =========================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def normalized_brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bs = brier_score_loss(y_true, y_prob)
    p0 = float(np.mean(y_true))
    bs_base = brier_score_loss(y_true, np.full_like(y_prob, p0, dtype=float))
    nbs = 1.0 - (bs / bs_base) if bs_base > 0 else float("nan")
    return float(nbs), float(bs), float(bs_base), float(p0)

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

    for gid in tqdm(game_ids, desc="Build samples", dynamic_ncols=True):
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
# OUTPUT: action-level meta (period_id, time_seconds, type_result)
# =========================================================
def build_action_meta(train_df: pd.DataFrame, teams_by_game: dict, code_to_str: dict):
    df = train_df.copy()
    df = df.sort_values(["game_id", "t"]).reset_index(drop=True)
    df["action_idx"] = df.groupby("game_id").cumcount()

    # type_result decoding (if numeric)
    if np.issubdtype(df["type_result"].dtype, np.number):
        df["type_result"] = df["type_result"].map(lambda x: code_to_str.get(int(x), str(int(x))) if pd.notna(x) else "NA")
    else:
        df["type_result"] = df["type_result"].astype(str)

    # team names (best effort)
    if "home_team_name" in df.columns and "away_team_name" in df.columns:
        game_tbl = df.groupby("game_id")[["home_team_name", "away_team_name"]].first().reset_index()
    else:
        # fallback empty names
        game_tbl = pd.DataFrame({
            "game_id": sorted(df["game_id"].unique().tolist()),
            "home_team_name": ["" for _ in sorted(df["game_id"].unique().tolist())],
            "away_team_name": ["" for _ in sorted(df["game_id"].unique().tolist())],
        })

    # home/away ids
    game_ids = sorted(df["game_id"].unique().tolist())
    home_ids, away_ids = [], []
    for gid in game_ids:
        h, a = teams_by_game.get(int(gid), (None, None))
        home_ids.append(h)
        away_ids.append(a)
    id_tbl = pd.DataFrame({"game_id": game_ids, "home_team_id": home_ids, "away_team_id": away_ids})

    out = df[["game_id", "action_idx", "period_id", "time_seconds", "type_result"]].copy()
    out = out.merge(id_tbl, on="game_id", how="left")
    out = out.merge(game_tbl, on="game_id", how="left")
    return out


# =========================================================
# MODEL
# =========================================================
def load_best_params(hpo_best_path: str) -> dict:
    if not os.path.exists(hpo_best_path):
        raise FileNotFoundError(f"hpo_best.json not found: {hpo_best_path}")

    with open(hpo_best_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "best_params" not in obj:
        raise KeyError(f"hpo_best.json missing best_params. keys={list(obj.keys())}")

    bp = obj["best_params"]
    if "n_prev" not in bp:
        raise KeyError(f"best_params missing n_prev. keys={list(bp.keys())}")

    return bp


def build_catboost_from_best(best_params: dict):
    task_type = "GPU" if USE_GPU else "CPU"

    model_kwargs = {}
    if task_type == "CPU" and ("rsm" in best_params):
        model_kwargs["rsm"] = best_params.get("rsm", 1.0)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=SEED,
        task_type=task_type,

        iterations=int(best_params["iterations"]),
        od_type="Iter",
        od_wait=int(best_params["od_wait"]),
        learning_rate=float(best_params["learning_rate"]),
        depth=int(best_params["depth"]),
        l2_leaf_reg=float(best_params["l2_leaf_reg"]),
        random_strength=float(best_params["random_strength"]),
        subsample=float(best_params["subsample"]),
        bootstrap_type="Bernoulli",
        min_data_in_leaf=int(best_params["min_data_in_leaf"]),
        border_count=int(best_params["border_count"]),

        verbose=False,
        allow_writing_files=False,
        use_best_model=True,
        **model_kwargs,
    )
    return model


# =========================================================
# MAIN
# =========================================================
def main():
    warnings.filterwarnings("ignore")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("[CONFIG]")
    print(" TRAIN_PATH    =", TRAIN_PATH)
    print(" MAP_PATH      =", MAP_PATH)
    print(" HPO_BEST_PATH =", HPO_BEST_PATH)
    print(" OUT_DIR       =", OUT_DIR)
    print(" TEMP_DIR      =", TEMP_DIR)
    print(" N_FUTURE      =", N_FUTURE_ACTIONS)
    print(" N_FOLDS       =", N_FOLDS)
    print(" USE_GPU       =", USE_GPU)

    best_params = load_best_params(HPO_BEST_PATH)
    n_prev = int(best_params["n_prev"])
    print("[BEST] n_prev =", n_prev)

    maps, code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes, shot_codes = build_code_maps_from_json(MAP_PATH)

    train = pd.read_csv(TRAIN_PATH).copy()
    train["t"] = [match_time_seconds(p, s) for p, s in zip(train["period_id"], train["time_seconds"])]

    if "dx" not in train.columns:
        train["dx"] = train["end_x"] - train["start_x"]
    if "dy" not in train.columns:
        train["dy"] = train["end_y"] - train["start_y"]

    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns], errors="ignore")

    teams_by_game = get_teams_by_game(train)
    attack_map = estimate_attack_direction(train, shot_codes)

    # samples cached per (k, n_prev)
    samples = load_samples_cache(N_FUTURE_ACTIONS, n_prev)
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
            n_prev_actions=n_prev,
        )
        save_samples_cache(samples, N_FUTURE_ACTIONS, n_prev)

    if len(samples) == 0:
        raise RuntimeError("No samples generated. Check train.csv required columns.")

    # build meta aligned with action_idx
    meta_df = build_action_meta(train, teams_by_game, code_to_str)

    # prepare X/y/groups
    base_drop = ["y", "game_id"]
    X = samples.drop(columns=[c for c in base_drop if c in samples.columns], errors="ignore").copy()
    y = samples["y"].astype(int).values
    groups = samples["game_id"].values

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    # OOF preds
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(samples), dtype=float)

    fold_info = []
    split_iter = list(gkf.split(X, y, groups))

    for fold, (tr_idx, va_idx) in enumerate(tqdm(split_iter, desc="OOF CV", dynamic_ncols=True)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        tr_pool = Pool(Xtr, label=ytr, cat_features=cat_idx)
        va_pool = Pool(Xva, label=yva, cat_features=cat_idx)

        model = build_catboost_from_best(best_params)
        model.fit(tr_pool, eval_set=va_pool)

        pred = model.predict_proba(va_pool)[:, 1]
        oof_pred[va_idx] = pred

        nbs, bs, bs_base, p0 = normalized_brier_score(yva, pred)
        auc = roc_auc_score(yva, pred) if len(np.unique(yva)) > 1 else float("nan")
        ll = log_loss(yva, np.clip(pred, 1e-6, 1 - 1e-6))

        fold_info.append({
            "fold": fold + 1,
            "brier": float(bs),
            "nbs": float(nbs),
            "auc": float(auc),
            "logloss": float(ll),
            "pos_rate": float(p0),
            "brier_base": float(bs_base),
        })

    nbs_all, bs_all, bs_base_all, p0_all = normalized_brier_score(y, oof_pred)
    auc_all = roc_auc_score(y, oof_pred) if len(np.unique(y)) > 1 else float("nan")
    ll_all = log_loss(y, np.clip(oof_pred, 1e-6, 1 - 1e-6))

    print(f"[OOF] brier={bs_all:.6f} nbs={nbs_all:.6f} auc={auc_all:.6f} logloss={ll_all:.6f} pos_rate={p0_all:.6f}")

    # make action-level preds (home/away per action_idx)
    oof_df = samples[["game_id", "team_id", "action_idx"]].copy()
    oof_df["pred"] = oof_pred

    home_preds = oof_df.rename(columns={"team_id": "home_team_id", "pred": "home_pred"})
    away_preds = oof_df.rename(columns={"team_id": "away_team_id", "pred": "away_pred"})

    out = meta_df.merge(
        home_preds[["game_id", "action_idx", "home_team_id", "home_pred"]],
        on=["game_id", "action_idx", "home_team_id"],
        how="left"
    ).merge(
        away_preds[["game_id", "action_idx", "away_team_id", "away_pred"]],
        on=["game_id", "action_idx", "away_team_id"],
        how="left"
    )

    out = out.rename(columns={"time_seconds": "time"})
    out_cols = ["game_id", "period_id", "time",
                "home_pred", "away_pred",
                "home_team_name", "away_team_name",
                "type_result"]
    for c in out_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[out_cols].sort_values(["game_id", "period_id", "time"], kind="mergesort").reset_index(drop=True)

    run_tag = datetime.now().strftime("%y%m%d_%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"{run_tag}_oof_preds.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] OOF preds csv -> {out_csv}")

    # train final model on all samples (for service)
    full_pool = Pool(X, label=y, cat_features=cat_idx)
    final_model = build_catboost_from_best(best_params)
    final_model.set_params(use_best_model=False)  # eval_set 없으면 의미없으니 꺼둠
    final_model.fit(full_pool)

    model_path = os.path.join(OUT_DIR, "catboost_goal_next10_full.cbm")
    final_model.save_model(model_path)
    print(f"[SAVED] final model -> {model_path}")

    meta_path = os.path.join(OUT_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "N_FUTURE_ACTIONS": int(N_FUTURE_ACTIONS),
            "N_FOLDS": int(N_FOLDS),
            "SEED": int(SEED),
            "USE_GPU": bool(USE_GPU),
            "best_params": best_params,
            "n_prev": int(n_prev),
            "feature_columns": list(X.columns),
            "cat_columns": cat_cols,
            "oof_metrics": {
                "brier": float(bs_all),
                "nbs": float(nbs_all),
                "auc": float(auc_all),
                "logloss": float(ll_all),
                "pos_rate": float(p0_all),
                "brier_base": float(bs_base_all),
                "folds": fold_info
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] model meta -> {meta_path}")


if __name__ == "__main__":
    main()
