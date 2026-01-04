#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import math
import xgboost as xgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


# =========================
# CONFIG (논문식: 다음 K actions)
# =========================
N_FUTURE_ACTIONS = 10      # ✅ 논문 VAEP 기본: 다음 10 actions 안에 득점
N_PREV_ACTIONS   = 3       # ✅ 상태는 최근 3 actions (a_i, a_{i-1}, a_{i-2})

N_FOLDS = 5
SEED    = 42

DATA_PATH       = "../data/"
TRAIN_PATH      = f"{DATA_PATH}train.csv"
MAP_PATH        = f"{DATA_PATH}preprocess_maps.json"   # type_result 매핑

MODEL_OUT       = f"SEED{SEED}_K{N_FUTURE_ACTIONS}_vaep_event_xgb.pkl"
TEMP_DIR        = "../temp"

# 누수 가능 컬럼 제거
LEAKAGE_COLS = ["home_score", "away_score"]

# pitch (StatsBomb 기준 가정)
PITCH_X_MAX = 105.0
PITCH_Y_MAX = 68.0
GOAL_X      = 105.0
GOAL_Y      = 34.0
GOAL_HALF_W = 3.66  # 7.32m / 2

MID_X = PITCH_X_MAX / 2.0


# =========================
# UTIL
# =========================
def match_time_seconds(period_id, time_seconds):
    return int(float(time_seconds) + max(int(period_id) - 1, 0) * 45 * 60)

def type_name_from_type_result_str(s: str) -> str:
    return str(s).split("__", 1)[0].strip()

def result_name_from_type_result_str(s: str) -> str:
    parts = str(s).split("__", 1)
    return parts[1].strip() if len(parts) == 2 else "NA"

def build_code_maps_from_json(map_path: str):
    """
    preprocess_maps.json 에서 type_result(str->int) 매핑을 읽어
    - code_to_str: int -> str
    - code_to_type: int -> type_name
    - code_to_result: int -> result_name
    - goal_codes / own_goal_codes / shot_codes
    """
    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    if "type_result" not in maps:
        raise KeyError(f"Missing key 'type_result' in {map_path}. keys={list(maps.keys())}")

    str_to_code = maps["type_result"]  # {"Shot__Goal": 12, ...} (0-based 가정)
    code_to_str = {int(v): str(k) for k, v in str_to_code.items()}
    code_to_type = {c: type_name_from_type_result_str(s) for c, s in code_to_str.items()}
    code_to_result = {c: result_name_from_type_result_str(s) for c, s in code_to_str.items()}

    goal_codes = set()
    own_goal_codes = set()
    shot_codes = set()

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
    """
    (game_id, period_id, team_id)별 공격 방향을 shot end_x 평균으로 추정.
    True: 오른쪽(큰 x) 공격
    """
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
    """
    x,y에서 골대(105, 34) 기준 '골대 두 포스트 사이 각도' (라디안)
    """
    x = float(x); y = float(y)
    dx = GOAL_X - x
    # dx가 0이거나 뒤쪽이면 각도 0 처리
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

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


# =========================
# EVENT-BASED LABEL (다음 K action 안에 득점)
# =========================
def build_goal_action_indices(df_g: pd.DataFrame,
                              goal_codes: set,
                              own_goal_codes: set,
                              teams_by_game: dict):
    """
    df_g: 특정 game_id의 이벤트(이미 시간 순으로 정렬된 DataFrame)
    return:
      goal_idx_by_team[(team_id)] = sorted list of action indices where that team "득점" 처리
    own goal은 상대팀 득점으로 귀속
    """
    gid = int(df_g["game_id"].iloc[0])
    home, away = teams_by_game.get(gid, (None, None))

    goal_idx_by_team = {}

    # goal로 판정되는 이벤트들
    gmask = df_g["type_result"].isin(goal_codes)
    idxs = np.where(gmask.values)[0]

    for i in idxs:
        actor = int(df_g.iloc[i]["team_id"])
        tr = int(df_g.iloc[i]["type_result"])

        # own goal이면 반대로 귀속
        if tr in own_goal_codes and (home is not None and away is not None):
            scorer = away if actor == home else home
        else:
            scorer = actor

        goal_idx_by_team.setdefault(scorer, []).append(int(i))

    for tid in list(goal_idx_by_team.keys()):
        goal_idx_by_team[tid] = sorted(goal_idx_by_team[tid])

    return goal_idx_by_team

def label_score_in_next_k_actions(goal_indices_sorted, cur_idx: int, k: int):
    """
    cur_idx 이후 (cur_idx+1 ~ cur_idx+k) 범위 안에 goal index가 있으면 1
    """
    if goal_indices_sorted is None or len(goal_indices_sorted) == 0:
        return 0
    # 첫 goal index > cur_idx
    j = np.searchsorted(goal_indices_sorted, cur_idx, side="right")
    if j >= len(goal_indices_sorted):
        return 0
    return 1 if goal_indices_sorted[j] <= (cur_idx + k) else 0


# =========================
# FEATURE: 최근 3 actions로 state 구성 (논문식)
# =========================
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
                         code_to_type, code_to_result,
                         score_diff: int):
    """
    idx 시점의 상태 S_idx를 (a_idx, a_{idx-1}, a_{idx-2})로 표현
    team_pov 관점(팀이 공격 오른쪽이 되도록 x flip)
    """
    feats = {}
    cur = actions_list[idx]
    gid = int(cur["game_id"])
    period_id = int(cur["period_id"])

    # POV팀 공격 방향
    attack_right = attack_map.get((gid, period_id, team_pov), True)

    # 최근 3개 action
    for k in range(N_PREV_ACTIONS):
        j = idx - k
        prefix = f"a{k}_"  # a0=현재, a1=직전, a2=직전2
        if j < 0:
            # padding
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

        # 좌표를 team_pov 공격방향 기준으로 정규화(논문에서 자주 하는 방식)
        sx = flip_x_if_needed(a["start_x"], attack_right)
        ex = flip_x_if_needed(a["end_x"], attack_right)
        # y는 그대로 사용(필요하면 y flip도 가능)
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

        feats[prefix + "dist_ex"]  = goal_distance(ex, ey)
        feats[prefix + "angle_ex"] = goal_angle(ex, ey)
        feats[prefix + "forward_ex"] = float(dx)  # 공격방향 기준 전진량

        # 시간 간격(현재 action 기준으로만 의미가 커서 a0에 특히 중요)
        if j - 1 >= 0:
            t_prev = int(actions_list[j - 1]["t"])
            feats[prefix + "dt"] = float(max(0, a["t"] - t_prev))
        else:
            feats[prefix + "dt"] = 0.0

    # 게임 컨텍스트 (논문도 score_diff 사용)
    feats["score_diff"] = int(score_diff)

    # possession change(간단 버전)
    if idx - 1 >= 0:
        feats["possession_changed"] = 1 if int(actions_list[idx]["team_id"]) != int(actions_list[idx - 1]["team_id"]) else 0
    else:
        feats["possession_changed"] = 0

    return feats


# =========================
# BUILD SAMPLES (team POV 2개씩)
# =========================
def make_event_based_samples(df: pd.DataFrame,
                             teams_by_game: dict,
                             attack_map: dict,
                             code_to_type: dict,
                             code_to_result: dict,
                             goal_codes: set,
                             own_goal_codes: set,
                             k_future_actions: int):
    rows = []
    game_ids = sorted(df["game_id"].unique().tolist())

    for gid in tqdm(game_ids, desc="Build samples (event-based)"):
        gid = int(gid)
        if gid not in teams_by_game:
            continue
        A, B = teams_by_game[gid]

        g = df[df["game_id"] == gid].sort_values(["t"]).reset_index(drop=True)

        # goal event indices per team (in this game's action order)
        goal_idx_by_team = build_goal_action_indices(g, goal_codes, own_goal_codes, teams_by_game)

        # score tracking by action index (누수 컬럼 대신 직접 누적)
        scoreA = 0
        scoreB = 0

        # goal indices sets (빠른 포함 검사)
        gA_set = set(goal_idx_by_team.get(A, []))
        gB_set = set(goal_idx_by_team.get(B, []))

        actions_list = g.to_dict(orient="records")

        for i in range(len(actions_list)):
            # action i가 goal이면 먼저 점수 반영 (state after action i 기준)
            if i in gA_set:
                scoreA += 1
            if i in gB_set:
                scoreB += 1

            # 라벨: 다음 K actions 안에 A가 득점? / B가 득점?
            yA = label_score_in_next_k_actions(goal_idx_by_team.get(A, []), i, k_future_actions)
            yB = label_score_in_next_k_actions(goal_idx_by_team.get(B, []), i, k_future_actions)

            cur_t = int(actions_list[i]["t"])
            minute = cur_t / 60.0

            # A POV
            fA = build_state_features(actions_list, i, A, attack_map, code_to_type, code_to_result, scoreA - scoreB)
            rows.append({
                "game_id": gid,
                "team_id": A,
                "t": cur_t,
                "minute": minute,
                "action_idx": i,
                "y": int(yA),
                **fA
            })

            # B POV
            fB = build_state_features(actions_list, i, B, attack_map, code_to_type, code_to_result, scoreB - scoreA)
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

def time_binned_brier(df_oof, bin_minutes=(0, 15, 30, 45, 60, 75, 90, 120)):
    bins = list(bin_minutes)
    out = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = df_oof[(df_oof["minute"] >= lo) & (df_oof["minute"] < hi)]
        if len(m) == 0:
            continue
        bs = brier_score_loss(m["y"].values, m["pred"].values)
        out.append({"min_lo": lo, "min_hi": hi, "n": len(m), "brier": float(bs), "pos_rate": float(m["y"].mean())})
    return pd.DataFrame(out)


# =========================
# CACHE
# =========================
def file_signature(path: str):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}

def cache_paths():
    os.makedirs(TEMP_DIR, exist_ok=True)
    tag = f"k{N_FUTURE_ACTIONS}_prev{N_PREV_ACTIONS}"
    samples_pkl = os.path.join(TEMP_DIR, f"samples_{tag}.pkl")
    meta_json   = os.path.join(TEMP_DIR, f"samples_{tag}.meta.json")
    return samples_pkl, meta_json

def load_samples_cache():
    samples_pkl, meta_json = cache_paths()
    if not (os.path.exists(samples_pkl) and os.path.exists(meta_json)):
        return None

    try:
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    sig_now = file_signature(TRAIN_PATH)
    if meta.get("N_FUTURE_ACTIONS") != N_FUTURE_ACTIONS:
        return None
    if meta.get("N_PREV_ACTIONS") != N_PREV_ACTIONS:
        return None

    ms = meta.get("TRAIN_SIG", {})
    if (ms.get("size") != sig_now["size"]) or (ms.get("mtime") != sig_now["mtime"]):
        return None

    try:
        return pd.read_pickle(samples_pkl)
    except Exception:
        return None

def save_samples_cache(samples: pd.DataFrame):
    samples_pkl, meta_json = cache_paths()
    samples.to_pickle(samples_pkl)
    meta = {
        "N_FUTURE_ACTIONS": N_FUTURE_ACTIONS,
        "N_PREV_ACTIONS": N_PREV_ACTIONS,
        "TRAIN_SIG": file_signature(TRAIN_PATH),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================
# MAIN
# =========================
def main():
    maps, code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes, shot_codes = build_code_maps_from_json(MAP_PATH)

    # ===== 샘플 캐시 =====
    samples = load_samples_cache()

    if samples is None:
        train = pd.read_csv(TRAIN_PATH).copy()

        # 필수 파생/정리
        train["t"] = [match_time_seconds(p, s) for p, s in zip(train["period_id"], train["time_seconds"])]

        # dx/dy 없으면 생성
        if "dx" not in train.columns:
            train["dx"] = train["end_x"] - train["start_x"]
        if "dy" not in train.columns:
            train["dy"] = train["end_y"] - train["start_y"]

        # 누수 컬럼 제거
        train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns], errors="ignore")

        teams_by_game = get_teams_by_game(train)
        attack_map = estimate_attack_direction(train, shot_codes)

        samples = make_event_based_samples(
            train,
            teams_by_game=teams_by_game,
            attack_map=attack_map,
            code_to_type=code_to_type,
            code_to_result=code_to_result,
            goal_codes=goal_codes,
            own_goal_codes=own_goal_codes,
            k_future_actions=N_FUTURE_ACTIONS
        )

        save_samples_cache(samples)

    # ===== Feature encoding (categorical one-hot) =====
    # y, group 제외
    base_drop = ["y", "game_id"]
    feature_df = samples.drop(columns=[c for c in base_drop if c in samples.columns], errors="ignore").copy()

    # categorical: a0_type/a0_result/... -> one-hot
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, dummy_na=False)

    feature_cols = feature_df.columns.tolist()

    X = feature_df
    y = samples["y"].astype(int).values
    groups = samples["game_id"].values

    # ===== XGBoost params =====
    train_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=50,

        # ✅ GPU
        tree_method="hist",
        device="cuda",

        seed=SEED,
    )
    NUM_BOOST_ROUND = 5000
    EARLY_STOP = 200

    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_pred = np.zeros(len(samples), dtype=float)
    fold_reports = []
    models = []

    for fold, (tr_idx, va_idx) in tqdm(enumerate(gkf.split(X, y, groups)), total=N_FOLDS, desc="Train CV (XGB)"):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=X.columns.tolist())
        dva = xgb.DMatrix(Xva, label=yva, feature_names=X.columns.tolist())

        bst = xgb.train(
            params=train_params,
            dtrain=dtr,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dva, "valid")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False
        )

        # ✅ 구버전 호환: ntree_limit 사용
        best_iter = getattr(bst, "best_iteration", None)
        if best_iter is not None and best_iter >= 0:
            pred = bst.predict(dva, iteration_range=(0, best_iter + 1))
        else:
            pred = bst.predict(dva)

        oof_pred[va_idx] = pred
        models.append(bst)

        auc = roc_auc_score(yva, pred) if len(np.unique(yva)) > 1 else float("nan")
        ll  = log_loss(yva, np.clip(pred, 1e-6, 1-1e-6))
        bs  = brier_score_loss(yva, pred)

        p0 = float(np.mean(yva))
        ll_base = log_loss(yva, np.full_like(pred, p0))
        bs_base = brier_score_loss(yva, np.full_like(pred, p0))

        nll = 1.0 - (ll / ll_base) if ll_base > 0 else float("nan")
        nbs = 1.0 - (bs / bs_base) if bs_base > 0 else float("nan")
        ece = ece_score(yva, pred, n_bins=10)

        fold_reports.append({
            "fold": int(fold),
            "n_val": int(len(va_idx)),
            "pos_rate_val": float(np.mean(yva)),
            "auc": float(auc),
            "logloss": float(ll),
            "brier": float(bs),
            "logloss_base": float(ll_base),
            "brier_base": float(bs_base),
            "NLL(1-LL/LLb)": float(nll),
            "NBS(1-BS/BSb)": float(nbs),
            "ECE(10bins)": float(ece),
            "best_iteration": int(getattr(bst, "best_iteration", -1)),
        })

    # ===== OOF metrics =====
    y_all = y
    pred_all = oof_pred

    auc_oof = roc_auc_score(y_all, pred_all) if len(np.unique(y_all)) > 1 else float("nan")
    ll_oof  = log_loss(y_all, np.clip(pred_all, 1e-6, 1-1e-6))
    bs_oof  = brier_score_loss(y_all, pred_all)

    p0_all = float(np.mean(y_all))
    ll_base_all = log_loss(y_all, np.full_like(pred_all, p0_all))
    bs_base_all = brier_score_loss(y_all, np.full_like(pred_all, p0_all))
    nll_oof = 1.0 - (ll_oof / ll_base_all) if ll_base_all > 0 else float("nan")
    nbs_oof = 1.0 - (bs_oof / bs_base_all) if bs_base_all > 0 else float("nan")
    ece_oof = ece_score(y_all, pred_all, n_bins=10)

    # 저장: 시간 구간별 Brier
    df_oof = samples[["game_id", "team_id", "t", "minute", "action_idx"]].copy()
    df_oof["y"] = y_all
    df_oof["pred"] = pred_all

    time_brier = time_binned_brier(df_oof, bin_minutes=(0, 15, 30, 45, 60, 75, 90, 120))
    time_brier.to_csv("oof_time_brier.csv", index=False, encoding="utf-8-sig")

    # 저장: OOF 전체
    df_oof_out = df_oof[["game_id", "team_id", "action_idx", "t", "minute", "y", "pred"]].copy()
    df_oof_out.to_csv("oof_predictions.csv", index=False, encoding="utf-8-sig")

    # 리포트 저장
    report = {
        "config": {
            "N_FUTURE_ACTIONS": N_FUTURE_ACTIONS,
            "N_PREV_ACTIONS": N_PREV_ACTIONS,
            "N_FOLDS": N_FOLDS,
            "SEED": SEED
        },
        "folds": fold_reports,
        "oof": {
            "auc": float(auc_oof),
            "logloss": float(ll_oof),
            "brier": float(bs_oof),
            "logloss_base": float(ll_base_all),
            "brier_base": float(bs_base_all),
            "NLL(1-LL/LLb)": float(nll_oof),
            "NBS(1-BS/BSb)": float(nbs_oof),
            "ECE(10bins)": float(ece_oof),
        }
    }
    with open("cv_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 모델 아티팩트 저장 (추론에서 동일하게 사용 가능)
    artifact = {
        "models_raw": [m.save_raw() for m in models],
        "feature_cols": feature_cols,
        "maps": maps,
        "code_to_str": code_to_str,
        "code_to_type": code_to_type,
        "code_to_result": code_to_result,
        "goal_codes": sorted(list(goal_codes)),
        "own_goal_codes": sorted(list(own_goal_codes)),
        "shot_codes": sorted(list(shot_codes)),
        "N_FUTURE_ACTIONS": N_FUTURE_ACTIONS,
        "N_PREV_ACTIONS": N_PREV_ACTIONS,
        "PITCH": {
            "PITCH_X_MAX": PITCH_X_MAX,
            "PITCH_Y_MAX": PITCH_Y_MAX,
            "GOAL_X": GOAL_X,
            "GOAL_Y": GOAL_Y
        }
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[OOF] AUC={auc_oof:.5f}  LogLoss={ll_oof:.5f}  Brier={bs_oof:.5f}  NBS={nbs_oof:.5f}  ECE={ece_oof:.5f}")


if __name__ == "__main__":
    main()
