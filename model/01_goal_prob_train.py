import os
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# =========================
# CONFIG
# =========================
TARGET_TIME_MIN = 0.2     # 타겟값 생성 feature - n분 내에 값을 쓸건지
LOOKBACK_MIN    = 0.1     # 입력값 생성 feature
N_FOLDS         = 5
SEED            = 42

DATA_PATH       = "../data/"
TRAIN_PATH      = f"{DATA_PATH}train.csv"
PREPROCESS_PATH = f"{DATA_PATH}preprocess.csv"         # 참고용(사용 안 해도 됨)
MAP_PATH        = f"{DATA_PATH}preprocess_maps.json"   # ✅ 여기서 type_result 매핑 읽음

MODEL_OUT       = f"SEED{SEED}TIME{TARGET_TIME_MIN}goal10m_model.pkl"

# Build samples 캐시 저장 폴더
TEMP_DIR        = "../temp"

# home_score/away_score는 누수 가능성이 높아 feature에서 제거
LEAKAGE_COLS = ["home_score", "away_score"]

PITCH_X_MAX = 105.0
MID_X = PITCH_X_MAX / 2.0


# =========================
# UTIL
# =========================
def match_time_seconds(period_id, time_seconds):
    return int(time_seconds + max(int(period_id) - 1, 0) * 45 * 60)

def type_name_from_type_result_str(s: str) -> str:
    return str(s).split("__", 1)[0].strip()

def build_code_maps_from_json(map_path: str):
    """
    preprocess_maps.json 에서 type_result(str->int) 매핑을 읽어
    - code_to_str: int -> str
    - code_to_type: int -> type_name
    - goal_codes / own_goal_codes / shot_codes 추출
    """
    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    if "type_result" not in maps:
        raise KeyError(f"Missing key 'type_result' in {map_path}. keys={list(maps.keys())}")

    str_to_code = maps["type_result"]  # {"Shot__Goal": 12, ...} (0-based 가정)
    code_to_str = {int(v): str(k) for k, v in str_to_code.items()}
    code_to_type = {c: type_name_from_type_result_str(s) for c, s in code_to_str.items()}

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

    return maps, code_to_str, code_to_type, goal_codes, own_goal_codes, shot_codes

def get_teams_by_game(df: pd.DataFrame):
    """
    home/away가 있으면 그걸로 game_id의 두 팀을 안정적으로 고정.
    없으면 team_id unique로 fallback.
    """
    if ("home_team_id" in df.columns) and ("away_team_id" in df.columns):
        g = df.groupby("game_id")[["home_team_id", "away_team_id"]].first()

        teams_by_game = {}
        for row in g.itertuples():
            gid = int(row.Index)  # index가 game_id
            home = int(row.home_team_id)
            away = int(row.away_team_id)
            teams_by_game[gid] = (home, away)

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
    """
    key_cols = ["game_id", "period_id", "team_id"]

    attack = {}
    df_shot = df[df["type_result"].isin(shot_codes)]
    if len(df_shot) > 0:
        m = df_shot.groupby(key_cols)["end_x"].mean()
        for k, v in m.items():
            attack[k] = bool(v > MID_X)

    # fallback: shot 없으면 전체 end_x 평균
    m2 = df.groupby(key_cols)["end_x"].mean()
    for k, v in m2.items():
        if k not in attack:
            attack[k] = bool(v > MID_X)

    return attack

def normalize_xy_df(df: pd.DataFrame, attack_map: dict):
    """
    row-wise flip. tqdm 진행률 표시.
    """
    cols = df.columns.tolist()
    out_rows = []

    it = df.itertuples(index=False, name=None)
    for row in tqdm(it, total=len(df), desc="Normalize XY", leave=False):
        r = dict(zip(cols, row))
        key = (r["game_id"], r["period_id"], r["team_id"])
        attack_right = attack_map.get(key, True)
        if not attack_right:
            r["start_x"] = PITCH_X_MAX - r["start_x"]
            r["end_x"]   = PITCH_X_MAX - r["end_x"]
            r["dx"]      = -r["dx"]
        out_rows.append(r)

    return pd.DataFrame(out_rows)

def build_goal_times(df: pd.DataFrame, goal_codes: set, own_goal_codes: set, teams_by_game: dict):
    """
    (game_id, team_id) -> sorted goal times
    own goal은 teams_by_game 기반으로 상대 팀 득점으로 귀속
    """
    g = df[df["type_result"].isin(goal_codes)][["game_id", "team_id", "t", "type_result"]].copy()

    if own_goal_codes:
        is_og = g["type_result"].isin(own_goal_codes)
        if is_og.any():
            for idx, row in g.loc[is_og].iterrows():
                gid = int(row["game_id"])
                actor = int(row["team_id"])
                if gid in teams_by_game:
                    home, away = teams_by_game[gid]
                    other = away if actor == home else home
                    g.at[idx, "team_id"] = other

    goal_times = {}
    for (gid, tid), sub in g.groupby(["game_id", "team_id"]):
        goal_times[(int(gid), int(tid))] = np.sort(sub["t"].astype(int).values)

    return goal_times

def label_score_in_future(goal_arr, t_now, horizon_sec):
    if goal_arr is None or len(goal_arr) == 0:
        return 0
    i = np.searchsorted(goal_arr, t_now, side="right")  # strictly after t_now
    if i >= len(goal_arr):
        return 0
    return 1 if goal_arr[i] <= t_now + horizon_sec else 0

def build_window_features(events, t_now, lookback_sec, type_names):
    t0 = t_now - lookback_sec
    cnt = {k: 0 for k in type_names}
    xs, dxs = [], []
    total = 0

    for e in events:
        if e["t"] <= t0:
            continue
        total += 1
        tn = e["type_name"]
        if tn in cnt:
            cnt[tn] += 1
        xs.append(e["start_x"])
        dxs.append(e["dx"])

    feat = {f"cnt_{k}": v for k, v in cnt.items()}
    feat["total_events"] = total
    feat["avg_x"] = float(np.mean(xs)) if xs else 0.0
    feat["avg_dx"] = float(np.mean(dxs)) if dxs else 0.0
    return feat

def make_team_perspective_samples(df, code_to_type, goal_times, teams_by_game, horizon_sec, lookback_sec):
    rows = []
    type_names = sorted(set(code_to_type.values()))
    game_ids = sorted(df["game_id"].unique().tolist())

    for gid in tqdm(game_ids, desc="Build samples"):
        gid = int(gid)
        if gid not in teams_by_game:
            continue
        A, B = teams_by_game[gid]
        g = df[df["game_id"] == gid].sort_values("t")

        team_events = {A: [], B: []}

        gA = goal_times.get((gid, A), np.array([], dtype=int))
        gB = goal_times.get((gid, B), np.array([], dtype=int))
        iA = iB = 0
        scoreA = scoreB = 0

        for r in g.itertuples(index=False):
            t_now = int(r.t)

            while iA < len(gA) and gA[iA] <= t_now:
                scoreA += 1; iA += 1
            while iB < len(gB) and gB[iB] <= t_now:
                scoreB += 1; iB += 1

            actor = int(r.team_id)
            tr = int(r.type_result)
            tn = code_to_type.get(tr, "UNK")

            if actor in team_events:
                team_events[actor].append({
                    "t": t_now,
                    "type_name": tn,
                    "start_x": float(r.start_x),
                    "dx": float(r.dx),
                })

            # A 관점
            yA = label_score_in_future(gA, t_now, horizon_sec)
            fA = build_window_features(team_events[A], t_now, lookback_sec, type_names)
            oA = build_window_features(team_events[B], t_now, lookback_sec, type_names)

            rows.append({
                "game_id": gid,
                "team_id": A,
                "t": t_now,
                "minute": t_now / 60.0,
                "score_diff": scoreA - scoreB,
                **{f"team_{k}": v for k, v in fA.items()},
                **{f"opp_{k}": v for k, v in oA.items()},
                "y": yA
            })

            # B 관점
            yB = label_score_in_future(gB, t_now, horizon_sec)
            fB = build_window_features(team_events[B], t_now, lookback_sec, type_names)
            oB = build_window_features(team_events[A], t_now, lookback_sec, type_names)

            rows.append({
                "game_id": gid,
                "team_id": B,
                "t": t_now,
                "minute": t_now / 60.0,
                "score_diff": scoreB - scoreA,
                **{f"team_{k}": v for k, v in fB.items()},
                **{f"opp_{k}": v for k, v in oB.items()},
                "y": yB
            })

    return pd.DataFrame(rows)

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

# ✅ early stopping을 위한 Brier metric (낮을수록 좋음)
def feval_brier(preds, dataset):
    y_true = dataset.get_label()
    bs = brier_score_loss(y_true, preds)
    return ("brier", bs, False)


# =========================
# CACHE
# =========================
def file_signature(path: str):
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}

def cache_paths():
    os.makedirs(TEMP_DIR, exist_ok=True)
    tag = f"t{TARGET_TIME_MIN}_lb{LOOKBACK_MIN}"
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

    if meta.get("TARGET_TIME_MIN") != TARGET_TIME_MIN:
        return None
    if meta.get("LOOKBACK_MIN") != LOOKBACK_MIN:
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
        "TARGET_TIME_MIN": TARGET_TIME_MIN,
        "LOOKBACK_MIN": LOOKBACK_MIN,
        "TRAIN_SIG": file_signature(TRAIN_PATH),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================
# MAIN
# =========================
def main():
    horizon_sec = TARGET_TIME_MIN * 60
    lookback_sec = LOOKBACK_MIN * 60

    maps, code_to_str, code_to_type, goal_codes, own_goal_codes, shot_codes = build_code_maps_from_json(MAP_PATH)

    # ===== 샘플 캐시 로드 시도 =====
    samples = load_samples_cache()

    if samples is None:
        train = pd.read_csv(TRAIN_PATH)

        train = train.copy()
        train["t"] = [match_time_seconds(p, s) for p, s in zip(train["period_id"], train["time_seconds"])]

        train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns], errors="ignore")

        teams_by_game = get_teams_by_game(train)

        attack_map = estimate_attack_direction(train, shot_codes)
        train = normalize_xy_df(train, attack_map)

        goal_times = build_goal_times(train, goal_codes, own_goal_codes, teams_by_game)

        samples = make_team_perspective_samples(train, code_to_type, goal_times, teams_by_game, horizon_sec, lookback_sec)

        save_samples_cache(samples)

    # ===== 학습/검증 =====
    feature_cols = [c for c in samples.columns if c not in ["y", "game_id"]]
    X = samples[feature_cols]
    y = samples["y"].astype(int).values
    groups = samples["game_id"].values

    params = dict(
        objective="binary",
        metric="None",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=SEED,
        verbosity=-1,
    )

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(samples), dtype=float)
    fold_reports = []
    models = []

    fold_iter = tqdm(
        enumerate(gkf.split(X, y, groups)),
        total=N_FOLDS,
        desc="Train CV"
    )

    for fold, (tr_idx, va_idx) in fold_iter:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva)

        model = lgb.train(
            params,
            dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            valid_names=["valid"],
            feval=feval_brier,
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(0),
            ]
        )

        pred = model.predict(Xva, num_iteration=model.best_iteration)
        oof_pred[va_idx] = pred
        models.append(model)

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
        })

    # ===== OOF metrics (마지막에만 출력) =====
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

    # 시간 구간별 Brier 저장
    df_oof = samples[["game_id", "team_id", "t", "minute"]].copy()
    df_oof["y"] = y_all
    df_oof["pred"] = pred_all

    time_brier = time_binned_brier(df_oof, bin_minutes=(0, 15, 30, 45, 60, 75, 90, 120))
    time_brier.to_csv("oof_time_brier.csv", index=False, encoding="utf-8-sig")

    # ✅ 추가 저장: OOF 샘플별 y/pred 결과
    # - 나중에 특정 "검증 경기(game_id)" 한 개만 필터해서 보면 됨
    df_oof_out = df_oof[["game_id", "team_id", "t", "minute", "y", "pred"]].copy()
    df_oof_out.to_csv("oof_predictions.csv", index=False, encoding="utf-8-sig")

    # 리포트 저장
    report = {
        "config": {
            "TARGET_TIME_MIN": TARGET_TIME_MIN,
            "LOOKBACK_MIN": LOOKBACK_MIN,
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

    # 모델 아티팩트 저장(추론 스크립트에서 사용)
    artifact = {
        "models": models,
        "feature_cols": feature_cols,
        "maps": maps,
        "code_to_str": code_to_str,
        "code_to_type": code_to_type,
        "goal_codes": sorted(list(goal_codes)),
        "own_goal_codes": sorted(list(own_goal_codes)),
        "shot_codes": sorted(list(shot_codes)),
        "TARGET_TIME_MIN": TARGET_TIME_MIN,
        "LOOKBACK_MIN": LOOKBACK_MIN,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(artifact, f)

    # ✅ 마지막 검증 결과만 출력
    print(f"[OOF] AUC={auc_oof:.5f}  LogLoss={ll_oof:.5f}  Brier={bs_oof:.5f}  NBS={nbs_oof:.5f}  ECE={ece_oof:.5f}")


if __name__ == "__main__":
    main()
