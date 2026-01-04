#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time (action-step) match outcome forecasting
- Sliding window: USER 방식(직전 N분 / 최근 K액션)으로 "입력"을 만든다.
- Model 이후: 논문(Ordered Probit + Bayesian + time-varying 이벤트 효과) 방식으로 설계한다.
- HPO: 첨부한 hpo_catboost.py 스타일(랜덤서치 + GroupKFold + trial별 CSV 저장)을 따른다.

논문 핵심(구현 반영):
- 결과 y ∈ {-1,0,1} (home perspective: loss/draw/win)
- latent Π = Xβ + ε, ε ~ N(0,1)
- cutoffs δ1 < δ2 로 y 매핑(ordered probit)
- 이벤트 타입별 "분(minute)별 카운트"를 covariate로 사용
- time-varying effect: 이벤트 타입 k의 계수는 minute index(1..90)에 따라 달라질 수 있게
  (여기서는 '분별 계수 벡터'에 AR(1) 형태(ρ^{|i-j|}) prior를 둬서
   시간 상관(지수감쇠) 구조를 논문처럼 반영)

사용자 요구(반영):
- 입력은 "슬라이딩 윈도우"로만 제어(직전 N분 또는 최근 K액션)
- 예측은 액션마다(실시간) 가능하도록, 현재 minute t에서 window를 업데이트해 feature를 만든다.
  (학습 데이터는 논문처럼 '매 분 스냅샷'으로 기본 구성. 필요 시 action-snapshot도 가능)

⚠️ 데이터 컬럼이 프로젝트마다 다르므로,
    아래 CONFIG의 COL_* 매핑만 맞추면 돌아가게 "강건하게" 작성함.
"""

import os
import json
import math
import random
import warnings
from contextlib import suppress
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss as sk_logloss
from sklearn.metrics import roc_auc_score

from scipy.stats import norm

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG (YOU EDIT HERE)
# =========================================================
DATA_PATH = "../../data/train.csv"   # ✅ action log CSV
OUT_DIR   = "../../../FiledNet_pkl_temp/hpo_results/bayesian"
CACHE_DIR = "./cache_bayes_outcome"

MAX_MINUTE = 90                 # 논문: Γ={1..90}
N_FOLDS = 3
SEED = 42

# HPO trials
N_ROUNDS = 50                   # 원하면 100으로(시간/자원 고려)

# -------- Sliding Window (HPO 대상) --------
WINDOW_MODE_R = ["ACTIONS"]  # 직전 N분 or 최근 K액션
WINDOW_MIN_R  = [3, 10, 1]              # MINUTES 모드일 때 직전 N분
WINDOW_ACT_R  = [50, 200, 25]           # ACTIONS 모드일 때 최근 K액션

# -------- Event types (논문 구조) --------
# 논문 스타일: event counts by type (Shot/Cross/Foul 등)
# 네 데이터에 따라 type_name이 다르면 여기 리스트만 조정하면 됨.
EVENT_TYPES_CANDIDATES = [
    ["Shot", "Cross", "Foul", "Pass"],                 # 가벼움(추천)
    ["Shot", "Cross", "Foul", "Pass", "Duel"],
    ["Shot", "Cross", "Foul", "Pass", "Duel", "Tackle"],
]

# -------- Bayesian / Gibbs (HPO 대상) --------
# AR(1) prior correlation rho in (0,1): rho^{|i-j|} ~ exp(-ω|i-j|)
RHO_R         = [0.70, 0.98, 0.04]      # 0.70~0.98
SIGMA_BETA_R  = [0.5, 3.0, 0.5]         # 이벤트계수 prior scale
SIGMA_GAMMA_R = [0.5, 3.0, 0.5]         # 정적 covariate prior scale

GIBBS_ITERS_R = [120, 300, 30]          # Gibbs iterations
BURN_IN_FRAC  = 0.5
THINNING      = 5                       # 5면 5스텝마다 1개 보관

# -------- Snapshots --------
# 논문은 "매 분 끝"에서 covariate 기록.
# 실시간 액션 예측은 동일 feature를 action마다 갱신해서 사용 가능.
TRAIN_SNAPSHOT_MODE = "END_OF_MINUTE"   # ["END_OF_MINUTE", "ALL_ACTIONS"]


# -------- Column mapping (데이터에 맞게 수정) --------
# 필수
COL_GAME   = "game_id"
COL_TEAM   = "team_id"
COL_PERIOD = "period_id"          # 없으면 None으로 두고 아래 match_time_seconds에서 처리
COL_TIME   = "time_seconds"

# home/away 정보(있으면 가장 좋음)
COL_HOME_TEAM = "home_team_id"    # 없으면 None
COL_AWAY_TEAM = "away_team_id"    # 없으면 None

# 이벤트 타입
# (1) type_name 컬럼이 있으면 그걸 사용
COL_TYPE_NAME   = "type_name"     # 없으면 None
# (2) 없으면 type_result 같은 "Shot__Goal" 형태에서 type/result 파싱
COL_TYPE_RESULT = "type_result"   # 없으면 None

# 골 판정(가능하면 result_name / code 기반)
COL_RESULT_NAME = "result_name"   # 없으면 None

# 최종 결과 라벨
# (A) 최종 스코어 컬럼이 있으면 사용
COL_HOME_SCORE_FT = "home_score"  # 없으면 None
COL_AWAY_SCORE_FT = "away_score"  # 없으면 None
# (B) 없으면 match-level result 컬럼(예: "H","D","A")가 있으면 매핑
COL_MATCH_RESULT  = None          # 예: "match_result"  (없으면 None)

# 좌표/instant raw feature를 논문 구조에 "추가 covariate"로 넣고 싶으면 True
# (정확히 논문만 하려면 False)
USE_INSTANT_RAW = False
COL_START_X = "start_x"
COL_START_Y = "start_y"
COL_END_X   = "end_x"
COL_END_Y   = "end_y"


# =========================================================
# Utils
# =========================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_values(rng, as_int=False):
    a, b, s = rng
    vals = []
    x = a
    while x <= b + 1e-12:
        vals.append(int(x) if as_int else float(x))
        x += s
    return vals


def match_time_seconds(period_id, time_seconds):
    # period_id가 없으면 그냥 time_seconds 사용
    if period_id is None or (isinstance(period_id, float) and np.isnan(period_id)):
        return int(float(time_seconds))
    return int(float(time_seconds) + max(int(period_id) - 1, 0) * 45 * 60)


def parse_type_result(s: str):
    # "Shot__Goal" -> ("Shot","Goal")
    s = str(s)
    if "__" in s:
        a, b = s.split("__", 1)
        return a.strip(), b.strip()
    return s.strip(), ""


def infer_teams_by_game(df: pd.DataFrame):
    # 1) home/away 컬럼 있으면 사용
    if COL_HOME_TEAM and COL_AWAY_TEAM and (COL_HOME_TEAM in df.columns) and (COL_AWAY_TEAM in df.columns):
        g = df.groupby(COL_GAME)[[COL_HOME_TEAM, COL_AWAY_TEAM]].first()
        teams = {}
        for row in g.itertuples():
            gid = int(row.Index)
            teams[gid] = (int(getattr(row, COL_HOME_TEAM)), int(getattr(row, COL_AWAY_TEAM)))
        return teams

    # 2) 없으면 game_id 내 team_id unique 2개로 추정 (정렬)
    tmp = df.groupby(COL_GAME)[COL_TEAM].unique().to_dict()
    teams = {}
    for gid, arr in tmp.items():
        arr = list(arr)
        if len(arr) < 2:
            continue
        arr_sorted = sorted([int(x) for x in arr])
        teams[int(gid)] = (arr_sorted[0], arr_sorted[1])  # ⚠️ home/away가 실제와 다를 수 있음
    return teams


def build_outcome_labels_by_game(df: pd.DataFrame, teams_by_game: dict):
    """
    y ∈ {-1,0,1} from home perspective:
      -1: home loss, 0: draw, 1: home win
    우선순위:
      (1) home_score/away_score FT 컬럼이 있으면 사용
      (2) match_result(예: H/D/A) 있으면 사용
      (3) 없으면 goal 이벤트 누적으로 FT 스코어를 계산(골 판정 로직 필요)
    """
    y_by_game = {}

    # (1) FT score
    if (COL_HOME_SCORE_FT in df.columns) and (COL_AWAY_SCORE_FT in df.columns):
        g = df.groupby(COL_GAME)[[COL_HOME_SCORE_FT, COL_AWAY_SCORE_FT]].first()
        for row in g.itertuples():
            gid = int(row.Index)
            hs = float(getattr(row, COL_HOME_SCORE_FT))
            as_ = float(getattr(row, COL_AWAY_SCORE_FT))
            if hs > as_:
                y_by_game[gid] = 1
            elif hs < as_:
                y_by_game[gid] = -1
            else:
                y_by_game[gid] = 0
        return y_by_game

    # (2) match_result like H/D/A
    if (COL_MATCH_RESULT is not None) and (COL_MATCH_RESULT in df.columns):
        g = df.groupby(COL_GAME)[COL_MATCH_RESULT].first()
        for gid, r in g.items():
            gid = int(gid)
            r = str(r).upper().strip()
            if r in ["H", "HOME", "1", "W"]:
                y_by_game[gid] = 1
            elif r in ["A", "AWAY", "2", "L"]:
                y_by_game[gid] = -1
            else:
                y_by_game[gid] = 0
        return y_by_game

    # (3) fallback: goal event 누적 (데이터에 맞게 보완 필요)
    # 여기서는 result_name == "Goal" 또는 type_result의 suffix == "Goal"이면 골로 간주
    tmp = {}
    for gid, gdf in df.groupby(COL_GAME):
        gid = int(gid)
        home, away = teams_by_game.get(gid, (None, None))
        if home is None:
            continue

        gh = 0
        ga = 0
        for r in gdf.itertuples(index=False):
            tid = int(getattr(r, COL_TEAM))
            # event type/result
            if COL_TYPE_NAME and (COL_TYPE_NAME in df.columns):
                tn = str(getattr(r, COL_TYPE_NAME))
                rn = str(getattr(r, COL_RESULT_NAME)) if (COL_RESULT_NAME and COL_RESULT_NAME in df.columns) else ""
            elif COL_TYPE_RESULT and (COL_TYPE_RESULT in df.columns):
                tn, rn = parse_type_result(getattr(r, COL_TYPE_RESULT))
            else:
                continue

            is_goal = False
            if rn.lower() == "goal":
                is_goal = True
            if isinstance(rn, str) and rn.endswith("Goal"):
                is_goal = True
            if isinstance(tn, str) and tn.lower() == "goal":
                is_goal = True

            if is_goal:
                if tid == home:
                    gh += 1
                elif tid == away:
                    ga += 1

        if gh > ga:
            tmp[gid] = 1
        elif gh < ga:
            tmp[gid] = -1
        else:
            tmp[gid] = 0

    return tmp


def compute_team_strength_points(y_by_game: dict, teams_by_game: dict):
    """
    간단 team strength: 승=3, 무=1, 패=0 (home/away 모두 동일하게 반영)
    fold leakage 방지 위해 fold별로 train subset에서 계산해서 사용.
    """
    pts = {}
    games = 0

    for gid, y in y_by_game.items():
        if gid not in teams_by_game:
            continue
        home, away = teams_by_game[gid]
        # home perspective y
        if y == 1:
            ph, pa = 3, 0
        elif y == -1:
            ph, pa = 0, 3
        else:
            ph, pa = 1, 1

        pts[home] = pts.get(home, 0) + ph
        pts[away] = pts.get(away, 0) + pa
        games += 1

    # normalize by matches played
    cnt = {}
    for gid, y in y_by_game.items():
        if gid not in teams_by_game:
            continue
        home, away = teams_by_game[gid]
        cnt[home] = cnt.get(home, 0) + 1
        cnt[away] = cnt.get(away, 0) + 1

    strength = {}
    for team, p in pts.items():
        strength[team] = p / max(cnt.get(team, 1), 1)

    return strength


# =========================================================
# Feature precompute (minute-level event counts)
# =========================================================
def preprocess_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # match time seconds
    if (COL_PERIOD is None) or (COL_PERIOD not in df.columns):
        df["_tsec"] = df[COL_TIME].astype(float).map(lambda x: match_time_seconds(None, x))
    else:
        df["_tsec"] = df.apply(lambda r: match_time_seconds(r[COL_PERIOD], r[COL_TIME]), axis=1)

    # minute index 1..MAX_MINUTE (cap)
    df["_minute"] = (df["_tsec"] // 60 + 1).astype(int)
    df["_minute"] = df["_minute"].clip(1, MAX_MINUTE)

    # type_name / result_name
    if (COL_TYPE_NAME is not None) and (COL_TYPE_NAME in df.columns):
        df["_type_name"] = df[COL_TYPE_NAME].astype(str)
        if (COL_RESULT_NAME is not None) and (COL_RESULT_NAME in df.columns):
            df["_result_name"] = df[COL_RESULT_NAME].astype(str)
        else:
            df["_result_name"] = ""
    elif (COL_TYPE_RESULT is not None) and (COL_TYPE_RESULT in df.columns):
        tmp = df[COL_TYPE_RESULT].astype(str).map(parse_type_result)
        df["_type_name"] = tmp.map(lambda x: x[0])
        df["_result_name"] = tmp.map(lambda x: x[1])
    else:
        raise ValueError("Need either type_name or type_result column. Check CONFIG COL_TYPE_NAME/COL_TYPE_RESULT.")

    # instant raw
    if USE_INSTANT_RAW:
        for c in [COL_START_X, COL_START_Y, COL_END_X, COL_END_Y]:
            if c not in df.columns:
                df[c] = np.nan

        df["_dx"] = (df[COL_END_X].astype(float) - df[COL_START_X].astype(float)).fillna(0.0)
        df["_dy"] = (df[COL_END_Y].astype(float) - df[COL_START_Y].astype(float)).fillna(0.0)

    return df


def build_minute_event_counts(df: pd.DataFrame, teams_by_game: dict, event_types: list):
    """
    base_counts[gid] = {
        "home": home_id, "away": away_id,
        "counts_home": (MAX_MINUTE, K) int,
        "counts_away": (MAX_MINUTE, K) int,
        "raw_last_row_idx_by_minute": (MAX_MINUTE,) -> end-of-minute snapshot index (optional)
    }
    """
    K = len(event_types)
    et2i = {e: i for i, e in enumerate(event_types)}

    base = {}
    for gid, gdf in df.groupby(COL_GAME):
        gid = int(gid)
        if gid not in teams_by_game:
            continue
        home, away = teams_by_game[gid]
        ch = np.zeros((MAX_MINUTE, K), dtype=np.int32)
        ca = np.zeros((MAX_MINUTE, K), dtype=np.int32)

        # end-of-minute snapshot row index (last action index within that minute)
        last_idx = np.full((MAX_MINUTE,), -1, dtype=np.int32)

        # sort by time
        gdf2 = gdf.sort_values("_tsec")
        for idx, r in enumerate(gdf2.itertuples(index=False)):
            tid = int(getattr(r, COL_TEAM))
            m = int(getattr(r, "_minute")) - 1
            tn = str(getattr(r, "_type_name"))

            if tn in et2i:
                j = et2i[tn]
                if tid == home:
                    ch[m, j] += 1
                elif tid == away:
                    ca[m, j] += 1

            last_idx[m] = idx  # 마지막 갱신

        base[gid] = dict(
            home=home, away=away,
            counts_home=ch, counts_away=ca,
            last_idx_by_minute=last_idx,
            gdf_sorted=gdf2.reset_index(drop=True) if USE_INSTANT_RAW else None
        )

    return base


def build_snapshot_table(df: pd.DataFrame,
                         teams_by_game: dict,
                         y_by_game: dict,
                         event_types: list,
                         window_mode: str,
                         window_min: int,
                         window_actions: int):
    """
    학습/평가용 snapshot 생성
    - 기본: match x minute (1..90) (논문 방식)
    - window_mode:
        * MINUTES : 직전 window_min분만 남기고 나머지 minutes의 feature는 0 처리
        * ACTIONS : 최근 window_actions 액션만 카운트 → minute별로 분해해 feature에 적재
    """
    dfp = preprocess_actions(df)

    base = build_minute_event_counts(dfp, teams_by_game, event_types)
    K = len(event_types)

    rows = []
    for gid, info in base.items():
        if gid not in y_by_game:
            continue
        y = int(y_by_game[gid])
        home, away = info["home"], info["away"]

        # 필요 시 action window를 위해 actions list
        gdf_sorted = info["gdf_sorted"] if USE_INSTANT_RAW else None
        gactions = None
        if window_mode == "ACTIONS":
            gactions = dfp[dfp[COL_GAME] == gid].sort_values("_tsec").reset_index(drop=True)

        for t in range(1, MAX_MINUTE + 1):
            # progress(0~1) - 사용자 요구(시간 정규화)
            prog = float(t) / float(MAX_MINUTE)

            # event counts per minute (window 적용)
            xh = np.zeros((MAX_MINUTE, K), dtype=np.float32)
            xa = np.zeros((MAX_MINUTE, K), dtype=np.float32)

            if window_mode == "MINUTES":
                start = max(1, t - window_min + 1)
                # base minute counts는 (0..89)
                xh[start-1:t, :] = info["counts_home"][start-1:t, :]
                xa[start-1:t, :] = info["counts_away"][start-1:t, :]

            else:  # ACTIONS
                # t분(초)까지의 액션 중 최근 K개만
                tsec_cut = t * 60
                sub = gactions[gactions["_tsec"] <= tsec_cut]
                if len(sub) > 0:
                    sub2 = sub.iloc[max(0, len(sub) - window_actions):]
                    # minute별로 다시 카운트
                    for r in sub2.itertuples(index=False):
                        m = int(getattr(r, "_minute")) - 1
                        tn = str(getattr(r, "_type_name"))
                        tid = int(getattr(r, COL_TEAM))
                        if tn in event_types:
                            j = event_types.index(tn)
                            if tid == home:
                                xh[m, j] += 1
                            elif tid == away:
                                xa[m, j] += 1

            # feature vector 구성(논문 구조):
            # z: [1, strength_diff, progress]  (strength_diff는 fold별로 채움 → 여기서는 team id만 저장)
            # time-varying: for each event type k, minute 1..90의 home/away count를 flatten
            feat_counts = np.concatenate([xh.reshape(-1), xa.reshape(-1)], axis=0).astype(np.float32)

            row = {
                "game_id": gid,
                "minute": t,
                "home_team_id": home,
                "away_team_id": away,
                "y": y,
                "progress": prog,
            }

            # counts를 벡터로 저장 (메모리 절약 위해 np.ndarray 그대로)
            row["feat_counts"] = feat_counts

            # (옵션) instant raw를 "현재 minute 끝 action" 기준으로 추가
            if USE_INSTANT_RAW and TRAIN_SNAPSHOT_MODE == "END_OF_MINUTE":
                li = info["last_idx_by_minute"][t-1]
                if li >= 0:
                    rr = info["gdf_sorted"].iloc[int(li)]
                    sx, sy = float(rr.get(COL_START_X, np.nan)), float(rr.get(COL_START_Y, np.nan))
                    ex, ey = float(rr.get(COL_END_X, np.nan)), float(rr.get(COL_END_Y, np.nan))
                    dx, dy = float(rr.get("_dx", 0.0)), float(rr.get("_dy", 0.0))
                else:
                    sx = sy = ex = ey = np.nan
                    dx = dy = 0.0
                row.update({"sx": sx, "sy": sy, "ex": ex, "ey": ey, "dx": dx, "dy": dy})

            rows.append(row)

    snap = pd.DataFrame(rows)
    return snap


# =========================================================
# Ordered Probit Bayesian (Gibbs)
# =========================================================
def ar1_precision(n: int, rho: float, sigma: float):
    """
    Cov: sigma^2 * rho^{|i-j|}
    Precision(Q) for AR(1) is tridiagonal:
      Q = (1/sigma^2) * (1/(1-rho^2)) * tridiag([1, 1+rho^2, ..., 1+rho^2, 1], off=-rho)
    """
    rho = float(rho)
    sigma = float(sigma)
    eps = 1e-9
    rho = min(max(rho, 0.0 + eps), 1.0 - eps)
    s2 = max(sigma * sigma, 1e-12)

    a = 1.0 / (1.0 - rho * rho)
    diag = np.full(n, (1.0 + rho * rho), dtype=np.float64)
    diag[0] = 1.0
    diag[-1] = 1.0
    off = np.full(n - 1, -rho, dtype=np.float64)

    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Q[i, i] = diag[i]
    for i in range(n - 1):
        Q[i, i + 1] = off[i]
        Q[i + 1, i] = off[i]
    Q *= (a / s2)
    return Q


def sample_truncnorm(mean, lo, hi, size=1):
    """
    N(mean,1) truncated to (lo,hi)
    via inverse CDF
    """
    a = (lo - mean) if np.isfinite(lo) else -np.inf
    b = (hi - mean) if np.isfinite(hi) else  np.inf
    Fa = norm.cdf(a) if np.isfinite(a) else 0.0
    Fb = norm.cdf(b) if np.isfinite(b) else 1.0
    u = np.random.uniform(Fa, Fb, size=size)
    z = norm.ppf(u)
    return mean + z


def ordered_probit_gibbs(X: np.ndarray,
                         y: np.ndarray,
                         rho: float,
                         sigma_beta: float,
                         sigma_gamma: float,
                         n_iters: int,
                         burn_frac: float,
                         thin: int,
                         event_types: list,
                         n_static: int,
                         seed: int = 42):
    """
    X: (n, d)
    y: (n,) in {-1,0,1} (ordered)
    - static covariates count = n_static (we use [1, strength_diff, progress] => 3)
    - remaining coefficients are event-count blocks:
        [home counts (MAX_MINUTE*K), away counts (MAX_MINUTE*K)]
      and each event type has minute-wise AR(1) prior with same rho, sigma_beta
      (home/away independent, different event types independent) - 논문 가정과 동일

    Returns:
      draws_beta: (S, d)
      draws_delta: (S, 2)
    """
    set_all_seeds(seed)

    n, d = X.shape
    y = y.astype(int)

    # prior precision matrix Q (d x d)
    # static covariates: iid N(0, sigma_gamma^2)
    Q = np.zeros((d, d), dtype=np.float64)
    Q[:n_static, :n_static] = np.eye(n_static, dtype=np.float64) / max(sigma_gamma * sigma_gamma, 1e-12)

    K = len(event_types)
    per_side = MAX_MINUTE * K
    start_home = n_static
    start_away = n_static + per_side

    # event type block: for each k, minutes 1..MAX_MINUTE have AR(1) prior
    Qm = ar1_precision(MAX_MINUTE, rho=rho, sigma=sigma_beta)  # (90x90)

    # fill home blocks
    for k in range(K):
        a = start_home + k * MAX_MINUTE
        b = a + MAX_MINUTE
        Q[a:b, a:b] = Qm

    # fill away blocks
    for k in range(K):
        a = start_away + k * MAX_MINUTE
        b = a + MAX_MINUTE
        Q[a:b, a:b] = Qm

    # init
    beta = np.zeros(d, dtype=np.float64)
    delta1, delta2 = -0.5, 0.5

    # latent Π
    Pi = np.zeros(n, dtype=np.float64)

    burn = int(n_iters * burn_frac)
    keep_idx = []
    for it in range(n_iters):
        # 1) sample Π | beta, delta
        mu = X @ beta

        # bounds by class
        for cls in [-1, 0, 1]:
            idx = np.where(y == cls)[0]
            if idx.size == 0:
                continue
            m = mu[idx]
            if cls == -1:
                Pi[idx] = sample_truncnorm(m, lo=-np.inf, hi=delta1, size=idx.size)
            elif cls == 0:
                Pi[idx] = sample_truncnorm(m, lo=delta1, hi=delta2, size=idx.size)
            else:
                Pi[idx] = sample_truncnorm(m, lo=delta2, hi=np.inf, size=idx.size)

        # 2) sample beta | Π
        # posterior precision: P = Q + X^T X
        XtX = (X.T @ X).astype(np.float64)
        P = Q + XtX
        bvec = (X.T @ Pi).astype(np.float64)

        # solve mean: P * mean = b
        # then sample using Cholesky of P
        # (P should be SPD)
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            # add jitter
            jitter = 1e-6 * np.eye(d, dtype=np.float64)
            L = np.linalg.cholesky(P + jitter)

        # mean = P^{-1} b using chol solve
        # solve L u = b ; solve L^T mean = u
        u = np.linalg.solve(L, bvec)
        mean = np.linalg.solve(L.T, u)

        z = np.random.normal(size=d)
        # sample: beta = mean + P^{-1/2} z
        # with chol(P)=L => P^{-1/2} z = solve(L.T, solve(L, z))
        v = np.linalg.solve(L, z)
        eps = np.linalg.solve(L.T, v)
        beta = mean + eps

        # 3) sample deltas | Π  (standard ordered probit Gibbs)
        # delta1 in (max Pi[y=-1], min Pi[y=0])
        # delta2 in (max Pi[y=0], min Pi[y=1])
        mL = np.max(Pi[y == -1]) if np.any(y == -1) else -2.0
        m0L = np.min(Pi[y == 0]) if np.any(y == 0) else 0.0
        lo1, hi1 = mL, m0L
        if not (np.isfinite(lo1) and np.isfinite(hi1) and lo1 < hi1):
            lo1, hi1 = -1.0, 0.0
        delta1 = np.random.uniform(lo1, hi1)

        m0U = np.max(Pi[y == 0]) if np.any(y == 0) else 0.0
        mW = np.min(Pi[y == 1]) if np.any(y == 1) else 2.0
        lo2, hi2 = m0U, mW
        if not (np.isfinite(lo2) and np.isfinite(hi2) and lo2 < hi2):
            lo2, hi2 = 0.0, 1.0
        delta2 = np.random.uniform(max(lo2, delta1 + 1e-3), hi2)

        if it >= burn and ((it - burn) % thin == 0):
            keep_idx.append(it)

    # collect draws (we only stored final beta,delta each kept iter)
    # To keep memory simple, rerun loop? No. We'll store during sampling.
    # -> modify: store in lists above
    # For now, minimal change: store progressively
    # (다시 구현)
    # ----------------------------------------------------------------
    # Re-run with proper storing (still deterministic seed)
    set_all_seeds(seed)

    beta = np.zeros(d, dtype=np.float64)
    delta1, delta2 = -0.5, 0.5
    Pi = np.zeros(n, dtype=np.float64)

    draws_beta = []
    draws_delta = []

    for it in range(n_iters):
        mu = X @ beta
        for cls in [-1, 0, 1]:
            idx = np.where(y == cls)[0]
            if idx.size == 0:
                continue
            m = mu[idx]
            if cls == -1:
                Pi[idx] = sample_truncnorm(m, lo=-np.inf, hi=delta1, size=idx.size)
            elif cls == 0:
                Pi[idx] = sample_truncnorm(m, lo=delta1, hi=delta2, size=idx.size)
            else:
                Pi[idx] = sample_truncnorm(m, lo=delta2, hi=np.inf, size=idx.size)

        XtX = (X.T @ X).astype(np.float64)
        P = Q + XtX
        bvec = (X.T @ Pi).astype(np.float64)

        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(P + 1e-6 * np.eye(d, dtype=np.float64))

        u = np.linalg.solve(L, bvec)
        mean = np.linalg.solve(L.T, u)

        z = np.random.normal(size=d)
        v = np.linalg.solve(L, z)
        eps = np.linalg.solve(L.T, v)
        beta = mean + eps

        mL = np.max(Pi[y == -1]) if np.any(y == -1) else -2.0
        m0L = np.min(Pi[y == 0]) if np.any(y == 0) else 0.0
        lo1, hi1 = mL, m0L
        if not (np.isfinite(lo1) and np.isfinite(hi1) and lo1 < hi1):
            lo1, hi1 = -1.0, 0.0
        delta1 = np.random.uniform(lo1, hi1)

        m0U = np.max(Pi[y == 0]) if np.any(y == 0) else 0.0
        mW = np.min(Pi[y == 1]) if np.any(y == 1) else 2.0
        lo2, hi2 = m0U, mW
        if not (np.isfinite(lo2) and np.isfinite(hi2) and lo2 < hi2):
            lo2, hi2 = 0.0, 1.0
        delta2 = np.random.uniform(max(lo2, delta1 + 1e-3), hi2)

        if it >= burn and ((it - burn) % thin == 0):
            draws_beta.append(beta.copy())
            draws_delta.append([delta1, delta2])

    draws_beta = np.asarray(draws_beta, dtype=np.float64)
    draws_delta = np.asarray(draws_delta, dtype=np.float64)
    return draws_beta, draws_delta


def ordered_probit_predict_proba(X: np.ndarray, draws_beta: np.ndarray, draws_delta: np.ndarray):
    """
    Return probs (n,3) for classes [-1,0,1]
    """
    n = X.shape[0]
    S = draws_beta.shape[0]
    probs = np.zeros((n, 3), dtype=np.float64)

    for s in range(S):
        beta = draws_beta[s]
        d1, d2 = draws_delta[s]
        mu = X @ beta
        pL = norm.cdf(d1 - mu)
        pD = norm.cdf(d2 - mu) - norm.cdf(d1 - mu)
        pW = 1.0 - norm.cdf(d2 - mu)
        probs[:, 0] += pL
        probs[:, 1] += pD
        probs[:, 2] += pW

    probs /= max(S, 1)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# =========================================================
# Metrics (Brier, RPS, ECE, LogLoss, AUC one-vs-rest)
# =========================================================
def y_to_onehot(y: np.ndarray):
    # order: [-1,0,1] -> [0,1,2]
    idx = np.where(y == -1, 0, np.where(y == 0, 1, 2))
    oh = np.zeros((len(y), 3), dtype=np.float64)
    oh[np.arange(len(y)), idx] = 1.0
    return oh, idx


def brier_multiclass(y: np.ndarray, p: np.ndarray):
    oh, _ = y_to_onehot(y)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def rps_3class(y: np.ndarray, p: np.ndarray):
    """
    Ranked Probability Score for ordered 3-class:
      RPS = (F1 - O1)^2 + (F2 - O2)^2
    where Fk is cumulative predicted up to class k, O cumulative observed.
    class order: [-1,0,1]
    """
    oh, idx = y_to_onehot(y)
    # cumulative predicted
    F1 = p[:, 0]
    F2 = p[:, 0] + p[:, 1]
    # cumulative observed
    O1 = (idx <= 0).astype(np.float64)   # y == -1
    O2 = (idx <= 1).astype(np.float64)   # y in {-1,0}
    return float(np.mean((F1 - O1) ** 2 + (F2 - O2) ** 2))


def logloss_3class(y: np.ndarray, p: np.ndarray):
    oh, idx = y_to_onehot(y)
    pt = p[np.arange(len(y)), idx]
    return float(-np.mean(np.log(np.clip(pt, 1e-12, 1.0))))


def ece_toplabel(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    """
    Top-label ECE:
      bin by confidence = max prob
      compare accuracy vs confidence
    """
    _, idx = y_to_onehot(y)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    acc = (pred == idx).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        m = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        w = np.mean(m)
        ece += w * abs(np.mean(acc[m]) - np.mean(conf[m]))
    return float(ece)


def nbs_from_brier(y: np.ndarray, p: np.ndarray):
    """
    Normalized Brier Score (skill): 1 - Brier / Brier_baseline
    baseline: class frequencies
    """
    b = brier_multiclass(y, p)
    oh, _ = y_to_onehot(y)
    freq = np.mean(oh, axis=0)  # baseline probs
    p0 = np.tile(freq, (len(y), 1))
    b0 = float(np.mean(np.sum((p0 - oh) ** 2, axis=1)))
    return float(1.0 - b / max(b0, 1e-12))


def auc_homewin_onevsrest(y: np.ndarray, p: np.ndarray):
    """
    one-vs-rest AUC for "home win (y=1)" probability p[:,2]
    """
    y_bin = (y == 1).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, p[:, 2]))


# =========================================================
# HPO Runner
# =========================================================
def sample_from(values):
    return random.choice(values)


def sample_from_range(rng, as_int=False):
    vals = build_values(rng, as_int=as_int)
    return random.choice(vals)


def make_design_matrix(snap: pd.DataFrame, strength_map: dict):
    """
    X = [1, strength_diff, progress] + counts(home+away minutes*types)
    """
    n = len(snap)
    # static: 3
    n_static = 3

    # counts length = 2 * MAX_MINUTE * K
    counts_len = len(snap.iloc[0]["feat_counts"])
    d = n_static + counts_len

    X = np.zeros((n, d), dtype=np.float64)

    # static
    home = snap["home_team_id"].values.astype(int)
    away = snap["away_team_id"].values.astype(int)

    sh = np.array([strength_map.get(int(t), 0.0) for t in home], dtype=np.float64)
    sa = np.array([strength_map.get(int(t), 0.0) for t in away], dtype=np.float64)
    sdiff = sh - sa

    X[:, 0] = 1.0
    X[:, 1] = sdiff
    X[:, 2] = snap["progress"].values.astype(np.float64)

    # counts
    # feat_counts is ndarray per row
    # stack
    C = np.stack(snap["feat_counts"].values, axis=0).astype(np.float64)
    X[:, n_static:] = C

    # optional raw
    if USE_INSTANT_RAW:
        # append at end (but then prior needs 확장. 여기서는 USE_INSTANT_RAW=False 권장)
        pass

    return X, n_static


def run_hpo():
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    set_all_seeds(SEED)

    df = pd.read_csv(DATA_PATH)
    teams_by_game = infer_teams_by_game(df)
    y_by_game = build_outcome_labels_by_game(df, teams_by_game)

    # HPO output file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUT_DIR, f"hpo_bayes_outcome_{ts}.csv")
    best_json = os.path.join(OUT_DIR, f"hpo_bayes_outcome_best_{ts}.json")

    # trial 기록
    header = [
        "trial", "window_mode", "window_min", "window_actions", "event_types",
        "rho", "sigma_beta", "sigma_gamma", "gibbs_iters",
        "brier", "nbs", "rps", "logloss", "ece", "auc_homewin",
        "n_samples", "n_features", "time_sec"
    ]
    pd.DataFrame(columns=header).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # cache key: (window_mode, window_min, window_actions, event_types_key)
    snap_cache = {}

    best_score = float("inf")  # brier 기준(낮을수록 좋음)
    best_pack = None

    outer = tqdm(range(1, N_ROUNDS + 1), desc="HPO", ncols=110)
    for trial in outer:
        t0 = datetime.now().timestamp()

        # ---- sample hyperparams
        window_mode = sample_from(WINDOW_MODE_R)
        window_min = int(sample_from_range(WINDOW_MIN_R, as_int=True))
        window_actions = int(sample_from_range(WINDOW_ACT_R, as_int=True))

        event_types = sample_from(EVENT_TYPES_CANDIDATES)
        event_types_key = "|".join(event_types)

        rho = float(sample_from_range(RHO_R, as_int=False))
        sigma_beta = float(sample_from_range(SIGMA_BETA_R, as_int=False))
        sigma_gamma = float(sample_from_range(SIGMA_GAMMA_R, as_int=False))
        gibbs_iters = int(sample_from_range(GIBBS_ITERS_R, as_int=True))

        # ---- load/build snapshots (cache)
        cache_key = (window_mode, window_min, window_actions, event_types_key)
        if cache_key in snap_cache:
            snap = snap_cache[cache_key]
        else:
            cache_path = os.path.join(
                CACHE_DIR,
                f"snap_{window_mode}_m{window_min}_a{window_actions}_e{len(event_types)}_{abs(hash(event_types_key))%999999}.pkl"
            )
            if os.path.exists(cache_path):
                snap = pd.read_pickle(cache_path)
            else:
                snap = build_snapshot_table(
                    df,
                    teams_by_game=teams_by_game,
                    y_by_game=y_by_game,
                    event_types=event_types,
                    window_mode=window_mode,
                    window_min=window_min,
                    window_actions=window_actions,
                )
                snap.to_pickle(cache_path)
            snap_cache[cache_key] = snap

        # ---- CV
        groups = snap["game_id"].values.astype(int)
        y = snap["y"].values.astype(int)

        gkf = GroupKFold(n_splits=N_FOLDS)

        fold_metrics = []
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(snap, y, groups=groups)):
            tr = snap.iloc[tr_idx].reset_index(drop=True)
            va = snap.iloc[va_idx].reset_index(drop=True)

            # fold-specific team strength (leakage 방지)
            y_train_games = {int(g): int(v) for g, v in y_by_game.items() if int(g) in set(tr["game_id"].unique())}
            strength_map = compute_team_strength_points(y_train_games, teams_by_game)

            Xtr, n_static = make_design_matrix(tr, strength_map)
            Xva, _ = make_design_matrix(va, strength_map)

            ytr = tr["y"].values.astype(int)
            yva = va["y"].values.astype(int)

            draws_beta, draws_delta = ordered_probit_gibbs(
                Xtr, ytr,
                rho=rho,
                sigma_beta=sigma_beta,
                sigma_gamma=sigma_gamma,
                n_iters=gibbs_iters,
                burn_frac=BURN_IN_FRAC,
                thin=THINNING,
                event_types=event_types,
                n_static=n_static,
                seed=SEED + trial * 100 + fold_idx
            )

            pva = ordered_probit_predict_proba(Xva, draws_beta, draws_delta)

            m = dict(
                brier=brier_multiclass(yva, pva),
                nbs=nbs_from_brier(yva, pva),
                rps=rps_3class(yva, pva),
                logloss=logloss_3class(yva, pva),
                ece=ece_toplabel(yva, pva),
                auc=auc_homewin_onevsrest(yva, pva)
            )
            fold_metrics.append(m)

        # ---- aggregate
        brier = float(np.mean([m["brier"] for m in fold_metrics]))
        nbs   = float(np.mean([m["nbs"] for m in fold_metrics]))
        rps   = float(np.mean([m["rps"] for m in fold_metrics]))
        ll    = float(np.mean([m["logloss"] for m in fold_metrics]))
        ece   = float(np.mean([m["ece"] for m in fold_metrics]))
        auc   = float(np.nanmean([m["auc"] for m in fold_metrics]))

        t1 = datetime.now().timestamp()
        elapsed = float(t1 - t0)

        # ---- record
        row = {
            "trial": trial,
            "window_mode": window_mode,
            "window_min": window_min,
            "window_actions": window_actions,
            "event_types": event_types_key,
            "rho": rho,
            "sigma_beta": sigma_beta,
            "sigma_gamma": sigma_gamma,
            "gibbs_iters": gibbs_iters,
            "brier": brier,
            "nbs": nbs,
            "rps": rps,
            "logloss": ll,
            "ece": ece,
            "auc_homewin": auc,
            "n_samples": int(len(snap)),
            "n_features": int(3 + len(snap.iloc[0]["feat_counts"])),
            "time_sec": elapsed
        }
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8-sig")

        outer.set_postfix_str(
            f"brier={brier:.6f} rps={rps:.6f} ece={ece:.6f} auc={auc:.4f} ({elapsed:.1f}s)",
            refresh=True
        )

        # ---- best (brier 기준)
        if brier < best_score:
            best_score = brier
            best_pack = dict(best_score=best_score, best_row=row)
            with open(best_json, "w", encoding="utf-8") as f:
                json.dump(best_pack, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] HPO results: {out_csv}")
    if best_pack:
        print(f"[Best] brier={best_pack['best_score']:.6f} saved: {best_json}")


# =========================================================
# Real-time action-step inference helper (서비스용)
# =========================================================
def build_realtime_feature_for_action(
    df_game: pd.DataFrame,
    action_idx: int,
    teams_by_game: dict,
    event_types: list,
    window_mode: str,
    window_min: int,
    window_actions: int,
    strength_map: dict
):
    """
    df_game: single game actions (preprocessed with preprocess_actions)
    action_idx: 현재 액션 인덱스(0-based)
    논문 feature 구조를 유지하되, 입력은 슬라이딩 윈도우로만 구성.
    반환: X(1,d), n_static
    """
    gid = int(df_game.iloc[0][COL_GAME])
    home, away = teams_by_game[gid]
    K = len(event_types)

    cur = df_game.iloc[action_idx]
    tsec = int(cur["_tsec"])
    t = int(cur["_minute"])  # 1..90
    prog = float(t) / float(MAX_MINUTE)

    # window로 minute-wise counts 만들기
    xh = np.zeros((MAX_MINUTE, K), dtype=np.float64)
    xa = np.zeros((MAX_MINUTE, K), dtype=np.float64)

    if window_mode == "MINUTES":
        start = max(1, t - window_min + 1)
        sub = df_game[df_game["_tsec"] <= tsec]
        # minute별 누적이 아니라, minute별 카운트
        for r in sub.itertuples(index=False):
            m = int(getattr(r, "_minute"))
            if m < start or m > t:
                continue
            tn = str(getattr(r, "_type_name"))
            if tn not in event_types:
                continue
            j = event_types.index(tn)
            tid = int(getattr(r, COL_TEAM))
            if tid == home:
                xh[m-1, j] += 1
            elif tid == away:
                xa[m-1, j] += 1

    else:  # ACTIONS
        sub = df_game.iloc[:action_idx+1]
        sub2 = sub.iloc[max(0, len(sub) - window_actions):]
        for r in sub2.itertuples(index=False):
            m = int(getattr(r, "_minute"))
            if m > t:
                continue
            tn = str(getattr(r, "_type_name"))
            if tn not in event_types:
                continue
            j = event_types.index(tn)
            tid = int(getattr(r, COL_TEAM))
            if tid == home:
                xh[m-1, j] += 1
            elif tid == away:
                xa[m-1, j] += 1

    feat_counts = np.concatenate([xh.reshape(-1), xa.reshape(-1)], axis=0)

    # static part
    n_static = 3
    d = n_static + feat_counts.size
    X = np.zeros((1, d), dtype=np.float64)
    X[0, 0] = 1.0
    X[0, 1] = strength_map.get(home, 0.0) - strength_map.get(away, 0.0)
    X[0, 2] = prog
    X[0, n_static:] = feat_counts
    return X, n_static, t


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    run_hpo()
