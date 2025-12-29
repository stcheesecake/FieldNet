#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FieldNet EDA Script (MINIMAL OUTPUT / 6 PNG)

- df = pd.read_csv("../data/raw_data.csv") 고정
- print 제거 + tqdm만 사용
- 결과물: 딱 6개 PNG만 저장
  1) eda_type_overview.png
  2) eda_top_types_dashboard.png            (핵심 type 3개만)
  3) goal_target_dashboard.png
  4) goal_target_feature_importance_top30.png
  5) goal_target_feature_importance_top30_no_coords.png
  6) goal_target_feature_importance_top30_no_coords_no_entities.png

핵심 수정사항:
- 한글 폰트 자동 설정 (Windows: Malgun Gothic 우선)
- goal_target 생성
- type별 target 매핑 (success_flag/duel_win/card_flag/goal_target)
- grid rate/count + 마스킹
- Feature importance는 "상위 30개만" 그리도록 수정 (글자 폭발 방지)
- importance는 3번 모두 "각각 drop한 컬럼으로 다시 학습"해서 출력
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# Font (Korean)
# =========================================================
def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in candidates:
        if font in available:
            mpl.rcParams["font.family"] = font
            break
    mpl.rcParams["axes.unicode_minus"] = False


set_korean_font()


# =========================================================
# CONFIG
# =========================================================
RESULT_DIR = "../EDA/result"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

# grid 설정
X_BINS = 12
Y_BINS = 8
MIN_CELL_COUNT = 30  # 성공률 grid에서 이 미만 셀은 마스킹

# 핵심 type 몇 개만 보여줄지
TOP_K_TYPES = 3
MIN_ROWS_PER_TYPE = 3000
MIN_POSITIVE_PER_TYPE = 50

# goal_target 옵션
INCLUDE_OWN_GOAL = False  # "골 발생 자체"로 보려면 True 권장

# importance drop 리스트
COORD_DROP_COLS = ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]
ENTITY_DROP_COLS = ["player_name_ko", "team_name_ko", "player_id", "team_id"]

# 모델/EDA용 shot 후보 타입
SHOT_CANDIDATE_TYPES = ["Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"]

# 공통적으로 모델에서 빼고 싶은 ID 류 (있으면 제거)
ID_DROP_COLS = ["game_id", "action_id"]


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("../data/raw_data.csv")


# =========================================================
# UTIL
# =========================================================
def has_cols(data: pd.DataFrame, cols) -> bool:
    return all(c in data.columns for c in cols)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_bins(series: pd.Series, n_bins: int = 12) -> pd.Series:
    """qcut 우선, 실패 시 cut fallback"""
    s = series.dropna()
    if s.nunique() < 2:
        return pd.Series([np.nan] * len(series), index=series.index)
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=min(n_bins, max(2, series.nunique())))


def grid_rate_and_count(data: pd.DataFrame, xcol: str, ycol: str, y: pd.Series,
                        x_bins=X_BINS, y_bins=Y_BINS, min_count=MIN_CELL_COUNT):
    tmp = data[[xcol, ycol]].copy()
    tmp["__y__"] = y.values
    tmp = tmp.dropna()

    tmp["x_bin"] = pd.cut(tmp[xcol], bins=x_bins)
    tmp["y_bin"] = pd.cut(tmp[ycol], bins=y_bins)

    cnt = tmp.groupby(["y_bin", "x_bin"], observed=True)["__y__"].size().unstack()
    rate = tmp.groupby(["y_bin", "x_bin"], observed=True)["__y__"].mean().unstack()

    cnt = cnt.fillna(0).astype(int)
    mask = (cnt < min_count) | rate.isna()
    return rate, cnt, mask


# =========================================================
# FEATURE: goal_target 생성
# =========================================================
def build_goal_target(data: pd.DataFrame, include_own_goal: bool = True) -> pd.Series:
    """
    goal_target = 1 조건:
      - (type_name == "Goal" && result_name is NaN)
      - (type_name in ["Penalty Kick","Shot","Shot_Corner","Shot_Freekick"] && result_name == "Goal")
      - (optional) (type_name == "Own Goal" && result_name is NaN)
    """
    t = data["type_name"].astype(str)
    rn = data["result_name"]

    cond_goal_event = (t.eq("Goal") & rn.isna())

    shot_goal_types = ["Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"]
    cond_shot_goal = (t.isin(shot_goal_types) &
                      rn.astype(str).str.strip().str.lower().eq("goal"))

    cond_own_goal = (t.eq("Own Goal") & rn.isna()) if include_own_goal else False

    return (cond_goal_event | cond_shot_goal | cond_own_goal).astype(int)


df["goal_target"] = build_goal_target(df, include_own_goal=INCLUDE_OWN_GOAL)


# =========================================================
# COLUMN DETECTION
# =========================================================
TIME_COL = "time_seconds"
PERIOD_COL = "period_id"
COORD_SET = ("start_x", "start_y", "end_x", "end_y")

has_time = TIME_COL in df.columns
has_period = PERIOD_COL in df.columns
has_coord = has_cols(df, COORD_SET)


# =========================================================
# TARGET MAPPING (type별)
# =========================================================
def resolve_type_target(sub: pd.DataFrame, type_name: str):
    """
    type별로 '해석 가능한 이진 타겟'을 선택.
    반환: (target_name, y or None)
    """
    t = str(type_name)

    # 샷/골 관련은 goal_target 사용
    shot_goal_types = {"Goal", "Own Goal", "Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"}
    if t in shot_goal_types:
        y = sub["goal_target"].astype(int)
        return "goal_target", (y if y.nunique() >= 2 else None)

    if "result_name" in sub.columns:
        rn = sub["result_name"].astype(str).str.strip().str.lower()
        uniq = set(rn.dropna().unique().tolist())

        # Successful/Unsuccessful
        if ("successful" in uniq) and ("unsuccessful" in uniq):
            y = rn.eq("successful").astype(int)
            return "success_flag", (y if y.nunique() >= 2 else None)

        # Duel won/lost
        if "duel" in t.lower():
            if ("won" in uniq) and ("lost" in uniq):
                y = rn.eq("won").astype(int)
                return "duel_win", (y if y.nunique() >= 2 else None)

    # Foul류: card_flag (yellow/red/card 포함)
    if "foul" in t.lower() and "result_name" in sub.columns:
        rn_raw = sub["result_name"].astype(str).fillna("")
        card_pat = re.compile(r"(yellow|red|card)", re.IGNORECASE)
        y = rn_raw.apply(lambda s: 1 if card_pat.search(s) else 0).astype(int)
        if y.sum() > 0 and y.nunique() >= 2:
            return "card_flag", y
        return "card_flag", None

    return "none", None


def aggregate_ohe_importance(imp_series: pd.Series) -> pd.Series:
    """
    OHE feature importance를 '원본 컬럼 단위'로 합산.
    예) team_name_ko_제주SK FC, team_name_ko_FC서울 ... -> team_name_ko
    """
    out = {}
    for feat, val in imp_series.items():
        # sklearn OHE 이름: "colname_category" (get_feature_names_out)
        base = feat.split("_", 1)[0] if "_" in feat else feat
        out[base] = out.get(base, 0.0) + float(val)
    return pd.Series(out).sort_values(ascending=False)


# =========================================================
# 1) EDA: type별 overview (1장)
# =========================================================
type_counts = df["type_name"].value_counts()
type_indices = df.groupby("type_name", sort=False).indices

rows = []
for t in type_counts.index:
    idx = type_indices.get(t)
    sub = df.loc[idx]
    target_name, y = resolve_type_target(sub, t)
    n = len(sub)

    if y is None:
        rows.append((t, n, target_name, np.nan, 0, 0))
        continue

    pos = int(y.sum())
    neg = int((1 - y).sum())
    rate = float(y.mean())
    rows.append((t, n, target_name, rate, pos, neg))

ov = pd.DataFrame(rows, columns=["type_name", "n", "target_name", "target_rate", "pos", "neg"])
ov_valid = ov.dropna(subset=["target_rate"]).sort_values("n", ascending=False)
topN = ov_valid.head(15).copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].barh(topN["type_name"][::-1], topN["n"][::-1])
axes[0].set_title("Top types by count (valid binary target only)")
axes[0].set_xlabel("count")

axes[1].barh(topN["type_name"][::-1], topN["target_rate"][::-1])
axes[1].set_title("Target rate by type (depends on mapped target)")
axes[1].set_xlabel("rate")
axes[1].set_xlim(0, 1)

fig.suptitle("Type Overview (what is worth analyzing)", y=1.02)
save_fig(f"{RESULT_DIR}/eda_type_overview.png")


# =========================================================
# 2) EDA: 핵심 type TOP_K_TYPES만 대시보드 (1장)
# =========================================================
cand = ov_valid[
    (ov_valid["n"] >= MIN_ROWS_PER_TYPE) &
    (ov_valid["pos"] >= MIN_POSITIVE_PER_TYPE) &
    (ov_valid["neg"] >= MIN_POSITIVE_PER_TYPE)
].copy()

top_types = cand.sort_values("n", ascending=False).head(TOP_K_TYPES)["type_name"].tolist()

n_rows = max(1, len(top_types))
fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5 * n_rows))
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for r, t in enumerate(top_types):
    sub = df.loc[type_indices[t]].copy()
    target_name, y = resolve_type_target(sub, t)

    if y is None:
        for c in range(4):
            axes[r, c].text(0.5, 0.5, f"{t}\nNo usable target", ha="center", va="center")
            axes[r, c].set_axis_off()
        continue

    # (1) time trend
    ax = axes[r, 0]
    if has_time and sub[TIME_COL].notna().sum() > 0 and sub[TIME_COL].nunique() >= 5:
        if has_period and sub[PERIOD_COL].notna().sum() > 0:
            tmp2 = sub[[TIME_COL, PERIOD_COL]].copy()
            tmp2["__y__"] = y.values
            tmp2 = tmp2.dropna()
            if len(tmp2) >= 100:
                for pid in sorted(tmp2[PERIOD_COL].dropna().unique()):
                    part = tmp2[tmp2[PERIOD_COL] == pid].copy()
                    if len(part) < 50:
                        continue
                    part["time_bin"] = make_bins(part[TIME_COL], n_bins=10)
                    trend = part.groupby("time_bin", observed=True)["__y__"].mean()
                    ax.plot(range(len(trend)), trend.values, marker="o", label=f"period {int(pid)}")
                ax.legend()
                ax.set_title(f"{t} | {target_name} rate over time (by period)")
                ax.set_ylim(0, 1)
                ax.set_xlabel("bin index")
                ax.set_ylabel("rate")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
        else:
            tmp = sub[[TIME_COL]].copy()
            tmp["__y__"] = y.values
            tmp = tmp.dropna()
            if len(tmp) >= 100:
                tmp["time_bin"] = make_bins(tmp[TIME_COL], n_bins=12)
                trend = tmp.groupby("time_bin", observed=True)["__y__"].mean()
                ax.plot(range(len(trend)), trend.values, marker="o")
                ax.set_title(f"{t} | {target_name} rate over time (binned)")
                ax.set_ylim(0, 1)
                ax.set_xlabel("bin index")
                ax.set_ylabel("rate")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()

    # (2) distance success curve
    ax = axes[r, 1]
    if has_coord:
        sx, sy, ex, ey = COORD_SET
        ok = sub[[sx, sy, ex, ey]].notna().all(axis=1).sum() >= 200
        if ok:
            tmp = sub[[sx, sy, ex, ey]].copy()
            tmp["__y__"] = y.values
            tmp = tmp.dropna()
            tmp["distance"] = np.sqrt((tmp[ex] - tmp[sx])**2 + (tmp[ey] - tmp[sy])**2)

            tmp["dist_bin"] = make_bins(tmp["distance"], n_bins=10)
            rate = tmp.groupby("dist_bin", observed=True)["__y__"].mean()
            cnt = tmp.groupby("dist_bin", observed=True)["__y__"].size()

            ax.plot(range(len(rate)), rate.values, marker="o")
            ax.set_ylim(0, 1)
            ax.set_title(f"{t} | {target_name} rate by distance (quantile bins)")
            ax.set_xlabel("distance bin index")
            ax.set_ylabel("rate")
            ax.text(0.02, 0.02, f"min bin count={int(cnt.min())}", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()

    # (3)(4) startpos rate grid + count grid
    ax_rate = axes[r, 2]
    ax_cnt = axes[r, 3]
    if has_coord:
        sx, sy, _, _ = COORD_SET
        ok = sub[[sx, sy]].notna().all(axis=1).sum() >= 500
        if ok:
            rate_grid, cnt_grid, mask = grid_rate_and_count(sub.dropna(subset=[sx, sy]), sx, sy, y)
            sns.heatmap(rate_grid, ax=ax_rate, cmap="viridis", vmin=0, vmax=1, mask=mask)
            ax_rate.set_title(f"{t} | startpos {target_name} rate (mask < {MIN_CELL_COUNT})")
            ax_rate.set_xlabel("x_bin"); ax_rate.set_ylabel("y_bin")

            sns.heatmap(cnt_grid, ax=ax_cnt, cmap="magma")
            ax_cnt.set_title(f"{t} | startpos count")
            ax_cnt.set_xlabel("x_bin"); ax_cnt.set_ylabel("y_bin")
        else:
            ax_rate.text(0.5, 0.5, "N/A", ha="center", va="center"); ax_rate.set_axis_off()
            ax_cnt.text(0.5, 0.5, "N/A", ha="center", va="center"); ax_cnt.set_axis_off()
    else:
        ax_rate.text(0.5, 0.5, "N/A", ha="center", va="center"); ax_rate.set_axis_off()
        ax_cnt.text(0.5, 0.5, "N/A", ha="center", va="center"); ax_cnt.set_axis_off()

fig.suptitle(f"Top {len(top_types)} Types Dashboard (only the most interpretable visuals)", y=1.01)
save_fig(f"{RESULT_DIR}/eda_top_types_dashboard.png")


# =========================================================
# 3) goal_target 전용 대시보드 (1장)
# =========================================================
shot_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# (a) goal_target 분포
gt_counts = df["goal_target"].value_counts().sort_index()
axes[0].barh(gt_counts.index.astype(str), gt_counts.values)
axes[0].set_title("goal_target Distribution (All Events)")
axes[0].set_xlabel("count")

# (b) 타입별 goal rate
if len(shot_df) > 0:
    rate_by_type = shot_df.groupby("type_name", observed=True)["goal_target"].mean().sort_values()
    axes[1].barh(rate_by_type.index.astype(str), rate_by_type.values)
    axes[1].set_title("Goal rate by shot-candidate type")
    axes[1].set_xlim(0, 1)
else:
    axes[1].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[1].set_axis_off()

# (c) 시간대별 goal rate (period 분리)
if has_time and len(shot_df) > 0 and shot_df[TIME_COL].notna().sum() > 0 and shot_df[TIME_COL].nunique() >= 5:
    if has_period and shot_df[PERIOD_COL].notna().sum() > 0:
        for pid in sorted(shot_df[PERIOD_COL].dropna().unique()):
            part = shot_df[shot_df[PERIOD_COL] == pid][[TIME_COL, "goal_target"]].dropna().copy()
            if len(part) < 50:
                continue
            part["time_bin"] = make_bins(part[TIME_COL], n_bins=10)
            trend = part.groupby("time_bin", observed=True)["goal_target"].mean()
            axes[2].plot(range(len(trend)), trend.values, marker="o", label=f"period {int(pid)}")
        axes[2].legend()
        axes[2].set_title("Goal rate over time (shot candidates, by period)")
    else:
        tmp = shot_df[[TIME_COL, "goal_target"]].dropna().copy()
        tmp["time_bin"] = make_bins(tmp[TIME_COL], n_bins=12)
        trend = tmp.groupby("time_bin", observed=True)["goal_target"].mean()
        axes[2].plot(range(len(trend)), trend.values, marker="o")
        axes[2].set_title("Goal rate over time (shot candidates, binned)")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("bin index")
    axes[2].set_ylabel("rate")
else:
    axes[2].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[2].set_axis_off()

# (d)(e) startpos goal rate grid + count grid
if has_coord and len(shot_df) > 0:
    sx, sy, _, _ = COORD_SET
    d2 = shot_df.dropna(subset=[sx, sy, "goal_target"])
    if len(d2) >= 300:
        rate_grid, cnt_grid, mask = grid_rate_and_count(d2, sx, sy, d2["goal_target"])
        sns.heatmap(rate_grid, ax=axes[3], cmap="viridis", vmin=0, vmax=1, mask=mask)
        axes[3].set_title(f"StartPos goal rate (mask < {MIN_CELL_COUNT})")
        axes[3].set_xlabel("x_bin"); axes[3].set_ylabel("y_bin")

        sns.heatmap(cnt_grid, ax=axes[4], cmap="magma")
        axes[4].set_title("StartPos count")
        axes[4].set_xlabel("x_bin"); axes[4].set_ylabel("y_bin")
    else:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
        axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()
else:
    axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
    axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()

# (f) distance by goal_target
if has_coord and len(shot_df) > 0:
    sx, sy, ex, ey = COORD_SET
    d3 = shot_df.dropna(subset=[sx, sy, ex, ey, "goal_target"]).copy()
    if len(d3) >= 300:
        d3["distance"] = np.sqrt((d3[ex] - d3[sx])**2 + (d3[ey] - d3[sy])**2)
        sns.boxplot(data=d3, x="goal_target", y="distance", ax=axes[5])
        axes[5].set_title("Distance by goal_target (shot candidates)")
        axes[5].set_xlabel("goal_target (0/1)")
    else:
        axes[5].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[5].set_axis_off()
else:
    axes[5].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[5].set_axis_off()

fig.suptitle(f"goal_target Dashboard (INCLUDE_OWN_GOAL={INCLUDE_OWN_GOAL})", y=1.01)
save_fig(f"{RESULT_DIR}/goal_target_dashboard.png")


# =========================================================
# 4/5/6) Tree importance (goal_target) - 3 runs (각각 다시 학습)
# =========================================================
def train_extratrees_and_plot_importance(model_df: pd.DataFrame, y: pd.Series, out_png: str):
    # 라벨/누수 컬럼 제거
    drop_cols = [c for c in ["result_name", "goal_target"] if c in model_df.columns]
    X = model_df.drop(columns=drop_cols, errors="ignore")

    # 범주/수치 분리
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    model = ExtraTreesClassifier(
        n_estimators=1200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
    )

    clf = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    # feature names 복원
    feature_names = []
    feature_names.extend(num_cols)

    if len(cat_cols) > 0:
        ohe = clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_names)

    importances = clf.named_steps["model"].feature_importances_
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # (핵심) 상위 30개만 plot (글자 폭발 방지)
    imp_top = imp.head(30).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    imp_top.plot(kind="barh")
    plt.title(out_png.replace(".png", ""))
    plt.xlabel("importance")
    save_fig(f"{RESULT_DIR}/{out_png}")


# goal_target 모델은 shot 후보에서만 학습
base_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()
y_goal = base_df["goal_target"].astype(int)

# 공통 drop(id)
base_df = base_df.drop(columns=[c for c in ID_DROP_COLS if c in base_df.columns], errors="ignore")

with tqdm(total=3, desc="Goal-target models (3 fits)", ncols=110) as pbar:
    # (1) 기본
    train_extratrees_and_plot_importance(
        model_df=base_df,
        y=y_goal,
        out_png="goal_target_feature_importance_top30.png"
    )
    pbar.update(1)

    # (2) 좌표 제거
    drop_coords = [c for c in COORD_DROP_COLS if c in base_df.columns]
    df_no_coords = base_df.drop(columns=drop_coords, errors="ignore")

    train_extratrees_and_plot_importance(
        model_df=df_no_coords,
        y=y_goal,
        out_png="goal_target_feature_importance_top30_no_coords.png"
    )
    pbar.update(1)

    # (3) 좌표 + 엔티티/ID 제거
    drop_entities = [c for c in ENTITY_DROP_COLS if c in df_no_coords.columns]
    df_no_coords_no_entities = df_no_coords.drop(columns=drop_entities, errors="ignore")

    train_extratrees_and_plot_importance(
        model_df=df_no_coords_no_entities,
        y=y_goal,
        out_png="goal_target_feature_importance_top30_no_coords_no_entities.png"
    )
    pbar.update(1)
