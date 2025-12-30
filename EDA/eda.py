#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FieldNet EDA Script (MINIMAL OUTPUT)

- df = pd.read_csv("../data/raw_data.csv") 고정
- print 제거 + tqdm만 사용
- 결과물: PNG 6개 저장
  1) eda_type_overview.png
  2) eda_top_types_dashboard.png  (지정 타입 7개)
  3) goal_target_dashboard.png
  4) goal_target_feature_importance_top30.png
  5) goal_target_feature_importance_top30_no_coords.png
  6) goal_target_feature_importance_top30_no_coords_no_entities.png
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# FONT
# =========================
def set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
    available = set(f.name for f in fm.fontManager.ttflist)
    for font in candidates:
        if font in available:
            mpl.rcParams["font.family"] = font
            break
    mpl.rcParams["axes.unicode_minus"] = False

set_korean_font()

# =========================
# CONFIG
# =========================
RESULT_DIR = "../EDA/result"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

# grid 설정
X_BINS = 12
Y_BINS = 8
MIN_CELL_COUNT = 30  # grid에서 이 미만 셀은 마스킹

# (자동 선정 모드용) 상위 K개
TOP_K_TYPES = 3
MIN_ROWS_PER_TYPE = 3000
MIN_POSITIVE_PER_TYPE = 50

# ✅ (핵심) 대시보드에서 "무조건" 보고 싶은 타입들
DASHBOARD_TYPES = [
    "Pass", "Tackle", "Throw-In",
    "Goal Kick", "Penalty Kick", "Shot", "Shot_Freekick"
]

# goal_target 정의 옵션
INCLUDE_OWN_GOAL = False  # "골 발생" 자체면 True 권장

# 좌표 제거 importance를 위한 컬럼(없으면 자동 무시)
COORD_DROP_COLS = ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]

# (추가) importance에서 제외할 ID/엔티티 컬럼
ENTITY_DROP_COLS = ["player_name_ko", "team_name_ko", "player_id", "team_id"]

# goal_target 학습/EDA 대상 (골 여부)
SHOT_CANDIDATE_TYPES = ["Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"]

# ✅ result_name 기반으로 "이진/다중" 자동 판단할 타입
RESULTNAME_TASK_TYPES = ["Goal Kick", "Penalty Kick", "Shot", "Shot_Freekick"]

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("../data/raw_data.csv")

# =========================
# UTIL
# =========================
def has_cols(data: pd.DataFrame, cols) -> bool:
    return all(c in data.columns for c in cols)

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def make_bins(series: pd.Series, n_bins: int = 12) -> pd.Series:
    """qcut 우선, 실패 시 cut"""
    s = series.dropna()
    if s.nunique() < 2:
        return pd.Series([np.nan] * len(series), index=series.index)
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=min(n_bins, max(2, series.nunique())))

def top1_share(cat_series: pd.Series) -> float:
    vc = cat_series.value_counts(normalize=True, dropna=True)
    return float(vc.iloc[0]) if len(vc) else np.nan

def pick_positive_label(labels_lower):
    """binary일 때 어떤 라벨을 1로 둘지 휴리스틱"""
    if "goal" in labels_lower:
        return "goal"
    if "successful" in labels_lower:
        return "successful"
    if "won" in labels_lower:
        return "won"
    # 그 외엔 그냥 첫 번째(정렬)로 고정
    return sorted(list(labels_lower))[0]

def grid_metric_and_count(data: pd.DataFrame, xcol: str, ycol: str, y: pd.Series,
                          task_kind: str,
                          x_bins=X_BINS, y_bins=Y_BINS, min_count=MIN_CELL_COUNT):
    tmp = data[[xcol, ycol]].copy()
    tmp["__y__"] = y.values
    tmp = tmp.dropna(subset=[xcol, ycol, "__y__"])

    tmp["x_bin"] = pd.cut(tmp[xcol], bins=x_bins)
    tmp["y_bin"] = pd.cut(tmp[ycol], bins=y_bins)

    grp = tmp.groupby(["y_bin", "x_bin"], observed=True)["__y__"]
    cnt = grp.size().unstack().fillna(0).astype(int)

    if task_kind == "binary":
        metric = grp.mean().unstack()
    else:
        metric = grp.apply(lambda s: top1_share(s)).unstack()

    metric = metric.astype(float)
    mask = (cnt < min_count) | metric.isna()
    return metric, cnt, mask

# =========================
# FEATURE: goal_target 생성
# =========================
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

# =========================
# COLUMN DETECTION
# =========================
TIME_COL = "time_seconds"
PERIOD_COL = "period_id"
COORD_SET = ("start_x", "start_y", "end_x", "end_y")

has_time = TIME_COL in df.columns
has_period = PERIOD_COL in df.columns
has_coord = has_cols(df, COORD_SET)

# =========================
# TARGET RESOLUTION
# =========================
def resolve_type_target(sub: pd.DataFrame, type_name: str):
    """
    반환:
      - task_kind: "binary" | "multiclass" | "none"
      - target_name: str
      - y: pd.Series or None
      - meta: dict (positive_label / n_classes / top_class 등)
    """
    tname = str(type_name)

    # ✅ 요청 반영: 특정 타입은 result_name 기준으로 이진/다중 자동 판단
    if tname in RESULTNAME_TASK_TYPES and "result_name" in sub.columns:
        rn_raw = sub["result_name"].astype(str).str.strip()
        rn_raw = rn_raw.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        rn = rn_raw.dropna()
        uniq = rn.unique().tolist()

        if len(uniq) == 2:
            labels_lower = set(rn.str.lower().unique().tolist())
            pos = pick_positive_label(labels_lower)
            y = rn_raw.astype(str).str.strip().str.lower().eq(pos).astype(int)
            if y.nunique() >= 2:
                return "binary", f"result_name=={pos}", y, {"positive_label": pos, "n_classes": 2}
            return "none", "result_name_binary", None, {}

        if len(uniq) >= 3:
            y = rn_raw.fillna("NA").astype(str)
            if y.nunique() >= 2:
                topc = y.value_counts().idxmax()
                return "multiclass", f"result_name_multiclass(k={y.nunique()})", y, {"top_class": topc, "n_classes": int(y.nunique())}
            return "none", "result_name_multiclass", None, {}

        return "none", "result_name", None, {}

    # goal / own goal 은 goal_target로 (result_name이 비어있는 케이스)
    if tname in {"Goal", "Own Goal"}:
        y = sub["goal_target"].astype(int)
        return "binary", "goal_target", (y if y.nunique() >= 2 else None), {"positive_label": "goal_target", "n_classes": 2}

    # 일반 성공/실패 이진
    if "result_name" in sub.columns:
        rn = sub["result_name"].astype(str).str.strip().str.lower()
        uniq = set(rn.dropna().unique().tolist())

        if ("successful" in uniq) and ("unsuccessful" in uniq):
            y = rn.eq("successful").astype(int)
            return "binary", "success_flag", (y if y.nunique() >= 2 else None), {"positive_label": "successful", "n_classes": 2}

        # Duel류: won/lost
        if "duel" in tname.lower():
            if ("won" in uniq) and ("lost" in uniq):
                y = rn.eq("won").astype(int)
                return "binary", "duel_win", (y if y.nunique() >= 2 else None), {"positive_label": "won", "n_classes": 2}

    # Foul류: 카드 여부
    if "foul" in tname.lower() and "result_name" in sub.columns:
        rn_raw = sub["result_name"].astype(str).fillna("")
        card_pat = re.compile(r"(yellow|red|card)", re.IGNORECASE)
        y = rn_raw.apply(lambda s: 1 if card_pat.search(s) else 0).astype(int)
        if y.sum() > 0 and y.nunique() >= 2:
            return "binary", "card_flag", y, {"positive_label": "card", "n_classes": 2}
        return "none", "card_flag", None, {}

    return "none", "none", None, {}

# =========================
# 1) EDA: type별 overview (1장)
# =========================
type_counts = df["type_name"].value_counts()
type_indices = df.groupby("type_name", sort=False).indices

rows = []
for t in type_counts.index:
    idx = type_indices.get(t)
    sub = df.loc[idx]

    task_kind, target_name, y, meta = resolve_type_target(sub, t)
    n = len(sub)

    if y is None:
        rows.append((t, n, task_kind, target_name, np.nan, np.nan, np.nan, np.nan, ""))
        continue

    if task_kind == "binary":
        pos = int(y.sum())
        neg = int((1 - y).sum())
        rate = float(y.mean())
        rows.append((t, n, task_kind, target_name, rate, pos, neg, 2, meta.get("positive_label", "")))
    else:
        # multiclass: target_rate = top1 share (한 클래스 쏠림 정도)
        rate = top1_share(y)
        ncls = int(y.nunique())
        topc = str(meta.get("top_class", y.value_counts().idxmax()))
        rows.append((t, n, task_kind, target_name, rate, np.nan, np.nan, ncls, topc))

ov = pd.DataFrame(rows, columns=[
    "type_name", "n", "task_kind", "target_name",
    "target_rate", "pos", "neg", "n_classes", "top_class_or_poslabel"
])

ov_valid = ov.dropna(subset=["target_rate"]).sort_values("n", ascending=False)
topN = ov_valid.head(15).copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].barh(topN["type_name"][::-1], topN["n"][::-1])
axes[0].set_title("Top types by count (valid target)")
axes[0].set_xlabel("count")

axes[1].barh(topN["type_name"][::-1], topN["target_rate"][::-1])
axes[1].set_title("Target rate by type (binary mean / multiclass top1 share)")
axes[1].set_xlabel("rate")
axes[1].set_xlim(0, 1)

fig.suptitle("Type Overview (what is worth analyzing)", y=1.02)
save_fig(f"{RESULT_DIR}/eda_type_overview.png")

# =========================
# 2) EDA: 지정 타입 대시보드 (1장)
# =========================
types_to_plot = DASHBOARD_TYPES[:] if DASHBOARD_TYPES else \
    ov_valid[
        (ov_valid["n"] >= MIN_ROWS_PER_TYPE) &
        (ov_valid["task_kind"] == "binary") &
        (ov_valid["pos"] >= MIN_POSITIVE_PER_TYPE) &
        (ov_valid["neg"] >= MIN_POSITIVE_PER_TYPE)
    ].sort_values("n", ascending=False).head(TOP_K_TYPES)["type_name"].tolist()

n_rows = max(1, len(types_to_plot))
fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5 * n_rows))
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for r, t in enumerate(types_to_plot):
    if t not in type_indices:
        for c in range(4):
            axes[r, c].text(0.5, 0.5, f"{t}\nNot in data", ha="center", va="center")
            axes[r, c].set_axis_off()
        continue

    sub = df.loc[type_indices[t]].copy()
    task_kind, target_name, y, meta = resolve_type_target(sub, t)

    if y is None:
        for c in range(4):
            axes[r, c].text(0.5, 0.5, f"{t}\nNo usable target", ha="center", va="center")
            axes[r, c].set_axis_off()
        continue

    metric_label = "rate" if task_kind == "binary" else "top1 share"

    # ---- (0) 전체 분포 텍스트(참고) ----
    # binary: positive rate, multiclass: top3 class share
    dist_txt = ""
    if task_kind == "binary":
        dist_txt = f"mean={y.mean():.3f}"
    else:
        vc = y.value_counts(normalize=True).head(3)
        dist_txt = "top3: " + ", ".join([f"{k}({v:.2f})" for k, v in vc.items()])

    # ---------- (1) time trend ----------
    ax = axes[r, 0]
    if has_time and sub[TIME_COL].notna().sum() > 0 and sub[TIME_COL].nunique() >= 5:
        tmp = sub[[TIME_COL]].copy()
        tmp["__y__"] = y.values
        tmp = tmp.dropna()

        if len(tmp) >= 100:
            if has_period and sub[PERIOD_COL].notna().sum() > 0:
                tmp2 = sub[[TIME_COL, PERIOD_COL]].copy()
                tmp2["__y__"] = y.values
                tmp2 = tmp2.dropna()

                for pid in sorted(tmp2[PERIOD_COL].unique()):
                    part = tmp2[tmp2[PERIOD_COL] == pid].copy()
                    part["time_bin"] = make_bins(part[TIME_COL], n_bins=10)

                    if task_kind == "binary":
                        trend = part.groupby("time_bin", observed=True)["__y__"].mean()
                    else:
                        trend = part.groupby("time_bin", observed=True)["__y__"].apply(top1_share)

                    ax.plot(range(len(trend)), trend.values, marker="o", label=f"period {int(pid)}")

                ax.legend()
                ax.set_title(f"{t} | {target_name} {metric_label} over time (by period)")
            else:
                tmp["time_bin"] = make_bins(tmp[TIME_COL], n_bins=12)
                if task_kind == "binary":
                    trend = tmp.groupby("time_bin", observed=True)["__y__"].mean()
                else:
                    trend = tmp.groupby("time_bin", observed=True)["__y__"].apply(top1_share)

                ax.plot(range(len(trend)), trend.values, marker="o")
                ax.set_title(f"{t} | {target_name} {metric_label} over time (binned)")

            ax.set_ylim(0, 1)
            ax.set_xlabel("bin index")
            ax.set_ylabel(metric_label)
            ax.text(0.02, 0.02, dist_txt, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()

    # ---------- (2) distance metric curve ----------
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
            if task_kind == "binary":
                rate = tmp.groupby("dist_bin", observed=True)["__y__"].mean()
            else:
                rate = tmp.groupby("dist_bin", observed=True)["__y__"].apply(top1_share)
            cnt = tmp.groupby("dist_bin", observed=True)["__y__"].size()

            ax.plot(range(len(rate)), rate.values, marker="o")
            ax.set_ylim(0, 1)
            ax.set_title(f"{t} | {target_name} {metric_label} by distance (quantile bins)")
            ax.set_xlabel("distance bin index")
            ax.set_ylabel(metric_label)
            ax.text(0.02, 0.02, f"min bin count={int(cnt.min())}", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()

    # ---------- (3) startpos metric grid + (4) count grid ----------
    ax = axes[r, 2]
    ax2 = axes[r, 3]
    if has_coord:
        sx, sy, ex, ey = COORD_SET
        ok = sub[[sx, sy]].notna().all(axis=1).sum() >= 500
        if ok:
            metric, cnt, mask = grid_metric_and_count(sub.dropna(subset=[sx, sy]), sx, sy, y, task_kind)
            sns.heatmap(metric, ax=ax, cmap="viridis", vmin=0, vmax=1, mask=mask)
            ax.set_title(f"{t} | startpos {metric_label} (mask < {MIN_CELL_COUNT})")
            ax.set_xlabel("x_bin"); ax.set_ylabel("y_bin")

            sns.heatmap(cnt, ax=ax2, cmap="magma")
            ax2.set_title(f"{t} | startpos count")
            ax2.set_xlabel("x_bin"); ax2.set_ylabel("y_bin")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center"); ax2.set_axis_off()
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
        ax2.text(0.5, 0.5, "N/A", ha="center", va="center"); ax2.set_axis_off()

fig.suptitle("Selected Types Dashboard (fixed list)", y=1.01)
save_fig(f"{RESULT_DIR}/eda_top_types_dashboard.png")

# =========================
# 3) goal_target 전용 대시보드 (1장)
# =========================
shot_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

gt_counts = df["goal_target"].value_counts().sort_index()
axes[0].barh(gt_counts.index.astype(str), gt_counts.values)
axes[0].set_title("goal_target Distribution (All Events)")
axes[0].set_xlabel("count")

if len(shot_df) > 0:
    rate_by_type = shot_df.groupby("type_name", observed=True)["goal_target"].mean().sort_values()
    axes[1].barh(rate_by_type.index.astype(str), rate_by_type.values)
    axes[1].set_title("Goal rate by shot-candidate type")
    axes[1].set_xlim(0, 1)
else:
    axes[1].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[1].set_axis_off()

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

if has_coord and len(shot_df) > 0:
    sx, sy, ex, ey = COORD_SET
    d2 = shot_df.dropna(subset=[sx, sy, "goal_target"])
    if len(d2) >= 300:
        metric, cnt, mask = grid_metric_and_count(d2, sx, sy, d2["goal_target"], "binary")
        sns.heatmap(metric, ax=axes[3], cmap="viridis", vmin=0, vmax=1, mask=mask)
        axes[3].set_title(f"StartPos goal rate (mask < {MIN_CELL_COUNT})")
        sns.heatmap(cnt, ax=axes[4], cmap="magma")
        axes[4].set_title("StartPos count")
    else:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
        axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()
else:
    axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
    axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()

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

# =========================
# 4~6) Feature importance (goal_target) 3종
#     - 드롭 컬럼셋이 달라서 "각각 모델을 다시 학습"함
# =========================
def aggregate_ohe_importance(imp_series: pd.Series, cat_cols: list) -> pd.Series:
    """
    OHE feature명을 원본 컬럼 단위로 합산.
    get_feature_names_out은 보통 "{col}_{category}"라서
    col명이 언더스코어 포함해도 안전하게 prefix 매칭으로 처리.
    """
    prefixes = sorted([c for c in cat_cols], key=len, reverse=True)
    out = {}
    for feat, val in imp_series.items():
        base = None
        for c in prefixes:
            if feat.startswith(c + "_"):
                base = c
                break
        if base is None:
            base = feat
        out[base] = out.get(base, 0.0) + float(val)
    return pd.Series(out).sort_values(ascending=False)

def train_extratrees_and_plot_importance(model_df: pd.DataFrame, y: pd.Series, out_png: str):
    drop_cols = [c for c in ["result_name", "goal_target"] if c in model_df.columns]
    X = model_df.drop(columns=drop_cols, errors="ignore")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(steps=[("imp", SimpleImputer(strategy="median"))])
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

    imp_agg = aggregate_ohe_importance(imp, cat_cols=cat_cols)
    imp_top = imp_agg.head(30).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    imp_top.plot(kind="barh")
    plt.title(out_png.replace(".png", ""))
    plt.xlabel("importance")
    save_fig(f"{RESULT_DIR}/{out_png}")

# goal_target 모델은 shot 후보에서만 학습
base_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()
y_goal = base_df["goal_target"].astype(int)

# 학습에 불필요한 키 제거(있으면)
base_df = base_df.drop(columns=[c for c in ["game_id", "action_id"] if c in base_df.columns], errors="ignore")

# 3회 학습 + importance 3장
train_extratrees_and_plot_importance(
    model_df=base_df,
    y=y_goal,
    out_png="goal_target_feature_importance_top30.png"
)

drop_coords = [c for c in COORD_DROP_COLS if c in base_df.columns]
no_coords_df = base_df.drop(columns=drop_coords, errors="ignore")
train_extratrees_and_plot_importance(
    model_df=no_coords_df,
    y=y_goal,
    out_png="goal_target_feature_importance_top30_no_coords.png"
)

drop_entities = [c for c in ENTITY_DROP_COLS if c in no_coords_df.columns]
no_coords_no_entities_df = no_coords_df.drop(columns=drop_entities, errors="ignore")
train_extratrees_and_plot_importance(
    model_df=no_coords_no_entities_df,
    y=y_goal,
    out_png="goal_target_feature_importance_top30_no_coords_no_entities.png"
)
