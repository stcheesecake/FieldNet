#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FieldNet EDA Script (MINIMAL OUTPUT, FIXED)

- df = pd.read_csv("../data/raw_data.csv") 고정
- print 제거 + tqdm만 사용
- 결과물: 딱 5개 PNG만 저장
  1) eda_type_overview.png
  2) eda_top_types_dashboard.png  (핵심 type 3개만)
  3) goal_target_dashboard.png
  4) goal_target_feature_importance_top30.png                 (컬럼 단위로 집계!)
  5) goal_target_feature_importance_top30_no_coords.png       (컬럼 단위로 집계!)

핵심 Fix
- OHE feature importance를 "원본 컬럼 단위"로 합산해서 TOP30만 그림
- 언더스코어 포함 컬럼명(team_name_ko 등)도 정확히 집계
- 한글 폰트: OS 폰트 파일 addfont 방식(DejaVu 경고 완화)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 먼저 backend 고정

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# FONT (Korean)
# =========================
def set_korean_font():
    """
    - conda/matplotlib이 폰트 캐시 때문에 DejaVu로 잡히는 경우가 잦아서
      '폰트 파일 경로'를 직접 addfont 하는 방식이 가장 안정적임.
    """
    font_paths = [
        # Windows
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        # macOS
        "/System/Library/Fonts/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        # Linux (설치돼 있으면)
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    ]

    chosen = None
    for fp in font_paths:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            chosen = fm.FontProperties(fname=fp).get_name()
            break

    if chosen:
        mpl.rcParams["font.family"] = chosen

    mpl.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

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
MIN_CELL_COUNT = 30  # 성공률 grid에서 이 미만 셀은 마스킹

# 핵심 type 몇 개만 보여줄지
TOP_K_TYPES = 3
MIN_ROWS_PER_TYPE = 3000
MIN_POSITIVE_PER_TYPE = 50

# goal_target 정의 옵션
INCLUDE_OWN_GOAL = False  # "골(득점 기록)"이면 True 권장

# 좌표 제거 importance를 위한 컬럼(없으면 자동 무시)
COORD_DROP_COLS = ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]

# 골 모델/EDA 대상(너가 정의한 샷 후보)
SHOT_CANDIDATE_TYPES = ["Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"]


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
    s = series.dropna()
    if s.nunique() < 2:
        return pd.Series([np.nan] * len(series), index=series.index)
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=min(n_bins, max(2, series.nunique())))

def midpoints_from_intervals(idx):
    mids = []
    for itv in idx:
        if hasattr(itv, "mid"):
            mids.append(float(itv.mid))
        else:
            mids.append(np.nan)
    return np.array(mids)

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


# =========================
# FEATURE: goal_target
# =========================
def build_goal_target(data: pd.DataFrame, include_own_goal: bool = True) -> pd.Series:
    """
    goal_target = 1 조건 (네가 정의한 그대로):
      - (type_name == "Goal" && result_name is NaN)
      - (type_name in ["Penalty Kick","Shot","Shot_Corner","Shot_Freekick"] && result_name == "Goal")
      - (optional) (type_name == "Own Goal" && result_name is NaN)
    """
    t = data["type_name"].astype(str)
    rn = data["result_name"]

    cond_goal_event = (t.eq("Goal") & rn.isna())

    shot_goal_types = ["Penalty Kick", "Shot", "Shot_Corner", "Shot_Freekick"]
    cond_shot_goal = (
        t.isin(shot_goal_types) &
        rn.astype(str).str.strip().str.lower().eq("goal")
    )

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
# TARGET MAPPING (type별)
# =========================
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

    # Successful/Unsuccessful
    if "result_name" in sub.columns:
        rn = sub["result_name"].astype(str).str.strip().str.lower()
        uniq = set(rn.dropna().unique().tolist())

        if ("successful" in uniq) and ("unsuccessful" in uniq):
            y = rn.eq("successful").astype(int)
            return "success_flag", (y if y.nunique() >= 2 else None)

        # Duel: won/lost
        if "duel" in t.lower():
            if ("won" in uniq) and ("lost" in uniq):
                y = rn.eq("won").astype(int)
                return "duel_win", (y if y.nunique() >= 2 else None)

    # Foul: 카드 여부
    if "foul" in t.lower() and "result_name" in sub.columns:
        rn_raw = sub["result_name"].astype(str).fillna("")
        card_pat = re.compile(r"(yellow|red|card)", re.IGNORECASE)
        y = rn_raw.apply(lambda s: 1 if card_pat.search(s) else 0).astype(int)
        if y.sum() > 0 and y.nunique() >= 2:
            return "card_flag", y
        return "card_flag", None

    return "none", None


# =========================
# 중요: OHE importance를 "원본 컬럼 단위"로 정확히 합산
# =========================
def aggregate_ohe_importance_to_columns(imp: pd.Series, cat_cols: list[str]) -> pd.Series:
    """
    OHE get_feature_names_out 결과는 보통 "{col}_{category}" 형태.
    col 자체가 언더스코어 포함(team_name_ko 등)이므로 split로 자르면 망함.
    => cat_cols 기반 prefix 매칭으로 원본 컬럼 단위 합산.
    """
    prefixes = sorted([(c, c + "_") for c in cat_cols], key=lambda x: len(x[1]), reverse=True)
    out = {}
    for feat, val in imp.items():
        base = None
        for c, pref in prefixes:
            if feat.startswith(pref):
                base = c
                break
        if base is None:
            base = feat  # numeric 또는 예상 밖 feature
        out[base] = out.get(base, 0.0) + float(val)
    return pd.Series(out).sort_values(ascending=False)


# =========================
# PLOTS (5개만)
# =========================
with tqdm(total=5, desc="EDA", ncols=110) as pbar:

    # =========================
    # 1) EDA: type별 overview (1장)
    # =========================
    type_counts = df["type_name"].value_counts()
    type_indices = df.groupby("type_name", sort=False).indices

    rows = []
    for t in type_counts.index:
        sub = df.loc[type_indices[t]]
        target_name, y = resolve_type_target(sub, t)
        n = len(sub)
        if y is None:
            rows.append((t, n, target_name, np.nan, 0, 0))
        else:
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
    axes[1].set_title("Target rate by type (mapped target)")
    axes[1].set_xlabel("rate")
    axes[1].set_xlim(0, 1)

    fig.suptitle("Type Overview (what is worth analyzing)", y=1.02)
    save_fig(f"{RESULT_DIR}/eda_type_overview.png")
    pbar.update(1)

    # =========================
    # 2) EDA: 핵심 type TOP_K_TYPES만 대시보드 (1장)
    # =========================
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
            tmp = sub[[TIME_COL]].copy()
            tmp["__y__"] = y.values
            tmp = tmp.dropna()
            if len(tmp) >= 200:
                if has_period and sub[PERIOD_COL].notna().sum() > 0:
                    tmp2 = sub[[TIME_COL, PERIOD_COL]].copy()
                    tmp2["__y__"] = y.values
                    tmp2 = tmp2.dropna()
                    for pid in sorted(tmp2[PERIOD_COL].dropna().unique()):
                        part = tmp2[tmp2[PERIOD_COL] == pid].copy()
                        part["time_bin"] = make_bins(part[TIME_COL], n_bins=10)
                        trend = part.groupby("time_bin", observed=True)["__y__"].mean()
                        xs = midpoints_from_intervals(trend.index)
                        ax.plot(xs, trend.values, marker="o", label=f"period {int(pid)}")
                    ax.legend()
                    ax.set_title(f"{t} | {target_name} over time (by period)")
                else:
                    tmp["time_bin"] = make_bins(tmp[TIME_COL], n_bins=12)
                    trend = tmp.groupby("time_bin", observed=True)["__y__"].mean()
                    xs = midpoints_from_intervals(trend.index)
                    ax.plot(xs, trend.values, marker="o")
                    ax.set_title(f"{t} | {target_name} over time (binned)")
                ax.set_ylim(0, 1)
                ax.set_xlabel("time (bin midpoint)")
                ax.set_ylabel("rate")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()

        # (2) distance success curve
        ax = axes[r, 1]
        if has_coord:
            sx, sy, ex, ey = COORD_SET
            ok = sub[[sx, sy, ex, ey]].notna().all(axis=1).sum() >= 500
            if ok:
                tmp = sub[[sx, sy, ex, ey]].copy()
                tmp["__y__"] = y.values
                tmp = tmp.dropna()
                tmp["distance"] = np.sqrt((tmp[ex] - tmp[sx])**2 + (tmp[ey] - tmp[sy])**2)

                tmp["dist_bin"] = make_bins(tmp["distance"], n_bins=10)
                rate = tmp.groupby("dist_bin", observed=True)["__y__"].mean()
                cnt = tmp.groupby("dist_bin", observed=True)["__y__"].size()

                xs = midpoints_from_intervals(rate.index)
                ax.plot(xs, rate.values, marker="o")
                ax.set_ylim(0, 1)
                ax.set_title(f"{t} | {target_name} by distance (binned)")
                ax.set_xlabel("distance (bin midpoint)")
                ax.set_ylabel("rate")
                ax.text(0.02, 0.02, f"min bin n={int(cnt.min())}", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()

        # (3) startpos success rate grid + (4) count grid
        ax = axes[r, 2]
        ax2 = axes[r, 3]
        if has_coord:
            sx, sy, ex, ey = COORD_SET
            ok = sub[[sx, sy]].notna().all(axis=1).sum() >= 800
            if ok:
                rate, cnt, mask = grid_rate_and_count(sub.dropna(subset=[sx, sy]), sx, sy, y)
                sns.heatmap(rate, ax=ax, cmap="viridis", vmin=0, vmax=1, mask=mask,
                            xticklabels=False, yticklabels=False)
                ax.set_title(f"{t} | startpos {target_name} rate (mask<{MIN_CELL_COUNT})")
                ax.set_xlabel("x_bin"); ax.set_ylabel("y_bin")

                sns.heatmap(cnt, ax=ax2, cmap="magma",
                            xticklabels=False, yticklabels=False)
                ax2.set_title(f"{t} | startpos count")
                ax2.set_xlabel("x_bin"); ax2.set_ylabel("y_bin")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
                ax2.text(0.5, 0.5, "N/A", ha="center", va="center"); ax2.set_axis_off()
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.set_axis_off()
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center"); ax2.set_axis_off()

    fig.suptitle(f"Top {len(top_types)} Types Dashboard", y=1.01)
    save_fig(f"{RESULT_DIR}/eda_top_types_dashboard.png")
    pbar.update(1)

    # =========================
    # 3) goal_target 전용 대시보드 (1장)
    # =========================
    shot_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # (a) goal_target 분포 (둘 다 보이게)
    gt_counts = df["goal_target"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    axes[0].bar(["0", "1"], gt_counts.values)
    axes[0].set_title("goal_target Distribution (All Events)")
    axes[0].set_xlabel("goal_target")
    axes[0].set_ylabel("count")

    # (b) 타입별 goal rate (shot candidates)
    if len(shot_df) > 0:
        rate_by_type = shot_df.groupby("type_name", observed=True)["goal_target"].mean().sort_values()
        axes[1].barh(rate_by_type.index.astype(str), rate_by_type.values)
        axes[1].set_title("Goal rate by shot-candidate type")
        axes[1].set_xlim(0, 1)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[1].set_axis_off()

    # (c) 시간대별 goal rate
    if has_time and len(shot_df) > 0 and shot_df[TIME_COL].notna().sum() > 0 and shot_df[TIME_COL].nunique() >= 5:
        if has_period and shot_df[PERIOD_COL].notna().sum() > 0:
            for pid in sorted(shot_df[PERIOD_COL].dropna().unique()):
                part = shot_df[shot_df[PERIOD_COL] == pid][[TIME_COL, "goal_target"]].dropna().copy()
                if len(part) < 80:
                    continue
                part["time_bin"] = make_bins(part[TIME_COL], n_bins=10)
                trend = part.groupby("time_bin", observed=True)["goal_target"].mean()
                xs = midpoints_from_intervals(trend.index)
                axes[2].plot(xs, trend.values, marker="o", label=f"period {int(pid)}")
            axes[2].legend()
            axes[2].set_title("Goal rate over time (shot candidates, by period)")
        else:
            tmp = shot_df[[TIME_COL, "goal_target"]].dropna().copy()
            tmp["time_bin"] = make_bins(tmp[TIME_COL], n_bins=12)
            trend = tmp.groupby("time_bin", observed=True)["goal_target"].mean()
            xs = midpoints_from_intervals(trend.index)
            axes[2].plot(xs, trend.values, marker="o")
            axes[2].set_title("Goal rate over time (shot candidates, binned)")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("time (bin midpoint)")
        axes[2].set_ylabel("rate")
    else:
        axes[2].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[2].set_axis_off()

    # (d)(e) startpos goal rate grid + count
    if has_coord and len(shot_df) > 0:
        sx, sy, ex, ey = COORD_SET
        d2 = shot_df.dropna(subset=[sx, sy, "goal_target"])
        if len(d2) >= 400:
            rate, cnt, mask = grid_rate_and_count(d2, sx, sy, d2["goal_target"])
            sns.heatmap(rate, ax=axes[3], cmap="viridis", vmin=0, vmax=1, mask=mask,
                        xticklabels=False, yticklabels=False)
            axes[3].set_title(f"StartPos goal rate (mask<{MIN_CELL_COUNT})")

            sns.heatmap(cnt, ax=axes[4], cmap="magma",
                        xticklabels=False, yticklabels=False)
            axes[4].set_title("StartPos count")
        else:
            axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
            axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()
    else:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[3].set_axis_off()
        axes[4].text(0.5, 0.5, "N/A", ha="center", va="center"); axes[4].set_axis_off()

    # (f) distance by goal_target (boxplot)
    if has_coord and len(shot_df) > 0:
        sx, sy, ex, ey = COORD_SET
        d3 = shot_df.dropna(subset=[sx, sy, ex, ey, "goal_target"]).copy()
        if len(d3) >= 400:
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
    pbar.update(1)

    # =========================
    # 4) / 5) Tree importance (goal_target)
    #     - TOP30 "컬럼 단위"로 집계해서 그림
    # =========================
    def train_extratrees_and_plot_importance(model_df: pd.DataFrame, y: pd.Series, out_png: str):
        drop_cols = [c for c in ["result_name", "goal_target"] if c in model_df.columns]
        X = model_df.drop(columns=drop_cols, errors="ignore")

        # 범주/수치 분리
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

        # AUC (표시용: print 안 함)
        proba = clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, proba)

        # feature names 복원
        feature_names = []
        feature_names.extend(num_cols)

        if len(cat_cols) > 0:
            ohe = clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
            cat_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(cat_names)

        importances = clf.named_steps["model"].feature_importances_
        imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        # 핵심: 원본 컬럼 단위로 합산 -> TOP30만 plot
        imp_col = aggregate_ohe_importance_to_columns(imp, cat_cols)
        imp_top = imp_col.head(30).sort_values()

        plt.figure(figsize=(10, 8))
        imp_top.plot(kind="barh")
        plt.title(f"{out_png.replace('.png','')} | AUC={auc:.3f} | top30 (column-aggregated)")
        plt.xlabel("importance")
        save_fig(f"{RESULT_DIR}/{out_png}")

    # goal_target 모델은 shot 후보에서만 학습
    model_df = df[df["type_name"].isin(SHOT_CANDIDATE_TYPES)].copy()
    y_goal = model_df["goal_target"].astype(int)

    # ID 컬럼(의미없거나 누수 가능) 제거
    base_df = model_df.drop(columns=[c for c in ["game_id", "action_id"] if c in model_df.columns], errors="ignore")

    train_extratrees_and_plot_importance(
        model_df=base_df,
        y=y_goal,
        out_png="goal_target_feature_importance_top30.png"
    )
    pbar.update(1)

    drop_coords = [c for c in COORD_DROP_COLS if c in base_df.columns]
    model_no_coords = base_df.drop(columns=drop_coords, errors="ignore")

    train_extratrees_and_plot_importance(
        model_df=model_no_coords,
        y=y_goal,
        out_png="goal_target_feature_importance_top30_no_coords.png"
    )
    pbar.update(1)
