#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BAYES_DIR = "../../../FiledNet_pkl_temp/results/results_bayes_winprob"
CAT_DIR   = "../../../FiledNet_pkl_temp/infer_results/catboost"

BAYES_CSV = os.path.join(BAYES_DIR, "oof_all_minutes_ext.csv")
CAT_CSV   = os.path.join(CAT_DIR,   "oof_preds_preprocessed.csv")

OUT_CSV   = os.path.join(BAYES_DIR, "preprocessed_oof_all_minutes_ext.csv")

DEBUG_N = 20


# =========================================================
# Helpers
# =========================================================
def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}\nAvailable: {list(df.columns)}")

def to_numeric_safe(s: pd.Series, colname: str):
    out = pd.to_numeric(s, errors="coerce")
    n_bad = out.isna().sum()
    if n_bad > 0:
        print(f"[WARN] {colname}: {n_bad} values could not be parsed to numeric (set to NaN).")
    return out

def pick_time_col(df: pd.DataFrame, candidates, name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"[{name}] time-like column not found. "
        f"Tried: {candidates}. Available: {list(df.columns)}"
    )

def summarize_max_minutes(df_in: pd.DataFrame, name: str, time_col_min: str) -> pd.DataFrame:
    gp = (
        df_in.dropna(subset=["game_id", "period_id", time_col_min])
            .groupby(["game_id", "period_id"], as_index=False)[time_col_min]
            .max()
    )
    p1 = gp[gp["period_id"] == 1][["game_id", time_col_min]].rename(columns={time_col_min: f"{name}_p1_max"})
    p2 = gp[gp["period_id"] == 2][["game_id", time_col_min]].rename(columns={time_col_min: f"{name}_p2_max"})
    return p1.merge(p2, on="game_id", how="outer")


# =========================================================
# Main
# =========================================================
def main():
    if not os.path.exists(BAYES_CSV):
        raise FileNotFoundError(f"Not found: {BAYES_CSV}")
    if not os.path.exists(CAT_CSV):
        raise FileNotFoundError(f"Not found: {CAT_CSV}")

    bayes = pd.read_csv(BAYES_CSV)
    cat   = pd.read_csv(CAT_CSV)

    require_cols(bayes, ["game_id", "period_id", "time"], "oof_all_minutes_ext.csv")
    require_cols(cat,   ["game_id", "period_id", "minutes"], "oof_preds_preprocessed.csv")

    print(f"[Bayes] rows={len(bayes):,} | unique game_id={bayes['game_id'].nunique():,}")
    print(f"[Cat]   rows={len(cat):,}   | unique game_id={cat['game_id'].nunique():,}")

    # numeric
    bayes["time"] = to_numeric_safe(bayes["time"], "bayes.time")         # seconds (period 내)
    bayes["period_id"] = to_numeric_safe(bayes["period_id"], "bayes.period_id").astype("Int64")
    cat["minutes"] = to_numeric_safe(cat["minutes"], "cat.minutes")      # match-global minutes
    cat["period_id"] = to_numeric_safe(cat["period_id"], "cat.period_id").astype("Int64")

    # ---------------------------------------------------------
    # 3) cat 기준 game_id별 period1_max / period2_max
    # ---------------------------------------------------------
    cat_gp = (
        cat.dropna(subset=["game_id", "period_id", "minutes"])
           .groupby(["game_id", "period_id"], as_index=False)["minutes"]
           .max()
           .rename(columns={"minutes": "cat_max_min"})
    )
    p1 = cat_gp[cat_gp["period_id"] == 1][["game_id", "cat_max_min"]].rename(columns={"cat_max_min": "p1_max"})
    p2 = cat_gp[cat_gp["period_id"] == 2][["game_id", "cat_max_min"]].rename(columns={"cat_max_min": "p2_max"})

    limits = p1.merge(p2, on="game_id", how="outer")
    limits["p1_max"] = limits["p1_max"].fillna(0.0)
    # 후반 길이(필터링용): p2_max - p1_max
    limits["p2_len"] = (limits["p2_max"] - limits["p1_max"]).fillna(0.0)

    print("\n[Cat limits preview]")
    print(limits.head(10).to_string(index=False))

    # ---------------------------------------------------------
    # 4) bayes time(seconds) -> period 내 minutes(float)
    # ---------------------------------------------------------
    bayes["minute_in_half"] = bayes["time"] / 60.0  # 0,1,2,... 형식(분)

    # ---------------------------------------------------------
    # 5) 필터: cat의 실제 경기 길이까지만 남기기
    #    - p1: minute_in_half <= p1_max
    #    - p2: minute_in_half <= p2_len
    # ---------------------------------------------------------
    bayes2 = bayes.merge(limits[["game_id", "p1_max", "p2_len"]], on="game_id", how="left")

    missing_gid_rows = bayes2["p1_max"].isna().sum()
    if missing_gid_rows > 0:
        print(f"\n[WARN] {missing_gid_rows:,} rows have game_id not found in cat limits -> will be dropped.")
    bayes2 = bayes2.dropna(subset=["p1_max", "p2_len"])

    mask_p1 = (bayes2["period_id"] == 1) & (bayes2["minute_in_half"] <= bayes2["p1_max"] + 1e-9)
    mask_p2 = (bayes2["period_id"] == 2) & (bayes2["minute_in_half"] <= bayes2["p2_len"] + 1e-9)

    filtered = bayes2[mask_p1 | mask_p2].copy()

    # ---------------------------------------------------------
    # ✅ 여기서 핵심 변경:
    # 결과 time을 cat의 minutes 형식(경기 전체 누적 분)으로 생성
    # - period1: time = minute_in_half
    # - period2: time = p1_max + minute_in_half
    # 그리고 period2에서 minute_in_half==0 row 삭제(중복 방지)
    # ---------------------------------------------------------
    # 중복 방지: 후반 시작 0분 제거
    filtered = filtered[~((filtered["period_id"] == 2) & (filtered["minute_in_half"].abs() < 1e-9))].copy()

    # cat-style time 생성
    filtered["time"] = filtered["minute_in_half"]
    p2_idx = (filtered["period_id"] == 2)
    filtered.loc[p2_idx, "time"] = filtered.loc[p2_idx, "p1_max"] + filtered.loc[p2_idx, "minute_in_half"]

    # 정리
    filtered = filtered.drop(columns=["minute_in_half", "p1_max", "p2_len"], errors="ignore")

    os.makedirs(BAYES_DIR, exist_ok=True)
    filtered.to_csv(OUT_CSV, index=False)

    print("\n====================")
    print(f"Saved: {OUT_CSV}")
    print(f"Before: {len(bayes):,} rows  -> After: {len(filtered):,} rows")
    print("====================")

    # ---------------------------------------------------------
    # ✅ 추가 진단: cat vs bayes_raw vs bayes_filtered max minutes 비교
    # ---------------------------------------------------------
    # cat: minutes 그대로
    cat_max = summarize_max_minutes(cat.assign(time_min=cat["minutes"]), "cat", "time_min")

    # bayes_raw: (period1은 minute_in_half) / (period2는 p1_max + minute_in_half) 기준으로 비교해야 일관됨
    # -> bayes_raw도 limits merge해서 cat-style time 만들어서 max를 보자
    bayes_raw = bayes.merge(limits[["game_id", "p1_max"]], on="game_id", how="left")
    bayes_raw["minute_in_half"] = bayes_raw["time"] / 60.0
    bayes_raw["time_cat_style"] = bayes_raw["minute_in_half"]
    bayes_raw.loc[bayes_raw["period_id"] == 2, "time_cat_style"] = bayes_raw.loc[bayes_raw["period_id"] == 2, "p1_max"] + bayes_raw.loc[bayes_raw["period_id"] == 2, "minute_in_half"]
    bayes_raw_max = summarize_max_minutes(bayes_raw.assign(time_min=bayes_raw["time_cat_style"]), "bayes_raw", "time_min")

    # filtered: 이미 time이 cat-style minutes임
    filtered_max = summarize_max_minutes(filtered.assign(time_min=filtered["time"]), "bayes_filtered", "time_min")

    diag = (
        cat_max.merge(bayes_raw_max, on="game_id", how="outer")
              .merge(filtered_max, on="game_id", how="outer")
    )

    cols_order = [
        "game_id",
        "cat_p1_max", "cat_p2_max",
        "bayes_raw_p1_max", "bayes_raw_p2_max",
        "bayes_filtered_p1_max", "bayes_filtered_p2_max",
    ]
    for c in cols_order:
        if c not in diag.columns:
            diag[c] = pd.NA

    print(f"\n[DEBUG] per-game max minutes comparison (head {DEBUG_N})")
    print(diag[cols_order].head(DEBUG_N).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        sys.exit(1)
