#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG (여기만 바꾸면 됨)
# =========================================================
THIS_DIR = "../../../FiledNet_pkl_temp/infer_results/catboost"

IN_OOF_PATH  = os.path.join(THIS_DIR, "oof_preds.csv")              # 입력
OUT_PATH     = os.path.join(THIS_DIR, "oof_preds_preprocessed.csv") # 출력

FUTURE_MIN_WINDOW = 5   # "향후 5분" -> m..m+5 inclusive => window=6
# =========================================================


def build_continuous_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    game_id별로 period를 이어붙인 연속 시간(cont_time_sec)을 만든다.
    offset(period p) = sum(max_time(period q)) for q < p  (stoppage time 반영)
    """
    df = df.copy()
    df["_row"] = np.arange(len(df), dtype=np.int64)

    # 안정 정렬
    df = df.sort_values(["game_id", "period_id", time_col, "_row"], kind="mergesort").reset_index(drop=True)

    # period_end = 각 (game, period)의 마지막 time
    period_end = (
        df.groupby(["game_id", "period_id"], as_index=False)[time_col]
          .max()
          .rename(columns={time_col: "period_end"})
    )

    # offset 계산
    period_end["offset"] = 0.0
    for gid, g in period_end.groupby("game_id"):
        g = g.sort_values("period_id")
        offsets = []
        cum = 0.0
        for _, r in g.iterrows():
            offsets.append(cum)
            cum += float(r["period_end"])
        period_end.loc[g.index, "offset"] = offsets

    df = df.merge(period_end[["game_id", "period_id", "offset"]], on=["game_id", "period_id"], how="left")
    df["cont_time_sec"] = df[time_col].astype(float) + df["offset"].astype(float)
    return df


def preprocess_oof_to_minute(in_path: str, out_path: str, future_min_window: int = 5):
    df = pd.read_csv(in_path)

    # time 컬럼 호환 (time / time_seconds)
    if "time" in df.columns:
        time_col = "time"
    elif "time_seconds" in df.columns:
        time_col = "time_seconds"
    else:
        raise KeyError(f"Expected 'time' or 'time_seconds'. got={df.columns.tolist()}")

    need = ["game_id", "period_id", time_col, "home_pred", "away_pred",
            "home_team_name", "away_team_name", "type_result"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")

    # 1) 필요한 컬럼만 유지
    df = df[need].copy()

    # 2) 연속 시간 + minutes
    df = build_continuous_time(df, time_col=time_col)
    df["minutes"] = np.floor(df["cont_time_sec"] / 60.0).astype(int)

    # 3) minute-level raw (분 안에서 max)
    preds_min = (
        df.groupby(["game_id", "minutes"], as_index=False)[["home_pred", "away_pred"]]
          .max()
          .rename(columns={"home_pred": "home_pred_raw", "away_pred": "away_pred_raw"})
    )

    # context는 "그 분의 마지막 액션" 기준으로 period_id/type_result를 채움 (프론트 표시용)
    df = df.sort_values(["game_id", "minutes", "cont_time_sec"], kind="mergesort")
    context = (
        df.groupby(["game_id", "minutes"], as_index=False)
          .tail(1)[["game_id", "minutes", "period_id", "type_result",
                    "home_team_name", "away_team_name"]]
    )

    min_df = context.merge(preds_min, on=["game_id", "minutes"], how="left")

    # 4) 향후 5분(max) => m..m+5 inclusive => window = future_min_window + 1
    W = int(future_min_window) + 1

    out_parts = []
    for gid, g in min_df.groupby("game_id"):
        g = g.sort_values("minutes").reset_index(drop=True)

        # forward max: 뒤집어서 rolling max 후 다시 뒤집기
        g["home_pred"] = g["home_pred_raw"].iloc[::-1].rolling(W, min_periods=1).max().iloc[::-1].values
        g["away_pred"] = g["away_pred_raw"].iloc[::-1].rolling(W, min_periods=1).max().iloc[::-1].values

        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)

    # 최종 컬럼
    out = out[[
        "game_id", "minutes", "period_id",
        "home_pred", "away_pred",
        "home_team_name", "away_team_name",
        "type_result"
    ]].sort_values(["game_id", "minutes"], kind="mergesort").reset_index(drop=True)

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved -> {out_path} | rows={len(out)}")


def main():
    if not os.path.exists(IN_OOF_PATH):
        raise FileNotFoundError(f"Input not found: {IN_OOF_PATH}")
    preprocess_oof_to_minute(IN_OOF_PATH, OUT_PATH, FUTURE_MIN_WINDOW)


if __name__ == "__main__":
    main()
