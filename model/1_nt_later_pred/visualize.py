#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# CONFIG (여기만 수정)
# =========================
INPUT_CSV = r"../../../FiledNet_pkl_temp/infer_results/catboost/oof_preds.csv"   # 네 oof_preds.csv 경로
OUT_DIR   = r"./plots_by_game"                # 결과 이미지 저장 폴더

# 너무 촘촘하면 보기 힘드니 옵션 제공
RESAMPLE_SEC = 5     # None이면 원본 그대로 / 5면 5초 단위 평균
ROLLING_SEC  = 0     # 0이면 미적용 / 예: 30이면 30초 롤링 평균

SHOW_FIRST_N = 0     # 0이면 화면 표시 안 함 / 예: 3이면 앞 3경기만 plt.show()

# 후반을 45분(2700초) 뒤에 붙여서 전체 타임라인으로 만들지 여부
USE_MATCH_TIME = True

# =========================
# Utils
# =========================
def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:120]

def maybe_resample(g: pd.DataFrame) -> pd.DataFrame:
    if RESAMPLE_SEC is None:
        return g

    tcol = "match_time_sec" if USE_MATCH_TIME else "time"
    # binning
    bin_sec = (np.floor(g[tcol].astype(float) / RESAMPLE_SEC) * RESAMPLE_SEC).astype(float)
    gg = g.copy()
    gg["bin_sec"] = bin_sec

    agg = (
        gg.groupby("bin_sec", as_index=False)
          .agg({
              "home_pred": "mean",
              "away_pred": "mean",
              "period_id": "max",
          })
    )
    agg.rename(columns={"bin_sec": tcol}, inplace=True)
    return agg

def maybe_rolling(g: pd.DataFrame) -> pd.DataFrame:
    if not ROLLING_SEC or ROLLING_SEC <= 0:
        return g

    tcol = "match_time_sec" if USE_MATCH_TIME else "time"
    gg = g.sort_values(tcol).copy()

    # rolling window를 "개수"로 바꾸기 위해, resample이 켜져있다면 그 간격을 사용
    if RESAMPLE_SEC is not None and RESAMPLE_SEC > 0:
        win = max(1, int(round(ROLLING_SEC / RESAMPLE_SEC)))
    else:
        # 원본이 초 단위가 아닐 수 있어 대략적으로 30개 같은 식으로는 위험 -> 보수적으로 30
        win = 30

    gg["home_pred"] = gg["home_pred"].rolling(win, min_periods=1).mean()
    gg["away_pred"] = gg["away_pred"].rolling(win, min_periods=1).mean()
    return gg

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    need_cols = ["game_id", "period_id", "time", "home_pred", "away_pred", "home_team_name", "away_team_name"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # 숫자 변환/정리
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["period_id"] = pd.to_numeric(df["period_id"], errors="coerce").fillna(1).astype(int)
    df["home_pred"] = pd.to_numeric(df["home_pred"], errors="coerce")
    df["away_pred"] = pd.to_numeric(df["away_pred"], errors="coerce")

    # 전후반 이어붙인 시간축
    if USE_MATCH_TIME:
        df["match_time_sec"] = df["time"] + (df["period_id"] - 1) * 45 * 60
        tcol = "match_time_sec"
    else:
        tcol = "time"

    # 경기별 plot
    game_ids = df["game_id"].dropna().unique().tolist()
    shown = 0

    for gid in tqdm(game_ids, desc="Plot games", dynamic_ncols=True):
        g = df[df["game_id"] == gid].copy()
        g = g.dropna(subset=[tcol, "home_pred", "away_pred"])
        if len(g) == 0:
            continue

        home = g["home_team_name"].iloc[0]
        away = g["away_team_name"].iloc[0]

        g = g.sort_values([tcol], kind="mergesort")

        # 보기 좋게: resample -> rolling
        g2 = maybe_resample(g)
        g2 = maybe_rolling(g2)
        g2 = g2.sort_values([tcol], kind="mergesort")

        x_min = g2[tcol].astype(float) / 60.0

        plt.figure(figsize=(12, 4))
        plt.plot(x_min, g2["home_pred"].values, label="home_pred")
        plt.plot(x_min, g2["away_pred"].values, label="away_pred")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Goal probability (next 10 actions)")
        plt.title(f"game_id={gid} | {home} vs {away}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        fname = safe_filename(f"{gid}_{home}_vs_{away}.png")
        out_path = os.path.join(OUT_DIR, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)

        # 화면 표시(옵션)
        if SHOW_FIRST_N and shown < SHOW_FIRST_N:
            plt.show()
            shown += 1

        plt.close()

    print(f"[DONE] saved plots -> {OUT_DIR}")

if __name__ == "__main__":
    main()