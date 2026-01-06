#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np


# =========================
# PATHS
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(THIS_DIR, "../data"))

DATA_CSV = os.path.join(DATA_DIR, "data.csv")
MATCH_INFO_CSV = os.path.join(DATA_DIR, "match_info.csv")
MAP_JSON = os.path.join(DATA_DIR, "preprocess_maps.json")

OUT_GOAL = os.path.join(THIS_DIR, "goal_contained.csv")
OUT_CMP  = os.path.join(THIS_DIR, "goal_vs_matchinfo_comparison.csv")
OUT_AUDIT = os.path.join(THIS_DIR, "goal_code_audit.csv")


# =========================
# MAP HELPERS (same as your code)
# =========================
def type_name_from_type_result_str(s: str) -> str:
    return str(s).split("__", 1)[0].strip()

def result_name_from_type_result_str(s: str) -> str:
    parts = str(s).split("__", 1)
    return parts[1].strip() if len(parts) == 2 else "NA"

def build_code_maps_from_json(map_path: str):
    with open(map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

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

    return code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes


# =========================
# MAIN
# =========================
def main():
    print("[PATHS]")
    print(" DATA_CSV      =", DATA_CSV)
    print(" MATCH_INFO    =", MATCH_INFO_CSV)
    print(" MAP_JSON      =", MAP_JSON)
    print(" OUT_GOAL      =", OUT_GOAL)
    print(" OUT_CMP       =", OUT_CMP)
    print(" OUT_AUDIT     =", OUT_AUDIT)

    code_to_str, code_to_type, code_to_result, goal_codes, own_goal_codes = build_code_maps_from_json(MAP_JSON)

    print("\n[GOAL RULE CHECK]")
    print(" goal_codes size     =", len(goal_codes))
    print(" own_goal_codes size =", len(own_goal_codes))
    print("\n[GOAL CODES]")
    for c in sorted(goal_codes):
        print(f" {c:>3} -> {code_to_str.get(c)}")

    risky = sorted(list(own_goal_codes - goal_codes))
    if risky:
        print("\n[NOTE] own_goal_codes that are NOT treated as goals by rule (own_goal_codes - goal_codes):")
        for c in risky:
            print(f" {c:>3} -> {code_to_str.get(c)}")

    df = pd.read_csv(DATA_CSV)
    mi = pd.read_csv(MATCH_INFO_CSV)

    # 정렬 + action_idx
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(["game_id", "period_id", "time_seconds", "_row_id"], kind="mergesort").reset_index(drop=True)
    df["action_idx"] = df.groupby("game_id").cumcount()

    # ---- (A) 룰 기반: 이 이벤트면 골 후보인가?
    df["is_goal_by_rule"] = df["type_result"].astype(int).isin(goal_codes).astype(int)
    df["is_own_goal_by_rule"] = df["type_result"].astype(int).isin(own_goal_codes).astype(int)

    # ---- (B) 실제 기반: 스코어 변화(delta)로 진짜 골 판정
    df["prev_home_score"] = df.groupby("game_id")["home_score"].shift(1).fillna(0).astype(int)
    df["prev_away_score"] = df.groupby("game_id")["away_score"].shift(1).fillna(0).astype(int)
    df["d_home"] = (df["home_score"].astype(int) - df["prev_home_score"]).astype(int)
    df["d_away"] = (df["away_score"].astype(int) - df["prev_away_score"]).astype(int)

    # 진짜 골 증가분(대부분 0 또는 1이지만 혹시 모를 n도 허용)
    df["goal_home_inc_true"] = df["d_home"].clip(lower=0)
    df["goal_away_inc_true"] = df["d_away"].clip(lower=0)
    df["is_goal_by_true_delta"] = ((df["goal_home_inc_true"] > 0) | (df["goal_away_inc_true"] > 0)).astype(int)

    # 룰로만 누적하면(지금 네가 돌린 것처럼) "2배 문제"가 터짐 → 디버그용 누적도 같이 계산
    # 룰 누적(naive)
    df["goal_home_inc_rule"] = 0
    df["goal_away_inc_rule"] = 0

    # scorer 판정(룰 기반): 기본은 actor 팀 득점, own_goal_codes면 반대로
    home_id = df["home_team_id"].astype(int)
    away_id = df["away_team_id"].astype(int)
    actor = df["team_id"].astype(int)

    # 룰로 골 이벤트인 경우에만 증가분 생성
    is_goal_evt = df["is_goal_by_rule"].astype(bool)
    is_own_evt  = df["is_own_goal_by_rule"].astype(bool)

    # scorer_team_id (룰 기반)
    scorer = actor.copy()
    scorer[is_goal_evt & is_own_evt] = np.where(actor[is_goal_evt & is_own_evt] == home_id[is_goal_evt & is_own_evt],
                                                away_id[is_goal_evt & is_own_evt],
                                                home_id[is_goal_evt & is_own_evt])

    # home/away에 1점씩 올림(룰 기반 naive)
    df.loc[is_goal_evt & (scorer == home_id), "goal_home_inc_rule"] = 1
    df.loc[is_goal_evt & (scorer == away_id), "goal_away_inc_rule"] = 1

    df["home_score_calc_rule"] = df.groupby("game_id")["goal_home_inc_rule"].cumsum().astype(int)
    df["away_score_calc_rule"] = df.groupby("game_id")["goal_away_inc_rule"].cumsum().astype(int)

    # true delta 기반 누적(항상 match_info와 맞아야 하는 기준선)
    df["home_score_calc_true"] = df.groupby("game_id")["goal_home_inc_true"].cumsum().astype(int)
    df["away_score_calc_true"] = df.groupby("game_id")["goal_away_inc_true"].cumsum().astype(int)

    # type_result 디코딩 컬럼(보기 편하게)
    tr_code = df["type_result"].astype(int)
    df["type_result_str"] = tr_code.map(lambda c: code_to_str.get(int(c), str(int(c))))
    df["type_name"] = tr_code.map(lambda c: code_to_type.get(int(c), "UNK"))
    df["result_name"] = tr_code.map(lambda c: code_to_result.get(int(c), "NA"))

    # 저장: goal_contained.csv (요청 컬럼 + 디버그 컬럼)
    out_cols = [
        "game_id","period_id","time_seconds","action_idx",
        "home_team_id","away_team_id","team_id",
        "type_result","type_result_str","type_name","result_name",
        "home_score","away_score",
        "d_home","d_away",
        "is_goal_by_rule","is_own_goal_by_rule","is_goal_by_true_delta",
        "goal_home_inc_rule","goal_away_inc_rule","home_score_calc_rule","away_score_calc_rule",
        "goal_home_inc_true","goal_away_inc_true","home_score_calc_true","away_score_calc_true",
    ]
    df[out_cols].to_csv(OUT_GOAL, index=False, encoding="utf-8-sig")
    print(f"\n[SAVED] goal_contained -> {OUT_GOAL} (rows={len(df):,})")

    # 비교: match_info vs (true 기반 최종) & (rule 기반 최종)
    final_true = mi[["game_id","home_score","away_score"]].copy()

    final_calc_true = df.groupby("game_id")[["home_score_calc_true","away_score_calc_true"]].max().reset_index()
    final_calc_rule = df.groupby("game_id")[["home_score_calc_rule","away_score_calc_rule"]].max().reset_index()

    cmp = final_true.merge(final_calc_true, on="game_id", how="left").merge(final_calc_rule, on="game_id", how="left")
    cmp["both_match_true"] = (cmp["home_score"] == cmp["home_score_calc_true"]) & (cmp["away_score"] == cmp["away_score_calc_true"])
    cmp["both_match_rule"] = (cmp["home_score"] == cmp["home_score_calc_rule"]) & (cmp["away_score"] == cmp["away_score_calc_rule"])

    cmp.to_csv(OUT_CMP, index=False, encoding="utf-8-sig")
    print(f"[SAVED] comparison -> {OUT_CMP}")

    print("\n[COMPARE SUMMARY]")
    print(" games:", int(cmp["game_id"].nunique()))
    print(" match_true (should be all):", int(cmp["both_match_true"].sum()))
    print(" match_rule (your current rule):", int(cmp["both_match_rule"].sum()))

    # audit: 코드별로 룰/진짜 골 교차표 (중복/false positive 찾기)
    audit = (
        df.groupby(["type_result_str","type_result"])
          .agg(
              n_rows=("type_result", "size"),
              n_goal_by_rule=("is_goal_by_rule", "sum"),
              n_goal_by_true=("is_goal_by_true_delta", "sum"),
          )
          .reset_index()
          .sort_values(["n_goal_by_true","n_goal_by_rule","n_rows"], ascending=False)
    )
    audit["rule_minus_true"] = audit["n_goal_by_rule"] - audit["n_goal_by_true"]
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")
    print(f"[SAVED] audit -> {OUT_AUDIT}")

    print("\n[TIP]")
    print("- rule_minus_true가 큰 코드 = 골이 아닌데 룰로 골로 잡히는(중복 포함) 가능성이 큼")
    print("- 보통 Goal__NA와 Shot__Goal이 동시에 기록되어 2배가 되는 케이스가 여기서 바로 드러남")


if __name__ == "__main__":
    main()
