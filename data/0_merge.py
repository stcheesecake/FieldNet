import pandas as pd

# =========================
# CONFIG
# =========================
RAW_PATH = "raw_data.csv"
MATCH_PATH = "match_info.csv"
JOIN_KEY = "game_id"
OUT_PATH = "merge.csv"  # 현재 디렉토리에 저장

# 쓸모없는 feature 제거 목록 (없으면 무시)
DROP_COLS = [
    "venue",
    "competition_name",
    "country_name",
    "season_name",
    "home_team_name_ko",
    "away_team_name_ko",
    "season_id",
    "competition_id",
    "game_day",
    "game_date",
    "action_id",
    "player_name_ko",
    "team_name_ko",
]

# =========================
# 1) CSV 두개 합치기
# =========================
raw = pd.read_csv(RAW_PATH)
match = pd.read_csv(MATCH_PATH)

df = raw.merge(match, on=JOIN_KEY, how="left", validate="m:1")

# =========================
# 2) 쓸모없는 feature 지우기
# =========================
df = df.drop(columns=DROP_COLS, errors="ignore")

# =========================
# 3) SAVE
# =========================
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("Saved:", OUT_PATH)
print("Shape:", df.shape)
