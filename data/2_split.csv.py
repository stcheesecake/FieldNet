import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
IN_PATH = "preprocess.csv"
TRAIN_OUT = "train.csv"
TEST_OUT  = "test.csv"

TEST_RATIO = 0.2
SEED = 42

# =========================
# LOAD
# =========================
df = pd.read_csv(IN_PATH)

# =========================
# GAME-ID SPLIT (8:2)
# =========================
game_ids = df["game_id"].dropna().unique()
rng = np.random.RandomState(SEED)
rng.shuffle(game_ids)

n_test = int(len(game_ids) * TEST_RATIO)
test_games = set(game_ids[:n_test])
train_games = set(game_ids[n_test:])

train_df = df[df["game_id"].isin(train_games)].reset_index(drop=True)
test_df  = df[df["game_id"].isin(test_games)].reset_index(drop=True)

# =========================
# SAVE
# =========================
train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test_df.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

print("Total rows:", len(df))
print("Train rows:", len(train_df), "| Train games:", train_df["game_id"].nunique())
print("Test  rows:", len(test_df),  "| Test  games:", test_df["game_id"].nunique())
print("Saved:", TRAIN_OUT, TEST_OUT)
