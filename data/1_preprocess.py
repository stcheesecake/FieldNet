import json
import pandas as pd

# =========================
# CONFIG
# =========================
IN_PATH = "merge.csv"
OUT_PATH = "data.csv"
MAP_PATH = "preprocess_maps.json"

# 인코딩에서 제외할 컬럼 리스트
EXCLUDE_COLS = ["home_team_name", "away_team_name", "home_team_name_ko", "away_team_name_ko"]

# =========================
# HELPERS
# =========================
def label_encode_0(series: pd.Series):
    """
    0..K-1 라벨 인코딩
    결측은 "NA"로 통일
    """
    s = series.fillna("NA").astype(str).str.strip()
    uniq = sorted(s.unique().tolist())
    m = {v: i for i, v in enumerate(uniq)}
    encoded = s.map(m).astype("int32")
    return encoded, m

# =========================
# LOAD
# =========================
df = pd.read_csv(IN_PATH)

maps = {}

# =========================
# 1) position_name, main_position -> 정수형 인코딩
# =========================
for col in ["position_name", "main_position"]:
    if col in df.columns:
        df[col], maps[col] = label_encode_0(df[col])

# =========================
# 2) type_name + result_name -> 합쳐서 정수형 인코딩
# =========================
need = ["type_name", "result_name"]
for c in need:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

type_s = df["type_name"].fillna("NA").astype(str).str.strip()
res_s  = df["result_name"].fillna("NA").astype(str).str.strip()

df["type_result"] = type_s + "__" + res_s
df["type_result"], maps["type_result"] = label_encode_0(df["type_result"])

# 원래 type_name / result_name은 더 이상 필요 없으면 제거
df = df.drop(columns=["type_name", "result_name"], errors="ignore")

# =========================
# 3) 남아있는 문자열(object) 컬럼 -> 제외 목록 빼고 전부 정수형 인코딩
# =========================
# 모든 object 컬럼을 가져온 뒤, EXCLUDE_COLS에 포함되지 않은 것만 리스트로 만듭니다.
obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in EXCLUDE_COLS]

for col in obj_cols:
    df[col], maps[col] = label_encode_0(df[col])

# =========================
# SAVE
# =========================
# 한글 이름이 포함되어 있을 수 있으므로 utf-8-sig로 저장합니다.
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

with open(MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(maps, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_PATH)
print("Saved maps:", MAP_PATH)
print("Shape:", df.shape)
print("Excluded from encoding:", [c for c in EXCLUDE_COLS if c in df.columns])
print("Encoded object cols:", len(obj_cols))