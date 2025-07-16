import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 상위 디렉터리(프로젝트 루트)를 파이썬 모듈 경로에 추가해 autointMLP를 import 가능하게 합니다.
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from autointMLP import AutoIntMLPModel

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MOVIELENS_DIR = os.path.join(DATA_DIR, "ml-1m")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 데이터 로딩
# -----------------------------------------------------------------------------
print("[INFO] 데이터 로딩 중 ...")
ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, "ratings_prepro.csv"))
movies = pd.read_csv(os.path.join(MOVIELENS_DIR, "movies_prepro.csv"))
users = pd.read_csv(os.path.join(MOVIELENS_DIR, "users_prepro.csv"))

# 평점 4 이상을 positive 로 간주
ratings["label"] = (ratings["rating"] >= 4).astype(int)

# 필드 리스트 (AutoInt 학습 시 사용된 컬럼과 동일해야 함)
FEATURE_COLS = [
    "user_id",
    "movie_id",
    "movie_decade",
    "movie_year",
    "rating_year",
    "rating_month",
    "rating_decade",
    "genre1",
    "genre2",
    "genre3",
    "gender",
    "age",
    "occupation",
    "zip",
]

# ❶ ratings + movies + users 조인
df = ratings.merge(movies, on="movie_id").merge(users, on="user_id")

# ❷ 필요한 feature 컬럼이 모두 존재하는지 확인
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"다음 컬럼이 데이터 프레임에 없습니다: {missing_cols}")

try:
    import joblib

    label_enc_path = os.path.join(DATA_DIR, "label_encoders.pkl")
    label_encoders = joblib.load(label_enc_path)

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
except FileNotFoundError:
    # 인코더 파일이 없으면 데이터가 이미 정수형이라고 가정
    pass

X = df[FEATURE_COLS].values.astype(np.int64)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

# field_dims.npy  로부터 각 필드 차원(cardinality) 로드
field_dims_path = os.path.join(DATA_DIR, "field_dims.npy")
field_dims = np.load(field_dims_path)

print("[INFO] 데이터 shape:", X.shape)

# -----------------------------------------------------------------------------
# Train / Validation split
# -----------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# -----------------------------------------------------------------------------
# 모델 정의
# -----------------------------------------------------------------------------
embed_dim = 16
model = AutoIntMLPModel(
    field_dims=field_dims,
    embedding_size=embed_dim,
    att_layer_num=3,
    att_head_num=2,
    dnn_hidden_units=(128, 64),
    dnn_dropout=0.4,
)
# Build
model(tf.constant(X_train[:1]))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
)

# -----------------------------------------------------------------------------
# 학습
# -----------------------------------------------------------------------------
BATCH_SIZE = 1024
EPOCHS = 5

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
)

# -----------------------------------------------------------------------------
# 검증 AUC 출력
# -----------------------------------------------------------------------------
val_pred = model.predict(X_val, batch_size=BATCH_SIZE)
val_auc = roc_auc_score(y_val, val_pred)
print(f"[RESULT] Validation AUC: {val_auc:.4f}")

# -----------------------------------------------------------------------------
# 가중치 저장
# -----------------------------------------------------------------------------
weights_path = os.path.join(MODEL_DIR, "autoIntMLP_model.weights.h5")
model.save_weights(weights_path)
print(f"[INFO] 가중치 저장 완료 -> {weights_path}") 