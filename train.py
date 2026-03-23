"""
IPL Score Prediction – Training Script
=======================================
Run this ONCE before launching app.py:

    pip install -r requirements.txt
    python train.py

Outputs
-------
  ipl_model.h5
  artifacts/batting_le.pkl
  artifacts/bowling_le.pkl
  artifacts/scaler.pkl
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── TF import guard ───────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print(f"[OK] TensorFlow {tf.__version__} loaded.")
except ModuleNotFoundError:
    sys.exit(
        "\n[ERROR] TensorFlow is not installed.\n"
        "Fix: pip install tensorflow\n"
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = "ipl_model.h5"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Teams (must be identical in app.py) ──────────────────────────────────────
ALL_TEAMS = sorted([
    "Chennai Super Kings",
    "Delhi Capitals",
    "Kings XI Punjab",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
])

NUMERICAL_COLS = ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
FEATURE_COLS   = ["batting_team_enc", "bowling_team_enc"] + NUMERICAL_COLS


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_or_generate(csv_path="ipl.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        print(f"[INFO] Reading '{csv_path}' …")
        df = pd.read_csv(csv_path)
        required = {"batting_team","bowling_team","runs","wickets",
                    "overs","runs_last_5","wickets_last_5","total"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"[ERROR] CSV is missing columns: {missing}")
        return df

    print("[INFO] ipl.csv not found → generating synthetic dataset …")
    rng = np.random.default_rng(42)
    n   = 80_000

    batting = rng.choice(ALL_TEAMS, n)
    bowling = np.array([rng.choice([t for t in ALL_TEAMS if t != b]) for b in batting])

    overs          = np.round(rng.uniform(5.1, 19.5, n), 1)
    runs           = (rng.uniform(20, 150, n) * (overs / 20)).astype(int)
    wickets        = np.clip((rng.uniform(0, 9, n) * (overs / 20)).astype(int), 0, 9)
    runs_last_5    = np.clip(rng.normal(35, 15, n), 0, 90).astype(int)
    wickets_last_5 = np.clip(rng.poisson(1.4, n), 0, 5).astype(int)

    base_rr = runs / np.maximum(overs, 0.1)
    total   = np.clip((base_rr * 20 + rng.normal(0, 20, n)).astype(int), 80, 265)

    return pd.DataFrame({
        "batting_team":   batting,
        "bowling_team":   bowling,
        "runs":           runs,
        "wickets":        wickets,
        "overs":          overs,
        "runs_last_5":    runs_last_5,
        "wickets_last_5": wickets_last_5,
        "total":          total,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 2. Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    df = df[df["overs"] >= 5.0].copy()

    # LabelEncoder – fit on fixed ALL_TEAMS list so encoding is stable
    batting_le = LabelEncoder().fit(ALL_TEAMS)
    bowling_le = LabelEncoder().fit(ALL_TEAMS)

    def safe_enc(le, col):
        return col.apply(
            lambda v: int(le.transform([v])[0]) if v in le.classes_ else 0
        )

    df["batting_team_enc"] = safe_enc(batting_le, df["batting_team"])
    df["bowling_team_enc"] = safe_enc(bowling_le, df["bowling_team"])

    scaler = MinMaxScaler()
    df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["total"].values.astype(np.float32)

    return X, y, batting_le, bowling_le, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(n_features: int):
    model = Sequential([
        Input(shape=(n_features,)),

        Dense(512, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(256, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(0.25),

        Dense(128, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(0.20),

        Dense(64, activation="relu", kernel_initializer="he_normal"),

        Dense(1, activation="linear"),
    ], name="IPL_ANN")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Train + evaluate
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    print(f"[INFO] Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")

    model = build_model(X_tr.shape[1])
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-6, verbose=1),
        ],
        verbose=1,
    )

    y_pred = model.predict(X_te, verbose=0).flatten()
    mae    = mean_absolute_error(y_te, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)

    print("\n" + "═"*45)
    print(f"  MAE        : {mae:.2f} runs")
    print(f"  RMSE       : {rmse:.2f} runs")
    print(f"  R²         : {r2:.4f}")
    print(f"  ±10 runs   : {np.mean(np.abs(y_te - y_pred)<=10)*100:.1f}%")
    print(f"  ±20 runs   : {np.mean(np.abs(y_te - y_pred)<=20)*100:.1f}%")
    print("═"*45 + "\n")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. Save
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, batting_le, bowling_le, scaler):
    model.save(MODEL_PATH)
    print(f"[✔] Model  → {MODEL_PATH}")

    for name, obj in [("batting_le", batting_le),
                       ("bowling_le", bowling_le),
                       ("scaler",     scaler)]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[✔] Saved  → {path}")

    print("\n[✔] All artifacts saved. You can now run:  streamlit run app.py\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*50)
    print("  IPL Score Predictor – Training")
    print("="*50)

    df = load_or_generate()
    print(f"[INFO] Dataset: {len(df):,} rows")

    X, y, batting_le, bowling_le, scaler = preprocess(df)
    print(f"[INFO] Features: {X.shape}  |  Target range: {int(y.min())}–{int(y.max())} runs")

    model = train_model(X, y)
    save_artifacts(model, batting_le, bowling_le, scaler)
