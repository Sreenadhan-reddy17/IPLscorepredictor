"""
IPL Score Prediction – Streamlit App
=====================================
Requirements:
  1. pip install -r requirements.txt
  2. python train.py          ← generates ipl_model.h5 + artifacts/
  3. streamlit run app.py
"""

import os
import pickle
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=" IPL Score Predictor",
    page_icon="🏏",
    layout="centered",
)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = "ipl_model.h5"

# ── Same team list as train.py ─────────────────────────────────────────────
ALL_TEAMS = sorted([
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
])

NUMERICAL_COLS = ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
FEATURE_COLS   = ["batting_team_enc", "bowling_team_enc"] + NUMERICAL_COLS


# ══════════════════════════════════════════════════════════════════════════════
# Artifact loaders – all errors caught and returned as strings
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_tf_model():
    """Load ipl_model.h5. Returns (model | None, error_str | None)."""
    # Step 1 – check file exists before even importing TF
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"`{MODEL_PATH}` not found. "
            "Please run **`python train.py`** first."
        )
    # Step 2 – import TensorFlow
    try:
        import tensorflow as tf                      # noqa: F401
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError:
        return None, (
            "TensorFlow is not installed.\n\n"
            "Fix:  `pip install tensorflow`"
        )
    except Exception as e:
        return None, f"TensorFlow import error: {e}"
    # Step 3 – load model
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:
        return None, f"Error loading `{MODEL_PATH}`: {e}"


@st.cache_resource(show_spinner=False)
def load_pkl(filename: str):
    """Load a pickle file from ARTIFACTS_DIR. Returns (obj | None, error | None)."""
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        return None, (
            f"`{path}` not found. "
            "Please run **`python train.py`** first."
        )
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, f"Error loading `{filename}`: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    border-radius:14px; padding:2rem 2.5rem;
    margin-bottom:1.6rem; text-align:center;
    box-shadow:0 8px 28px rgba(0,0,0,.35);
}
.hero h1 { color:#e94560; font-size:2rem; margin:0; }
.hero p  { color:#a8b2d8; margin:.4rem 0 0; }

/* Setup box */
.setup-box {
    background:#1e1e2e; border:1px solid #e94560;
    border-radius:12px; padding:1.4rem 1.8rem;
}
.setup-box h3 { color:#f1fa8c; margin-top:0; }
.setup-box code {
    background:#282a36; color:#50fa7b;
    padding:.15rem .5rem; border-radius:5px; font-size:.9rem;
}

/* Prediction result */
.result-box {
    background:linear-gradient(135deg,#0f3460,#533483);
    border-radius:14px; padding:1.8rem; text-align:center;
    box-shadow:0 6px 24px rgba(83,52,131,.4);
    margin-top:1rem;
}
.result-box .label { color:#a8b2d8; font-size:.95rem; margin-bottom:.4rem; }
.result-box .score { color:#f1fa8c; font-size:3rem; font-weight:700; line-height:1; }
.result-box .range { color:#8be9fd; font-size:1.15rem; margin-top:.5rem; }

/* Live stat tiles */
.tile-row { display:flex; gap:1rem; margin:.8rem 0; }
.tile {
    flex:1; background:#1e1e2e; border:1px solid #2d2d44;
    border-radius:10px; padding:.8rem; text-align:center;
}
.tile .tlabel { color:#6272a4; font-size:.72rem; text-transform:uppercase; }
.tile .tvalue { color:#f8f8f2; font-size:1.25rem; font-weight:600; }

/* Predict button */
div.stButton > button {
    background:linear-gradient(90deg,#e94560,#a855f7);
    color:white; border:none; border-radius:10px;
    padding:.75rem 2rem; font-size:1.05rem; font-weight:600;
    width:100%;
}
div.stButton > button:hover { opacity:.88; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🏏 IPL Score Predictor</h1>
  <p>Deep Learning (ANN) · TensorFlow / Keras</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load all artifacts – collect errors
# ══════════════════════════════════════════════════════════════════════════════

model,      model_err   = load_tf_model()
batting_le, bat_err     = load_pkl("batting_le.pkl")
bowling_le, bowl_err    = load_pkl("bowling_le.pkl")
scaler,     scaler_err  = load_pkl("scaler.pkl")

errors = [e for e in [model_err, bat_err, bowl_err, scaler_err] if e]

# ── Show setup guide if anything is missing ───────────────────────────────────
if errors:
    st.markdown("""
<div class="setup-box">
<h3>⚙️ Setup Required</h3>
<p>One or more required files are missing. Follow these steps in your terminal:</p>
<ol>
  <li>Install dependencies:<br><code>pip install -r requirements.txt</code></li>
  <li>Train the model (generates .h5 &amp; .pkl files):<br><code>python train.py</code></li>
  <li>Re-launch the app:<br><code>streamlit run app.py</code></li>
</ol>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### ❌ Errors Detected")
    for err in errors:
        st.error(err)

    st.markdown("---")
    st.info(
        "📌 **Quick check:** Make sure `ipl_model.h5` and the `artifacts/` folder "
        "exist in the same directory as `app.py` after running `train.py`."
    )
    st.stop()   # Do not render the rest of the app

# ── All good ──────────────────────────────────────────────────────────────────
st.success("✅ Model & encoders loaded successfully!", icon="✔")


# ══════════════════════════════════════════════════════════════════════════════
# Prediction helper
# ══════════════════════════════════════════════════════════════════════════════

def safe_encode(le, value):
    return int(le.transform([value])[0]) if value in le.classes_ else 0


def predict_score(batting_team, bowling_team, runs, wickets,
                  overs, runs_last_5, wickets_last_5):
    bat_enc  = safe_encode(batting_le,  batting_team)
    bowl_enc = safe_encode(bowling_le, bowling_team)

    num = np.array([[runs, wickets, overs, runs_last_5, wickets_last_5]],
                   dtype=np.float32)
    num_scaled = scaler.transform(num)

    features = np.hstack([[bat_enc, bowl_enc], num_scaled[0]]).reshape(1, -1).astype(np.float32)

    raw = float(model.predict(features, verbose=0)[0][0])
    mid  = max(int(round(raw)), runs)        # can't predict less than current
    low  = max(mid - 10, runs)
    high = mid + 10
    return low, mid, high


# ══════════════════════════════════════════════════════════════════════════════
# Input form
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📋 Match Snapshot")

c1, c2 = st.columns(2)
with c1:
    batting_team = st.selectbox(
        "🏏 Batting Team", ALL_TEAMS,
        index=ALL_TEAMS.index("Mumbai Indians"),
    )
with c2:
    bowling_options = [t for t in ALL_TEAMS if t != batting_team]
    bowling_team = st.selectbox("⚾ Bowling Team", bowling_options)

st.markdown("**📊 Current Innings Stats**")

c3, c4, c5 = st.columns(3)
with c3:
    runs = st.number_input("🔢 Current Score", 0, 300, 78)
with c4:
    overs = st.number_input("🕐 Overs Completed", 5.0, 19.5, 10.0, step=0.1, format="%.1f")
with c5:
    wickets = st.number_input("🎯 Wickets Fallen", 0, 9, 2)

c6, c7 = st.columns(2)
with c6:
    runs_last_5 = st.number_input("📈 Runs (Last 5 Overs)", 0, 120, 40)
with c7:
    wickets_last_5 = st.number_input("📉 Wickets (Last 5 Overs)", 0, 5, 1)

# ── Live stat tiles ───────────────────────────────────────────────────────────
run_rate   = runs / max(overs, 0.1)
balls_left = int((20 - overs) * 6)
wkts_left  = 10 - wickets
req_rr_180 = max(0, (180 - runs) / max(20 - overs, 0.01))

st.markdown(f"""
<div class="tile-row">
  <div class="tile"><div class="tlabel">Run Rate</div><div class="tvalue">{run_rate:.2f}</div></div>
  <div class="tile"><div class="tlabel">Balls Left</div><div class="tvalue">{balls_left}</div></div>
  <div class="tile"><div class="tlabel">Wickets Left</div><div class="tvalue">{wkts_left}</div></div>
  <div class="tile"><div class="tlabel">Req. RR (180)</div><div class="tvalue">{req_rr_180:.2f}</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Predict
# ══════════════════════════════════════════════════════════════════════════════

if st.button("🚀 Predict Final Score"):
    try:
        with st.spinner("Predicting …"):
            low, mid, high = predict_score(
                batting_team, bowling_team, runs, wickets,
                overs, runs_last_5, wickets_last_5,
            )

        st.markdown(f"""
<div class="result-box">
  <div class="label">🏆 Predicted Final Score</div>
  <div class="score">{mid}</div>
  <div class="range">Likely Range &nbsp;|&nbsp; {low} – {high} runs</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if mid >= 180:
            st.success(f"🔥 Big total on the cards! {batting_team} looking explosive.")
        elif mid >= 155:
            st.info(f"📊 Competitive score. {batting_team} on track for a solid total.")
        else:
            st.warning(f"📉 Below-par total predicted. {bowling_team} in firm control.")

        with st.expander("📐 Breakdown"):
            ca, cb = st.columns(2)
            ca.metric("Batting Team",   batting_team)
            ca.metric("Runs Scored",    runs)
            ca.metric("Overs Done",     f"{overs:.1f}")
            cb.metric("Bowling Team",   bowling_team)
            cb.metric("Wickets Fallen", wickets)
            cb.metric("Current RR",    f"{run_rate:.2f}")
            st.caption(
                "The ±10 run range accounts for in-game variability "
                "(power hitting, pitch changes, death-over surprises)."
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info(
            "Ensure `ipl_model.h5` and `artifacts/` are in the same folder as "
            "`app.py`, then restart the app."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6272a4;font-size:.82rem;'>"
    "IPL Score Predictor · TensorFlow / Keras ANN · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
