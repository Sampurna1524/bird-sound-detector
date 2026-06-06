# ============================================================
# BIRD DETECTOR ULTRA - PREMIUM UI
# 182 BIRD SPECIES AI
# REAL-TIME STYLE FRONTEND
# ============================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time
from io import BytesIO

from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bird Detector Ultra",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #0B0F19,
        #121826,
        #1C2333
    );
    color: white;
}

/* =========================================================
TITLE
========================================================= */

.main-title {
    text-align: center;
    font-size: 65px;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #00F5A0,
        #00D9F5,
        #7F5AF0
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: -30px;
}

.subtitle {
    text-align: center;
    color: #B0B7C3;
    font-size: 20px;
    margin-bottom: 40px;
}

/* =========================================================
GLASS CARDS
========================================================= */

.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.1);

    border-radius: 20px;

    padding: 25px;

    margin-bottom: 20px;

    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* =========================================================
PREDICTION CARDS
========================================================= */

.prediction-card {

    background: linear-gradient(
        135deg,
        rgba(0,245,160,0.15),
        rgba(0,217,245,0.15)
    );

    border-radius: 18px;

    padding: 20px;

    margin-bottom: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    transition: 0.3s;
}

.prediction-card:hover {

    transform: scale(1.02);

    box-shadow: 0 0 20px rgba(0,245,160,0.2);
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card {

    background: rgba(255,255,255,0.05);

    border-radius: 16px;

    padding: 18px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background: #0F172A;

    border-right: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
BUTTON
========================================================= */

.stButton>button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #00F5A0,
        #00D9F5
    );

    color: black;

    border: none;

    padding: 14px;

    border-radius: 14px;

    font-weight: 700;

    font-size: 16px;

    transition: 0.3s;
}

.stButton>button:hover {

    transform: scale(1.02);

    box-shadow: 0 0 25px rgba(0,245,160,0.3);
}

/* =========================================================
UPLOAD BOX
========================================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

    border-radius: 18px;

    padding: 15px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_ai():

    model = load_model(
        "bird_detector_ultra_v4.keras"
    )

    with open(
        "bird_detector_ultra_v4_encoder.pkl",
        "rb"
    ) as f:

        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_ai()

# ============================================================
# BIRD INFO DATABASE
# ============================================================

bird_info = {

    "hoopoe": {

        "common_name": "Eurasian Hoopoe",

        "scientific_name": "Upupa epops",

        "habitat": "Woodlands, grasslands, farmlands",

        "diet": "Insects, larvae, worms",

        "status": "Least Concern",

        "fun_fact": "Its crown opens like a royal fan when excited.",

        "image": "https://upload.wikimedia.org/wikipedia/commons/3/32/Hoopoe_%28Upupa_epops%29.jpg"
    },

    "commyn": {

        "common_name": "Common Myna",

        "scientific_name": "Acridotheres tristis",

        "habitat": "Urban areas, villages, forests",

        "diet": "Fruits, insects, food scraps",

        "status": "Least Concern",

        "fun_fact": "Can mimic sounds and human speech.",

        "image": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Common_Myna.jpg"
    },

    "greegr": {

        "common_name": "Green Bee-eater",

        "scientific_name": "Merops orientalis",

        "habitat": "Open grasslands and fields",

        "diet": "Bees, insects, dragonflies",

        "status": "Least Concern",

        "fun_fact": "Catches insects mid-air with incredible precision.",

        "image": "https://upload.wikimedia.org/wikipedia/commons/9/9d/Green_bee-eater_%28Merops_orientalis%29.jpg"
    }

}

# ============================================================
# FEATURE EXTRACTION
# ============================================================

N_MELS = 64
SPEC_WIDTH = 128

def extract_features(audio_source):

    # =====================================================
    # HANDLE STREAMLIT UPLOADS / MIC AUDIO
    # =====================================================

    if isinstance(audio_source, bytes):

        audio_file = BytesIO(audio_source)

    else:

        audio_file = audio_source

    # =====================================================
    # LOAD AUDIO
    # =====================================================

    audio, sr = librosa.load(

        audio_file,

        sr=None,

        mono=True,

        duration=5

    )

    # =====================================================
    # MEL SPECTROGRAM
    # =====================================================

    mel_spec = librosa.feature.melspectrogram(

        y=audio,

        sr=sr,

        n_mels=N_MELS

    )

    mel_spec = librosa.power_to_db(

        mel_spec,

        ref=np.max

    )

    # =====================================================
    # FIX WIDTH
    # =====================================================

    if mel_spec.shape[1] < SPEC_WIDTH:

        pad_width = SPEC_WIDTH - mel_spec.shape[1]

        mel_spec = np.pad(

            mel_spec,

            pad_width=((0,0),(0,pad_width)),

            mode='constant'

        )

    else:

        mel_spec = mel_spec[:, :SPEC_WIDTH]

    # =====================================================
    # NORMALIZE
    # =====================================================

    mel_spec = mel_spec / (

        np.max(np.abs(mel_spec)) + 1e-6

    )

    mel_spec = mel_spec[..., np.newaxis]

    mel_spec = np.expand_dims(

        mel_spec,

        axis=0

    )

    return mel_spec, audio, sr

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_bird(features):

    prediction = model.predict(
        features,
        verbose=0
    )[0]

    top_indices = prediction.argsort()[-5:][::-1]

    results = []

    for idx in top_indices:

        bird_name = encoder.inverse_transform([idx])[0]

        confidence = float(prediction[idx])

        results.append(
            (bird_name, confidence)
        )

    return results

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
# 🐦 Bird Detector Ultra
""")

st.sidebar.markdown("""
### 🌍 AI Features

✅ 182 Bird Species  
✅ Residual CNN  
✅ Attention Mechanisms  
✅ Spectrogram Intelligence  
✅ Deep Audio Learning  
✅ 81.5% Accuracy  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### ⚡ Model Stats

- Parameters: 1.47M
- Audio Duration: 5 sec
- Spectrograms: Mel Scale
- Architecture: Residual Attention CNN
""")

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="main-title">
🐦 Bird Detector Ultra
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Large-Scale Environmental Audio Intelligence System
</div>
""", unsafe_allow_html=True)

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="metric-card">
        <h2>182</h2>
        <p>Bird Species</p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-card">
        <h2>81.5%</h2>
        <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="metric-card">
        <h2>24K+</h2>
        <p>Samples</p>
    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown("""
    <div class="metric-card">
        <h2>1.47M</h2>
        <p>Parameters</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "🎧 Upload Bird Audio",
    type=["wav", "mp3", "ogg"]
)

# ============================================================
# REAL-TIME MICROPHONE
# ============================================================

st.markdown("""
## 🎤 Real-Time Bird Listening
""")

audio_bytes = audio_recorder(

    text="🎙️ Click To Record",

    recording_color="#FF4B4B",

    neutral_color="#00F5A0",

    icon_name="microphone",

    icon_size="2x"

)

# ============================================================
# HANDLE MIC AUDIO
# ============================================================

if audio_bytes:

    st.success("✅ Recording Captured!")

    st.audio(audio_bytes)

    uploaded_file = BytesIO(audio_bytes)

# ============================================================
# MAIN PREDICTION
# ============================================================

if uploaded_file:


    with st.spinner("🧠 AI is listening to the forest..."):

        time.sleep(2)

        features, audio, sr = extract_features(
            uploaded_file
        )

        results = predict_bird(features)

    top_bird = results[0][0]

    top_conf = results[0][1]

    # =====================================================
    # MAIN RESULT
    # =====================================================

    st.markdown(f"""
    <div class="glass-card">

    <h1 style="text-align:center;">
    🐦 {top_bird}
    </h1>

    <h3 style="text-align:center;color:#00F5A0;">
    Confidence: {top_conf:.2%}
    </h3>

    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # BIRD INFORMATION
    # =====================================================

    # =====================================================
    # ADVANCED BIRD INFO
    # =====================================================

    if top_bird in bird_info:

        info = bird_info[top_bird]

        st.markdown("""
        ## 🦜 Bird Information
        """)

        col1, col2 = st.columns([1,2])

        # =================================================
        # IMAGE
        # =================================================

        with col1:

            st.image(
                info["image"],
                use_container_width=True
            )

        # =================================================
        # DETAILS
        # =================================================

        with col2:

            st.markdown(f"""
            <div class="glass-card">

            <h1>
            🐦 {info['common_name']}
            </h1>

            <h4 style="color:#A0AEC0;">
            {info['scientific_name']}
            </h4>

            <hr>

            <p>
            <b>🌍 Habitat:</b><br>
            {info['habitat']}
            </p>

            <p>
            <b>🍽️ Diet:</b><br>
            {info['diet']}
            </p>

            <p>
            <b>🛡️ Conservation Status:</b><br>
            {info['status']}
            </p>

            <p>
            <b>✨ Fun Fact:</b><br>
            {info['fun_fact']}
            </p>

            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # TOP 5 PREDICTIONS
        # =====================================================

        st.markdown("""
        ## 🔍 Top Predictions
        """)

        for bird, conf in results:

            st.markdown(f"""
            <div class="prediction-card">

            <h3>🐦 {bird}</h3>

            <p>Confidence: {conf:.2%}</p>

            </div>
            """, unsafe_allow_html=True)

            st.progress(conf)

        # =====================================================
        # CHART
        # =====================================================

        st.markdown("""
        ## 📊 Prediction Confidence
        """)

        chart_data = pd.DataFrame({

            "Bird": [x[0] for x in results],

            "Confidence": [x[1] for x in results]

        })

        st.bar_chart(
            chart_data.set_index("Bird")
        )

        # =====================================================
        # MEL SPECTROGRAM
        # =====================================================

        st.markdown("""
        ## 🎼 Mel Spectrogram
        """)

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        img = librosa.display.specshow(
            mel_db,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            ax=ax
        )

        plt.colorbar(img, ax=ax)

        st.pyplot(fig)

        # =====================================================
        # AUDIO WAVEFORM
        # =====================================================

        st.markdown("""
        ## 📈 Audio Waveform
        """)

        fig2, ax2 = plt.subplots(
            figsize=(12,3)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax2
        )

        st.pyplot(fig2)

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>

<h3>🌿 Built with Deep Learning & Bioacoustics</h3>

<p>
Residual CNN • Attention Mechanisms • Spectrogram Intelligence
</p>

</center>
""", unsafe_allow_html=True)