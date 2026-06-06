import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Bird Detector Pro",
    page_icon="🐦",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #00FFB3;
    text-align: center;
    font-size: 50px;
}

h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #00FFB3;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.prediction-box {
    background-color: #1A1D24;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    model = load_model("bird_detector_pro.keras")

    with open("bird_detector_pro_encoder.pkl", "rb") as f:

        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_ai_model()

# =========================================================
# BIRD INFORMATION
# =========================================================

bird_info = {

    "hoopoe": {
        "name": "Eurasian Hoopoe",
        "description": "Known for its beautiful crown feathers and unique call.",
        "habitat": "Woodlands, grasslands, gardens"
    },

    "commyn": {
        "name": "Common Myna",
        "description": "Highly vocal urban bird commonly found near humans.",
        "habitat": "Cities, towns, villages"
    },

    "greegr": {
        "name": "Green Bee-eater",
        "description": "Small colorful bird famous for catching insects mid-air.",
        "habitat": "Open fields and forests"
    }
}

# =========================================================
# TITLE
# =========================================================

st.title("🐦 Bird Detector Pro")

st.markdown("""
### Deep Learning Bird Sound Classification System

Detect bird species using:
- 🎧 Audio Analysis
- 📊 Mel Spectrograms
- 🧠 CNN Deep Learning
- 🚀 SpecAugment Technology
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About")

st.sidebar.info("""
Bird Detector Pro uses a deep convolutional neural network trained on BirdCLEF environmental audio data.

Model Features:
- 50 Bird Species
- 84.9% Accuracy
- Mel Spectrogram Processing
- SpecAugment Augmentation
""")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "🎧 Upload Bird Audio File",
    type=["wav", "mp3", "ogg"]
)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
        duration=5
    )

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # =====================================================
    # FIX SIZE
    # =====================================================

    if mel_spec_db.shape[1] < 216:

        pad_width = 216 - mel_spec_db.shape[1]

        mel_spec_db = np.pad(
            mel_spec_db,
            pad_width=((0,0),(0,pad_width)),
            mode='constant'
        )

    else:

        mel_spec_db = mel_spec_db[:, :216]

    # =====================================================
    # NORMALIZE
    # =====================================================

    mel_spec_db = mel_spec_db / np.max(np.abs(mel_spec_db))

    # =====================================================
    # RESHAPE
    # =====================================================

    mel_spec_db = mel_spec_db[..., np.newaxis]

    mel_spec_db = np.expand_dims(
        mel_spec_db,
        axis=0
    )

    return mel_spec_db, audio, sr

# =========================================================
# PREDICTION
# =========================================================

def predict_bird(features):

    prediction = model.predict(features)[0]

    top_indices = prediction.argsort()[-5:][::-1]

    results = []

    for idx in top_indices:

        bird_name = encoder.inverse_transform([idx])[0]

        confidence = prediction[idx]

        results.append((bird_name, confidence))

    return results

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("🧠 AI is analyzing bird sounds..."):

        features, audio, sr = extract_features(
            uploaded_file
        )

        results = predict_bird(features)

    # =====================================================
    # TOP PREDICTION
    # =====================================================

    top_bird = results[0][0]
    top_conf = results[0][1]

    st.success(
        f"🐦 Predicted Bird: {top_bird}"
    )

    st.write(
        f"### Confidence: {top_conf:.2%}"
    )

    # =====================================================
    # BIRD INFORMATION
    # =====================================================

    if top_bird in bird_info:

        st.subheader("📖 Bird Information")

        st.write(
            f"### {bird_info[top_bird]['name']}"
        )

        st.write(
            bird_info[top_bird]['description']
        )

        st.write(
            f"**Habitat:** {bird_info[top_bird]['habitat']}"
        )

    # =====================================================
    # TOP 5 PREDICTIONS
    # =====================================================

    st.subheader("🔍 Top 5 Predictions")

    for bird, conf in results:

        st.markdown(
            f"""
            <div class="prediction-box">
                <h3>🐦 {bird}</h3>
                <p>Confidence: {conf:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(conf))

    # =====================================================
    # SPECTROGRAM
    # =====================================================

    st.subheader("📊 Mel Spectrogram")

    fig, ax = plt.subplots(figsize=(12,5))

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
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
    # RAW AUDIO WAVEFORM
    # =====================================================

    st.subheader("📈 Audio Waveform")

    fig2, ax2 = plt.subplots(figsize=(12,3))

    librosa.display.waveshow(
        audio,
        sr=sr,
        ax=ax2
    )

    st.pyplot(fig2)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<center>
Made with ❤️ using Deep Learning + Streamlit
</center>
""", unsafe_allow_html=True)