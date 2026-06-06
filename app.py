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
    page_title="Bird Sound Detector AI",
    page_icon="🐦",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stApp {
    background-color: #0E1117;
}

h1 {
    color: #00FFB3;
    text-align: center;
}

h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model("advanced_bird_classifier.keras")

# =========================================================
# LOAD LABEL ENCODER
# =========================================================

with open("advanced_label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# =========================================================
# TITLE
# =========================================================

st.title("🐦 Bird Sound Detector AI")

st.write("""
Deep Learning based Bird Sound Classification System  
using Mel Spectrograms + SpecAugment + CNN
""")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "🎧 Upload Bird Audio",
    type=["wav", "mp3", "ogg"]
)

# =========================================================
# CREATE MEL SPECTROGRAM
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

    # FIX SIZE
    if mel_spec_db.shape[1] < 216:

        pad_width = 216 - mel_spec_db.shape[1]

        mel_spec_db = np.pad(
            mel_spec_db,
            pad_width=((0,0),(0,pad_width)),
            mode='constant'
        )

    else:
        mel_spec_db = mel_spec_db[:, :216]

    # NORMALIZE
    mel_spec_db = mel_spec_db / np.max(np.abs(mel_spec_db))

    # RESHAPE
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

    top_indices = prediction.argsort()[-3:][::-1]

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

    with st.spinner("🧠 Analyzing Bird Sound..."):

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
        f"Confidence: {top_conf:.2%}"
    )

    # =====================================================
    # TOP 3 PREDICTIONS
    # =====================================================

    st.subheader("🔍 Top 3 Predictions")

    for bird, conf in results:

        st.write(f"### {bird}")

        st.progress(float(conf))

        st.write(f"{conf:.2%}")

    # =====================================================
    # SPECTROGRAM VISUALIZATION
    # =====================================================

    st.subheader("📊 Mel Spectrogram")

    fig, ax = plt.subplots(figsize=(10,4))

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