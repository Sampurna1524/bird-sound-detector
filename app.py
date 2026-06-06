import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model

# ============================================
# LOAD MODEL
# ============================================

model = load_model("bird_sound_classifier.h5")

# LOAD LABEL ENCODER
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ============================================
# PAGE TITLE
# ============================================

st.title("🐦 Bird Sound Detector AI")

st.write(
    "Upload a bird sound audio file and the AI will predict the bird species."
)

# ============================================
# FILE UPLOADER
# ============================================

uploaded_file = st.file_uploader(
    "Upload Bird Audio",
    type=["wav", "mp3", "ogg"]
)

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_bird(audio_file):

    # LOAD AUDIO
    audio, sr = librosa.load(
        audio_file,
        duration=5
    )

    # EXTRACT MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # FIX SIZE
    if mfcc.shape[1] < 216:

        pad_width = 216 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0,0),(0,pad_width)),
            mode='constant'
        )

    else:
        mfcc = mfcc[:, :216]

    # NORMALIZE
    mfcc = mfcc / np.max(mfcc)

    # RESHAPE
    mfcc = mfcc[..., np.newaxis]

    # ADD BATCH DIMENSION
    mfcc = np.expand_dims(mfcc, axis=0)

    # PREDICT
    prediction = model.predict(mfcc)

    predicted_index = np.argmax(prediction)

    bird_name = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = np.max(prediction)

    return bird_name, confidence

# ============================================
# RUN PREDICTION
# ============================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("Analyzing Bird Sound..."):

        bird_name, confidence = predict_bird(
            uploaded_file
        )

    st.success(f"Predicted Bird: {bird_name}")

    st.write(
        f"Confidence: {confidence:.2f}"
    )