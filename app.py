# app.py
import os
import io
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
from audiorecorder import audiorecorder
import pandas as pd
import speech_recognition as sr
from fpdf import FPDF
from datetime import datetime

from predict_emotion import predict_emotion_with_confidence

# Emotion mapping
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOJI_MAP = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}

# Ensure FFmpeg path (adjust if your ffmpeg-downloader path differs)
os.environ["PATH"] += os.pathsep + r"C:\Users\RISHI\AppData\Local\ffmpegio\ffmpeg-downloader\ffmpeg\bin"

st.set_page_config(page_title="Voice Emotion Detector", page_icon="🎧", layout="centered")

st.title("🎙️ Voice Emotion Detection & Report")
st.write("Upload or record a voice note — get waveform, transcript, emotion confidence chart, and downloadable report.")

# Input selection
option = st.radio("🎤 Choose Input Method:", ["Upload Audio File", "Record with Microphone"])
audio_path = None
temp_file = "temp.wav"

# ----- UPLOAD -----
if option == "Upload Audio File":
    uploaded = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a"])
    if uploaded:
        audio = AudioSegment.from_file(uploaded)
        audio.export(temp_file, format="wav")
        audio_path = temp_file
        st.audio(io.BytesIO(open(temp_file, "rb").read()), format="audio/wav")

# ----- RECORD -----
else:
    st.write("Press record, then stop. After recording, press Analyze.")
    audio_segment = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")
    if len(audio_segment) > 0:
        audio_segment.export(temp_file, format="wav")
        audio_path = temp_file
        buf = io.BytesIO()
        audio_segment.export(buf, format="wav")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

# ----- PROCESS & ANALYZE -----
if audio_path and st.button("🔍 Analyze & Generate Report"):
    start = time.time()
    with st.spinner("Processing audio — extracting features and predicting..."):
        # 1️⃣ Waveform
        try:
            y, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            fig, ax = plt.subplots(figsize=(8, 2.5))
            librosa.display.waveshow(y, sr=sample_rate, ax=ax, color="#1f77b4")
            ax.set_title("Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot waveform: {e}")

        # 2️⃣ Transcription
        recognizer = sr.Recognizer()
        transcript = ""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
                st.subheader("🗣️ Transcription")
                st.write(transcript)
        except sr.UnknownValueError:
            st.info("Transcription: Could not understand audio.")
        except sr.RequestError:
            st.info("Transcription: Google Speech Recognition service unavailable.")
        except Exception as ex:
            st.info(f"Transcription failed: {ex}")

        # 3️⃣ Emotion Prediction
        try:
            pred_label, confidences = predict_emotion_with_confidence(audio_path)
            pred_label_name = EMOTION_MAP.get(pred_label, pred_label)
            confidences_named = {EMOTION_MAP.get(k, k): v for k, v in confidences.items()}

            emoji = EMOJI_MAP.get(pred_label_name, "")
            display_label = f"{emoji} {pred_label_name.capitalize()}"

            st.subheader("💭 Emotion Prediction")
            st.markdown(f"**Detected Emotion:** :orange[{display_label}]")

            # Confidence bar chart
            conf_df = pd.DataFrame(list(confidences_named.items()), columns=["Emotion", "Confidence"])
            conf_df = conf_df.sort_values("Confidence", ascending=False)
            st.bar_chart(conf_df.set_index("Emotion")["Confidence"])
            st.dataframe(conf_df.style.format({"Confidence": "{:.3f}"}))
        except Exception as e:
            st.error(f"Emotion prediction failed: {e}")
            pred_label_name = "N/A"
            confidences_named = {}

        elapsed = time.time() - start
        st.caption(f"Processing time: {elapsed:.2f} sec")

        # 4️⃣ Prepare Report
        report_df = pd.DataFrame({
            "emotion": [pred_label_name],
            "confidence_top": [max(confidences_named.values()) if confidences_named else None],
            "transcript": [transcript],
            "duration_sec": [librosa.get_duration(y=y, sr=sample_rate) if 'y' in locals() else None],
        })

        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("📄 Download CSV Report", data=csv, file_name="emotion_report.csv", mime="text/csv")

        # PDF generation
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Voice Emotion Detection Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 8, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.line(10, 30, 200, 30)
            pdf.ln(6)

            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, txt=f"Detected Emotion: {pred_label_name.capitalize()}", ln=True)
            pdf.cell(0, 8, txt=f"Top Confidence: {round(max(confidences_named.values()),3) if confidences_named else 'N/A'}", ln=True)
            pdf.cell(0, 8, txt=f"Duration (s): {round(librosa.get_duration(y=y, sr=sample_rate),2) if 'y' in locals() else 'N/A'}", ln=True)
            pdf.ln(4)
            pdf.multi_cell(0, 6, txt=f"Transcription:\n{transcript if transcript else 'N/A'}")
            pdf.ln(4)
            pdf.cell(0, 6, txt="Confidence Scores:", ln=True)
            for emo, c in confidences_named.items():
                pdf.cell(0, 6, txt=f" - {emo.capitalize()}: {round(c,4)}", ln=True)

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("🧾 Download PDF Report", data=pdf_bytes, file_name="emotion_report.pdf", mime="application/pdf")
        except Exception as e:
            st.warning(f"Could not create PDF: {e}")

        st.success("✅ Analysis complete!")
