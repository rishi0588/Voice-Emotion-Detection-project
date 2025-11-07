# 🎙️ Voice Emotion Detection using Deep Learning

## 🧠 Overview
This project detects **human emotions from voice recordings** using **audio signal processing** and a **deep learning model** built with TensorFlow and Keras.  
It features a **Streamlit web app** that lets users upload or record voice samples, visualize waveforms, view emotion confidence scores, and download reports in **CSV** and **PDF** format.

---

## 🚀 Features
✅ Upload or record audio using a built-in microphone interface  
✅ Automatic waveform visualization  
✅ Real-time **emotion classification** (Angry, Happy, Calm, Sad, Fearful, Neutral, etc.)  
✅ Speech transcription using **Google SpeechRecognition API**  
✅ Confidence-based emotion probability chart  
✅ Exportable **PDF and CSV reports**  
✅ Lightweight, intuitive **Streamlit** interface  

---

## 🧩 Tech Stack

| Component | Technology / Library |
|------------|----------------------|
| **Frontend UI** | Streamlit |
| **Audio Processing** | Librosa, PyDub, FFmpeg |
| **Model Training** | TensorFlow / Keras |
| **Feature Extraction** | MFCCs, Mel Spectrogram, Spectral Contrast |
| **Speech Transcription** | SpeechRecognition (Google Web API) |
| **Reporting** | Pandas, FPDF |
| **Language/NLP Connection** | Speech-to-text and semantic understanding |

---

## 🧰 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rishi0588/Voice-Emotion-Detection-project.git
cd voice-emotion-detection
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Install FFmpeg (Required for PyDub)
bash
Copy code
python -m ffmpeg_downloader install
4️⃣ Run the App
bash
Copy code
streamlit run app.py
🧠 How It Works
Audio Input — User either uploads a .wav file or records audio via Streamlit’s microphone.

Feature Extraction — feature_extraction.py extracts MFCCs, Mel Spectrograms, and Spectral Contrast features using Librosa.

Emotion Classification — The features are passed into a feed-forward neural network trained on emotional speech data (e.g. RAVDESS or custom dataset).

Confidence Estimation — The model outputs class probabilities; temperature scaling is applied to reduce bias.

Speech Recognition — Audio is transcribed into text using Google SpeechRecognition API.

Visualization & Reports — Streamlit displays the waveform, emotion chart, and transcript, and generates downloadable PDF + CSV reports.

🧪 Model Details
Architecture: 3-layer Dense neural network with ReLU activations and Dropout regularization.

Trained using: RAVDESS dataset or similar emotional speech data.

Input Features: 200-dimensional vector of MFCC, Mel, and Spectral Contrast features.

Output Labels: 8 emotion classes — ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'].

Loss: Categorical Cross-Entropy

Optimizer: Adam

📊 Output Example
Detected Emotion: Happy 😄

Confidence Scores:

Emotion	Confidence
Happy	0.72
Calm	0.15
Neutral	0.09
Angry	0.04

Transcription: “Hello, I’m Rishi and I’m very happy today!”

Duration: 3.5 seconds

Generated Files:

emotion_report.csv

emotion_report.pdf

🧭 NLP Connection
Although primarily audio-based, this project integrates NLP concepts through speech transcription.
The recognized text can be used for:

Sentiment analysis,

Semantic context understanding,

Multimodal emotion recognition (voice + text).

Thus, it bridges speech signal processing and Natural Language Processing (NLP) for richer emotional insight.

🏁 Future Enhancements
🔹 Use Transformer models (e.g., Wav2Vec2 or Whisper) for direct end-to-end speech emotion recognition.

🔹 Combine voice tone + text sentiment for multi-modal emotion analysis.

🔹 Add timeline emotion visualization across longer recordings.

🔹 Optimize for Indian regional languages.

🧑‍💻 Author
Rishi Ponda
Shivansh Khandelwal
Priyanshu P/adhi
🎓 MBA (Tech) — Data Science, MPSTME NMIMS

🪪 License
This project is open source and available under the MIT License.

⭐ If you found this project useful, consider giving it a star on GitHub! ⭐
