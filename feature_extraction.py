# feature_extraction.py
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def ensure_wav(file_path):
    """If file is not wav, convert to converted_temp.wav and return that path."""
    if file_path.lower().endswith(".wav"):
        return file_path
    audio = AudioSegment.from_file(file_path)
    out_path = "converted_temp.wav"
    audio.export(out_path, format="wav")
    return out_path

def extract_features(file_path, target_len=200):
    """
    Returns a 1D numpy array of length target_len (pads/truncates).
    Features: MFCC(40) mean + Mel-spectrogram mean + spectral contrast mean
    """
    file_to_load = ensure_wav(file_path)

    try:
        audio, sample_rate = librosa.load(file_to_load, sr=None, mono=True)
    except Exception:
        audio, sample_rate = sf.read(file_to_load)

    # Feature extraction
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)

    features = np.hstack((mfccs, mel, contrast)).astype(np.float32)

    # Pad or truncate to target_len
    if len(features) < target_len:
        features = np.pad(features, (0, target_len - len(features)), mode='constant')
    else:
        features = features[:target_len]

    return features
