# predict_emotion.py
import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features
import os

# Load model once (expect model path in working dir models/emotion_model.h5)
MODEL_PATH = os.path.join("models", "emotion_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train or place model there.")

model = load_model(MODEL_PATH)

# Try to load classes.npy if available; otherwise fallback to default labels
CLASSES_PATH = os.path.join("models", "classes.npy")
if os.path.exists(CLASSES_PATH):
    LABELS = list(np.load(CLASSES_PATH, allow_pickle=True))
else:
    # Fallback: change this to match your trained model labels if different
    LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


def predict_emotion_with_confidence(file_path):
    """
    Returns (pred_label_str, confidence_dict)
    confidence_dict maps label->probability (0..1)
    """
    # 1️⃣ Extract features
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)

    # 2️⃣ Model prediction
    raw_pred = model.predict(features, verbose=0)[0]

    # ✅ Temperature scaling (smooths overconfidence)
    temperature = 1.5
    probs = np.exp(np.log(raw_pred + 1e-9) / temperature)
    probs = probs / np.sum(probs)

    # 3️⃣ Handle output-label mismatches
    if len(probs) != len(LABELS):
        if len(probs) > len(LABELS):
            probs = probs[:len(LABELS)]
        else:
            probs = np.pad(probs, (0, len(LABELS) - len(probs)), mode='constant')

    # 4️⃣ Normalize to ensure total = 1
    sum_probs = float(np.sum(probs))
    if sum_probs <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / sum_probs

    # 5️⃣ Get predicted label
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]

    # Map to confidence dict
    confidence_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    # ✅ 6️⃣ Post-correction rule to reduce "angry" bias
    # Only trigger if angry < 0.6 and another calm/happy/neutral is strong
    if pred_label.lower() == "angry":
        angry_conf = confidence_dict.get("angry", 0)
        happy_conf = confidence_dict.get("happy", 0)
        neutral_conf = confidence_dict.get("neutral", 0)
        calm_conf = confidence_dict.get("calm", 0)

        if angry_conf < 0.6:
            if happy_conf > 0.3 or neutral_conf > 0.3 or calm_conf > 0.3:
                # pick whichever of happy/neutral/calm is strongest
                alt = max(
                    [("happy", happy_conf), ("neutral", neutral_conf), ("calm", calm_conf)],
                    key=lambda x: x[1]
                )[0]
                pred_label = alt

    return pred_label, confidence_dict
