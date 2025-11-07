import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from feature_extraction import extract_features

DATASET_PATH = "data/"  # make sure this folder exists

def load_data():
    features, labels = [], []
    print("🎧 Extracting features from dataset...")

    for file in os.listdir(DATASET_PATH):
        if not file.lower().endswith(('.wav', '.mp3', '.m4a')):
            continue

        path = os.path.join(DATASET_PATH, file)
        try:
            feat = extract_features(path)
            features.append(feat)

            # Extract emotion label more flexibly
            # Example: '03-01-05-02-02-01-01.wav' → emotion='05'
            parts = file.split("-")
            emotion = parts[2] if len(parts) > 2 else "unknown"
            labels.append(emotion)
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")

    features = np.array(features)
    labels = np.array(labels)
    print(f"✅ Extracted {len(features)} samples, {len(labels)} labels")
    return features, labels

# --- MAIN SCRIPT ---
X, y = load_data()

if len(X) == 0:
    raise ValueError("🚨 No valid audio files found in 'data/'! Check your dataset path and formats.")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

os.makedirs("models", exist_ok=True)
model.save("models/emotion_model.h5")
np.save("models/classes.npy", le.classes_)
print("✅ Model and label classes saved successfully!")
