"""
live_detection.py — Optimized real-time emotion detection
Fixes:
  - Missing comma bug in emotion_labels (root cause of wrong predictions + crash)
  - Prediction smoothing (reduces flickering)
  - Confidence threshold (suppresses low-confidence noise)
  - Safe memory-efficient predict call
  - Graceful crash handling
"""

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ==============================
# Config
# ==============================
MODEL_PATH        = "models/emotion_model.h5"
CONFIDENCE_THRESH = 0.40     # ignore predictions below 40% confidence
SMOOTH_WINDOW     = 5        # average over last 5 frames to reduce flickering

# ==============================
# FIX 1: Correct emotion labels with ALL commas present
# Original bug: 'Fear' 'Happy' (no comma) → Python merges them into 'FearHappy'
# → only 6 labels for 7 classes → index out of range on Surprise → crash
# ==============================
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Distinct colors per emotion (BGR format for OpenCV)
EMOTION_COLORS = {
    'Angry':    (0,   0,   255),   # Red
    'Disgust':  (0,   140, 0  ),   # Dark Green
    'Fear':     (128, 0,   128),   # Purple
    'Happy':    (0,   215, 255),   # Gold
    'Neutral':  (200, 200, 200),   # Light Gray
    'Sad':      (255, 100, 0  ),   # Blue-Orange
    'Surprise': (0,   165, 255),   # Orange
}

# ==============================
# Load Model
# ==============================
print("Loading emotion model...")
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"   Input shape expected: {model.input_shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# ==============================
# Load Haar Cascade
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
if face_cascade.empty():
    print("Haar cascade not found. Check your OpenCV installation.")
    exit(1)

# ==============================
# Prediction Smoothing Buffer
# ==============================
prediction_buffer = deque(maxlen=SMOOTH_WINDOW)

# ==============================
# Start Webcam
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Webcam started. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame read failed, retrying...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        prediction_buffer.clear()

    for (x, y, w, h) in faces:
        # ---- Preprocess face ----
        face_roi        = gray[y:y+h, x:x+w]
        face_resized    = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input      = np.reshape(face_normalized, (1, 48, 48, 1))

        # ---- Predict ----
        try:
            prediction = model.predict(face_input, verbose=0)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        # ---- Smoothing ----
        prediction_buffer.append(prediction)
        avg_prediction = np.mean(prediction_buffer, axis=0)
        smooth_index   = int(np.argmax(avg_prediction))
        smooth_conf    = float(np.max(avg_prediction))

        # ---- Confidence Gate ----
        if smooth_conf < CONFIDENCE_THRESH:
            emotion = "Uncertain"
            color   = (100, 100, 100)
        else:
            emotion = EMOTION_LABELS[smooth_index]
            color   = EMOTION_COLORS[emotion]

        # ---- Draw bounding box ----
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # ---- Emotion label ----
        label_text = f"{emotion} ({smooth_conf:.0%})"
        label_y    = y - 10 if y - 10 > 20 else y + h + 25
        cv2.putText(
            frame, label_text,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
        )

        # ---- Probability bars ----
        bar_y_start = y + h + 5
        bar_height  = 8
        bar_max_w   = w

        for i, (em, prob) in enumerate(zip(EMOTION_LABELS, avg_prediction)):
            bar_w   = int(prob * bar_max_w)
            bar_y   = bar_y_start + i * (bar_height + 2)
            bar_col = EMOTION_COLORS[em]

            # Background
            cv2.rectangle(frame,
                          (x, bar_y),
                          (x + bar_max_w, bar_y + bar_height),
                          (50, 50, 50), -1)
            # Fill
            if bar_w > 0:
                cv2.rectangle(frame,
                              (x, bar_y),
                              (x + bar_w, bar_y + bar_height),
                              bar_col, -1)
            # Label
            cv2.putText(frame, f"{em[:3]} {prob:.0%}",
                        (x + bar_max_w + 4, bar_y + bar_height - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, bar_col, 1, cv2.LINE_AA)

    # ---- HUD ----
    cv2.putText(frame, "Press 'q' to quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {len(faces)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n Exiting...")
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")