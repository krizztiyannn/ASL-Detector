import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pickle
from collections import Counter, deque
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

st.set_page_config(
    page_title="ASL Real-Time Recognition",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time ASL Alphabet Detection Web App")
st.markdown("### CPE126L AI Summative Project")

MODEL_FILE = "asl_model.p"

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file '{MODEL_FILE}' not found! Run train_model.py first.")
    st.stop()


@st.cache_resource
def load_trained_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


model = load_trained_model()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Controls")
    run_cam = st.toggle("Activate Webcam", value=True)
    confidence_threshold = st.slider("Min Confidence (%)", 30, 90, 50, step=5)
    st.divider()
    prediction_metric = st.empty()
    confidence_metric = st.empty()

with col1:
    frame_placeholder = st.empty()

if run_cam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Camera feed unavailable. Verify camera access.")
        st.stop()

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    prediction_history = deque(maxlen=6)

    try:
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Video stream ended.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            detected_char = "No Hand Detected"
            confidence_pct = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y

                    coords = []
                    x_pixels = []
                    y_pixels = []

                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x - wrist_x, lm.y - wrist_y])
                        x_pixels.append(int(lm.x * w))
                        y_pixels.append(int(lm.y * h))

                    max_val = max(map(abs, coords))
                    if max_val > 0:
                        coords = [c / max_val for c in coords]

                    probs = model.predict_proba(np.array(coords).reshape(1, -1))[0]
                    max_idx = np.argmax(probs)
                    confidence_pct = float(probs[max_idx] * 100)
                    raw_prediction = str(model.classes_[max_idx])

                    if confidence_pct >= confidence_threshold:
                        prediction_history.append(raw_prediction)
                        detected_char = Counter(prediction_history).most_common(1)[0][0]
                        box_color = (0, 255, 0)
                    else:
                        prediction_history.clear()
                        detected_char = f"Uncertain ({raw_prediction})"
                        box_color = (255, 165, 0)

                    x_min, x_max = max(0, min(x_pixels) - 20), min(w, max(x_pixels) + 20)
                    y_min, y_max = max(0, min(y_pixels) - 20), min(h, max(y_pixels) + 20)

                    cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                    cv2.putText(
                        rgb_frame,
                        f"{detected_char} ({confidence_pct:.1f}%)",
                        (x_min, max(30, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        box_color,
                        2
                    )
            else:
                prediction_history.clear()

            prediction_metric.metric("Predicted Sign", detected_char)
            confidence_metric.metric("Confidence", f"{confidence_pct:.1f}%")
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    finally:
        cap.release()
        hands.close()
else:
    frame_placeholder.info("Webcam is inactive. Flip the toggle on the right to start.")