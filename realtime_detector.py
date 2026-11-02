import os
import pickle
from collections import Counter, deque
import cv2
import numpy as np
import mediapipe as mp

MODEL_FILE = "asl_model.p"

if not os.path.exists(MODEL_FILE):
    print(f"Error: '{MODEL_FILE}' not found. Run 'train_model.py' first.")
    exit()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Window & Fullscreen setup
WINDOW_NAME = "Real-Time ASL Alphabet Detector"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

CONFIDENCE_THRESHOLD = 50.0
prediction_window = deque(maxlen=6)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera feed unavailable.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            coords = []
            x_pixels = []
            y_pixels = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x - wrist_x, lm.y - wrist_y])
                x_pixels.append(int(lm.x * w))
                y_pixels.append(int(lm.y * h))

            # Scale normalization matching the extraction pipeline
            max_val = max(map(abs, coords))
            if max_val > 0:
                coords = [c / max_val for c in coords]

            x_min, x_max = max(0, min(x_pixels) - 20), min(w, max(x_pixels) + 20)
            y_min, y_max = max(0, min(y_pixels) - 20), min(h, max(y_pixels) + 20)

            # Predict probabilities
            input_vector = np.array(coords).reshape(1, -1)
            probs = model.predict_proba(input_vector)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx] * 100
            current_sign = model.classes_[max_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_window.append(current_sign)
                stable_sign = Counter(prediction_window).most_common(1)[0][0]
                box_color = (0, 255, 0)
                label_text = f"{stable_sign} ({confidence:.1f}%)"
            else:
                prediction_window.clear()
                box_color = (0, 165, 255)
                label_text = f"Uncertain ({confidence:.1f}%)"

            # Draw bounding box and prediction text
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(
                frame,
                label_text,
                (x_min, max(35, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                box_color,
                2
            )
    else:
        prediction_window.clear()

    # Hotkey display
    cv2.putText(
        frame,
        "Press 'F' to Toggle Fullscreen | 'ESC' to Quit",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key
        break
    elif key in (ord('f'), ord('F')):  # Toggle Fullscreen
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
hands.close()