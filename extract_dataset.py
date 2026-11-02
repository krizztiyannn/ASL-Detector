import os
import csv
import cv2
import mediapipe as mp

DATASET_DIRS = [
    "asl_dataset2"
]

OUTPUT_CSV = "asl_landmarks.csv"
MAX_SAMPLES_PER_LETTER = 300

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

valid_dataset_dirs = [d for d in DATASET_DIRS if os.path.exists(d)]
if not valid_dataset_dirs:
    print(f"Error: None of the dataset directories exist: {DATASET_DIRS}")
    exit()

print(f"Datasets found: {valid_dataset_dirs}")

TARGET_LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in ['J', 'Z']]

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y')]
    writer.writerow(header)

    for letter in TARGET_LETTERS:
        print(f"Processing letter: '{letter}'...")
        count = 0

        for dataset_dir in valid_dataset_dirs:
            if count >= MAX_SAMPLES_PER_LETTER:
                break

            matching_subfolders = [
                os.path.join(dataset_dir, d)
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d)) and d.upper() == letter
            ]

            for class_folder in matching_subfolders:
                for root, _, files in os.walk(class_folder):
                    for file_name in files:
                        if count >= MAX_SAMPLES_PER_LETTER:
                            break

                        img_path = os.path.join(root, file_name)
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                wrist_x = hand_landmarks.landmark[0].x
                                wrist_y = hand_landmarks.landmark[0].y

                                coords = []
                                for lm in hand_landmarks.landmark:
                                    coords.extend([lm.x - wrist_x, lm.y - wrist_y])

                                max_val = max(map(abs, coords))
                                if max_val > 0:
                                    coords = [c / max_val for c in coords]

                                # Normal
                                writer.writerow([letter] + coords)

                                # Mirror augmentation
                                mirrored = coords.copy()
                                for i in range(0, len(mirrored), 2):
                                    mirrored[i] = -mirrored[i]
                                writer.writerow([letter] + mirrored)

                                count += 1

        print(f"Total samples for '{letter}': {count * 2} (including mirrors)")

hands.close()
print(f"\nLandmark extraction complete! Output saved to '{OUTPUT_CSV}'.")