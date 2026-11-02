import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CSV_FILE = "asl_landmarks.csv"
MODEL_FILE = "asl_model.p"

if not os.path.exists(CSV_FILE):
    print(f"Error: '{CSV_FILE}' does not exist! Run 'extract_dataset.py' first.")
    exit()

print("Loading dataset...")
data = pd.read_csv(CSV_FILE)

# Exclude classes with too few detections
class_counts = data['label'].value_counts()
valid_classes = class_counts[class_counts >= 15].index
filtered_data = data[data['label'].isin(valid_classes)].copy()

dropped = class_counts[class_counts < 15].index.tolist()
if dropped:
    print(f"Excluded low-sample classes: {dropped}")

X = filtered_data.drop('label', axis=1).values
y = filtered_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Random Forest on {len(X_train)} samples across {len(valid_classes)} classes...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluation report
y_pred = clf.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save trained model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(clf, f)

print(f"\nModel successfully saved to '{MODEL_FILE}'.")