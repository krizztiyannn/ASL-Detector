# Real-Time ASL Alphabet Detection Web App

A machine learning web application built with Streamlit, MediaPipe Hands, OpenCV, and Scikit-Learn that classifies American Sign Language (ASL) alphabet signs in real time via webcam.

## Features
- **MediaPipe Landmark Extraction:** Captures 21 3D hand landmarks normalized relative to the wrist.
- **Random Forest Classifier:** Predicts letters with live confidence scoring.
- **Streamlit Web UI:** Browser-based interface featuring dynamic threshold controls and metric cards.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/krizztiyannn/ASL-Detector.git](https://github.com/krizztiyannn/ASL-Detector.git)
   cd ASL-Detector
