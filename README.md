# Filipino Sign Language (FSL) Recognition Training Project

This repository contains the training pipelines and reusable libraries for both **Static Sign Recognition** (single-frame hand landmarks classification) and **Dynamic Sign Recognition** (sequence-based sign gesture recognition) using MediaPipe and Keras/TensorFlow.

---

## Project Structure

```
c:/Projects/signia-fsl-recognition/
├── csv/
│   ├── expanded_labels.csv        # Complete label mapping CSV (with modality details)
│   └── labels.csv                 # Core labels list
├── notebooks/                     # Modality-grouped notebooks
│   ├── 00_setup/                  # Setup and diagnostic notebooks
│   ├── 01_static/                 # Static modality training pipeline
│   │   ├── 01_extract_static_landmarks.ipynb
│   │   ├── 02_train_static_model.ipynb
│   │   └── 03_test_static_webcam.ipynb
│   ├── 02_dynamic/                # Dynamic modality training pipeline
│   │   ├── 01_extract_dynamic_landmarks.ipynb
│   │   ├── 02_train_dynamic_model.ipynb
│   │   └── 03_test_dynamic_webcam.ipynb
│   ├── experiments/               # Action recognition and custom experiments
│   │   └── 04_ad.ipynb
│   └── archive/                   # Archive for legacy python scripts
├── src/                           # Reusable core Python package
│   ├── labels.py                  # Label mapping and active filter helpers
│   ├── manifest.py                # Dataset raw file scanner
│   ├── landmarks/
│   │   ├── static_extractor.py    # MediaPipe HandLandmarker wrapper for images
│   │   └── dynamic_extractor.py   # MediaPipe Hand tracking sequence extractor
│   ├── preprocessing/
│   │   ├── normalization.py       # Relative wrist normalization and scaling
│   │   ├── sequence_utils.py      # Padding and truncation helpers
│   │   ├── static_preprocessing.py # Preprocessing pipeline for static frames
│   │   └── dynamic_preprocessing.py# Sequence normalization & scaling pipeline
│   ├── models/
│   │   ├── static_model.py        # MLP Keras model definition
│   │   └── dynamic_model.py       # LSTM sequence model definition
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy metrics and scikit reports
│   │   └── plots.py               # Confusion matrix & accuracy curves plotting
│   └── inference/
│       ├── stabilizer.py          # Real-time sliding voting window filter
│       ├── static_inference.py    # Live static model prediction wrapper
│       └── dynamic_inference.py   # Live sequence model prediction wrapper
├── static_raw/                    # Place raw static image folders here (e.g. A, B, C...)
├── dynamic_raw/                   # Place raw dynamic video folders here (named by Label ID)
├── static_landmarks/              # Extracted static landmark arrays (.npy)
├── dynamic_landmarks/             # Extracted dynamic landmark sequence arrays (.npy)
├── models/                        # Saved training checkpoints and label encoders
└── requirements.txt               # Dependencies list
```

---

## Modality Pipelines & Execution Order

### 1. Static Sign Pipeline
Static signs are recognized by detecting a single-frame posture of the hand.
*   **01_extract_static_landmarks.ipynb**: Reads images from `static_raw/`, detects hand landmarks, applies wrist-centric normalization/scaling, and saves features as `.npy` arrays into `static_landmarks/`.
*   **02_train_static_model.ipynb**: Loads `.npy` arrays, performs a train/test split, builds and compiles a Keras MLP model, trains it, and saves the trained model (`models/static_fsl_model.keras`) and label encoder (`models/label_encoder.pkl`).
*   **03_test_static_webcam.ipynb**: Starts a live webcam capture to track hand landmarks, feed them into the model, and display prediction overlays.

### 2. Dynamic Sign Pipeline
Dynamic signs (gestures) are recognized over a sequence of video frames (default sequence length of 30 frames).
*   **01_extract_dynamic_landmarks.ipynb**: Reads video files from `dynamic_raw/` directories, runs sequence extraction frame-by-frame, pads/truncates sequences to 30 frames, and saves sequence features into `dynamic_landmarks/`.
*   **02_train_dynamic_model.ipynb**: Loads dynamic landmark sequences, trains an LSTM neural network, and saves the sequence classifier.
*   **03_test_dynamic_webcam.ipynb**: Runs live sequence predictions from a webcam using a sliding queue and stabilizes predictions with a voting stabilizer.

---

## Data Setup Guide

To run the notebooks:
1.  **Static Data**: Populate `static_raw/` with sign subdirectories containing image datasets:
    ```
    static_raw/
    ├── A/
    │   ├── img_1.jpg
    │   └── img_2.jpg
    ├── B/
    ...
    ```
2.  **Dynamic Data**: Populate `dynamic_raw/` with subdirectories named by their label ID (from `csv/expanded_labels.csv`), containing gesture video clips:
    ```
    dynamic_raw/
    ├── 0/
    │   ├── video_1.mp4
    │   └── video_2.mp4
    ├── 3/
    ...
    ```
