# Codebase Refactoring Walkthrough

We have successfully restructured the Filipino Sign Language (FSL) Recognition training project, extracted reusable modules into the `src/` library, organized the notebook folders, and updated imports to make notebooks self-contained and beginner-readable.

## Summary of Changes

### 1. Created Core Package Library (`src/`)
We modularized the project by separating the preprocessing, extraction, modeling, evaluation, and inference code into `src/`:

*   **`src/labels.py`**: Handles loading, mapping, and filtering of active class labels for both static and dynamic modalities using `csv/expanded_labels.csv`.
*   **`src/manifest.py`**: Scans dataset directories and maps raw files to labels.
*   **`src/landmarks/`**:
    *   `static_extractor.py`: MediaPipe HandLandmarker wrapper for single images.
    *   `dynamic_extractor.py`: MediaPipe tracking wrapper for sequence-based frames or videos.
*   **`src/preprocessing/`**:
    *   `normalization.py`: Relocalizes wrists to origin and scales landmarks (bounding box/max distance).
    *   `sequence_utils.py`: Sequence padding and truncation for dynamic frames.
    *   `static_preprocessing.py`: Wrapper for static data preprocessing.
    *   `dynamic_preprocessing.py`: Wrapper for dynamic data sequence preprocessing.
*   **`src/models/`**:
    *   `static_model.py`: MLP network architecture configuration builder.
    *   `dynamic_model.py`: LSTM network architecture configuration builder.
*   **`src/evaluation/`**:
    *   `metrics.py`: Wraps reports and confusion matrix calculations.
    *   `plots.py`: Generates confusion matrix heatmaps and training accuracy/loss curves.
*   **`src/inference/`**:
    *   `stabilizer.py`: Prediction stabilizer implementing deque-based voting filters.
    *   `static_inference.py`: Pipeline for real-time frame classification.
    *   `dynamic_inference.py`: Pipeline for real-time sequence gesture classification.

---

### 2. Reorganized and Renamed Notebooks
We structured the `notebooks/` folder by modality folders and renamed files into logical sequential flows:

*   **Static modality (`notebooks/01_static/`)**:
    *   `01_static_landmark_extraction.ipynb` → [01_extract_static_landmarks.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/01_static/01_extract_static_landmarks.ipynb)
    *   `02_static_training.ipynb` → [02_train_static_model.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/01_static/02_train_static_model.ipynb)
    *   `02_static_webcam_test.ipynb` → [03_test_static_webcam.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/01_static/03_test_static_webcam.ipynb)
*   **Dynamic modality (`notebooks/02_dynamic/`)**:
    *   `03_dynamic_landmark_extraction.ipynb` → [01_extract_dynamic_landmarks.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/02_dynamic/01_extract_dynamic_landmarks.ipynb)
    *   `03_dynamic_training.ipynb` → [02_train_dynamic_model.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/02_dynamic/02_train_dynamic_model.ipynb)
    *   `03_dynamic_webcam_test.ipynb` → [03_test_dynamic_webcam.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/02_dynamic/03_test_dynamic_webcam.ipynb)
*   **Experiments (`notebooks/experiments/`)**:
    *   `04_ad.ipynb` → [04_ad.ipynb](file:///c:/Projects/signia-fsl-recognition/notebooks/experiments/04_ad.ipynb)
*   **Archive (`notebooks/archive/`)**:
    *   Moved legacy scripts [fsl_preprocessing.py](file:///c:/Projects/signia-fsl-recognition/notebooks/archive/fsl_preprocessing.py) and [fsl_dynamic_utils.py](file:///c:/Projects/signia-fsl-recognition/notebooks/archive/fsl_dynamic_utils.py) here.

---

### 3. Updated Imports and Added `sys.path` Configuration
*   Notebooks have been programmatically edited to inject path adjustments (`sys.path.append(os.path.abspath(os.path.join('..', '..')))`), enabling clean imports from `src.*`.
*   Root test scripts `static_testing.py` and `dynamic_webcam_test.py` have been updated to cleanly import from `src.preprocessing.normalization`.

---

### 4. Added README Documentation
A new root [README.md](file:///c:/Projects/signia-fsl-recognition/README.md) has been created to guide users on:
*   Project file tree
*   Pipeline modalities
*   Raw dataset directory layouts (`static_raw/` vs `dynamic_raw/`)
*   Sequential notebook execution steps
