Implemented a full rewrite of 02_static_training.ipynb to make training and webcam inference use the same preprocessing and saved artifacts.

What was changed
Added reusable preprocessing functions:

extract_keypoints_from_hand_landmarks(...)
normalize_landmarks(...) (wrist-relative + configurable scale mode: bbox, max_dist, none)
preprocess_sample(...)
preprocess_dataset(...)
See 02_static_training.ipynb (line 83) and 02_static_training.ipynb (line 114).
Added proper split pipeline:

Train/validation/test = 70/15/15 with stratification.
See 02_static_training.ipynb (line 222).
Added callbacks:

EarlyStopping and ModelCheckpoint (best model by val_loss).
See 02_static_training.ipynb (line 286) and 02_static_training.ipynb (line 292).
Added artifact saving:

Model: static_fsl_model.keras
Label encoder: label_encoder.pkl
Preprocessing config: preprocess_config.pkl
See 02_static_training.ipynb (line 58), 02_static_training.ipynb (line 59), and 02_static_training.ipynb (line 418).
Added organized markdown sections explaining each stage (imports, preprocessing, split, training, evaluation, saving, inference).

Added inference example cell that:

Loads model + encoder + preprocessing config
Extracts MediaPipe landmarks
Applies the exact same normalization used in training
Predicts from webcam input
See 02_static_training.ipynb (line 457), 02_static_training.ipynb (line 470), and 02_static_training.ipynb (line 491).