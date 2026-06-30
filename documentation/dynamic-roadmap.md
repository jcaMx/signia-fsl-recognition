
Your dynamic model workflow should be:

```txt
videos
→ video_manifest.csv
→ landmark extraction
→ preprocessing
→ sequence dataset
→ train model
→ evaluate errors
→ export model + label maps
→ Flask/Godot inference
```

Use Jupyter notebooks for readability, but keep reusable functions in `.py` files so your project does not become messy.

---

# Recommended notebook roadmap

## Folder structure

```txt
dynamic_fsl_training/
  labels.csv
  video_manifest.csv

  notebooks/
    00_dataset_audit.ipynb
    01_create_video_manifest.ipynb
    02_extract_landmarks.ipynb
    03_preprocess_sequences.ipynb
    04_visualize_dataset.ipynb
    05_train_baseline_model.ipynb
    06_evaluate_model.ipynb
    07_train_final_model.ipynb
    08_realtime_inference_test.ipynb

  src/
    config.py
    labels.py
    manifest.py
    landmarks.py
    preprocessing.py
    dataset.py
    models.py
    evaluation.py
    inference.py

  dataset/
    videos/
      GREETING/
        GOOD_MORNING/
        GOOD_AFTERNOON/
        GOOD_EVENING/

    processed/
      GREETING/
        raw_sequences/
        X.npy
        y.npy
        metadata.csv
        label_map.json
        preprocess_config.json

  models/
    GREETING/
      greeting_dynamic_model.keras
      label_map.json
      preprocess_config.json
```

The notebooks should explain and run the pipeline. The `src/` files should contain reusable code.

---

# Phase 1: Dataset audit

## `00_dataset_audit.ipynb`

Goal: check if your CSV and video folders are clean.

This notebook should verify:

```txt
- labels.csv loads correctly
- every label has a unique id
- no duplicate labels
- no empty category values
- no disabled labels included in training
- every video folder matches a label
- every class has enough videos
```

Example checks:

```python
import pandas as pd

labels = pd.read_csv("../labels.csv")

labels.head()
labels["category"].value_counts()
labels["modality"].value_counts()
labels[labels.duplicated("id")]
labels[labels.duplicated("label")]
```

You should also check per-category class counts:

```python
labels.groupby("category")["label"].count()
```

Expected result:

```txt
GREETING         10
SURVIVAL         10
NUMBER           10
CALENDAR         12
DAYS             10
FAMILY           10
RELATIONSHIPS    10
COLOR            13
FOOD             10
DRINK            10
```

---

# Phase 2: Create `video_manifest.csv`

## `01_create_video_manifest.ipynb`

Your `labels.csv` defines the signs, but your model needs to know the actual video files.

Create this:

```csv
video_path,label_id,label,display_label,category,signer_id,take,enabled
dataset/videos/GREETING/GOOD_MORNING/001.mp4,0,GOOD MORNING,Good Morning,GREETING,signer01,1,true
dataset/videos/GREETING/GOOD_MORNING/002.mp4,0,GOOD MORNING,Good Morning,GREETING,signer01,2,true
```

This file is very important because it lets you track:

```txt
which file belongs to which class
which signer performed it
which category it belongs to
which samples are disabled/bad
```

Do not train directly from folders forever. Use the manifest.

---

# Phase 3: Choose your first training target

Do not train all 105 labels first.

Start with:

```txt
Category: GREETING
Classes: 5 first, then 10
```

First 5-class model:

```txt
GOOD MORNING
GOOD AFTERNOON
GOOD EVENING
HELLO
THANK YOU
```

Then full GREETING model:

```txt
GOOD MORNING
GOOD AFTERNOON
GOOD EVENING
HELLO
HOW ARE YOU
IM FINE
NICE TO MEET YOU
THANK YOU
YOURE WELCOME
SEE YOU TOMORROW
```

This is your first milestone.

---

# Phase 4: Extract landmarks from videos

## `02_extract_landmarks.ipynb`

Goal: convert each video into a sequence of landmarks.

Pipeline:

```txt
video file
→ read frames with OpenCV
→ detect hands with MediaPipe
→ extract left hand landmarks
→ extract right hand landmarks
→ save raw sequence
```

Recommended first feature shape:

```txt
2 hands × 21 landmarks × 3 coordinates = 126 features per frame
```

So one video becomes:

```txt
num_frames × 126
```

Example:

```txt
GOOD_MORNING_001.mp4 → 52 frames × 126
GOOD_MORNING_002.mp4 → 38 frames × 126
HELLO_001.mp4        → 44 frames × 126
```

Save raw extracted files:

```txt
dataset/processed/GREETING/raw_sequences/
  GOOD_MORNING_001.npy
  GOOD_MORNING_002.npy
  HELLO_001.npy
```

Also save extraction metadata:

```csv
video_path,label_id,label,total_frames,detected_frames,detection_rate,status
dataset/videos/GREETING/GOOD_MORNING/001.mp4,0,GOOD MORNING,52,50,0.96,ok
dataset/videos/GREETING/HELLO/004.mp4,3,HELLO,41,18,0.43,bad_detection
```

Recommended rule:

```txt
If detection_rate < 0.70, review or exclude the video.
```

---

# Phase 5: Preprocess sequences

## `03_preprocess_sequences.ipynb`

Goal: make every video have the same input shape.

Your model cannot train on random lengths like:

```txt
38 × 126
52 × 126
91 × 126
```

So convert every sequence to:

```txt
40 × 126
```

Recommended steps:

```txt
1. Fill missing hand landmarks with zeros.
2. Normalize coordinates.
3. Resample every video to 40 frames.
4. Save X.npy and y.npy.
```

Final output:

```txt
X shape = samples × 40 × 126
y shape = samples
```

Example:

```txt
X.shape = (200, 40, 126)
y.shape = (200,)
```

For GREETING with 10 signs and 20 videos each:

```txt
10 classes × 20 videos = 200 samples
```

Save:

```txt
dataset/processed/GREETING/X.npy
dataset/processed/GREETING/y.npy
dataset/processed/GREETING/metadata.csv
dataset/processed/GREETING/label_map.json
dataset/processed/GREETING/preprocess_config.json
```

Example `preprocess_config.json`:

```json
{
  "sequence_length": 40,
  "features_per_frame": 126,
  "hands": 2,
  "landmarks_per_hand": 21,
  "coordinates": ["x", "y", "z"],
  "normalization": "wrist_relative_scale_normalized",
  "missing_hand_value": 0
}
```

---

# Phase 6: Visualize the dataset

## `04_visualize_dataset.ipynb`

This notebook helps you catch problems before training.

Check:

```txt
- class distribution
- video length distribution
- detection rate distribution
- sample landmark movement
- bad samples
```

You want to know things like:

```txt
GOOD MORNING has 20 videos
HELLO has 20 videos
THANK YOU has 20 videos
```

But also:

```txt
HELLO has 8 bad videos with poor detection
GOOD AFTERNOON videos are too short
THANK YOU and YOURE WELCOME look too similar
```

This notebook is where you diagnose dataset issues.

---

# Phase 7: Train baseline model

## `05_train_baseline_model.ipynb`

Start simple.

Recommended first model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense

model = Sequential([
    Input(shape=(40, 126)),

    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(32)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])
```

Compile:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Use callbacks:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        patience=5,
        factor=0.5
    ),
    ModelCheckpoint(
        "../models/GREETING/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]
```

Train:

```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
```

---

# Phase 8: Evaluate properly

## `06_evaluate_model.ipynb`

Do not only look at accuracy.

Check:

```txt
- validation accuracy
- test accuracy
- confusion matrix
- per-class precision
- per-class recall
- wrong predictions
```

The confusion matrix is very important.

Example:

```txt
GOOD MORNING confused with GOOD AFTERNOON
DONT KNOW confused with DONT UNDERSTAND
THANK YOU confused with YOURE WELCOME
```

That tells you what to fix.

Possible fixes:

```txt
- collect more samples for confused signs
- trim videos better
- improve lighting
- add more signers
- add pose landmarks
- split signs into separate lessons
```

Good first target:

```txt
5 classes: 85%+ test accuracy
10 classes: 75%+ test accuracy
```

If your 10-class model is below 70%, do not expand yet.

---

# Phase 9: Train final category model

## `07_train_final_model.ipynb`

Once the baseline works, train the final model for the category.

Output files:

```txt
models/GREETING/greeting_dynamic_model.keras
models/GREETING/label_map.json
models/GREETING/preprocess_config.json
models/GREETING/training_report.json
```

Example `label_map.json`:

```json
{
  "0": {
    "global_id": 0,
    "label": "GOOD MORNING",
    "display_label": "Good Morning",
    "category": "GREETING"
  },
  "1": {
    "global_id": 1,
    "label": "GOOD AFTERNOON",
    "display_label": "Good Afternoon",
    "category": "GREETING"
  }
}
```

Important: keep both local index and global ID.

The model output is local:

```txt
model output index 0
```

But your game should receive global:

```txt
label_id 0
label GOOD MORNING
category GREETING
```

For another category like SURVIVAL:

```txt
model output index 0 → global_id 10 → UNDERSTAND
model output index 1 → global_id 11 → DONT UNDERSTAND
```

---

# Phase 10: Realtime inference test

## `08_realtime_inference_test.ipynb`

Before Flask/Godot, test the model using your webcam.

Realtime logic:

```txt
1. Open webcam.
2. Extract landmarks frame by frame.
3. Keep last 40 frames in a buffer.
4. Run prediction.
5. Apply stabilization.
6. Show stable result.
```

Important: do not predict from one frame.

Use a rolling sequence:

```txt
frame buffer = last 40 frames
model input = 1 × 40 × 126
```

Prediction stabilization:

```txt
- confidence must be >= 0.75
- same prediction must appear several times
- use majority vote over recent predictions
```

Example output:

```json
{
  "prediction": "GOOD MORNING",
  "display_label": "Good Morning",
  "label_id": 0,
  "category": "GREETING",
  "confidence": 0.91,
  "stable": true
}
```

---

# Phase 11: Flask integration

After the notebook test works, move inference to Flask.

Recommended backend structure:

```txt
backend/
  app.py

  recognition/
    model_registry.py
    landmark_extractor.py
    sequence_buffer.py
    prediction_service.py
    stabilizer.py

  models/
    GREETING/
      greeting_dynamic_model.keras
      label_map.json
      preprocess_config.json
```

Flask should receive:

```json
{
  "category": "GREETING",
  "frame": "base64 image"
}
```

Flask should return:

```json
{
  "prediction": "GOOD MORNING",
  "display_label": "Good Morning",
  "label_id": 0,
  "category": "GREETING",
  "confidence": 0.91,
  "stable": true
}
```

Your game should tell Flask which category is active.

Example:

```txt
Lesson 1: Greetings
→ category = GREETING
→ load greeting_dynamic_model.keras
```

This avoids making the model choose from all 105 signs.

---

# The exact roadmap I recommend

## Milestone 1 — Make the pipeline work

```txt
Goal: Train 3–5 GREETING signs
Classes:
- GOOD MORNING
- GOOD AFTERNOON
- GOOD EVENING
- HELLO
- THANK YOU
```

Deliverables:

```txt
X.npy
y.npy
label_map.json
baseline model
confusion matrix
webcam prediction test
```

Do not worry about perfect accuracy yet. Just prove the full pipeline works.

---

## Milestone 2 — Full GREETING model

```txt
Goal: Train all 10 GREETING signs
```

Deliverables:

```txt
greeting_dynamic_model.keras
test accuracy report
Flask prediction endpoint
Godot integration test
```

This is your first real usable dynamic lesson.

---

## Milestone 3 — Improve dataset quality

Before adding many categories, improve GREETING data.

Target:

```txt
30–50 videos per sign
at least 3 signers if possible
different speeds
different lighting
different camera distance
```

Your current 19–20 videos per sign is okay for testing, but not ideal for reliable gameplay.

---

## Milestone 4 — Add SURVIVAL model

Next category:

```txt
UNDERSTAND
DONT UNDERSTAND
KNOW
DONT KNOW
NO
YES
WRONG
CORRECT
SLOW
FAST
```

This category may be harder because some signs are conceptually and visually similar.

Train it as:

```txt
survival_dynamic_model.keras
```

Do not mix it with GREETING yet.

---

## Milestone 5 — Decide static vs dynamic categories

Not everything should use the dynamic model.

Recommended classification:

```txt
GREETING        dynamic
SURVIVAL        dynamic
NUMBER          static or short dynamic
CALENDAR        dynamic
DAYS            dynamic
FAMILY          dynamic
RELATIONSHIPS   dynamic
COLOR           static or dynamic depending on your signs
FOOD            dynamic
DRINK           dynamic
```

For gameplay, you can use both:

```txt
static_model.keras
dynamic_greeting_model.keras
dynamic_survival_model.keras
```

---

# Important rules while training

## Rule 1: Do not expand too fast

Bad path:

```txt
Train all 105 signs immediately
```

Good path:

```txt
5 signs → 10 signs → 1 category → 2 categories → more categories
```

## Rule 2: Fix the data before blaming the model

Most dynamic sign problems come from:

```txt
bad lighting
unclear hand detection
inconsistent start/end timing
too few samples
too many similar signs
same signer only
```

## Rule 3: Save everything

Every training run should save:

```txt
model
label map
preprocess config
accuracy/loss plot
confusion matrix
classification report
training notes
```

Example:

```txt
models/GREETING/runs/run_001/
  model.keras
  label_map.json
  preprocess_config.json
  training_history.csv
  confusion_matrix.png
  classification_report.txt
  notes.md
```

This helps you compare experiments.

---

# Your immediate next steps

Do these next:

```txt
1. Create video_manifest.csv.
2. Choose GREETING as the first category.
3. Temporarily enable only 5 GREETING signs.
4. Build 02_extract_landmarks.ipynb.
5. Save raw landmark sequences.
6. Build 03_preprocess_sequences.ipynb.
7. Train first LSTM model.
8. Evaluate confusion matrix.
9. Test with webcam.
10. Integrate into Flask only after notebook inference works.
```

Your next file should be:

```txt
video_manifest.csv
```

Your next notebook should be:

```txt
01_create_video_manifest.ipynb
```

The most important thing: **do not jump to Flask/Godot until the notebook webcam test works.**
