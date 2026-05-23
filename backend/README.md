# Signia Backend

## Overview

This backend serves hand-sign predictions over HTTP using Flask, MediaPipe Hands, and TensorFlow/Keras models.

The current alphabet pipeline is designed to stay close to the training and testing setup used in `static_testing.py`:

- MediaPipe extracts `21 x (x, y, z)` landmarks
- landmarks are normalized with the same preprocessing logic used during training
- the saved `label_encoder.pkl` is used to decode model outputs
- prediction stabilization is available for streaming use cases

The `/predict` endpoint supports two frame-source styles without changing predictor code:

- web client flow: the client owns the webcam and sends a base64 image
- Godot flow: Flask owns the webcam and the client sends only prediction options

## Architecture

```text
frontend / game / web client
    -> POST /predict
    -> PredictionRouter
    -> ModelManager selects predictor by mode
    -> frame source resolution
       - `frame_source: "client"` -> decode payload image
       - `frame_source: "server"` -> read from Flask webcam
       - missing `frame_source` -> auto-detect from payload
    -> MediaPipeHandsPipeline
    -> predictor-specific preprocessing + model inference
    -> optional PredictionStabilizer
    -> JSON response
```


## Project Structure

```text
backend/
|-- app.py
|-- README.md
|-- core/
|   |-- model_manager.py
|   |-- prediction_router.py
|   |-- prediction_stabilizer.py
|   |-- preprocessing.py
|   `-- __init__.py
|-- models/
|   |-- alphabet_model/
|   |   `-- predictor.py
|   |-- command_model/
|   |   `-- predictor.py
|   |-- dynamic_model/
|   |   `-- predictor.py
|   |-- phrase_model/
|   |   `-- predictor.py
|   `-- word_model/
|       `-- predictor.py
|-- templates/
|   `-- test.html
`-- webcam/
    |-- camera_manager.py
    `-- mediapipe_pipeline.py
```

## Core Components

### `app.py`

Creates the Flask app and wires together shared services:

- `ModelManager`
- `CameraManager`
- `MediaPipeHandsPipeline`
- `PredictionStabilizer`
- `PredictionRouter`

This keeps the app bootstrap small and makes the services reusable.

### `core/model_manager.py`

Handles predictor discovery and registration.

Each mode is implemented as a predictor module under `backend/models/<mode>/predictor.py`.
At startup, `discover_models()` scans those modules and registers them automatically.

### `core/prediction_router.py`

This is the HTTP layer.

Responsibilities:

- parse requests
- resolve frame source
- decode base64 images for client-owned webcam flows
- capture frames from `CameraManager` for server-owned webcam flows
- run the MediaPipe pipeline
- invoke the selected predictor
- optionally stabilize predictions
- return a JSON response for the frontend

### `core/preprocessing.py`

Shared single-frame preprocessing helpers, including:

- first-hand extraction
- hand count helpers
- flattening MediaPipe landmarks into 63 features
- training-style normalization
- model input shaping and validation

### `core/prediction_stabilizer.py`

Provides lightweight in-memory stabilization for stream-like inference.

Features:

- rolling history buffer
- confidence threshold
- short consistency window
- cooldown between emitted predictions
- rapid-change suppression
- stale stream expiry

The stabilizer is applied per stream id, so multiple clients can coexist without mixing histories.

### `webcam/camera_manager.py`

Handles backend webcam access when the router needs a server-owned frame source.

Current behavior includes:

- lazy camera open
- frame size hints
- warmup reads
- reduced buffer lag

### `webcam/mediapipe_pipeline.py`

Wraps MediaPipe Hands so predictors do not need to manage the hand detector directly.

## Current Alphabet Pipeline

The alphabet predictor is the most complete implementation right now.

It loads:

- `models/static_fsl_model.keras`
- `models/label_encoder.pkl`
- `models/preprocess_config.pkl`

It then:

1. extracts the first hand landmarks from MediaPipe
2. flattens them into `63` values
3. normalizes them using the saved preprocessing config
4. reshapes them to `(1, 63)`
5. runs model inference
6. decodes the predicted class using the saved label encoder

This means backend inference now matches the training/testing path much more closely than a raw-landmark-only approach.

## API

### `GET /`

Health/status endpoint.

Example response:

```json
{
  "status": "running",
  "available_modes": ["alphabet", "command", "dynamic", "phrase", "word"]
}
```

### `GET /modes`

Returns the registered model modes.

### `POST /predict`

Primary prediction endpoint.

Example payload for a web client:

```json
{
  "mode": "alphabet",
  "frame_source": "client",
  "image": "data:image/jpeg;base64,..."
}
```

Example payload for Godot or another HTTP client that wants Flask to own the webcam:

```json
{
  "mode": "alphabet",
  "frame_source": "server",
  "stabilize": true,
  "stream_id": "godot-player-1"
}
```

Optional fields:

- `frame_source`: `"client"` or `"server"`
- `stabilize`: force enable or disable stabilization
- `stream_id`: client stream key for stabilization
- `session_id`: alternate stream key
- `client_id`: alternate stream key

If `frame_source` is omitted, the router auto-detects:

- `image` present -> `client`
- no `image` -> `server`

### `GET /predict_letter`

Shortcut route for alphabet mode using the backend webcam.

## Response Shape

Typical response fields:

```json
{
  "prediction": "W",
  "confidence": 0.999,
  "mode": "alphabet",
  "raw_prediction": "W",
  "raw_confidence": 0.999,
  "frame_source": "client",
  "stabilization_enabled": false,
  "frame_shape": [480, 640, 3],
  "hand_detected": true,
  "hand_count": 1,
  "landmark_count": 21,
  "feature_count": 63,
  "model_input_features": 63,
  "preprocess_scale_mode": "bbox"
}
```

Notes:

- `prediction` is the final output the frontend should usually display
- `raw_prediction` is the immediate model result before stabilization
- if stabilization is enabled, `prediction` may lag behind `raw_prediction`
- if stabilization is disabled, `prediction` and `raw_prediction` will usually match

## Frontend Integration

### 1. Web client flow

Recommended payload:

```json
{
  "mode": "alphabet",
  "frame_source": "client",
  "image": "data:image/jpeg;base64,...",
  "stream_id": "web-client-1"
}
```

Behavior:

- browser captures frames locally
- Flask decodes the image payload
- use a stable `stream_id` for stabilization
- best for browser-based live preview and testing

### 2. Godot or server-webcam flow

Recommended payload:

```json
{
  "mode": "alphabet",
  "frame_source": "server",
  "stabilize": true,
  "stream_id": "godot-player-1"
}
```

Recommendations:

- Godot does not need to manage camera capture here
- Flask uses `CameraManager.read_frame()` internally
- keep the same `stream_id` for the same user/session
- display `prediction`, not `raw_prediction`, if you want flicker reduction
- display `raw_prediction` only for debugging

### 3. Auto-detection behavior

If the client does not send `frame_source`:

- sending an `image` makes the router use `client`
- omitting `image` makes the router use `server`

This keeps the endpoint flexible without hardcoding different predictor logic.

### 4. Which field should the frontend display?

Use:

- `prediction` for user-facing UI
- `raw_prediction` for debugging or development overlays

If `stabilization_enabled` is `false`, the final and raw outputs should be effectively the same.

### 5. Error handling

The frontend should handle:

- no hand detected
- unsupported mode
- invalid client image when `frame_source` is `client`
- camera unavailable when `frame_source` is `server`
- unimplemented predictor modes

## Extension Strategy

The backend is intentionally open for extension through predictor modules.

To add a new mode:

1. create a folder under `backend/models/`, for example `intent_model/`
2. add `predictor.py`
3. implement a class that inherits from `BasePredictor`
4. define a unique `mode` string
5. implement `predict(context) -> PredictionResponse`
6. expose `register(model_manager)` and call `model_manager.register(...)`
7. restart the backend

`ModelManager.discover_models()` will automatically load it at startup.

### What a predictor owns

A predictor can own its own:

- model files
- label encoder
- preprocessing config
- preprocessing logic
- inference policy
- response metadata

This makes each mode self-contained and avoids hardcoding model-specific logic into `app.py` or the router.

### Example responsibilities for a future predictor

- `dynamic_model`: sequence buffering, temporal preprocessing, sequence classifier
- `word_model`: custom vocabulary and decoder
- `phrase_model`: phrase-level aggregation or language-model post-processing
- `command_model`: command mapping for game actions

## Design Principles

The current backend is structured around a few practical rules:

- shared transport, mode-specific inference
- preprocessing should mirror training artifacts whenever possible
- frame acquisition and prediction logic should stay separate
- stabilization should improve UX, not hide valid raw predictions
- new modes should be pluggable without editing the app bootstrap

## Local Development

From the project root:

```powershell
.\fsl39\Scripts\activate
cd backend
python app.py
```

Open:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/modes
http://127.0.0.1:5000/test
```

## Stabilizer Configuration

The stabilizer can be tuned with environment variables:

- `FSL_STABILIZER_HISTORY_SIZE`
- `FSL_STABILIZER_WINDOW_MS`
- `FSL_STABILIZER_CONFIDENCE_THRESHOLD`
- `FSL_STABILIZER_MIN_CONSISTENT_FRAMES`
- `FSL_STABILIZER_COOLDOWN_MS`
- `FSL_STABILIZER_CHANGE_SUPPRESSION_MS`
- `FSL_STABILIZER_STALE_TTL_MS`

These are especially useful when tuning live webcam UX.

## Practical Notes

- The backend alphabet predictor currently uses the root-level trained artifacts, not the older local `backend/models/alphabet_model/model.keras` path.
- The router supports both client-owned and server-owned webcam flows through `frame_source`, while predictors remain unchanged.
- The `backend/models/<mode>/predictor.py` layout still matters because it is how modes are discovered and extended.
- Some registered modes are placeholders and may return `501 Not Implemented` until their predictors are completed.
