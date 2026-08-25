import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Projects\signia-fsl-recognition").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.camera.webcam_recognizer import WebcamRecognizer
from src.models.sign_lstm import SignLSTM
from src.preprocessing.greeting_features import (
    add_motion_features,
    normalize_sequence,
)


GREETING_CLASSES = {
    0: "Good Morning",
    1: "Good Afternoon",
    2: "Good Evening",
    3: "Hello",
    4: "How Are You",
}

MODEL_PATH = REPO_ROOT / "notebooks" / "02_dynamic" / "best_greetings_lstm.pt"


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_size = int(checkpoint.get("input_size", 252))
        num_classes = int(checkpoint.get("num_classes", len(GREETING_CLASSES)))
        state_dict = checkpoint["model_state_dict"]
    else:
        input_size = 252
        num_classes = len(GREETING_CLASSES)
        state_dict = checkpoint

    model = SignLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    model = load_model(MODEL_PATH)

    recognizer = WebcamRecognizer(
        model=model,
        labels=GREETING_CLASSES,
        seq_length=30,
        input_size=252,
        normalize_fn=normalize_sequence,
        feature_fn=add_motion_features,
        camera_index=0,
        confidence_threshold=0.0,
        window_name="FSL Greeting Recognition",
        max_missed_frames=4,
    )

    recognizer.run()


if __name__ == "__main__":
    main()
