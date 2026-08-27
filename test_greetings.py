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


class_names = {
    0: "Good Morning",
    1: "Good Afternoon",
    2: "Good Evening",
    3: "Hello",
    4: "How Are You",
    5: "Im Fine",
    6: "Nice To Meet You",
    7: "Thank You",
    8: "Youre Welcome",
    9: "See You Tomorrow",
}

MODEL_PATH = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "greeting"
    / "greeting_lstm_best.pt"
)


def load_model(model_path):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_size = int(checkpoint.get("input_size", 126))
        num_classes = int(
            checkpoint.get("num_classes", len(class_names))
        )
        state_dict = checkpoint["model_state_dict"]
    else:
        input_size = 126
        num_classes = len(class_names)
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

    return model, input_size


def main():
    model, input_size = load_model(MODEL_PATH)

    print(f"Model input size: {input_size}")

    if input_size == 252:
        feature_fn = add_motion_features
        print("Using pipeline: normalize_sequence + add_motion_features")
    elif input_size == 126:
        feature_fn = None
        print("Using pipeline: normalize_sequence only")
    else:
        raise ValueError(
            f"Unsupported Survival input size: {input_size}. "
            "Expected 126 or 252."
        )

    recognizer = WebcamRecognizer(
        model=model,
        labels=class_names,
        seq_length=30,

        # Use the actual model input size
        input_size=input_size,

        # Match preprocessing to the checkpoint input size.
        normalize_fn=normalize_sequence,
        feature_fn=feature_fn,

        camera_index=0,
        confidence_threshold=0.0,
        window_name="FSL Greeting Recognition",
        max_missed_frames=4,
    )

    recognizer.run()


if __name__ == "__main__":
    main()
# python test_survival.py