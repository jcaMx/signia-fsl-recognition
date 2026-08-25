# scripts/diagnose_models.py

import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.models.factory import create_model
from src.training.label_encoder import ModelLabelEncoder


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(
    r"C:\Projects\signia-fsl-recognition"
).resolve()

DATASET_PATH = (
    REPO_ROOT
    / "notebooks"
    / "02_dynamic"
    / "fsl_dataset_clean.pt"
)

LABELS_CSV = (
    REPO_ROOT
    / "csv"
    / "labels.csv"
)

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42


# Add/change paths here.
MODELS = {

    "greeting": {
        "label_ids": list(range(0, 10)),
        "model": (
            REPO_ROOT
            / "artifacts/models/greeting/"
            / "greetings_lstm_best.pt"
        ),
    },

    "survival": {
        "label_ids": list(range(10, 20)),
        "model": (
            REPO_ROOT
            / "artifacts/models/survival/"
            / "survival_lstm_best.pt"
        ),
    },

    "number": {
        "label_ids": list(range(20, 30)),
        "model": (
            REPO_ROOT
            / "artifacts/models/number/"
            / "numbers_lstm_best.pt"
        ),
    },

    "calendar": {
        "label_ids": list(range(30, 42)),
        "model": (
            REPO_ROOT
            / "artifacts/models/calendar/"
            / "calendar_lstm_best.pt"
        ),
    },
}


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    import pandas as pd

    df = pd.read_csv(LABELS_CSV)

    return {
        int(row.id): str(row.label)
        for _, row in df.iterrows()
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    bundle = torch.load(
        DATASET_PATH,
        map_location="cpu",
        weights_only=False,
    )

    X = np.asarray(bundle["X"])
    y = np.asarray(bundle["y"]).reshape(-1)

    return X, y


# ============================================================
# SPLIT — EXACT COPY OF TRAINER
# ============================================================

def split_dataset(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(
            TEST_SIZE +
            VALIDATION_SIZE
        ),
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_ratio = (
        VALIDATION_SIZE /
        (
            TEST_SIZE +
            VALIDATION_SIZE
        )
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# LANDMARK QUALITY
# ============================================================

def landmark_quality(X, y, label_names):

    print("\nLANDMARK QUALITY")
    print("-" * 90)

    for label_id in sorted(np.unique(y)):

        samples = X[y == label_id]

        total = samples.size

        zeros = np.sum(
            samples == 0
        )

        zero_pct = (
            zeros / total * 100
        )

        # 63 left + 63 right
        left = samples[:, :, :63]
        right = samples[:, :, 63:]

        left_detected = np.any(
            left != 0,
            axis=2
        )

        right_detected = np.any(
            right != 0,
            axis=2
        )

        both = (
            left_detected &
            right_detected
        )

        detected = (
            left_detected |
            right_detected
        )

        print(
            f"{label_names.get(label_id, str(label_id)):25s}"
            f" samples={len(samples):3d}"
            f" detect={detected.mean()*100:6.1f}%"
            f" left={left_detected.mean()*100:6.1f}%"
            f" right={right_detected.mean()*100:6.1f}%"
            f" both={both.mean()*100:6.1f}%"
            f" zero={zero_pct:6.2f}%"
        )


# ============================================================
# MODEL
# ============================================================

def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    input_size = checkpoint["input_size"]
    num_classes = checkpoint["num_classes"]

    model = create_model(
        model_type="sign_lstm",
        num_classes=num_classes,
        input_size=input_size,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    label_ids,
    label_names,
):

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    with torch.no_grad():

        outputs = model(
            X_tensor
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        ).numpy()

    accuracy = (
        predictions == y_test
    ).mean()

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print("\nPER-CLASS TEST ACCURACY")
    print("-" * 70)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=range(len(label_ids)),
    )

    results = []

    for encoded_id, global_id in enumerate(
        label_ids
    ):

        mask = (
            y_test == encoded_id
        )

        total = mask.sum()

        if total == 0:
            continue

        correct = (
            predictions[mask] == encoded_id
        ).sum()

        class_acc = (
            correct / total
        )

        results.append(
            (
                class_acc,
                global_id,
                correct,
                total,
            )
        )

        print(
            f"{label_names.get(global_id, str(global_id)):25s}"
            f" {correct:2d}/{total:2d}"
            f" = {class_acc*100:6.2f}%"
        )

    # --------------------------------------------------------
    # Confusions
    # --------------------------------------------------------

    print("\nTOP CONFUSIONS")
    print("-" * 70)

    confusions = []

    for true_id in range(
        len(label_ids)
    ):

        for pred_id in range(
            len(label_ids)
        ):

            if true_id == pred_id:
                continue

            count = cm[
                true_id,
                pred_id
            ]

            if count > 0:

                confusions.append(
                    (
                        count,
                        label_ids[true_id],
                        label_ids[pred_id],
                    )
                )

    confusions.sort(
        reverse=True
    )

    for count, true_id, pred_id in confusions[:10]:

        print(
            f"{label_names.get(true_id, str(true_id))}"
            f" -> "
            f"{label_names.get(pred_id, str(pred_id))}"
            f" : {count}"
        )

    # --------------------------------------------------------
    # Worst classes
    # --------------------------------------------------------

    print("\nWORST CLASSES")
    print("-" * 70)

    for acc, global_id, correct, total in sorted(
        results
    )[:5]:

        print(
            f"{label_names.get(global_id, str(global_id)):25s}"
            f" {correct}/{total}"
            f" ({acc*100:.2f}%)"
        )

    return accuracy


# ============================================================
# DIAGNOSE MODEL
# ============================================================

def diagnose_model(
    name,
    model_config,
    X,
    y,
    label_names,
):

    print("\n")
    print("=" * 90)
    print(f"MODEL: {name.upper()}")
    print("=" * 90)

    label_ids = model_config[
        "label_ids"
    ]

    model_path = model_config[
        "model"
    ]

    # --------------------------------------------------------
    # Select classes
    # --------------------------------------------------------

    mask = np.isin(
        y,
        label_ids
    )

    X_selected = X[mask]
    y_global = y[mask]

    print(
        f"Samples: {len(X_selected)}"
    )

    print(
        f"Input shape: {X_selected.shape}"
    )

    print(
        f"Classes: {len(label_ids)}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\nCLASS DISTRIBUTION")
    print("-" * 70)

    for label_id in label_ids:

        count = np.sum(
            y_global == label_id
        )

        print(
            f"{label_id:3d} "
            f"{label_names.get(label_id, str(label_id)):25s}"
            f" {count}"
        )

    # --------------------------------------------------------
    # Landmark quality
    # --------------------------------------------------------

    landmark_quality(
        X_selected,
        y_global,
        label_names,
    )

    # --------------------------------------------------------
    # Encode labels exactly like trainer
    # --------------------------------------------------------

    encoder = ModelLabelEncoder()

    encoder.fit(
        sorted(
            np.unique(y_global)
        )
    )

    y_encoded = encoder.transform(
        y_global
    )

    # --------------------------------------------------------
    # Split exactly like trainer
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        X_selected,
        y_encoded,
    )

    print("\nSPLIT")
    print("-" * 70)

    print(
        f"Train:      {len(X_train)}"
    )

    print(
        f"Validation: {len(X_val)}"
    )

    print(
        f"Test:       {len(X_test)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if not model_path.exists():

        print(
            f"\nMODEL NOT FOUND:\n{model_path}"
        )

        return

    model, checkpoint = load_model(
        model_path
    )

    print("\nCHECKPOINT")
    print("-" * 70)

    print(
        f"Input size: "
        f"{checkpoint.get('input_size')}"
    )

    print(
        f"Classes: "
        f"{checkpoint.get('num_classes')}"
    )

    print(
        f"Best val accuracy: "
        f"{checkpoint.get('best_val_accuracy')}"
    )

    print(
        f"Dataset input size: "
        f"{X_selected.shape[-1]}"
    )

    if (
        checkpoint["input_size"]
        != X_selected.shape[-1]
    ):

        print(
            "\nWARNING:"
            " checkpoint input size does not "
            "match dataset input size."
        )

        return

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    accuracy = evaluate_model(
        model,
        X_test,
        y_test,
        label_ids,
        label_names,
    )

    print(
        "\nOVERALL TEST ACCURACY:"
        f" {accuracy*100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("FSL MODEL / DATASET DIAGNOSTIC")
    print("=" * 90)

    print(
        f"\nRepository:\n{REPO_ROOT}"
    )

    print(
        f"\nDataset:\n{DATASET_PATH}"
    )

    label_names = load_labels()

    X, y = load_dataset()

    print("\nDATASET")
    print("-" * 70)

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"y shape: {y.shape}"
    )

    print(
        f"Samples: {len(X)}"
    )

    print(
        f"Unique labels: "
        f"{len(np.unique(y))}"
    )

    for name, config in MODELS.items():

        diagnose_model(
            name,
            config,
            X,
            y,
            label_names,
        )

    print("\n")
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()