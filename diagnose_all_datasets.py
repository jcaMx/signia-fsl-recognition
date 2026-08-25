"""
Diagnose all FSL dynamic-sign datasets and optionally trained models.

Checks:
- Dataset structure
- Class distribution
- Train/val/test distribution
- Missing classes
- Sequence lengths
- Feature dimensions
- Zero/padded frames
- Landmark detection rate
- Left/right/both-hand detection
- Per-class statistics
- Optional model evaluation
- Confusion matrix
- Per-class accuracy

Usage:
    python diagnose_all_models.py

Adjust MODEL_CONFIGS below to match your project.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from src.preprocessing.greeting_features import (
    normalize_sequence,
    add_motion_features,
)

# ============================================================
# PATH
# ============================================================

REPO_ROOT = Path(
    r"C:\Projects\signia-fsl-recognition"
).resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# CONFIG
# ============================================================

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


# ------------------------------------------------------------
# Models to diagnose
#
# Change label_ids/output/model paths as needed.
# ------------------------------------------------------------

MODEL_CONFIGS = {

    "greetings": {
        "label_ids": list(range(0,10)),
        "model_path": (
            REPO_ROOT
            / "artifacts"
            / "models"
            / "greeting"
            / "greetings_lstm_best.pt"
        ),
        "model_type": "sign_lstm",
        "input_size": 252,
        "num_classes": 10,
    },

    "survival": {
        "label_ids": list(range(10,20)),
        "model_path": (
            REPO_ROOT
            / "artifacts"
            / "models"
            / "survival"
            / "survival_lstm_best.pt"
        ),
        "model_type": "sign_lstm",
        "input_size": 252,
        "num_classes": 10,
    },

    "numbers": {
        "label_ids": list(range(20,30)),
        "model_path": (
            REPO_ROOT
            / "artifacts"
            / "models"
            / "number"
            / "numbers_lstm_best.pt"
        ),
        "model_type": "sign_lstm",
        "input_size": 252,
        "num_classes": 10,
    },

    "calendar": {
        "label_ids": list(range(30,42)),
        "model_path": (
            REPO_ROOT
            / "artifacts"
            / "models"
            / "calendar"
            / "calendar_lstm_best.pt"
        ),
        "model_type": "sign_lstm",
        "input_size": 252,
        "num_classes": 12,
    },
}


# ============================================================
# HELPERS
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title):

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def get_label_names(labels_csv):

    """
    Read labels.csv if possible.

    Expected common formats:
        id,label
        0,HELLO

    Adapt this function if your CSV uses different columns.
    """

    import csv

    labels = {}

    if not labels_csv.exists():

        print(
            f"WARNING: labels CSV not found: {labels_csv}"
        )

        return labels

    with open(
        labels_csv,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            # Try common column names.

            id_key = None
            label_key = None

            for key in row:

                lower = key.lower().strip()

                if lower in {
                    "id",
                    "label_id",
                    "class_id"
                }:
                    id_key = key

                if lower in {
                    "label",
                    "name",
                    "class",
                    "class_name"
                }:
                    label_key = key

            if id_key is None or label_key is None:
                continue

            try:

                label_id = int(row[id_key])

                labels[label_id] = (
                    row[label_key]
                    .strip()
                )

            except (
                ValueError,
                TypeError
            ):
                continue

    return labels


def label_name(label_id, label_names):

    return label_names.get(
        label_id,
        f"UNKNOWN_{label_id}"
    )


def is_frame_detected(frame):

    """
    A frame is considered detected if it contains
    at least one non-zero landmark value.
    """

    return np.any(frame != 0)


def get_hand_detection(frame):

    """
    Assumes:

        first 63  = left hand
        next 63   = right hand

    Returns:
        left_detected
        right_detected
    """

    if frame.shape[-1] < 126:

        return False, False

    left = frame[:63]

    right = frame[63:126]

    left_detected = np.any(left != 0)

    right_detected = np.any(right != 0)

    return (
        bool(left_detected),
        bool(right_detected)
    )


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(path):

    print_header("LOADING DATASET")

    print("Path:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataset = torch.load(
        path,
        map_location="cpu"
    )

    print(
        "\nDataset type:",
        type(dataset)
    )

    if isinstance(dataset, dict):

        print(
            "\nDataset keys:"
        )

        for key in dataset.keys():

            value = dataset[key]

            if hasattr(value, "shape"):

                print(
                    f"  {key}: "
                    f"shape={value.shape}"
                )

            else:

                print(
                    f"  {key}: "
                    f"type={type(value)}"
                )

    return dataset


# ============================================================
# DATASET STRUCTURE
# ============================================================

def find_dataset_arrays(dataset):

    """
    Try to identify features and labels from common
    dictionary formats.

    Modify this if your dataset uses different keys.
    """

    if not isinstance(dataset, dict):

        raise ValueError(
            "Expected dataset to be a dictionary."
        )

    feature_keys = [
        "X",
        "x",
        "features",
        "sequences",
        "data",
        "inputs",
    ]

    label_keys = [
        "y",
        "labels",
        "targets",
        "target",
    ]

    X = None
    y = None

    for key in feature_keys:

        if key in dataset:

            X = dataset[key]

            break

    for key in label_keys:

        if key in dataset:

            y = dataset[key]

            break

    if X is None:

        raise KeyError(
            "Could not find feature array in dataset."
        )

    if y is None:

        raise KeyError(
            "Could not find label array in dataset."
        )

    if isinstance(X, torch.Tensor):

        X = X.cpu().numpy()

    if isinstance(y, torch.Tensor):

        y = y.cpu().numpy()

    return X, y


# ============================================================
# BASIC DATASET STATS
# ============================================================

def diagnose_basic(X, y, label_names):

    print_header("BASIC DATASET INFORMATION")

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Label shape:",
        y.shape
    )

    print(
        "Number of samples:",
        len(X)
    )

    if X.ndim >= 2:

        print(
            "Sequence length:",
            X.shape[1]
        )

    if X.ndim >= 3:

        print(
            "Features per frame:",
            X.shape[2]
        )

    print(
        "Number of unique labels:",
        len(np.unique(y))
    )

    print_section("CLASS DISTRIBUTION")

    counts = Counter(
        y.tolist()
    )

    for label_id in sorted(counts):

        print(
            f"{label_id:4d}  "
            f"{label_name(label_id, label_names):25s} "
            f"{counts[label_id]:5d}"
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def diagnose_expected_classes(
    y,
    label_ids,
    label_names
):

    print_section(
        "EXPECTED CLASS COVERAGE"
    )

    counts = Counter(
        y.tolist()
    )

    for label_id in label_ids:

        count = counts.get(
            label_id,
            0
        )

        status = "OK"

        if count == 0:

            status = "MISSING"

        elif count < 5:

            status = "VERY LOW"

        elif count < 10:

            status = "LOW"

        print(
            f"{label_id:4d}  "
            f"{label_name(label_id, label_names):25s} "
            f"{count:5d}  "
            f"{status}"
        )


# ============================================================
# SEQUENCE QUALITY
# ============================================================

def diagnose_sequences(
    X,
    y,
    label_names
):

    print_section(
        "SEQUENCE / LANDMARK QUALITY"
    )

    if X.ndim != 3:

        print(
            "WARNING: Expected "
            "(samples, frames, features)."
        )

        return

    per_class = defaultdict(
        lambda: {
            "samples": 0,
            "total_frames": 0,
            "detected_frames": 0,
            "left": 0,
            "right": 0,
            "both": 0,
            "none": 0,
        }
    )

    for sequence, label in zip(
        X,
        y
    ):

        stats = per_class[
            int(label)
        ]

        stats["samples"] += 1

        for frame in sequence:

            stats["total_frames"] += 1

            left, right = get_hand_detection(
                frame
            )

            if left or right:

                stats[
                    "detected_frames"
                ] += 1

            if left:
                stats["left"] += 1

            if right:
                stats["right"] += 1

            if left and right:
                stats["both"] += 1

            if not left and not right:
                stats["none"] += 1

    print(
        f"{'Class':25s} "
        f"{'Samples':>7s} "
        f"{'Detect':>8s} "
        f"{'Left':>8s} "
        f"{'Right':>8s} "
        f"{'Both':>8s} "
        f"{'None':>8s}"
    )

    print("-" * 90)

    for label_id in sorted(per_class):

        stats = per_class[label_id]

        total = stats["total_frames"]

        detection_rate = (
            stats["detected_frames"]
            / total
            * 100
            if total > 0
            else 0
        )

        left_rate = (
            stats["left"]
            / total
            * 100
            if total > 0
            else 0
        )

        right_rate = (
            stats["right"]
            / total
            * 100
            if total > 0
            else 0
        )

        both_rate = (
            stats["both"]
            / total
            * 100
            if total > 0
            else 0
        )

        none_rate = (
            stats["none"]
            / total
            * 100
            if total > 0
            else 0
        )

        print(
            f"{label_name(label_id, label_names):25s} "
            f"{stats['samples']:7d} "
            f"{detection_rate:7.1f}% "
            f"{left_rate:7.1f}% "
            f"{right_rate:7.1f}% "
            f"{both_rate:7.1f}% "
            f"{none_rate:7.1f}%"
        )


# ============================================================
# ZERO FRAME ANALYSIS
# ============================================================

def diagnose_zero_frames(
    X,
    y,
    label_names
):

    print_section(
        "ZERO / EMPTY FRAME ANALYSIS"
    )

    for label_id in sorted(
        np.unique(y)
    ):

        class_sequences = X[
            y == label_id
        ]

        total_frames = (
            class_sequences.shape[0]
            * class_sequences.shape[1]
        )

        zero_frames = 0

        for sequence in class_sequences:

            for frame in sequence:

                if not np.any(frame != 0):

                    zero_frames += 1

        rate = (
            zero_frames
            / total_frames
            * 100
            if total_frames > 0
            else 0
        )

        status = ""

        if rate > 20:
            status = "  ⚠ HIGH"

        elif rate > 5:
            status = "  ⚠"

        print(
            f"{label_name(label_id, label_names):25s} "
            f"{zero_frames:6d}/"
            f"{total_frames:<6d} "
            f"({rate:6.2f}%){status}"
        )


# ============================================================
# SPLIT DIAGNOSTICS
# ============================================================

def diagnose_splits(
    dataset,
    label_names
):

    print_section(
        "TRAIN / VALIDATION / TEST SPLITS"
    )

    possible_splits = [
        ("train", "train_labels"),
        ("val", "val_labels"),
        ("validation", "validation_labels"),
        ("test", "test_labels"),
    ]

    found = False

    for split_name, label_key in possible_splits:

        if label_key not in dataset:

            continue

        found = True

        labels = dataset[label_key]

        if isinstance(
            labels,
            torch.Tensor
        ):

            labels = labels.cpu().numpy()

        counts = Counter(
            labels.tolist()
        )

        print(
            f"\n{split_name.upper()}: "
            f"{len(labels)} samples"
        )

        for label_id in sorted(counts):

            print(
                f"  {label_name(label_id, label_names):25s}"
                f"{counts[label_id]:5d}"
            )

        # Detect missing test classes.

        if split_name == "test":

            missing = [
                label_id
                for label_id in label_names
                if counts.get(
                    label_id,
                    0
                ) == 0
            ]

            if missing:

                print(
                    "\n  ⚠ TEST SET MISSING CLASSES:"
                )

                for label_id in missing:

                    print(
                        f"    - "
                        f"{label_name(label_id, label_names)}"
                    )

    if not found:

        print(
            "No explicit train/val/test labels "
            "found in dataset."
        )


# ============================================================
# WARNINGS
# ============================================================

def generate_warnings(
    X,
    y,
    label_ids,
    label_names
):

    print_header(
        "AUTOMATIC DIAGNOSTIC WARNINGS"
    )

    warnings = []

    counts = Counter(
        y.tolist()
    )

    for label_id in label_ids:

        count = counts.get(
            label_id,
            0
        )

        name = label_name(
            label_id,
            label_names
        )

        if count == 0:

            warnings.append(
                f"{name}: no samples found"
            )

        elif count < 10:

            warnings.append(
                f"{name}: only {count} samples"
            )

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    if X.ndim == 3 and X.shape[-1] >= 126:

        for label_id in sorted(
            np.unique(y)
        ):

            sequences = X[
                y == label_id
            ]

            total = (
                sequences.shape[0]
                * sequences.shape[1]
            )

            detected = 0

            for sequence in sequences:

                for frame in sequence:

                    if np.any(
                        frame != 0
                    ):

                        detected += 1

            rate = (
                detected / total * 100
                if total
                else 0
            )

            if rate < 70:

                warnings.append(
                    f"{label_name(label_id, label_names)}: "
                    f"low landmark detection "
                    f"({rate:.1f}%)"
                )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if not warnings:

        print(
            "No major automatic warnings."
        )

    else:

        for warning in warnings:

            print(
                f"⚠ {warning}"
            )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    label_ids,
    label_names
):

    print_header(
        "MODEL EVALUATION"
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long
    )

    correct = Counter()
    total = Counter()

    predictions = []

    with torch.no_grad():

        batch_size = 32

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ]

            targets = y_tensor[
                start:start + batch_size
            ]

            outputs = model(
                batch
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            for true, pred in zip(
                targets.tolist(),
                predicted.tolist()
            ):

                total[true] += 1

                if true == pred:

                    correct[true] += 1

    print(
        f"{'Class':25s} "
        f"{'Correct':>8s} "
        f"{'Total':>8s} "
        f"{'Accuracy':>10s}"
    )

    print("-" * 60)

    for label_id in label_ids:

        c = correct[label_id]

        t = total[label_id]

        if t == 0:

            accuracy = float("nan")

        else:

            accuracy = (
                c / t * 100
            )

        print(
            f"{label_name(label_id, label_names):25s} "
            f"{c:8d} "
            f"{t:8d} "
            f"{accuracy:9.2f}%"
        )

    return np.array(
        predictions
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    y_true,
    y_pred,
    label_ids,
    label_names
):

    print_header(
        "CONFUSION MATRIX"
    )

    label_ids = list(
        label_ids
    )

    matrix = np.zeros(
        (
            len(label_ids),
            len(label_ids)
        ),
        dtype=int
    )

    id_to_index = {
        label_id: i
        for i, label_id
        in enumerate(label_ids)
    }

    for true, pred in zip(
        y_true,
        y_pred
    ):

        if (
            true in id_to_index
            and pred in id_to_index
        ):

            matrix[
                id_to_index[true],
                id_to_index[pred]
            ] += 1

    names = [
        label_name(
            label_id,
            label_names
        )
        for label_id in label_ids
    ]

    short_names = [
        name[:12]
        for name in names
    ]

    print(
        "\nActual \\ Predicted"
    )

    print(
        f"{'':18s}",
        end=""
    )

    for name in short_names:

        print(
            f"{name:>13s}",
            end=""
        )

    print()

    for i, name in enumerate(
        short_names
    ):

        print(
            f"{name:<18s}",
            end=""
        )

        for j in range(
            len(label_ids)
        ):

            print(
                f"{matrix[i, j]:13d}",
                end=""
            )

        print()


# ============================================================
# MODEL LOADING
# ============================================================
def load_model(config):

    model_path = config.get("model_path")

    if model_path is None:
        return None, None

    if not model_path.exists():

        print(
            f"\nModel not found:\n{model_path}"
        )

        return None, None

    print(
        f"\nLoading model:\n{model_path}"
    )

    try:

        from src.models.sign_lstm import SignLSTM

        checkpoint = torch.load(
            model_path,
            map_location="cpu"
        )

        print("\nCheckpoint information:")

        if isinstance(checkpoint, dict):

            print(
                "Keys:",
                list(checkpoint.keys())
            )

            print(
                "input_size:",
                checkpoint.get("input_size")
            )

            print(
                "num_classes:",
                checkpoint.get("num_classes")
            )

            print(
                "classes:",
                checkpoint.get("classes")
            )

            print(
                "best_val_accuracy:",
                checkpoint.get(
                    "best_val_accuracy"
                )
            )

        # ----------------------------------------------------
        # Get architecture from checkpoint
        # ----------------------------------------------------

        input_size = checkpoint.get(
            "input_size",
            config["input_size"]
        )

        num_classes = checkpoint.get(
            "num_classes",
            config["num_classes"]
        )

        model_config = checkpoint.get(
            "config",
            {}
        )

        hidden_size = model_config.get(
            "hidden_size",
            128
        )

        num_layers = model_config.get(
            "num_layers",
            2
        )

        model = SignLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # checkpoint contains metadata
        # ----------------------------------------------------

        state_dict = checkpoint[
            "model_state_dict"
        ]

        model.load_state_dict(
            state_dict
        )

        model.eval()

        print(
            "\nModel loaded successfully."
        )

        print(
            f"Architecture: "
            f"input={input_size}, "
            f"hidden={hidden_size}, "
            f"layers={num_layers}, "
            f"classes={num_classes}"
        )

        return model, checkpoint

    except Exception as e:

        print(
            "\nWARNING: Could not load model."
        )

        print(
            type(e).__name__,
            str(e)
        )

        import traceback

        traceback.print_exc()

        return None, None
    
# ============================================================
# RUN ONE MODEL
# ============================================================

def diagnose_model(
    model_name,
    config,
    dataset,
    label_names
):

    print_header(
        f"DIAGNOSING: {model_name.upper()}"
    )

    X, y = find_dataset_arrays(
        dataset
    )

    label_ids = config[
        "label_ids"
    ]

    diagnose_basic(
        X,
        y,
        label_names
    )

    diagnose_expected_classes(
        y,
        label_ids,
        label_names
    )

    diagnose_sequences(
        X,
        y,
        label_names
    )

    diagnose_zero_frames(
        X,
        y,
        label_names
    )

    diagnose_splits(
        dataset,
        label_names
    )

    generate_warnings(
        X,
        y,
        label_ids,
        label_names
    )

    # --------------------------------------------------------
    # Optional model evaluation
    # --------------------------------------------------------

    model = load_model(
        config
    )

    if model is not None:

        predictions = evaluate_model(
            model,
            X,
            y,
            label_ids,
            label_names
        )

        print_confusion_matrix(
            y,
            predictions,
            label_ids,
            label_names
        )

def preprocess_for_model(X):

    normalized = normalize_sequence(
        X
    )

    features = add_motion_features(
        normalized
    )

    return features
# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "FSL MODEL / DATASET DIAGNOSTIC"
    )

    print(
        "Repository:",
        REPO_ROOT
    )

    print(
        "Dataset:",
        DATASET_PATH
    )

    label_names = get_label_names(
        LABELS_CSV
    )

    dataset = load_dataset(
        DATASET_PATH
    )

    for model_name, config in (
        MODEL_CONFIGS.items()
    ):

        try:

            diagnose_model(
                model_name,
                config,
                dataset,
                label_names
            )

        except Exception as e:

            print(
                f"\nERROR diagnosing "
                f"{model_name}:"
            )

            print(
                type(e).__name__
            )

            print(
                str(e)
            )

            import traceback

            traceback.print_exc()

    print_header(
        "DIAGNOSTICS COMPLETE"
    )


if __name__ == "__main__":

    main()