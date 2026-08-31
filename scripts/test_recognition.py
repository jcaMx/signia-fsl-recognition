"""
test_recognition.py
-------------------
Phase 1 FSL Live Recognition CLI

Unified replacement for test_greetings.py / test_survival.py.

Loads available categories from LabelRegistry, presents a numbered menu,
discovers the trained model for the selected category, and launches
WebcamRecognizer for real-time prediction.

Usage:
    python scripts/test_recognition.py

    python scripts/test_recognition.py          # interactive menu
    python scripts/test_recognition.py GREETING # skip category menu
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Imports
# ============================================================

import torch

from src.camera.webcam_recognizer import WebcamRecognizer
from src.data_collection.label_registry import LabelRegistry
from src.models.sign_lstm import SignLSTM
from src.preprocessing.greeting_features import (
    add_motion_features,
    normalize_sequence,
)

# ============================================================
# Constants
# ============================================================

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
DIVIDER = "=" * 44


# ============================================================
# UI helpers  (same style as demo_collection.py)
# ============================================================


def _print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f" {title}")
    print(DIVIDER)


def _numbered_menu(prompt: str, items: List[str]) -> Optional[int]:
    """
    Print a numbered menu and return the 0-based index of the chosen item.
    Returns None on EOF / Ctrl-C.
    """
    print(f"\n{prompt}\n")
    for i, item in enumerate(items, start=1):
        print(f"  {i}. {item}")
    print()

    while True:
        try:
            raw = input("Enter number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not raw:
            print("  Please enter a number.")
            continue

        if not raw.isdigit():
            print("  Invalid choice. Please enter a number.")
            continue

        choice = int(raw)
        if 1 <= choice <= len(items):
            return choice - 1  # 0-based

        print(f"  Out of range. Enter a number between 1 and {len(items)}.")


# ============================================================
# Model discovery
# ============================================================


def find_model_path(category: str) -> Optional[Path]:
    """
    Resolve the best-checkpoint path for *category*.

    Convention: artifacts/models/<category_lower>/<category_lower>_lstm_best.pt

    If the expected file is absent, lists all .pt files in that folder
    and asks the user to pick one.

    Returns
    -------
    Path or None
        Resolved path, or None if the user cancels.
    """
    cat_lower = category.lower()
    cat_dir = MODELS_DIR / cat_lower
    preferred = cat_dir / f"{cat_lower}_lstm_best.pt"

    if preferred.exists():
        return preferred

    # Fallback: list all .pt files in the folder
    if not cat_dir.exists():
        logger.error(
            "No model directory found at: %s", cat_dir
        )
        return None

    pt_files = sorted(cat_dir.glob("*.pt"))
    if not pt_files:
        logger.error(
            "No .pt checkpoints found in: %s", cat_dir
        )
        return None

    logger.warning(
        "Expected '%s' not found. Available checkpoints:", preferred.name
    )
    options = [p.name for p in pt_files]
    idx = _numbered_menu("Select checkpoint:", options)
    if idx is None:
        return None
    return pt_files[idx]


# ============================================================
# Model loading
# ============================================================


def load_checkpoint(model_path: Path) -> dict:
    """
    Load a checkpoint and return its metadata dict.

    The checkpoint must contain at minimum:
        model_state_dict, input_size, num_classes, original_label_ids
    """
    logger.info("Loading checkpoint: %s", model_path)
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unexpected checkpoint format in {model_path}. "
            "Expected a dict with model_state_dict."
        )

    required = {"model_state_dict", "input_size", "num_classes"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Checkpoint {model_path} is missing keys: {missing}"
        )

    return checkpoint


def build_model(checkpoint: dict) -> SignLSTM:
    """Reconstruct SignLSTM from checkpoint metadata and load weights."""
    input_size = int(checkpoint["input_size"])
    num_classes = int(checkpoint["num_classes"])

    model = SignLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_labels_dict(
    checkpoint: dict,
    registry: LabelRegistry,
) -> Dict[int, str]:
    """
    Build the {argmax_index: label_name} dict for WebcamRecognizer.

    Uses `original_label_ids` from the checkpoint to resolve each
    argmax output index (0-based) to the correct label name from
    labels.csv — regardless of which category or ID range the model covers.

    Parameters
    ----------
    checkpoint : dict
        Loaded .pt checkpoint.
    registry : LabelRegistry
        Loaded LabelRegistry for name resolution.

    Returns
    -------
    dict[int, str]
        e.g. {0: "UNDERSTAND", 1: "DONT UNDERSTAND", ...}
    """
    # Build a full id→label lookup from the registry's internal DataFrame
    id_to_label: Dict[int, str] = dict(
        zip(registry._df["id"].astype(int), registry._df["label"])
    )

    original_ids: List[int] = list(checkpoint["original_label_ids"])

    labels: Dict[int, str] = {}
    for argmax_idx, label_id in enumerate(original_ids):
        name = id_to_label.get(int(label_id))
        if name is None:
            logger.warning(
                "Label ID %d from checkpoint not found in labels.csv "
                "— using ID as placeholder.",
                label_id,
            )
            name = str(label_id)
        labels[argmax_idx] = name

    return labels


def resolve_preprocessing(input_size: int):
    """
    Return (normalize_fn, feature_fn) appropriate for *input_size*.

    126  → normalize only
    252  → normalize + add_motion_features
    other → warn and use normalize only
    """
    if input_size == 126:
        logger.info("Preprocessing: normalize_sequence only (input_size=126)")
        return normalize_sequence, None
    elif input_size == 252:
        logger.info(
            "Preprocessing: normalize_sequence + add_motion_features (input_size=252)"
        )
        return normalize_sequence, add_motion_features
    else:
        logger.warning(
            "Unrecognised input_size=%d — falling back to normalize only.",
            input_size,
        )
        return normalize_sequence, None


# ============================================================
# Category selection
# ============================================================


def select_category(registry: LabelRegistry, preselected: Optional[str] = None) -> Optional[str]:
    """
    Return the selected category (UPPERCASE), or None if the user exits.

    If *preselected* is given and matches a known category, it is returned
    directly without showing the menu.
    """
    categories = registry.get_categories()

    if preselected:
        key = preselected.strip().upper()
        if key in categories:
            return key
        logger.warning(
            "Category '%s' not found in labels.csv. Showing menu.", preselected
        )

    _print_header("FSL Live Recognition")
    idx = _numbered_menu("Select category:", categories)
    if idx is None:
        return None
    return categories[idx]


# ============================================================
# Main
# ============================================================


def run_recognition(registry: LabelRegistry, category: str) -> None:
    """
    Load model for *category* and run WebcamRecognizer.
    """
    # ---- Discover model ----
    model_path = find_model_path(category)
    if model_path is None:
        logger.error("No model found for category '%s'. Aborting.", category)
        return

    # ---- Load checkpoint ----
    try:
        checkpoint = load_checkpoint(model_path)
    except (ValueError, RuntimeError) as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        return

    input_size = int(checkpoint["input_size"])
    num_classes = int(checkpoint["num_classes"])
    best_val_acc = checkpoint.get("best_val_accuracy")

    # ---- Build model ----
    model = build_model(checkpoint)

    # ---- Build label dict ----
    labels = build_labels_dict(checkpoint, registry)

    # ---- Resolve preprocessing ----
    normalize_fn, feature_fn = resolve_preprocessing(input_size)

    # ---- Summary ----
    _print_header(f"Category: {category}")
    print(f"\n  Model     : {model_path.name}")
    print(f"  Classes   : {num_classes}")
    print(f"  Input size: {input_size}")
    if best_val_acc is not None:
        print(f"  Best val  : {best_val_acc:.1%}")
    print(f"\n  Labels ({num_classes}):")
    for idx, name in labels.items():
        print(f"    {idx:2d}. {name}")
    print("\n  Press Q in the webcam window to quit.\n")

    # ---- Launch recognizer ----
    try:
        recognizer = WebcamRecognizer(
            model=model,
            labels=labels,
            seq_length=30,
            input_size=input_size,
            normalize_fn=normalize_fn,
            feature_fn=feature_fn,
            camera_index=0,
            confidence_threshold=0.0,
            window_name=f"FSL Recognition — {category}",
            max_missed_frames=4,
        )
        recognizer.run()
    except RuntimeError as exc:
        logger.error("Webcam error: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error during recognition: %s", exc)
        raise


def main() -> None:
    # ---- Load registry ----
    try:
        registry = LabelRegistry()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot start: %s", exc)
        return

    logger.info(
        "Loaded %d categories from labels.csv.",
        len(registry.get_categories()),
    )

    # Optional: pass category as CLI argument to skip the menu
    preselected: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None

    while True:
        category = select_category(registry, preselected)
        preselected = None  # only use CLI arg on first iteration

        if category is None:
            logger.info("Exiting.")
            break

        run_recognition(registry, category)

        # ---- After webcam closes: loop prompt ----
        print()
        try:
            again = input("Test another category? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if again not in ("y", "yes"):
            logger.info("Exiting.")
            break


if __name__ == "__main__":
    main()

    # python scripts/test_recognition.py
    # python scripts/test_recognition.py GREETING
    # python scripts/test_recognition.py SURVIVAL
