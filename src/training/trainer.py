# src/training/trainer.py

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.models.factory import create_model
from src.training.label_encoder import ModelLabelEncoder
from src.training import train_one_epoch, evaluate



# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainingConfig:

    # Dataset
    manifest_path: Optional[str] = None
    dataset_path: Optional[str] = None
    artifact_root: str = "."

    # Output
    output_dir: str = "artifacts/models"
    model_name: str = "model"

    # Manifest
    artifact_column: str = "artifact"
    category_column: str = "category"

    # Training
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10

    # Dataset splitting
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42

    # Model
    model_type: str = "sign_lstm"

    # Subset selection
    category: Optional[str] = None

    # Label filtering (list of integer IDs to keep from the bundle)
    label_ids: Optional[list] = None
    # Optional path to labels CSV for logging class names
    labels_csv: Optional[str] = None

    # Hardware
    device: Optional[str] = None


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        augment=False,
        transform: Optional[Callable] = None,
    ):
        self.X = X
        self.y = y
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        sequence = self.X[index]
        label = self.y[index]

        if self.transform is not None:
            sequence = self.transform(
                sequence,
                augment=self.augment,
            )

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32,
        )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return sequence, label


# ============================================================
# Artifact loading
# ============================================================

def load_artifact(path):
    """
    Load a single training artifact.

    Currently supports torch/pickle based artifacts.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact does not exist: {path}"
        )

    if path.suffix in {".pt", ".pth"}:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)

    raise ValueError(
        f"Unsupported artifact type: {path.suffix}"
    )


# ============================================================
# Manifest
# ============================================================

def load_manifest(config: TrainingConfig):

    if not config.manifest_path:
        raise ValueError(
            "manifest_path is required when dataset_path is not set."
        )

    manifest = pd.read_csv(
        config.manifest_path
    )

    required_columns = {
        config.artifact_column,
        config.category_column,
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {missing}"
        )

    manifest = manifest.dropna(
        subset=[
            config.artifact_column,
            config.category_column,
        ]
    ).reset_index(drop=True)

    return manifest


def load_labels_csv(path):
    labels = pd.read_csv(path)

    required_columns = {"id", "label", "category"}
    missing = required_columns - set(labels.columns)
    if missing:
        raise ValueError(
            f"labels_csv is missing columns: {missing}"
        )

    return labels


def resolve_label_ids(config: TrainingConfig):
    labels = None
    if config.labels_csv:
        labels = load_labels_csv(config.labels_csv)

    if config.label_ids is not None:
        label_ids = [int(value) for value in config.label_ids]
    elif config.category is not None:
        if labels is None:
            raise ValueError(
                "labels_csv is required when category is set."
            )

        category_mask = (
            labels["category"].astype(str).str.upper()
            == str(config.category).upper()
        )
        label_ids = labels.loc[category_mask, "id"].astype(int).tolist()
        if not label_ids:
            raise ValueError(
                f"No labels found for category: {config.category}"
            )
    else:
        if labels is not None:
            label_ids = labels["id"].astype(int).tolist()
        else:
            return None

    if labels is not None:
        available_ids = set(labels["id"].astype(int).tolist())
        missing = sorted(set(label_ids) - available_ids)
        if missing:
            raise ValueError(
                f"Requested label IDs do not exist in labels_csv: {missing}"
            )

    return sorted(dict.fromkeys(label_ids))


def filter_manifest_by_labels(manifest, label_ids):
    if label_ids is None:
        return manifest

    if "enabled" in manifest.columns:
        manifest = manifest[manifest["enabled"].astype(bool)]

    if "label_id" not in manifest.columns:
        raise ValueError("Manifest is missing required column: label_id")

    filtered = manifest[
        manifest["label_id"].astype(int).isin(label_ids)
    ].reset_index(drop=True)

    if filtered.empty:
        raise ValueError("No manifest rows remain after label filtering.")

    return filtered


def load_dataset_bundle(config: TrainingConfig):
    if not config.dataset_path:
        raise ValueError(
            "dataset_path is required when manifest_path is not set."
        )

    bundle = load_artifact(config.dataset_path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "dataset_path must point to a torch bundle or dict "
            "containing 'X' and 'y'."
        )

    if "X" not in bundle or "y" not in bundle:
        raise ValueError(
            "dataset bundle must contain 'X' and 'y' keys."
        )

    X, y = bundle["X"], bundle["y"]
    y = np.asarray(y)

    label_ids = resolve_label_ids(config)
    if label_ids is not None:
        mask = np.isin(y, label_ids)
        if not np.any(mask):
            raise ValueError(
                f"No samples found for selected labels: {label_ids}"
            )
        X = X[mask]
        y = y[mask]

    return X, y


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(
    manifest,
    config,
    label_encoder,
    label_ids=None,
    preprocess=None,
):
    """
    Loads artifacts referenced by the manifest and
    converts categories into integer labels.
    """

    X = []
    categories = []
    selected_ids = set(label_ids) if label_ids is not None else None

    artifact_root = Path(config.artifact_root)

    for _, row in manifest.iterrows():

        if selected_ids is not None:
            label_id = int(row["label_id"])
            if label_id not in selected_ids:
                continue

        artifact_path = (
            artifact_root /
            str(row[config.artifact_column])
        )

        sequence = load_artifact(
            artifact_path
        )

        if preprocess is not None:
            sequence = preprocess(sequence)

        X.append(sequence)

        categories.append(
            str(row[config.category_column])
        )

    if not X:
        raise ValueError("No samples found after applying label selection.")

    X = np.asarray(X)

    y = label_encoder.transform(
        categories
    )

    return X, y


# ============================================================
# Split dataset
# ============================================================

def split_dataset(
    X,
    y,
    config,
):
    """
    Creates stratified train/validation/test splits.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(
            config.test_size +
            config.validation_size
        ),
        random_state=config.random_state,
        stratify=y,
    )

    # Relative validation ratio within temp
    validation_ratio = (
        config.validation_size /
        (
            config.test_size +
            config.validation_size
        )
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=config.random_state,
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
# Model creation
# ============================================================
# ============================================================
# Training
# ============================================================

def train(
    config: TrainingConfig,
    dataset_factory=None,
    preprocess: Optional[Callable] = None,
):
    """
    Main reusable training function.

    Parameters
    ----------
    config:
        Training configuration.

    model_factory:
        Function that receives num_classes and returns
        an initialized model.

    dataset_factory:
        Optional function for creating custom datasets.

    preprocess:
        Optional preprocessing function.
    """

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if config.device is not None:
        device = torch.device(config.device)

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Label encoder / data loading
    # --------------------------------------------------------

    label_encoder = ModelLabelEncoder()
    label_ids = resolve_label_ids(config)

    if label_ids is not None:
        print(f"Selected labels: {label_ids}")

    if config.dataset_path:

        X, y = load_dataset_bundle(config)

        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 1:
            y = np.asarray(y).reshape(-1)

        if np.issubdtype(y.dtype, np.integer):
            selected_original_ids = sorted(np.unique(y).tolist())
            label_encoder.fit(selected_original_ids)
            y = label_encoder.transform(y)
        else:
            y = y.astype(str)
            label_encoder.fit(y)
            y = label_encoder.transform(y)

        print(
            f"Loaded dataset bundle: {config.dataset_path}"
        )
        print(
            f"Dataset samples: {len(X)}"
        )

    else:

        manifest = load_manifest(config)

        manifest = filter_manifest_by_labels(manifest, label_ids)

        print(
            f"Manifest samples: {len(manifest)}"
        )

        label_encoder.fit(
            manifest[
                config.category_column
            ].astype(str)
        )

        print(
            f"Classes: {label_encoder.classes}"
        )

        print(
            f"Number of classes: "
            f"{label_encoder.num_classes}"
        )

        X, y = prepare_dataset(
            manifest,
            config,
            label_encoder,
            label_ids=label_ids,
            preprocess=preprocess,
        )

    print(f"Classes: {label_encoder.classes}")
    print(f"Number of classes: {label_encoder.num_classes}")
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    if y.min() < 0 or y.max() >= label_encoder.num_classes:
        raise ValueError(
            f"Encoded labels must be in range [0, {label_encoder.num_classes - 1}], "
            f"but got range [{y.min()}, {y.max()}]."
        )

    if X.ndim != 3:
        raise ValueError(
            "Expected X to have shape "
            "(samples, sequence_length, features), "
            f"but got {X.shape}"
        )

    input_size = X.shape[-1]

    print(
        f"Detected input size: {input_size}"
    )

    print(
        f"Encoded labels: {sorted(np.unique(y).tolist())}"
    )

    if label_ids is not None and config.labels_csv is not None:
        labels_df = load_labels_csv(config.labels_csv)
        filtered = labels_df[
            labels_df["id"].astype(int).isin(label_ids)
        ][["id", "label", "category"]]
        print("\nClasses to be trained:")
        print(filtered.to_string(index=False))

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        X,
        y,
        config,
    )

    print(
        f"Train: {len(X_train)}"
    )

    print(
        f"Validation: {len(X_val)}"
    )

    print(
        f"Test: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if dataset_factory is None:

        dataset_factory = SequenceDataset

    train_dataset = dataset_factory(
        X_train,
        y_train,
        augment=True,
    )

    val_dataset = dataset_factory(
        X_val,
        y_val,
        augment=False,
    )

    test_dataset = dataset_factory(
        X_test,
        y_test,
        augment=False,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        model_type=config.model_type,
        num_classes=label_encoder.num_classes,
        input_size=input_size,
    ).to(device)

    if model.fc.out_features != label_encoder.num_classes:
        raise ValueError(
            "Model output classes do not equal label encoder classes."
        )

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir = Path(
        config.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        output_dir /
        f"{config.model_name}_best.pt"
    )

    label_encoder_path = (
        output_dir /
        f"{config.model_name}_labels.json"
    )

    # Save label mapping immediately
    label_encoder.save(
        label_encoder_path
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Sanity check before training

    print(
        f"Model classes: {label_encoder.num_classes}"
    )

    print(
        f"Encoded labels: "
        f"{sorted(np.unique(y).tolist())}"
    )
    
    for epoch in range(config.epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        print(
            f"Epoch {epoch + 1:02d}/"
            f"{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        input_size,

                    "num_classes":
                        label_encoder.num_classes,

                    "classes":
                        label_encoder.classes,

                    "best_val_accuracy":
                        best_val_acc,

                    "config":
                        vars(config),
                },
                best_model_path,
            )

            print(
                f"  Saved best model -> "
                f"{best_model_path}"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= config.patience
        ):

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"\nBest validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    return {
        "model": model,
        "label_encoder": label_encoder,
        "history": history,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "best_val_accuracy": best_val_acc,
        "model_path": best_model_path,
        "label_encoder_path": label_encoder_path,
        "test_loader": test_loader,
        
    }
