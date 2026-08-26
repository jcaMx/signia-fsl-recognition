import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(r"C:\Projects\signia-fsl-recognition")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.sign_lstm import SignLSTM


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    REPO_ROOT
    / "artifacts"
    / "models"
    / "greeting"
    / "greetings_lstm_best.pt"
)

DATASET_PATH = (
    REPO_ROOT
    / "notebooks"
    / "02_dynamic"
    / "fsl_dataset_clean.pt"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("LSTM CHECKPOINT VERIFICATION")
print("=" * 70)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)

print("\nCheckpoint keys:")
for key in checkpoint:
    print(" ", key)


input_size = checkpoint["input_size"]
num_classes = checkpoint["num_classes"]
classes = checkpoint["classes"]
state_dict = checkpoint["model_state_dict"]

print("\nCheckpoint metadata:")
print("Input size:", input_size)
print("Classes:", classes)
print("Num classes:", num_classes)
print("Best validation accuracy:",
      checkpoint["best_val_accuracy"])


# ============================================================
# LOAD DATASET
# ============================================================

bundle = torch.load(
    DATASET_PATH,
    map_location="cpu",
    weights_only=False,
)

X = bundle["X"]
y = bundle["y"]

X = torch.as_tensor(X).float()
y = torch.as_tensor(y).long()

print("\nDataset:")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# VERIFY SHAPES
# ============================================================

assert X.ndim == 3

sequence_length = X.shape[1]
dataset_input_size = X.shape[2]

print("\nShape verification:")
print("Sequence length:", sequence_length)
print("Dataset input size:", dataset_input_size)
print("Checkpoint input size:", input_size)

assert dataset_input_size == input_size, (
    f"Input mismatch: dataset={dataset_input_size}, "
    f"checkpoint={input_size}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = SignLSTM(
    input_size=input_size,
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
)

model.load_state_dict(state_dict)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# VERIFY OUTPUT SHAPE
# ============================================================

with torch.no_grad():

    sample = X[:4]

    output = model(sample)

print("\nForward-pass verification:")
print("Input:", sample.shape)
print("Output:", output.shape)

assert output.shape == (
    4,
    num_classes,
), (
    f"Unexpected output shape: {output.shape}"
)

print("Forward pass: OK")


# ============================================================
# FULL DATASET EVALUATION
# ============================================================

print("\nRunning predictions...")

predictions = []

with torch.no_grad():

    for start in range(0, len(X), 32):

        batch = X[start:start + 32]

        logits = model(batch)

        predicted = torch.argmax(
            logits,
            dim=1,
        )

        predictions.extend(
            predicted.cpu().numpy()
        )

predictions = np.asarray(predictions)
targets = y.cpu().numpy()


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    predictions == targets
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nDataset accuracy: {accuracy:.4f}"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\nPer-class accuracy:")

for class_id in range(num_classes):

    mask = targets == class_id

    total = mask.sum()

    if total == 0:
        print(
            f"{class_id:>3}: NO SAMPLES"
        )
        continue

    correct = (
        predictions[mask] == class_id
    ).sum()

    class_accuracy = correct / total

    print(
        f"{class_id:>3}: "
        f"{class_accuracy:7.2%} "
        f"({correct}/{total})"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report:")

print(
    classification_report(
        targets,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion matrix:")

cm = confusion_matrix(
    targets,
    predictions,
    labels=list(range(num_classes)),
)

print(cm)


# ============================================================
# WORST CLASSES
# ============================================================

print("\nWorst classes:")

results = []

for class_id in range(num_classes):

    mask = targets == class_id

    total = mask.sum()

    if total == 0:
        continue

    correct = (
        predictions[mask] == class_id
    ).sum()

    acc = correct / total

    results.append(
        (acc, class_id, correct, total)
    )

results.sort()

for acc, class_id, correct, total in results:

    print(
        f"{class_id:>3}: "
        f"{acc:7.2%} "
        f"({correct}/{total})"
    )