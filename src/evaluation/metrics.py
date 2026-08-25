from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import classification_report
from pathlib import Path
def get_test_predictions(
    model,
    test_loader,
    device="cpu",
):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                y_batch.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    return y_true, y_pred


def get_classification_report(
    y_true,
    y_pred,
    class_names,
):
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        output_dict=True,
        zero_division=0,
    )

def plot_confusion_matrix(
    model,
    test_loader,
    class_names,
    device="cpu",
    save_path=None,
):
    """
    Generate and optionally save a confusion matrix.

    Args:
        model: Trained PyTorch classification model.
        test_loader: DataLoader containing test data.
        class_names: Names corresponding to model output indices.
        device: "cpu" or "cuda".
        save_path: Optional path to save the figure.

    Returns:
        cm: NumPy confusion matrix.
    """

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_targets.extend(
                y_batch.cpu().numpy()
            )

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d",
    )

    ax.set_title("Survival Sign Recognition - Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return cm