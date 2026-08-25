# src/training/label_encoder.py

from pathlib import Path
import json

from sklearn.preprocessing import LabelEncoder


class ModelLabelEncoder:
    """
    Reusable label encoder for classification models.

    Converts string category labels into integer class IDs
    and stores the mapping so it can be reused during inference.
    """

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, labels):
        self.encoder.fit(labels)
        return self

    def transform(self, labels):
        return self.encoder.transform(labels)

    def fit_transform(self, labels):
        return self.encoder.fit_transform(labels)

    def inverse_transform(self, labels):
        return self.encoder.inverse_transform(labels)

    @property
    def classes(self):
        return self.encoder.classes_.tolist()

    @property
    def num_classes(self):
        return len(self.encoder.classes_)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "classes": self.classes
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encoder = cls()
        encoder.encoder.fit(data["classes"])

        return encoder