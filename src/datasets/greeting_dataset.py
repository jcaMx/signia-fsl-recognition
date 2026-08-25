try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - lets the module import without torch installed
    Dataset = object

from src.preprocessing.greeting_features import augment_sequence


class GreetingDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype("float32")
        self.y = y.astype("int64")
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch

        sequence = self.X[idx].copy()
        label = self.y[idx]

        if self.augment:
            sequence = augment_sequence(sequence)

        return (
            torch.tensor(sequence, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
