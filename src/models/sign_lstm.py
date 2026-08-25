try:
    import torch.nn as nn
except ImportError:  # pragma: no cover - lets the module import without torch installed
    nn = None


_BaseModule = nn.Module if nn is not None else object


class SignLSTM(_BaseModule):
    def __init__(
        self,
        input_size=252,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
    ):
        if nn is None:
            raise ImportError("PyTorch is required to instantiate SignLSTM.")

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)
