from src.models.sign_lstm import SignLSTM


def create_model(model_type, num_classes, input_size):
    model_type = str(model_type).lower()

    if model_type == "sign_lstm":
        return SignLSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        )

    raise ValueError(
        f"Unsupported model_type: {model_type}"
    )
