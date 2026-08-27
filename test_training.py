import torch
from src.training.trainer import train, TrainingConfig

if __name__ == "__main__":
    # Create a small training config
    config = TrainingConfig(
        dataset_path="data/test_dataset.pt",  # Use the small test dataset if it exists
        epochs=3,
        batch_size=4,
        patience=2,
        output_dir="artifacts/test_models",
        model_name="test_model",
        category="GREETING",
        labels_csv="csv/labels.csv"
    )
    
    # Run training
    print("Starting test training to verify pipeline changes...")
    results = train(config)
    print("Test training completed successfully!")

