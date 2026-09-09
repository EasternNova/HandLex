import yaml
import torch

from .model import ModelB


CONFIG_PATH = "backend/modelB/configs/WLASL-100.yaml"


def main():
    # Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("Configuration loaded.")
    print(f"Number of classes: {config['num_labels']}")
    print(f"Model dimension: {config['d_model']}")

    # Build model
    model = ModelB(config)

    print("Model created successfully.")

    # Dummy input
    batch_size = 2
    sequence_length = 32
    num_joints = 75
    coordinates = 2

    keypoints = torch.randn(
        batch_size,
        sequence_length,
        num_joints,
        coordinates
    )

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long
    )

    labels = torch.randint(
        0,
        config["num_labels"],
        (batch_size,)
    )

    # Forward pass
    loss, logits = model(
        keypoints=keypoints,
        attention_mask=attention_mask,
        labels=labels
    )

    print("\nForward pass successful.")
    print(f"Input shape:  {keypoints.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss:         {loss.item():.4f}")


if __name__ == "__main__":
    main()