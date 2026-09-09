import torch

from torch.utils.data import DataLoader

from .model import ModelB
from .data import Datasets


CHECKPOINT_PATH = "checkpoints/modelB/epoch_10.pth"
DATA_ROOT = "dataset/WLASL"

EPOCHS = 100
LEARNING_RATE = 1e-3


def main():

    device = torch.device("cpu")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

    dataset = Datasets(
        root=DATA_ROOT,
        split="train",
        shuffle=False,
        joint_idxs=config["joint_idx"],
        augment=False,
    )

    # Only first 8 samples
    indices = list(range(min(8, len(dataset))))

    samples = [dataset[i] for i in indices]

    batch = dataset.data_collator(samples)

    keypoints = batch["keypoints"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print("Samples:", len(indices))
    print("Keypoints:", keypoints.shape)
    print("Labels:", labels.tolist())

    # ---------------------------------------------------------
    # Create fresh model
    # ---------------------------------------------------------

    model = ModelB(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ---------------------------------------------------------
    # Overfit
    # ---------------------------------------------------------

    model.train()

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()

        loss, logits = model(
            keypoints=keypoints,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        accuracy = (
            (predictions == labels)
            .float()
            .mean()
            .item()
        )

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy * 100:.1f}%"
            )

    print()
    print("=" * 50)
    print("OVERFIT TEST COMPLETE")
    print("=" * 50)

    model.eval()

    with torch.no_grad():

        _, logits = model(
            keypoints=keypoints,
            attention_mask=attention_mask,
            labels=labels,
        )

        predictions = logits.argmax(dim=1)

        accuracy = (
            (predictions == labels)
            .float()
            .mean()
            .item()
        )

    print("Final accuracy:", f"{accuracy * 100:.1f}%")
    print("Expected:       close to 100%")
    print("=" * 50)


if __name__ == "__main__":
    main()