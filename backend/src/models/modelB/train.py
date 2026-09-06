import os
import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import ModelB
from .dataset import Datasets


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = "backend/src/models/modelB/configs/WLASL-100.yaml"
DATA_ROOT = "dataset/WLASL"


BATCH_SIZE = 8
EPOCHS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "backend/checkpoints/modelB"


def main():

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("MODEL B TRAINING")
    print("=" * 60)

    print(f"Using device: {device}")


    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("Configuration loaded.")
    print(f"Classes: {config['num_labels']}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")


    # ========================================================
    # DATASET
    # ========================================================

    joint_idxs = config["joint_idx"]

    train_dataset = Datasets(
        root=DATA_ROOT,
        split="train",
        shuffle=True,
        joint_idxs=joint_idxs,
        augment=True,
    )

    val_dataset = Datasets(
        root=DATA_ROOT,
        split="val",
        shuffle=False,
        joint_idxs=joint_idxs,
        augment=False,
    )

    print()
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")


    # ========================================================
    # CLASS-BALANCED SAMPLING
    # ========================================================
    #
    # Important:
    # The sampler MUST be created BEFORE the DataLoader.
    #
    # We calculate how frequently each class appears and give
    # rare classes a higher sampling probability.
    #

    print()
    print("Creating class-balanced sampler...")

    labels = []

    for i in range(len(train_dataset)):

        _, label = train_dataset[i]

        labels.append(label.item())

    labels_tensor = torch.tensor(labels)

    class_counts = torch.bincount(
        labels_tensor,
        minlength=config["num_labels"]
    )

    class_weights = torch.zeros(
        config["num_labels"],
        dtype=torch.float
    )

    for class_id in range(config["num_labels"]):

        if class_counts[class_id] > 0:

            class_weights[class_id] = (
                1.0 / class_counts[class_id].float()
            )

    sample_weights = torch.tensor(
        [
            class_weights[label].item()
            for label in labels
        ],
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    print("Class-balanced sampler created.")


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=train_dataset.data_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.data_collator,
    )

    print()
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")


    # ========================================================
    # MODEL
    # ========================================================

    model = ModelB(config).to(device)

    print()
    print("Model created.")


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )


    # ========================================================
    # CHECKPOINT DIRECTORY
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )


    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_val_accuracy = 0.0
    best_val_loss = float("inf")


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print()
        print("=" * 60)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 60)


        # ----------------------------------------------------
        # TRAINING MODE
        # ----------------------------------------------------

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0


        # ----------------------------------------------------
        # TRAINING BATCHES
        # ----------------------------------------------------

        for batch_idx, batch in enumerate(train_loader):

            keypoints = batch["keypoints"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)


            # ------------------------------------------------
            # ZERO GRADIENTS
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # FORWARD PASS
            # ------------------------------------------------

            loss, logits = model(
                keypoints=keypoints,
                attention_mask=attention_mask,
                labels=labels,
            )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # GRADIENT CLIPPING
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )


            # ------------------------------------------------
            # OPTIMIZER UPDATE
            # ------------------------------------------------

            optimizer.step()


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            batch_correct = (
                predictions == labels
            ).sum().item()

            correct += batch_correct

            total += labels.size(0)


            # ------------------------------------------------
            # BATCH LOG
            # ------------------------------------------------

            if batch_idx % 10 == 0:

                batch_accuracy = (
                    batch_correct / labels.size(0)
                )

                print(
                    f"Batch {batch_idx + 1:3d}/"
                    f"{len(train_loader):3d} "
                    f"| Loss: {loss.item():.4f} "
                    f"| Accuracy: {batch_accuracy:.4f}"
                )


        # ====================================================
        # TRAINING METRICS
        # ====================================================

        train_loss = (
            total_loss / len(train_loader)
        )

        train_accuracy = (
            correct / total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0
        val_loss_total = 0.0


        with torch.no_grad():

            for batch in val_loader:

                keypoints = batch["keypoints"].to(device)

                attention_mask = batch["attention_mask"].to(device)

                labels = batch["labels"].to(device)


                # ------------------------------------------------
                # VALIDATION FORWARD PASS
                # ------------------------------------------------

                loss, logits = model(
                    keypoints=keypoints,
                    attention_mask=attention_mask,
                    labels=labels,
                )


                val_loss_total += loss.item()


                # ------------------------------------------------
                # VALIDATION PREDICTIONS
                # ------------------------------------------------

                predictions = logits.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        val_loss = (
            val_loss_total / len(val_loader)
        )

        val_accuracy = (
            val_correct / val_total
        )


        # ====================================================
        # LEARNING RATE UPDATE
        # ====================================================

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]


        # ====================================================
        # EPOCH RESULTS
        # ====================================================

        print()
        print("-" * 60)
        print(f"Epoch {epoch + 1} Results")
        print("-" * 60)

        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Train Accuracy:   {train_accuracy:.4f}")

        print(f"Validation Loss:  {val_loss:.4f}")
        print(f"Validation Acc:   {val_accuracy:.4f}")

        print(f"Learning Rate:    {current_lr:.8f}")


        # ====================================================
        # REGULAR CHECKPOINT
        # ====================================================

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch + 1}.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "train_accuracy":
                    train_accuracy,

                "val_accuracy":
                    val_accuracy,

                "config":
                    config,
            },
            checkpoint_path,
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_val_loss = val_loss


            best_checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )


            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "train_loss":
                        train_loss,

                    "val_loss":
                        val_loss,

                    "train_accuracy":
                        train_accuracy,

                    "val_accuracy":
                        val_accuracy,

                    "config":
                        config,
                },
                best_checkpoint_path,
            )


            print()
            print("★ NEW BEST MODEL")

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f}"
            )

            print(
                f"Best checkpoint: "
                f"{best_checkpoint_path}"
            )


        print()
        print(
            f"Checkpoint saved: "
            f"{checkpoint_path}"
        )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best Validation Loss:     "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best Model:               "
        f"{os.path.join(CHECKPOINT_DIR, 'best_model.pth')}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()