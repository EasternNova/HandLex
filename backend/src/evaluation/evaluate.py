import os
import json
import csv

import torch
from torch.utils.data import DataLoader

from backend.src.models.modelB.model import ModelB
from backend.src.models.modelB.dataset import Datasets


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "backend/checkpoints/modelB/best_model.pth"
DATA_ROOT = "dataset/WLASL"

BATCH_SIZE = 8

RESULTS_DIR = "backend/checkpoints/modelB/evaluation"
RESULTS_JSON = os.path.join(RESULTS_DIR, "test_results.json")
PER_CLASS_CSV = os.path.join(RESULTS_DIR, "per_class_results.csv")
CONFUSION_CSV = os.path.join(RESULTS_DIR, "confusion_matrix.csv")


# ============================================================
# HELPERS
# ============================================================

def top_k_accuracy(logits, labels, k):
    """
    Calculate top-k accuracy.
    """
    k = min(k, logits.size(1))

    _, predictions = torch.topk(logits, k=k, dim=1)

    correct = predictions.eq(labels.unsqueeze(1))

    return correct.any(dim=1).sum().item()


def get_class_name(id2label, class_id):
    """
    Safely convert class ID to class name.
    """
    return id2label.get(
        str(class_id),
        id2label.get(class_id, str(class_id))
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MODEL B - WLASL-100 EVALUATION")
    print("=" * 70)
    print(f"Using device: {device}")
    print()


    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    print("Loading checkpoint:")
    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    config = checkpoint["config"]

    print()
    print(f"Checkpoint epoch:       {checkpoint['epoch']}")
    print(f"Checkpoint train acc:   {checkpoint['train_accuracy']:.4f}")
    print(f"Checkpoint val acc:     {checkpoint['val_accuracy']:.4f}")
    print(f"Checkpoint val loss:    {checkpoint['val_loss']:.4f}")
    print(f"Number of classes:      {config['num_labels']}")
    print()


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = Datasets(
        root=DATA_ROOT,
        split="test",
        shuffle=False,
        joint_idxs=config["joint_idx"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=test_dataset.data_collator,
    )

    print(f"Test samples:            {len(test_dataset)}")
    print(f"Test batches:            {len(test_loader)}")
    print()


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ModelB(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")
    print()


    # --------------------------------------------------------
    # Class information
    # --------------------------------------------------------

    id2label = test_dataset.id2label

    num_classes = config["num_labels"]

    class_total = [0] * num_classes
    class_correct = [0] * num_classes

    confusion_matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]


    # --------------------------------------------------------
    # Evaluation accumulators
    # --------------------------------------------------------

    total = 0
    correct = 0

    top5_correct = 0

    total_loss = 0.0

    confidence_sum = 0.0


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("=" * 70)
    print("RUNNING TEST SET")
    print("=" * 70)

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            keypoints = batch["keypoints"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                keypoints=keypoints,
                attention_mask=attention_mask,
                labels=labels,
            )

            # ----------------------------------------------
            # Predictions
            # ----------------------------------------------

            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)

            total += batch_size

            total_loss += loss.item() * batch_size

            correct += (
                predictions == labels
            ).sum().item()

            top5_correct += top_k_accuracy(
                logits,
                labels,
                k=5
            )

            # ----------------------------------------------
            # Confidence
            # ----------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            max_confidence = probabilities.max(
                dim=1
            ).values

            confidence_sum += (
                max_confidence.sum().item()
            )

            # ----------------------------------------------
            # Per-class statistics
            # ----------------------------------------------

            for true_label, predicted_label in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist()
            ):

                class_total[true_label] += 1

                if true_label == predicted_label:
                    class_correct[true_label] += 1

                confusion_matrix[
                    true_label
                ][predicted_label] += 1


            # ----------------------------------------------
            # Progress
            # ----------------------------------------------

            print(
                f"Batch {batch_idx + 1:3d}/{len(test_loader):3d} | "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    test_loss = total_loss / total

    test_accuracy = correct / total

    top5_accuracy = top5_correct / total

    average_confidence = confidence_sum / total


    print()
    print("=" * 70)
    print("MODEL B TEST RESULTS")
    print("=" * 70)

    print(f"Checkpoint epoch:        {checkpoint['epoch']}")
    print(f"Test samples:            {total}")
    print()

    print(f"Test Loss:               {test_loss:.4f}")
    print(f"Top-1 Accuracy:          {test_accuracy:.4f}")
    print(f"Top-5 Accuracy:          {top5_accuracy:.4f}")
    print(f"Correct Top-1:           {correct}/{total}")
    print(f"Correct Top-5:           {top5_correct}/{total}")
    print(f"Average Confidence:      {average_confidence:.4f}")

    print("=" * 70)


    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    represented_classes = 0
    class_accuracy_sum = 0.0

    per_class_results = []

    for class_id in range(num_classes):

        total_class = class_total[class_id]

        if total_class == 0:
            continue

        represented_classes += 1

        accuracy = (
            class_correct[class_id]
            / total_class
        )

        class_accuracy_sum += accuracy

        class_name = get_class_name(
            id2label,
            class_id
        )

        per_class_results.append({
            "class_id": class_id,
            "class_name": class_name,
            "samples": total_class,
            "correct": class_correct[class_id],
            "accuracy": accuracy,
        })


    macro_accuracy = (
        class_accuracy_sum
        / represented_classes
        if represented_classes > 0
        else 0.0
    )


    print()
    print("=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)

    print(
        f"Represented test classes: {represented_classes}/{num_classes}"
    )

    print(
        f"Macro-average accuracy:   {macro_accuracy:.4f}"
    )

    print()


    # --------------------------------------------------------
    # Best classes
    # --------------------------------------------------------

    best_classes = sorted(
        per_class_results,
        key=lambda x: x["accuracy"],
        reverse=True
    )

    print("BEST PERFORMING CLASSES")
    print("-" * 70)

    for item in best_classes[:10]:

        print(
            f"{item['class_name']:<20} "
            f"{item['correct']}/{item['samples']} "
            f"({item['accuracy']:.2%})"
        )


    # --------------------------------------------------------
    # Worst classes
    # --------------------------------------------------------

    worst_classes = sorted(
        per_class_results,
        key=lambda x: x["accuracy"]
    )

    print()
    print("WORST PERFORMING CLASSES")
    print("-" * 70)

    for item in worst_classes[:10]:

        print(
            f"{item['class_name']:<20} "
            f"{item['correct']}/{item['samples']} "
            f"({item['accuracy']:.2%})"
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    # Find strongest incorrect confusions.

    confusions = []

    for true_id in range(num_classes):

        for predicted_id in range(num_classes):

            if true_id == predicted_id:
                continue

            count = confusion_matrix[
                true_id
            ][predicted_id]

            if count > 0:

                confusions.append({
                    "true_id": true_id,
                    "true_class": get_class_name(
                        id2label,
                        true_id
                    ),
                    "predicted_id": predicted_id,
                    "predicted_class": get_class_name(
                        id2label,
                        predicted_id
                    ),
                    "count": count,
                })


    confusions.sort(
        key=lambda x: x["count"],
        reverse=True
    )


    print()
    print("=" * 70)
    print("TOP CONFUSIONS")
    print("=" * 70)

    if confusions:

        for item in confusions[:15]:

            print(
                f"{item['true_class']:<20} -> "
                f"{item['predicted_class']:<20} "
                f"{item['count']} time(s)"
            )

    else:

        print("No incorrect predictions.")


    # ========================================================
    # SAVE PER-CLASS CSV
    # ========================================================

    with open(
        PER_CLASS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "samples",
                "correct",
                "accuracy",
            ]
        )

        writer.writeheader()
        writer.writerows(per_class_results)


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    with open(
        CONFUSION_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        header = ["true/predicted"]

        header.extend(
            get_class_name(id2label, i)
            for i in range(num_classes)
        )

        writer.writerow(header)

        for true_id in range(num_classes):

            row = [
                get_class_name(
                    id2label,
                    true_id
                )
            ]

            row.extend(
                confusion_matrix[true_id]
            )

            writer.writerow(row)


    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    results = {
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_epoch": checkpoint["epoch"],

        "num_classes": num_classes,

        "test_samples": total,

        "test_loss": test_loss,

        "top1_accuracy": test_accuracy,

        "top5_accuracy": top5_accuracy,

        "correct_top1": correct,

        "correct_top5": top5_correct,

        "average_confidence": average_confidence,

        "represented_test_classes": represented_classes,

        "macro_accuracy": macro_accuracy,

        "best_validation_accuracy":
            checkpoint["val_accuracy"],

        "best_validation_loss":
            checkpoint["val_loss"],
    }


    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Results JSON:       {RESULTS_JSON}")
    print(f"Per-class CSV:      {PER_CLASS_CSV}")
    print(f"Confusion matrix:   {CONFUSION_CSV}")

    print("=" * 70)


if __name__ == "__main__":
    main()