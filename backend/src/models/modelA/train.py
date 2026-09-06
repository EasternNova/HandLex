import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from .model import ModelA


# -----------------------------
# SETTINGS
# -----------------------------

DATA_DIR = "data/ASL/asl_alphabet_train/asl_alphabet_train"

MODEL_PATH = "src/models/modelA/modelA.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 29
TRAIN_RATIO = 0.8


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# TRANSFORMS
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# -----------------------------
# DATASET
# -----------------------------

base_dataset = datasets.ImageFolder(
    DATA_DIR
)

print("Classes:", base_dataset.classes)
print("Total images:", len(base_dataset))


# -----------------------------
# DATASET CHECK
# -----------------------------

if len(base_dataset.classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(base_dataset.classes)}: "
        f"{base_dataset.classes}"
    )

expected_classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del",
    "nothing",
    "space"
]

if base_dataset.classes != expected_classes:
    raise ValueError(
        "Class order is not what ModelA expects.\n"
        f"Found:    {base_dataset.classes}\n"
        f"Expected: {expected_classes}"
    )

print("Dataset check: PASSED")


# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------

train_size = int(TRAIN_RATIO * len(base_dataset))
val_size = len(base_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_indices, val_indices = random_split(
    range(len(base_dataset)),
    [train_size, val_size],
    generator=generator
)

train_indices = train_indices.indices
val_indices = val_indices.indices


# -----------------------------
# SEPARATE TRAIN / VAL DATASETS
# -----------------------------

train_base = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform
)

val_base = datasets.ImageFolder(
    DATA_DIR,
    transform=val_transform
)

train_dataset = Subset(
    train_base,
    train_indices
)

val_dataset = Subset(
    val_base,
    val_indices
)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# DATA LOADERS
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# MODEL
# -----------------------------

model = ModelA(num_classes=NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# BEST MODEL TRACKING
# -----------------------------

best_val_accuracy = 0.0


# -----------------------------
# TRAINING
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total


    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total
    average_val_loss = val_loss / len(val_loader)


    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": base_dataset.classes,
                "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1
            },
            MODEL_PATH
        )

        saved_text = " <-- BEST MODEL SAVED"

    else:
        saved_text = ""


    # -----------------------------
    # PROGRESS
    # -----------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Val Loss: {average_val_loss:.4f} "
        f"Val Acc: {val_accuracy:.2f}%"
        f"{saved_text}"
    )


# -----------------------------
# COMPLETE
# -----------------------------

print()
print("Training complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Best model saved to: {MODEL_PATH}")