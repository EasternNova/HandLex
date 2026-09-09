import os

import torch
from tqdm import tqdm


# MediaPipe Holistic landmark indices used by ModelB.
total_body_idx = 33
total_hand = 42

body_idx = list(range(11, 17))
lefthand_idx = [x + total_body_idx for x in range(21)]
righthand_idx = [x + 21 for x in lefthand_idx]

total_idx = body_idx + lefthand_idx + righthand_idx


def accuracy(logits, labels):
    """Calculate top-1 classification accuracy."""
    preds = torch.argmax(logits, dim=1)

    correct = (preds == labels).sum().item()
    total = labels.size(0)

    return correct / total if total > 0 else 0.0


def top_k_accuracy(logits, labels, k=5):
    """Calculate top-k classification accuracy."""
    k = min(k, logits.size(1))

    top_k_preds = torch.topk(logits, k, dim=1).indices

    correct = (
        (top_k_preds == labels.unsqueeze(1))
        .any(dim=1)
        .sum()
        .item()
    )

    total = labels.size(0)

    return correct / total if total > 0 else 0.0


def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    """Save model and optimizer checkpoint."""
    os.makedirs(path_dir, exist_ok=True)

    if name is None:
        filename = os.path.join(
            path_dir,
            f"checkpoints_{epoch}.pth"
        )
    else:
        filename = os.path.join(
            path_dir,
            f"checkpoints_{epoch}_{name}.pth"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        filename,
    )


def load_checkpoints(model, optimizer, path, resume=True):
    """Load a checkpoint from a file or checkpoint directory."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {path}"
        )

    if os.path.isdir(path):
        checkpoint_files = [
            x
            for x in os.listdir(path)
            if x.startswith("checkpoints_") and x.endswith(".pth")
        ]

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in: {path}"
            )

        def get_epoch(filename):
            name = filename[len("checkpoints_"):-4]

            try:
                return int(name)
            except ValueError:
                return -1

        valid_files = [
            x for x in checkpoint_files
            if get_epoch(x) >= 0
        ]

        if not valid_files:
            raise FileNotFoundError(
                f"No valid epoch checkpoints found in: {path}"
            )

        filename = os.path.join(
            path,
            max(valid_files, key=get_epoch)
        )

        print(f"Loaded latest checkpoint: {get_epoch(os.path.basename(filename))}")

        checkpoints = torch.load(
            filename,
            map_location="cpu",
        )

    else:
        print(f"Load checkpoint from file: {path}")

        checkpoints = torch.load(
            path,
            map_location="cpu",
        )

    model.load_state_dict(checkpoints["model"])

    if optimizer is not None and "optimizer" in checkpoints:
        optimizer.load_state_dict(checkpoints["optimizer"])

    if resume:
        return checkpoints["epoch"] + 1

    return 1


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler=None,
    epoch=0,
    epochs=0,
):
    """Run one training epoch."""
    all_loss = 0.0
    all_acc = 0.0
    all_top_5_acc = 0.0

    loop = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=True,
        desc=f"Training epoch {epoch + 1}/{epochs}: ",
    )

    for _, data in loop:
        labels = data["labels"]

        optimizer.zero_grad()

        loss, logits = model(**data)

        loss.backward()
        optimizer.step()

        all_loss += loss.item()

        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(
            f"Loss: {loss.item():.3f}, "
            f"Acc: {acc:.3f}, "
            f"Top 5 Acc: {top_5_acc:.3f}"
        )

    if scheduler is not None:
        scheduler.step()

    num_batches = len(dataloader)

    all_loss /= num_batches
    all_acc /= num_batches
    all_top_5_acc /= num_batches

    return all_loss, all_acc, all_top_5_acc


@torch.no_grad()
def evaluate(model, dataloader, epoch=0, epochs=0):
    """Evaluate the model without calculating gradients."""
    all_loss = 0.0
    all_acc = 0.0
    all_top_5_acc = 0.0

    loop = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=True,
        desc=f"Evaluation epoch {epoch + 1}/{epochs}: ",
    )

    for _, data in loop:
        labels = data["labels"]

        loss, logits = model(**data)

        all_loss += loss.item()

        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(
            f"Loss: {loss.item():.3f}, "
            f"Acc: {acc:.3f}, "
            f"Top 5 Acc: {top_5_acc:.3f}"
        )

    num_batches = len(dataloader)

    all_loss /= num_batches
    all_acc /= num_batches
    all_top_5_acc /= num_batches

    return all_loss, all_acc, all_top_5_acc


def create_attention_mask(mask, dtype, tgt_len=None):
    """
    Convert a binary padding mask into an additive attention mask.

    Input:
        mask:
            (batch, source_length)

            1 = valid frame
            0 = padding frame

    Output:
        (batch, 1, target_length, source_length)

            valid position  -> 0
            padding position -> dtype minimum value
    """
    bsz, src_len = mask.size()

    if tgt_len is None:
        tgt_len = src_len

    expanded_mask = (
        mask[:, None, None, :]
        .expand(bsz, 1, tgt_len, src_len)
        .to(dtype)
    )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.bool(),
        torch.finfo(dtype).min,
    )


def create_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
):
    """
    Create a combined padding + causal attention mask.

    Input:
        attention_mask:
            (batch, sequence_length)

            1 = valid token/frame
            0 = padding

    Output:
        (batch, 1, sequence_length, sequence_length)

            allowed position       -> 0
            padding/future position -> dtype minimum value
    """
    batch_size, query_length = input_shape[0], input_shape[1]

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device

    # ---------------------------------------------------------
    # Padding mask
    # ---------------------------------------------------------
    padding_mask = (
        attention_mask[:, None, None, :]
        .expand(
            batch_size,
            1,
            query_length,
            query_length,
        )
        .to(dtype)
    )

    padding_mask = 1.0 - padding_mask

    padding_mask = padding_mask.masked_fill(
        padding_mask.bool(),
        torch.finfo(dtype).min,
    )

    # ---------------------------------------------------------
    # Causal mask
    # ---------------------------------------------------------
    causal_blocked = torch.triu(
        torch.ones(
            query_length,
            query_length,
            device=device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    causal_mask = torch.zeros(
        (query_length, query_length),
        device=device,
        dtype=dtype,
    )

    causal_mask = causal_mask.masked_fill(
        causal_blocked,
        torch.finfo(dtype).min,
    )

    causal_mask = causal_mask[None, None, :, :]

    # ---------------------------------------------------------
    # Combine padding + causal masks
    # ---------------------------------------------------------
    return padding_mask + causal_mask


def count_model_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return {
        "total": total_params,
        "trainable": trainable_params,
    }


if __name__ == "__main__":
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ]
    )

    input_embeds = torch.randn(
        2,
        5,
        768,
    )

    expand_mask = create_causal_attention_mask(
        mask,
        (2, 5),
        input_embeds,
    )

    print("Causal attention mask shape:")
    print(expand_mask.shape)

    print("\nCausal attention mask:")
    print(expand_mask)