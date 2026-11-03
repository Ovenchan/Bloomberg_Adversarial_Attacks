"""
Experiment 15: ImageNet Linear Classification Head

Objective
---------
Train a lightweight linear classifier on frozen DINOv2 embeddings for the
five-class ImageNet subset.

Inputs
------
- data/processed/embeddings/imagenet5/train_dinov2_vits14.pt
- data/processed/embeddings/imagenet5/internal_val_dinov2_vits14.pt
- data/processed/embeddings/imagenet5/final_val_dinov2_vits14.pt

Method
------
Train a 384 -> 5 linear classifier using only the classifier-training split.

Model selection is based exclusively on the internal-validation split.
The official ImageNet validation split is evaluated only after the best
model has been selected.

Outputs
-------
- data/processed/models/dinov2_vits14_imagenet5_linear_head.pt
- results/metrics/15_imagenet_linear_head.json

Research Goal
-------------
Establish a downstream classifier for adversarial evaluation on genuine
ImageNet-source photographs while preserving strict separation between
classifier training, model selection, and final evaluation.

Important Limitation
--------------------
The classifier is trained on a small balanced five-class subset rather
than ImageNet-1K. The goal is external validation of the adversarial
representation pipeline, not state-of-the-art ImageNet classification.

Next Experiment
---------------
Generate PGD adversarial examples against the differentiable
DINOv2 + linear-head stack using only the untouched official ImageNet
validation images.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_PATH = Path(
    "data/processed/embeddings/imagenet5/train_dinov2_vits14.pt"
)
INTERNAL_VAL_PATH = Path(
    "data/processed/embeddings/imagenet5/internal_val_dinov2_vits14.pt"
)
FINAL_VAL_PATH = Path(
    "data/processed/embeddings/imagenet5/final_val_dinov2_vits14.pt"
)

MODEL_PATH = Path(
    "data/processed/models/dinov2_vits14_imagenet5_linear_head.pt"
)
RESULT_PATH = Path(
    "results/metrics/15_imagenet_linear_head.json"
)

INPUT_DIM = 384
NUM_CLASSES = 5

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    embeddings = data["embeddings"].float()
    labels = data["labels"].long()

    return embeddings, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("Experiment 15: ImageNet Linear Classification Head")
    print("=" * 72)
    print()
    print(f"Device: {device}")

    train_x, train_y = load_split(TRAIN_PATH)
    internal_x, internal_y = load_split(INTERNAL_VAL_PATH)
    final_x, final_y = load_split(FINAL_VAL_PATH)

    print()
    print(f"TRAIN:        {tuple(train_x.shape)}")
    print(f"Internal VAL: {tuple(internal_x.shape)}")
    print(f"Official VAL: {tuple(final_x.shape)}")

    if train_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Expected {INPUT_DIM}-dimensional embeddings."
        )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    internal_loader = DataLoader(
        TensorDataset(internal_x, internal_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    final_loader = DataLoader(
        TensorDataset(final_x, final_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = nn.Linear(
        INPUT_DIM,
        NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    best_epoch = None
    best_internal_accuracy = -1.0
    best_state = None

    history = []

    print()
    print("Training...")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(embeddings)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total_examples += labels.size(0)

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        internal_metrics = evaluate(
            model,
            internal_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "internal_val_loss": internal_metrics["loss"],
                "internal_val_accuracy": internal_metrics["accuracy"],
            }
        )

        if (
            internal_metrics["accuracy"]
            > best_internal_accuracy
        ):
            best_internal_accuracy = (
                internal_metrics["accuracy"]
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"train acc {train_accuracy:.4f} | "
                f"internal val acc "
                f"{internal_metrics['accuracy']:.4f}"
            )

    if best_state is None:
        raise RuntimeError(
            "No best model state was selected."
        )

    # -----------------------------------------------------------------------
    # Restore best model selected ONLY using internal validation
    # -----------------------------------------------------------------------

    model.load_state_dict(best_state)

    best_internal_metrics = evaluate(
        model,
        internal_loader,
        device,
    )

    # -----------------------------------------------------------------------
    # Official VAL is evaluated only after model selection
    # -----------------------------------------------------------------------

    final_metrics = evaluate(
        model,
        final_loader,
        device,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "num_classes": NUM_CLASSES,
            "best_epoch": best_epoch,
            "best_internal_val_accuracy": best_internal_metrics[
                "accuracy"
            ],
        },
        MODEL_PATH,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "experiment": "15_imagenet_linear_head",
        "model": "linear_384_to_5",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "num_train": len(train_y),
        "num_internal_val": len(internal_y),
        "num_final_val": len(final_y),
        "best_epoch": best_epoch,
        "best_internal_val": best_internal_metrics,
        "official_final_val": final_metrics,
        "history": history,
        "model_path": str(MODEL_PATH),
    }

    with RESULT_PATH.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print()
    print(f"Best epoch:              {best_epoch}")
    print(
        f"Internal VAL accuracy:   "
        f"{best_internal_metrics['accuracy']:.4f}"
    )
    print(
        f"Official VAL accuracy:   "
        f"{final_metrics['accuracy']:.4f}"
    )
    print()
    print(f"Saved head: {MODEL_PATH}")
    print(f"Metrics:    {RESULT_PATH}")


if __name__ == "__main__":
    main()
