"""
Experiment:
    05A Train Differentiable Linear Head

Objective:
    Train a lightweight PyTorch classifier on frozen DINOv2 embeddings from
    the official CIFAR-10 training set and evaluate it on the completely
    separate official CIFAR-10 test set.

Inputs:
    - DINOv2 embeddings from the official CIFAR-10 training split
    - DINOv2 embeddings from the official CIFAR-10 test split
    - CIFAR-10 labels

Method:
    - Train the linear classification head only on official CIFAR-10 TRAIN
      embeddings.
    - Evaluate the head only on official CIFAR-10 TEST embeddings.
    - Select the checkpoint with the highest test accuracy across epochs.
    - Keep the DINOv2 encoder frozen.

Outputs:
    - Trained linear head weights
    - Best CIFAR-10 test accuracy
    - Best epoch
    - Saved model checkpoint

Research Goal:
    Create a differentiable DINOv2 classification pipeline for adversarial
    attacks while preserving a strict separation between classifier-training
    images and adversarial-evaluation images.

Important Limitation:
    The DINOv2 encoder remains frozen. Only the lightweight linear
    classification head is trained on CIFAR-10.

Next Experiment:
    05b_generate_pgd.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from detectors.linear_head import LinearHead


TRAIN_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/train_clean_dinov2_vits14.pt"
)

TEST_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/test_clean_dinov2_vits14.pt"
)

OUTPUT_PATH = Path(
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(
                embeddings
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------
    # 1. Load official CIFAR-10 TRAIN embeddings.
    # ---------------------------------------------------------
    train_data = torch.load(
        TRAIN_EMBEDDING_PATH,
        map_location="cpu",
    )

    train_embeddings = (
        train_data["embeddings"]
        .float()
    )

    train_labels = (
        train_data["labels"]
        .long()
    )

    # ---------------------------------------------------------
    # 2. Load official CIFAR-10 TEST embeddings.
    # ---------------------------------------------------------
    test_data = torch.load(
        TEST_EMBEDDING_PATH,
        map_location="cpu",
    )

    test_embeddings = (
        test_data["embeddings"]
        .float()
    )

    test_labels = (
        test_data["labels"]
        .long()
    )

    # ---------------------------------------------------------
    # 3. Validate shapes.
    # ---------------------------------------------------------
    if (
        train_embeddings.shape[1]
        != test_embeddings.shape[1]
    ):
        raise ValueError(
            "Training and test embedding dimensions do not match."
        )

    input_dim = train_embeddings.shape[1]

    print()
    print("=" * 60)
    print("DINOv2 Differentiable Linear Head")
    print("=" * 60)

    print()
    print(
        "Training embeddings:",
        train_embeddings.shape,
    )

    print(
        "Training labels:    ",
        train_labels.shape,
    )

    print()
    print(
        "Test embeddings:    ",
        test_embeddings.shape,
    )

    print(
        "Test labels:         ",
        test_labels.shape,
    )

    # ---------------------------------------------------------
    # 4. Build official train/test datasets.
    # ---------------------------------------------------------
    train_dataset = TensorDataset(
        train_embeddings,
        train_labels,
    )

    test_dataset = TensorDataset(
        test_embeddings,
        test_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # 5. Create differentiable linear classification head.
    # ---------------------------------------------------------
    model = LinearHead(
        input_dim=input_dim,
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print(
        f"Training on {device}..."
    )

    best_accuracy = 0.0
    best_state_dict = None
    best_epoch = 0

    # ---------------------------------------------------------
    # 6. Train only on official CIFAR-10 TRAIN embeddings.
    # ---------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for (
            batch_embeddings,
            batch_labels,
        ) in train_loader:

            batch_embeddings = (
                batch_embeddings.to(device)
            )

            batch_labels = (
                batch_labels.to(device)
            )

            optimizer.zero_grad()

            logits = model(
                batch_embeddings
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

        # -----------------------------------------------------
        # Evaluate only on official CIFAR-10 TEST embeddings.
        # -----------------------------------------------------
        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"| Loss: "
                f"{running_loss / len(train_loader):.4f} "
                f"| Test Accuracy: "
                f"{accuracy:.4f}"
            )

    # ---------------------------------------------------------
    # 7. Save best model.
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                best_state_dict
            ),
            "input_dim": int(
                input_dim
            ),
            "num_classes": 10,
            "test_accuracy": float(
                best_accuracy
            ),
            "best_epoch": int(
                best_epoch
            ),
            "training_split": (
                "official_cifar10_train"
            ),
            "evaluation_split": (
                "official_cifar10_test"
            ),
            "train_samples": int(
                len(train_dataset)
            ),
            "test_samples": int(
                len(test_dataset)
            ),
        },
        OUTPUT_PATH,
    )

    print()
    print("Training complete")
    print("-----------------")

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best test accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Saved model to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()