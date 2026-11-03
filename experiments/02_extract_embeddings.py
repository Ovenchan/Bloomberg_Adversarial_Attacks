"""
Experiment:
    02 DINOv2 Embedding Extraction

Objective:
    Extract frozen DINOv2 feature representations for both the official
    CIFAR-10 training and test datasets using the pretrained DINOv2 ViT-S/14
    encoder.

Inputs:
    - CIFAR-10 training dataset
    - CIFAR-10 test dataset
    - Pretrained DINOv2 ViT-S/14 encoder

Outputs:
    - 384-dimensional embedding for each CIFAR-10 training image
    - 384-dimensional embedding for each CIFAR-10 test image
    - Corresponding CIFAR-10 labels
    - Separate saved embedding files for train and test splits

Research Goal:
    Build a clean train/test-separated representation baseline so that the
    downstream classification head is trained only on official CIFAR-10
    training images and adversarial evaluation is performed only on unseen
    CIFAR-10 test images.

Important Limitation:
    These embeddings are extracted from frozen DINOv2 features. The DINOv2
    encoder itself is not fine-tuned on CIFAR-10.

Next Experiment:
    03_pca_clean.py
"""

from encoders.dinov2 import DinoV2Encoder
from pipelines.dinov2_pipeline import (
    extract_embeddings,
    save_embeddings,
)
from utils.datasets import get_cifar10_loader


TRAIN_OUTPUT_PATH = (
    "data/processed/embeddings/"
    "cifar10/train_clean_dinov2_vits14.pt"
)

TEST_OUTPUT_PATH = (
    "data/processed/embeddings/"
    "cifar10/test_clean_dinov2_vits14.pt"
)

BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_WORKERS = 0


def extract_split(
    encoder,
    train,
    split_name,
    output_path,
):
    print()
    print("=" * 60)
    print(f"Extracting CIFAR-10 {split_name} embeddings")
    print("=" * 60)

    loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=train,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    embeddings, labels = extract_embeddings(
        encoder=encoder,
        loader=loader,
    )

    print()
    print(
        f"{split_name.capitalize()} embeddings:",
        embeddings.shape,
    )

    print(
        f"{split_name.capitalize()} labels:",
        labels.shape,
    )

    save_embeddings(
        embeddings=embeddings,
        labels=labels,
        output_path=output_path,
    )

    print(
        f"Saved {split_name} embeddings to "
        f"{output_path}"
    )


def main():
    encoder = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    # ---------------------------------------------------------
    # 1. Official CIFAR-10 training split.
    #
    # This split will be used to train downstream classifiers.
    # ---------------------------------------------------------
    extract_split(
        encoder=encoder,
        train=True,
        split_name="train",
        output_path=TRAIN_OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # 2. Official CIFAR-10 test split.
    #
    # This split remains unseen during classifier training and
    # is reserved for evaluation and adversarial experiments.
    # ---------------------------------------------------------
    extract_split(
        encoder=encoder,
        train=False,
        split_name="test",
        output_path=TEST_OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("Embedding extraction complete")
    print("=" * 60)

    print()
    print(
        "Train embeddings saved to:"
    )
    print(TRAIN_OUTPUT_PATH)

    print()
    print(
        "Test embeddings saved to:"
    )
    print(TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()