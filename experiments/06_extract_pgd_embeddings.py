"""
Experiment:
    06 Extract PGD Embeddings

Objective:
    Extract DINOv2 feature embeddings for the clean and PGD-adversarial
    CIFAR-10 images generated in Experiment 05B.

Inputs:
    - Saved clean/adversarial image pairs from Experiment 05B
    - Pretrained DINOv2 ViT-S/14 encoder
    - Number of samples passed through --num-samples

Outputs:
    - Clean DINOv2 embeddings
    - PGD-adversarial DINOv2 embeddings
    - CIFAR-10 labels
    - Clean and adversarial predictions
    - Saved paired embedding file

Research Goal:
    Create paired clean/adversarial representation data so we can directly
    measure how successful PGD attacks alter DINOv2 embedding space.

Next Experiment:
    07_compare_pgd_embeddings.py
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from encoders.dinov2 import DinoV2Encoder


BATCH_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of previously generated PGD samples to encode.",
    )

    return parser.parse_args()


def extract_embeddings(
    encoder,
    images,
    batch_size,
):
    """
    Extract DINOv2 embeddings from raw [0, 1] image tensors.

    The images saved by Experiment 05B are intentionally kept in pixel
    space. ImageNet normalization is applied here before passing them
    through DINOv2.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=images.dtype,
    ).view(1, 3, 1, 1)

    dataset = TensorDataset(images)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings = []

    for (batch_images,) in tqdm(
        loader,
        desc="Extracting embeddings",
    ):
        normalized_images = (
            batch_images - mean
        ) / std

        embeddings = encoder.encode(
            normalized_images
        )

        all_embeddings.append(
            embeddings
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )


def main():
    args = parse_args()

    num_samples = args.num_samples

    input_path = (
        "data/processed/adversarial/"
        f"cifar10/pgd_{num_samples}.pt"
    )

    output_path = Path(
        "data/processed/embeddings/"
        f"cifar10/pgd_{num_samples}_dinov2_vits14.pt"
    )

    # ---------------------------------------------------------
    # 1. Load clean/adversarial image pairs generated in 05B.
    # ---------------------------------------------------------
    data = torch.load(
        input_path,
        map_location="cpu",
    )

    clean_images = (
        data["clean_images"]
        .float()
    )

    adversarial_images = (
        data["adversarial_images"]
        .float()
    )

    labels = (
        data["labels"]
        .long()
    )

    clean_predictions = (
        data["clean_predictions"]
        .long()
    )

    adversarial_predictions = (
        data["adversarial_predictions"]
        .long()
    )

    print(
        "Clean images:",
        clean_images.shape,
    )

    print(
        "Adversarial images:",
        adversarial_images.shape,
    )

    print(
        "Labels:",
        labels.shape,
    )

    # ---------------------------------------------------------
    # 2. Validate that the requested sample count matches the
    #    saved attack file.
    # ---------------------------------------------------------
    actual_samples = clean_images.shape[0]

    if actual_samples != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, "
            f"but {input_path} contains "
            f"{actual_samples}."
        )

    # ---------------------------------------------------------
    # 3. Load DINOv2.
    # ---------------------------------------------------------
    encoder = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    # ---------------------------------------------------------
    # 4. Extract embeddings for clean images.
    # ---------------------------------------------------------
    print()
    print("Encoding clean images...")

    clean_embeddings = extract_embeddings(
        encoder=encoder,
        images=clean_images,
        batch_size=BATCH_SIZE,
    )

    # ---------------------------------------------------------
    # 5. Extract embeddings for adversarial images.
    # ---------------------------------------------------------
    print()
    print("Encoding adversarial images...")

    adversarial_embeddings = extract_embeddings(
        encoder=encoder,
        images=adversarial_images,
        batch_size=BATCH_SIZE,
    )

    # ---------------------------------------------------------
    # 6. Sanity checks.
    # ---------------------------------------------------------
    if (
        clean_embeddings.shape
        != adversarial_embeddings.shape
    ):
        raise ValueError(
            "Clean and adversarial embedding shapes do not match."
        )

    if (
        clean_embeddings.shape[0]
        != labels.shape[0]
    ):
        raise ValueError(
            "Embedding count does not match label count."
        )

    print()
    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "Adversarial embeddings:",
        adversarial_embeddings.shape,
    )

    # ---------------------------------------------------------
    # 7. Save paired embedding data.
    # ---------------------------------------------------------
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "clean_embeddings": clean_embeddings,
            "adversarial_embeddings": adversarial_embeddings,
            "labels": labels,
            "clean_predictions": clean_predictions,
            "adversarial_predictions": adversarial_predictions,
            "epsilon": data["epsilon"],
            "alpha": data["alpha"],
            "steps": data["steps"],
        },
        output_path,
    )

    print()
    print(
        f"Saved paired embeddings to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()