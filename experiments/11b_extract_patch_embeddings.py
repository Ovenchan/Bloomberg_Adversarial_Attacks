"""
Experiment:
    11B Extract Localized Patch Embeddings

Objective:
    Extract frozen DINOv2 representations for the clean and localized
    patch-adversarial CIFAR-10 image pairs generated in Experiment 11A.

Inputs:
    - Saved clean/patch image pairs from official CIFAR-10 TEST
    - Pretrained DINOv2 ViT-S/14 encoder
    - Number of samples passed through --num-samples

Method:
    - Load the clean and localized patch image pairs from Experiment 11A.
    - Verify that the saved attack data follows the corrected protocol:
        classifier trained on official CIFAR-10 TRAIN
        attack images drawn from official CIFAR-10 TEST
    - Apply ImageNet normalization.
    - Extract frozen 384-dimensional DINOv2 ViT-S/14 embeddings for
      both clean and localized patch images.
    - Preserve attack metadata and patch locations for downstream audit.

Outputs:
    - Clean DINOv2 embeddings
    - Patch DINOv2 embeddings
    - Labels and classifier predictions
    - Patch locations and attack metadata
    - Saved paired embedding file

Research Goal:
    Produce representation-space data needed to test whether the linear
    detector trained only on PGD transfers to a spatially localized
    adversarial attack.

Important Limitation:
    The localized patch attack modifies a small spatial region but permits
    large pixel changes inside that region. Its perturbation geometry is
    therefore fundamentally different from PGD, APGD, and CW.

Next Experiment:
    11C Evaluate the frozen PGD detector on localized patch embeddings
    using only the exact detector-held-out image IDs from Experiment 08.
"""

import argparse
from pathlib import Path

import torch

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from encoders.dinov2 import DinoV2Encoder


BATCH_SIZE = 20

EXPECTED_SOURCE_SPLIT = "official_cifar10_test"
EXPECTED_CLASSIFIER_TRAINING_SPLIT = "official_cifar10_train"
EXPECTED_ATTACK = "localized_patch"
EXPECTED_ATTACK_TYPE = "per_image_optimized_patch"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of localized patch image pairs to encode.",
    )

    return parser.parse_args()


def extract_embeddings(
    encoder,
    images,
    batch_size,
    description,
):
    """
    Apply ImageNet normalization and extract frozen DINOv2 embeddings.
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=images.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=images.dtype,
    ).view(
        1,
        3,
        1,
        1,
    )

    dataset = TensorDataset(
        images
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings = []

    for (batch_images,) in tqdm(
        loader,
        desc=description,
    ):
        normalized_images = (
            batch_images - mean
        ) / std

        embeddings = encoder.encode(
            normalized_images
        )

        all_embeddings.append(
            embeddings.detach().cpu()
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
        f"cifar10/patch_{num_samples}.pt"
    )

    output_path = Path(
        "data/processed/embeddings/"
        f"cifar10/patch_{num_samples}_dinov2_vits14.pt"
    )

    # ============================================================
    # 1. Load localized patch data from Experiment 11A.
    # ============================================================

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

    # ============================================================
    # 2. Load and validate attack metadata.
    # ============================================================

    attack = data.get(
        "attack"
    )

    attack_type = data.get(
        "attack_type"
    )

    source_split = data.get(
        "source_split"
    )

    classifier_training_split = data.get(
        "classifier_training_split"
    )

    classifier_evaluation_split = data.get(
        "classifier_evaluation_split"
    )

    patch_size = data.get(
        "patch_size"
    )

    image_size = data.get(
        "image_size"
    )

    patch_area_ratio = data.get(
        "patch_area_ratio"
    )

    steps = data.get(
        "steps"
    )

    learning_rate = data.get(
        "learning_rate"
    )

    seed = data.get(
        "seed"
    )

    patch_location_generation = data.get(
        "patch_location_generation"
    )

    patch_locations = data.get(
        "patch_locations"
    )

    top_positions = data.get(
        "top_positions"
    )

    left_positions = data.get(
        "left_positions"
    )

    protocol_valid = (
        source_split
        == EXPECTED_SOURCE_SPLIT
        and classifier_training_split
        == EXPECTED_CLASSIFIER_TRAINING_SPLIT
        and attack
        == EXPECTED_ATTACK
        and attack_type
        == EXPECTED_ATTACK_TYPE
    )

    print()
    print("=" * 70)
    print("LOCALIZED PATCH EMBEDDING EXTRACTION")
    print("=" * 70)

    print()
    print("Data protocol")
    print("-------------")

    print(
        f"Attack image source:        {source_split}"
    )

    print(
        f"Classifier training split:  {classifier_training_split}"
    )

    print(
        f"Classifier evaluation split:{classifier_evaluation_split}"
    )

    print(
        f"Attack:                     {attack}"
    )

    print(
        f"Attack type:                {attack_type}"
    )

    print(
        f"Corrected protocol valid:   {protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "Localized patch input file does not match the corrected "
            "CIFAR-10 TRAIN -> TEST protocol."
        )

    # ============================================================
    # 3. Validate tensor shapes.
    # ============================================================

    if clean_images.shape[0] != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but input contains "
            f"{clean_images.shape[0]} clean images."
        )

    if adversarial_images.shape[0] != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but input contains "
            f"{adversarial_images.shape[0]} patch images."
        )

    if clean_images.shape != adversarial_images.shape:
        raise ValueError(
            "Clean and localized patch image shapes do not match."
        )

    if labels.shape[0] != num_samples:
        raise ValueError(
            "Label count does not match requested sample count."
        )

    if clean_predictions.shape[0] != num_samples:
        raise ValueError(
            "Clean prediction count does not match requested sample count."
        )

    if adversarial_predictions.shape[0] != num_samples:
        raise ValueError(
            "Patch prediction count does not match requested sample count."
        )

    if top_positions is not None:
        if len(top_positions) != num_samples:
            raise ValueError(
                "Top-position count does not match requested sample count."
            )

    if left_positions is not None:
        if len(left_positions) != num_samples:
            raise ValueError(
                "Left-position count does not match requested sample count."
            )

    if patch_locations is not None:
        if patch_locations.shape != (
            num_samples,
            2,
        ):
            raise ValueError(
                "Patch-location tensor must have shape "
                f"({num_samples}, 2), found {tuple(patch_locations.shape)}."
            )

    print()
    print("Input tensors")
    print("-------------")

    print(
        "Clean images:",
        clean_images.shape,
    )

    print(
        "Patch images:",
        adversarial_images.shape,
    )

    print(
        "Labels:      ",
        labels.shape,
    )

    if patch_locations is not None:
        print(
            "Patch locations:",
            patch_locations.shape,
        )

    print()
    print("Patch configuration")
    print("-------------------")

    print(
        f"Image size:        {image_size}"
    )

    print(
        f"Patch size:        {patch_size}"
    )

    if patch_area_ratio is not None:
        print(
            f"Patch area:        "
            f"{float(patch_area_ratio) * 100:.2f}%"
        )

    print(
        f"Optimization steps:{steps}"
    )

    print(
        f"Learning rate:     {learning_rate}"
    )

    print(
        f"Seed:              {seed}"
    )

    print(
        f"Location method:   {patch_location_generation}"
    )

    # ============================================================
    # 4. Load frozen DINOv2 encoder.
    # ============================================================

    encoder = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    # ============================================================
    # 5. Extract clean embeddings.
    # ============================================================

    print()
    print("Encoding clean images...")

    clean_embeddings = extract_embeddings(
        encoder=encoder,
        images=clean_images,
        batch_size=BATCH_SIZE,
        description="Clean embeddings",
    )

    # ============================================================
    # 6. Extract localized patch embeddings.
    # ============================================================

    print()
    print("Encoding localized patch images...")

    adversarial_embeddings = extract_embeddings(
        encoder=encoder,
        images=adversarial_images,
        batch_size=BATCH_SIZE,
        description="Patch embeddings",
    )

    if (
        clean_embeddings.shape
        != adversarial_embeddings.shape
    ):
        raise ValueError(
            "Clean and localized patch embedding shapes do not match."
        )

    # ============================================================
    # 7. Verify expected DINOv2 representation size.
    # ============================================================

    embedding_dim = int(
        clean_embeddings.shape[1]
    )

    if embedding_dim != 384:
        raise ValueError(
            f"Expected 384-dimensional DINOv2 ViT-S/14 embeddings, "
            f"found {embedding_dim}."
        )

    # ============================================================
    # 8. Save paired representation data.
    # ============================================================

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
            "attack": attack,
            "attack_type": attack_type,
            "patch_size": patch_size,
            "image_size": image_size,
            "patch_area_ratio": patch_area_ratio,
            "steps": steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "patch_location_generation": patch_location_generation,
            "patch_locations": patch_locations,
            "top_positions": top_positions,
            "left_positions": left_positions,
            "source_split": source_split,
            "classifier_training_split": (
                classifier_training_split
            ),
            "classifier_evaluation_split": (
                classifier_evaluation_split
            ),
            "encoder": "dinov2_vits14",
            "embedding_dim": embedding_dim,
        },
        output_path,
    )

    # ============================================================
    # 9. Print summary.
    # ============================================================

    print()
    print("Embedding extraction results")
    print("----------------------------")

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "Patch embeddings:",
        adversarial_embeddings.shape,
    )

    print()

    print(
        f"Saved paired embeddings to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()