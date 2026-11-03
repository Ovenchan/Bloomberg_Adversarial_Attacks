"""
Experiment:
    10B Extract CW Embeddings

Objective:
    Extract frozen DINOv2 representations for the clean and CW
    adversarial CIFAR-10 image pairs generated in Experiment 10A.

Inputs:
    - Saved clean/CW image pairs from official CIFAR-10 TEST
    - Pretrained DINOv2 ViT-S/14 encoder
    - Number of samples passed through --num-samples

Method:
    - Load the clean/CW image pairs generated in Experiment 10A.
    - Verify the adversarial file follows the corrected data protocol:
        classifier trained on official CIFAR-10 TRAIN
        attack images drawn from official CIFAR-10 TEST
    - Apply ImageNet normalization.
    - Extract frozen 384-dimensional DINOv2 ViT-S/14 embeddings for
      both clean and CW images.

Outputs:
    - Clean DINOv2 embeddings
    - CW DINOv2 embeddings
    - Labels and predictions
    - Attack and data-lineage metadata
    - Saved paired embedding file

Research Goal:
    Produce CW representation data for cross-attack evaluation against
    the detector trained only on PGD.

Important Limitation:
    CW uses an L2-based optimization objective while PGD/APGD use
    L-infinity perturbation constraints. Perturbation magnitudes should
    therefore not be compared as though they share the same norm budget.

Next Experiment:
    10C Evaluate the frozen PGD detector on CW using only the exact
    detector-held-out image IDs from Experiment 08.
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of CW image pairs to encode.",
    )

    return parser.parse_args()


def extract_embeddings(
    encoder,
    images,
    batch_size,
):
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
        desc="Extracting embeddings",
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
        f"cifar10/cw_{num_samples}.pt"
    )

    output_path = Path(
        "data/processed/embeddings/"
        f"cifar10/cw_{num_samples}_dinov2_vits14.pt"
    )

    # ============================================================
    # 1. Load CW data from Experiment 10A.
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
    # 2. Verify corrected data protocol.
    # ============================================================

    source_split = data.get(
        "source_split"
    )

    classifier_training_split = data.get(
        "classifier_training_split"
    )

    classifier_evaluation_split = data.get(
        "classifier_evaluation_split"
    )

    protocol_valid = (
        source_split
        == EXPECTED_SOURCE_SPLIT
        and classifier_training_split
        == EXPECTED_CLASSIFIER_TRAINING_SPLIT
    )

    print()
    print("=" * 70)
    print("CW EMBEDDING EXTRACTION")
    print("=" * 70)

    print()
    print("Data protocol")
    print("-------------")

    print(
        f"Attack image source:       {source_split}"
    )

    print(
        f"Classifier training split: {classifier_training_split}"
    )

    print(
        f"Classifier eval split:     {classifier_evaluation_split}"
    )

    print(
        "Corrected protocol valid: "
        f"{protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "CW input file does not match the corrected "
            "CIFAR-10 TRAIN -> TEST protocol."
        )

    # ============================================================
    # 3. Validate shapes.
    # ============================================================

    if clean_images.shape[0] != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, "
            f"but input contains {clean_images.shape[0]}."
        )

    if clean_images.shape != adversarial_images.shape:
        raise ValueError(
            "Clean and CW image shapes do not match."
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
            "CW prediction count does not match requested sample count."
        )

    print()
    print("Input tensors")
    print("-------------")

    print(
        "Clean images:",
        clean_images.shape,
    )

    print(
        "CW images:   ",
        adversarial_images.shape,
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
        encoder,
        clean_images,
        BATCH_SIZE,
    )

    # ============================================================
    # 6. Extract CW embeddings.
    # ============================================================

    print()
    print("Encoding CW images...")

    adversarial_embeddings = extract_embeddings(
        encoder,
        adversarial_images,
        BATCH_SIZE,
    )

    if (
        clean_embeddings.shape
        != adversarial_embeddings.shape
    ):
        raise ValueError(
            "Clean and CW embedding shapes do not match."
        )

    # ============================================================
    # 7. Save paired embeddings.
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
            "attack": data.get(
                "attack",
                "cw",
            ),
            "norm": data.get(
                "norm",
                "L2",
            ),
            "c": data.get(
                "c"
            ),
            "kappa": data.get(
                "kappa"
            ),
            "steps": data.get(
                "steps"
            ),
            "learning_rate": data.get(
                "learning_rate"
            ),
            "seed": data.get(
                "seed"
            ),
            "source_split": source_split,
            "classifier_training_split": (
                classifier_training_split
            ),
            "classifier_evaluation_split": (
                classifier_evaluation_split
            ),
            "encoder": "dinov2_vits14",
            "embedding_dim": int(
                clean_embeddings.shape[1]
            ),
        },
        output_path,
    )

    # ============================================================
    # 8. Print summary.
    # ============================================================

    print()
    print("Embedding extraction results")
    print("----------------------------")

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "CW embeddings:   ",
        adversarial_embeddings.shape,
    )

    print()

    print(
        f"Saved paired embeddings to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()