"""
Experiment:
    11A Generate Localized Adversarial Patch Examples

Objective:
    Generate per-image optimized localized adversarial patch examples
    against the corrected differentiable DINOv2 + linear-head CIFAR-10
    classifier.

Inputs:
    - CIFAR-10 official TEST images in raw [0, 1] pixel space
    - Pretrained DINOv2 ViT-S/14 encoder
    - Linear classification head trained only on official CIFAR-10 TRAIN
    - Number of samples passed through --num-samples

Method:
    - Load the first N images from official CIFAR-10 TEST.
    - Verify that the classification head follows the corrected
      official TRAIN -> TEST protocol.
    - Assign each image a deterministic random patch location.
    - Generate each image's (top, left) patch location together so that
      image IDs receive the same location across different sample counts.
    - Optimize the pixels inside a small localized patch independently
      for each image using gradient ascent on classification loss.
    - Pixels outside the patch remain identical to the clean image.
    - Measure attack success only among samples classified correctly
      before the attack.

Outputs:
    - Clean and patch-adversarial images
    - Clean and adversarial predictions
    - Patch locations
    - Clean accuracy
    - Patch adversarial accuracy
    - Attack success rate
    - Perturbation statistics
    - Saved attack metrics

Research Goal:
    Test whether the representation-space signature learned from PGD
    transfers to a spatially localized adversarial attack whose
    perturbation geometry differs substantially from PGD, APGD, and CW.

Important Limitation:
    This is a per-image optimized localized adversarial patch, not a
    universal adversarial patch. Patch pixels are unrestricted within
    [0, 1], so this attack is not directly comparable to the L-infinity
    perturbation budgets used for PGD and APGD.

Next Experiment:
    11B Extract patch DINOv2 embeddings.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

from detectors.linear_head import LinearHead
from encoders.dinov2 import DinoV2Encoder
from models.dinov2_classifier import DinoV2Classifier
from utils.datasets import get_cifar10_loader
from utils.results import save_metrics


MODEL_PATH = (
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)

BATCH_SIZE = 20

IMAGE_SIZE = 224

# 48 x 48 on 224 x 224 is approximately 4.6% of image area.
PATCH_SIZE = 48

STEPS = 100

LEARNING_RATE = 0.05

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of official CIFAR-10 TEST images to attack.",
    )

    return parser.parse_args()


def get_predictions(model, images):
    with torch.no_grad():
        logits = model(images)

    return logits.argmax(
        dim=1
    )


def apply_patches(
    images,
    patch_values,
    top_positions,
    left_positions,
):
    """
    Insert one independently optimized patch into each image.

    The operation preserves gradients from the model output back
    to patch_values.
    """

    adversarial_images = images.clone()

    for index in range(images.shape[0]):
        top = int(
            top_positions[index]
        )

        left = int(
            left_positions[index]
        )

        adversarial_images[
            index,
            :,
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE,
        ] = patch_values[index]

    return adversarial_images


def generate_patch_attack(
    model,
    images,
    labels,
    top_positions,
    left_positions,
):
    """
    Optimize one localized patch independently for every image.

    We maximize cross-entropy with respect to the true label while
    leaving every pixel outside the patch unchanged.
    """

    batch_size = images.shape[0]

    # Random initialization makes the patch optimization independent
    # of the original pixel values inside the patch.
    patch_values = torch.rand(
        (
            batch_size,
            3,
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        device=images.device,
        dtype=images.dtype,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam(
        [patch_values],
        lr=LEARNING_RATE,
    )

    for _ in range(STEPS):
        optimizer.zero_grad()

        adversarial_images = apply_patches(
            images,
            patch_values,
            top_positions,
            left_positions,
        )

        logits = model(
            adversarial_images
        )

        # Adam minimizes by default.
        # Negative CE therefore maximizes classification loss.
        loss = -F.cross_entropy(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        # Keep patch pixels in valid image range.
        with torch.no_grad():
            patch_values.clamp_(
                0.0,
                1.0,
            )

    adversarial_images = apply_patches(
        images,
        patch_values.detach(),
        top_positions,
        left_positions,
    )

    return adversarial_images.detach()


def main():
    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    args = parse_args()

    num_samples = args.num_samples

    output_path = Path(
        "data/processed/adversarial/"
        f"cifar10/patch_{num_samples}.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"11a_patch_{num_samples}_metrics.json"
    )

    # ============================================================
    # 1. Load official CIFAR-10 TEST images.
    # ============================================================

    base_loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=False,
        image_size=IMAGE_SIZE,
        num_workers=0,
        normalize=False,
    )

    if num_samples > len(base_loader.dataset):
        raise ValueError(
            f"Requested {num_samples} samples, but CIFAR-10 TEST "
            f"contains only {len(base_loader.dataset)} images."
        )

    subset = Subset(
        base_loader.dataset,
        range(num_samples),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ============================================================
    # 2. Deterministically generate patch locations.
    #
    # Each image receives a (top, left) pair generated together.
    # Therefore image ID i receives the same location whether this
    # experiment is run with 100, 1000, or more samples.
    # ============================================================

    rng = np.random.default_rng(
        SEED
    )

    max_position = (
        IMAGE_SIZE
        - PATCH_SIZE
    )

    patch_locations = rng.integers(
        low=0,
        high=max_position + 1,
        size=(num_samples, 2),
    )

    all_top_positions = (
        patch_locations[:, 0]
    )

    all_left_positions = (
        patch_locations[:, 1]
    )

    # ============================================================
    # 3. Load corrected DINOv2 classification model.
    # ============================================================

    encoder_wrapper = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    device = encoder_wrapper.device

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    training_split = checkpoint.get(
        "training_split"
    )

    evaluation_split = checkpoint.get(
        "evaluation_split"
    )

    train_samples = checkpoint.get(
        "train_samples"
    )

    test_samples = checkpoint.get(
        "test_samples"
    )

    protocol_valid = (
        training_split
        == "official_cifar10_train"
        and evaluation_split
        == "official_cifar10_test"
        and train_samples == 50000
        and test_samples == 10000
    )

    print()
    print("=" * 70)
    print("LOCALIZED ADVERSARIAL PATCH GENERATION")
    print("=" * 70)

    print()
    print("Classification-head protocol")
    print("----------------------------")

    print(
        f"Training split:     {training_split}"
    )

    print(
        f"Evaluation split:   {evaluation_split}"
    )

    print(
        f"Training samples:   {train_samples}"
    )

    print(
        f"Evaluation samples: {test_samples}"
    )

    print(
        "Official TRAIN -> TEST protocol valid: "
        f"{protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "Saved classification head does not match the corrected "
            "official CIFAR-10 TRAIN -> TEST protocol."
        )

    head = LinearHead(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    )

    head.load_state_dict(
        checkpoint["model_state_dict"]
    )

    head.to(
        device
    )

    head.eval()

    model = DinoV2Classifier(
        encoder=encoder_wrapper.model,
        classifier=head,
    ).to(
        device
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    # ============================================================
    # 4. Generate adversarial patches.
    # ============================================================

    all_clean_images = []
    all_adversarial_images = []
    all_labels = []
    all_clean_predictions = []
    all_adversarial_predictions = []

    clean_correct = 0
    adversarial_correct = 0
    total = 0

    print()

    print(
        f"Generating localized adversarial patches "
        f"for {num_samples} official CIFAR-10 TEST images..."
    )

    print(
        f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}"
    )

    patch_area_ratio = (
        PATCH_SIZE * PATCH_SIZE
    ) / (
        IMAGE_SIZE * IMAGE_SIZE
    )

    print(
        f"Patch area: {patch_area_ratio * 100:.2f}% of image"
    )

    print(
        f"Optimization steps: {STEPS}"
    )

    for batch_index, (images, labels) in enumerate(
        loader
    ):
        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        batch_start = (
            batch_index
            * BATCH_SIZE
        )

        batch_end = (
            batch_start
            + images.shape[0]
        )

        top_positions = (
            all_top_positions[
                batch_start:batch_end
            ]
        )

        left_positions = (
            all_left_positions[
                batch_start:batch_end
            ]
        )

        clean_predictions = get_predictions(
            model,
            images,
        )

        adversarial_images = generate_patch_attack(
            model=model,
            images=images,
            labels=labels,
            top_positions=top_positions,
            left_positions=left_positions,
        )

        adversarial_predictions = get_predictions(
            model,
            adversarial_images,
        )

        clean_correct += (
            clean_predictions
            == labels
        ).sum().item()

        adversarial_correct += (
            adversarial_predictions
            == labels
        ).sum().item()

        total += labels.size(
            0
        )

        all_clean_images.append(
            images.detach().cpu()
        )

        all_adversarial_images.append(
            adversarial_images.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        all_clean_predictions.append(
            clean_predictions.detach().cpu()
        )

        all_adversarial_predictions.append(
            adversarial_predictions.detach().cpu()
        )

        print(
            f"Processed batch "
            f"{batch_index + 1}/{len(loader)}"
        )

    # ============================================================
    # 5. Combine batches.
    # ============================================================

    clean_images = torch.cat(
        all_clean_images,
        dim=0,
    )

    adversarial_images = torch.cat(
        all_adversarial_images,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    clean_predictions = torch.cat(
        all_clean_predictions,
        dim=0,
    )

    adversarial_predictions = torch.cat(
        all_adversarial_predictions,
        dim=0,
    )

    # ============================================================
    # 6. Attack effectiveness metrics.
    # ============================================================

    clean_accuracy = (
        clean_correct
        / total
    )

    adversarial_accuracy = (
        adversarial_correct
        / total
    )

    originally_correct_mask = (
        clean_predictions
        == labels
    )

    successful_attack_mask = (
        originally_correct_mask
        & (
            adversarial_predictions
            != labels
        )
    )

    originally_correct = (
        originally_correct_mask
        .sum()
        .item()
    )

    successful_attacks = (
        successful_attack_mask
        .sum()
        .item()
    )

    attack_success_rate = (
        successful_attacks
        / originally_correct
        if originally_correct > 0
        else 0.0
    )

    # ============================================================
    # 7. Perturbation statistics.
    # ============================================================

    perturbations = (
        adversarial_images
        - clean_images
    )

    absolute_perturbations = (
        perturbations.abs()
    )

    flattened_perturbations = (
        perturbations.reshape(
            perturbations.shape[0],
            -1,
        )
    )

    l2_per_image = torch.linalg.vector_norm(
        flattened_perturbations,
        ord=2,
        dim=1,
    )

    mean_l2_perturbation = float(
        l2_per_image.mean().item()
    )

    median_l2_perturbation = float(
        l2_per_image.median().item()
    )

    max_l2_perturbation = float(
        l2_per_image.max().item()
    )

    max_linf_perturbation = float(
        absolute_perturbations
        .amax()
        .item()
    )

    mean_absolute_perturbation = float(
        absolute_perturbations
        .mean()
        .item()
    )

    # Compute mean perturbation specifically inside patch regions.
    patch_absolute_sum = 0.0
    patch_value_count = 0

    for index in range(total):
        top = int(
            all_top_positions[index]
        )

        left = int(
            all_left_positions[index]
        )

        patch_difference = absolute_perturbations[
            index,
            :,
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE,
        ]

        patch_absolute_sum += float(
            patch_difference.sum().item()
        )

        patch_value_count += (
            patch_difference.numel()
        )

    mean_absolute_patch_perturbation = (
        patch_absolute_sum
        / patch_value_count
    )

    # ============================================================
    # 8. Save attack data.
    # ============================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "clean_images": clean_images,
            "adversarial_images": adversarial_images,
            "labels": labels,
            "clean_predictions": clean_predictions,
            "adversarial_predictions": adversarial_predictions,
            "attack": "localized_patch",
            "attack_type": "per_image_optimized_patch",
            "patch_size": PATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "patch_area_ratio": patch_area_ratio,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "patch_location_generation": (
                "paired_top_left_per_image"
            ),
            "patch_locations": torch.tensor(
                patch_locations,
                dtype=torch.long,
            ),
            "top_positions": torch.tensor(
                all_top_positions,
                dtype=torch.long,
            ),
            "left_positions": torch.tensor(
                all_left_positions,
                dtype=torch.long,
            ),
            "source_split": "official_cifar10_test",
            "classifier_training_split": training_split,
            "classifier_evaluation_split": evaluation_split,
        },
        output_path,
    )

    # ============================================================
    # 9. Save metrics.
    # ============================================================

    metrics = {
        "experiment": "11a_generate_patch",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "attack": "localized_patch",
        "attack_type": "per_image_optimized_patch",
        "source_split": "official_cifar10_test",
        "classifier_training_split": training_split,
        "classifier_evaluation_split": evaluation_split,
        "num_samples": total,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "patch_area_ratio": patch_area_ratio,
        "patch_area_percent": (
            patch_area_ratio
            * 100.0
        ),
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "patch_location_generation": (
            "paired_top_left_per_image"
        ),
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "originally_correct": originally_correct,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success_rate,
        "mean_l2_perturbation": mean_l2_perturbation,
        "median_l2_perturbation": median_l2_perturbation,
        "max_l2_perturbation": max_l2_perturbation,
        "max_linf_perturbation": max_linf_perturbation,
        "mean_absolute_perturbation": (
            mean_absolute_perturbation
        ),
        "mean_absolute_patch_perturbation": (
            mean_absolute_patch_perturbation
        ),
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ============================================================
    # 10. Print summary.
    # ============================================================

    print()
    print("Localized adversarial patch results")
    print("-----------------------------------")

    print(
        f"Samples:                         {total}"
    )

    print(
        f"Clean accuracy:                  "
        f"{clean_accuracy:.4f}"
    )

    print(
        f"Adversarial accuracy:            "
        f"{adversarial_accuracy:.4f}"
    )

    print(
        f"Originally correct:              "
        f"{originally_correct}"
    )

    print(
        f"Successful attacks:              "
        f"{successful_attacks}"
    )

    print(
        f"Attack success rate:             "
        f"{attack_success_rate:.4f}"
    )

    print(
        f"Patch size:                      "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Patch area:                      "
        f"{patch_area_ratio * 100:.2f}%"
    )

    print(
        f"Mean L2 perturbation:            "
        f"{mean_l2_perturbation:.6f}"
    )

    print(
        f"Median L2 perturbation:          "
        f"{median_l2_perturbation:.6f}"
    )

    print(
        f"Max L2 perturbation:             "
        f"{max_l2_perturbation:.6f}"
    )

    print(
        f"Max L-inf perturbation:          "
        f"{max_linf_perturbation:.6f}"
    )

    print(
        f"Mean absolute perturbation:      "
        f"{mean_absolute_perturbation:.6f}"
    )

    print(
        f"Mean absolute patch perturbation:"
        f" {mean_absolute_patch_perturbation:.6f}"
    )

    print()

    print(
        f"Saved adversarial samples to "
        f"{output_path}"
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()