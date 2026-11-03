"""
Experiment:
    09A Generate APGD Adversarial Examples

Objective:
    Generate Auto-PGD (APGD) adversarial examples against the corrected
    differentiable DINOv2 + linear-head CIFAR-10 classifier.

Inputs:
    - CIFAR-10 official TEST images in raw [0, 1] pixel space
    - Pretrained DINOv2 ViT-S/14 encoder
    - Linear classification head trained only on official CIFAR-10 TRAIN
    - Number of samples passed through --num-samples

Method:
    - Load the first N images from the official CIFAR-10 TEST split.
    - Verify the saved classification head was trained using the official
      CIFAR-10 TRAIN split and evaluated on official CIFAR-10 TEST.
    - Generate APGD L-infinity adversarial examples using epsilon = 8/255.
    - Measure attack success only among images originally classified correctly.

Outputs:
    - Clean and APGD-adversarial images
    - Clean and adversarial predictions
    - Clean accuracy
    - APGD accuracy
    - Attack success rate
    - Perturbation statistics
    - Saved attack metrics

Research Goal:
    Test whether the strong DINOv2 adversarial representation shift observed
    with PGD also occurs under a related but stronger adaptive gradient-based
    attack, enabling cross-attack detector evaluation.

Important Limitation:
    APGD is still closely related to PGD and therefore does not by itself test
    generalization to structurally different attack families.

Next Experiment:
    09B Extract APGD DINOv2 embeddings.
"""

import argparse
from pathlib import Path

import torch
import torchattacks

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

EPSILON = 8 / 255

STEPS = 20

N_RESTARTS = 1

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


def main():
    args = parse_args()

    num_samples = args.num_samples

    output_path = Path(
        "data/processed/adversarial/"
        f"cifar10/apgd_{num_samples}.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"09a_apgd_{num_samples}_metrics.json"
    )

    # ============================================================
    # 1. Load official CIFAR-10 TEST images in raw [0, 1] space.
    # ============================================================

    base_loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=False,
        image_size=224,
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
    # 2. Load DINOv2 and corrected linear head.
    # ============================================================

    encoder_wrapper = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    device = encoder_wrapper.device

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    # ------------------------------------------------------------
    # Validate corrected TRAIN -> TEST protocol.
    # ------------------------------------------------------------

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

    expected_training_split = (
        "official_cifar10_train"
    )

    expected_evaluation_split = (
        "official_cifar10_test"
    )

    protocol_valid = (
        training_split
        == expected_training_split
        and evaluation_split
        == expected_evaluation_split
        and train_samples == 50000
        and test_samples == 10000
    )

    print()
    print("=" * 70)
    print("APGD ADVERSARIAL GENERATION")
    print("=" * 70)

    print()
    print("Classification-head protocol")
    print("----------------------------")

    print(
        f"Training split:    {training_split}"
    )

    print(
        f"Evaluation split:  {evaluation_split}"
    )

    print(
        f"Training samples:  {train_samples}"
    )

    print(
        f"Evaluation samples:{test_samples}"
    )

    print(
        "Official TRAIN -> TEST protocol valid: "
        f"{protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "The saved classification head does not match the corrected "
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
    # 3. Configure APGD.
    #
    # Same L-infinity epsilon as the PGD experiment so the two
    # attacks are directly comparable.
    # ============================================================

    attack = torchattacks.APGD(
        model=model,
        norm="Linf",
        eps=EPSILON,
        steps=STEPS,
        n_restarts=N_RESTARTS,
        seed=SEED,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
    )

    # ============================================================
    # 4. Generate APGD examples.
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
        f"Generating APGD adversarial examples "
        f"for {num_samples} official CIFAR-10 TEST images..."
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

        clean_predictions = get_predictions(
            model,
            images,
        )

        adversarial_images = attack(
            images,
            labels,
        )

        adversarial_predictions = get_predictions(
            model,
            adversarial_images,
        )

        clean_correct += (
            clean_predictions == labels
        ).sum().item()

        adversarial_correct += (
            adversarial_predictions == labels
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
    # 6. Compute attack metrics.
    # ============================================================

    clean_accuracy = (
        clean_correct / total
    )

    adversarial_accuracy = (
        adversarial_correct / total
    )

    originally_correct_mask = (
        clean_predictions == labels
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

    perturbations = (
        adversarial_images
        - clean_images
    ).abs()

    max_linf_perturbation = (
        perturbations
        .amax()
        .item()
    )

    mean_absolute_perturbation = (
        perturbations
        .mean()
        .item()
    )

    # ============================================================
    # 7. Save attack data.
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
            "epsilon": EPSILON,
            "steps": STEPS,
            "n_restarts": N_RESTARTS,
            "seed": SEED,
            "attack": "apgd",
            "norm": "Linf",
            "source_split": "official_cifar10_test",
            "classifier_training_split": training_split,
            "classifier_evaluation_split": evaluation_split,
        },
        output_path,
    )

    # ============================================================
    # 8. Save metrics.
    # ============================================================

    metrics = {
        "experiment": "09a_generate_apgd",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "attack": "apgd",
        "norm": "Linf",
        "source_split": "official_cifar10_test",
        "classifier_training_split": training_split,
        "classifier_evaluation_split": evaluation_split,
        "num_samples": total,
        "batch_size": BATCH_SIZE,
        "epsilon": EPSILON,
        "steps": STEPS,
        "n_restarts": N_RESTARTS,
        "seed": SEED,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "originally_correct": originally_correct,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success_rate,
        "max_linf_perturbation": max_linf_perturbation,
        "mean_absolute_perturbation": (
            mean_absolute_perturbation
        ),
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ============================================================
    # 9. Print summary.
    # ============================================================

    print()

    print("APGD results")

    print("------------")

    print(
        f"Samples:               {total}"
    )

    print(
        f"Clean accuracy:        "
        f"{clean_accuracy:.4f}"
    )

    print(
        f"Adversarial accuracy:  "
        f"{adversarial_accuracy:.4f}"
    )

    print(
        f"Originally correct:    "
        f"{originally_correct}"
    )

    print(
        f"Successful attacks:    "
        f"{successful_attacks}"
    )

    print(
        f"Attack success rate:   "
        f"{attack_success_rate:.4f}"
    )

    print(
        f"Max L-inf perturbation: "
        f"{max_linf_perturbation:.6f}"
    )

    print(
        f"Mean absolute perturbation: "
        f"{mean_absolute_perturbation:.6f}"
    )

    print(
        f"Configured epsilon:     "
        f"{EPSILON:.6f}"
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