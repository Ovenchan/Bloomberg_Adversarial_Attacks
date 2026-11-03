"""
Experiment:
    05B PGD Adversarial Sanity / Scale Test

Objective:
    Generate PGD adversarial examples against the differentiable
    DINOv2 + linear-head CIFAR-10 classifier and verify that the
    attack reduces model accuracy.

Inputs:
    - CIFAR-10 test images in raw [0, 1] pixel space
    - Pretrained DINOv2 ViT-S/14 encoder
    - Trained CIFAR-10 linear classification head
    - Number of samples passed through --num-samples

Outputs:
    - Clean accuracy
    - Adversarial accuracy
    - Attack success rate
    - Saved clean/adversarial image pairs and predictions
    - Saved experiment metrics

Research Goal:
    Validate and scale the PGD attack pipeline before comparing
    representation shifts and training adversarial detectors.

Next Experiment:
    06_extract_pgd_embeddings.py
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from attacks.pgd import pgd_attack
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
ALPHA = 2 / 255
STEPS = 10


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of CIFAR-10 test images to attack.",
    )

    return parser.parse_args()


def get_predictions(model, images):
    with torch.no_grad():
        logits = model(images)

    return logits.argmax(dim=1)


def main():
    args = parse_args()

    num_samples = args.num_samples

    output_path = Path(
        "data/processed/adversarial/"
        f"cifar10/pgd_{num_samples}.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"05b_pgd_{num_samples}_metrics.json"
    )

    base_loader = get_cifar10_loader(
        batch_size=BATCH_SIZE,
        train=False,
        image_size=224,
        num_workers=0,
        normalize=False,
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

    encoder_wrapper = DinoV2Encoder(
        model_name="dinov2_vits14",
    )

    device = encoder_wrapper.device

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    head = LinearHead(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    )

    head.load_state_dict(
        checkpoint["model_state_dict"]
    )

    head.to(device)
    head.eval()

    model = DinoV2Classifier(
        encoder=encoder_wrapper.model,
        classifier=head,
    ).to(device)

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    all_clean_images = []
    all_adversarial_images = []

    all_labels = []

    all_clean_predictions = []
    all_adversarial_predictions = []

    clean_correct = 0
    adversarial_correct = 0
    total = 0

    print(
        f"Generating PGD adversarial examples "
        f"for {num_samples} images..."
    )

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        clean_predictions = get_predictions(
            model,
            images,
        )

        adversarial_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=EPSILON,
            alpha=ALPHA,
            steps=STEPS,
            random_start=True,
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

        total += labels.size(0)

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
        & (adversarial_predictions != labels)
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
        successful_attacks / originally_correct
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
            "alpha": ALPHA,
            "steps": STEPS,
        },
        output_path,
    )

    metrics = {
        "experiment": "05b_generate_pgd",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_samples": total,
        "batch_size": BATCH_SIZE,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "steps": STEPS,
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

    print()
    print("PGD results")
    print("-----------")

    print(
        f"Samples:               "
        f"{total}"
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
        f"Max L-inf perturbation:"
        f" {max_linf_perturbation:.6f}"
    )

    print(
        f"Configured epsilon:    "
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