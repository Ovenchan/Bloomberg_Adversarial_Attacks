"""
Experiment 16: ImageNet PGD Adversarial Evaluation

Objective
---------
Generate PGD adversarial examples against the differentiable
DINOv2 ViT-S/14 + linear classifier stack on the untouched official
ImageNet validation subset.

Inputs
------
- data/raw/imagenet/meta/imagenet5_final_val_50_per_class.txt
- data/raw/imagenet/val/<WNID>/*.JPEG
- data/processed/models/dinov2_vits14_imagenet5_linear_head.pt

Method
------
All 250 official ImageNet validation images are resized to 256 and
center-cropped to 224 x 224. PGD operates on raw image tensors in [0, 1].

ImageNet normalization is applied inside the differentiable classifier
wrapper so gradients propagate through normalization, DINOv2, and the
linear classification head.

PGD parameters:
    epsilon = 8/255
    step size = 2/255
    steps = 20

Outputs
-------
- data/processed/adversarial/imagenet5/pgd_8_255.pt
- data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt
- results/metrics/16_imagenet_pgd.json

Research Goal
-------------
Test whether PGD remains effective against the DINOv2-based classifier
on genuine ImageNet-source photographs and produce paired clean/adversarial
representations for the final detector and geometry analysis.

Important Limitation
--------------------
This experiment evaluates a balanced five-class ImageNet subset rather
than ImageNet-1K. The attack is non-adaptive with respect to the later
adversarial detector.

Next Experiment
---------------
Measure clean-to-PGD representation geometry and train/evaluate a linear
clean-versus-PGD detector using identity-separated image pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from encoders.dinov2 import DinoV2Encoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path(
    "data/raw/imagenet/meta/imagenet5_final_val_50_per_class.txt"
)
VAL_ROOT = Path("data/raw/imagenet/val")

HEAD_PATH = Path(
    "data/processed/models/dinov2_vits14_imagenet5_linear_head.pt"
)

ADV_OUTPUT = Path(
    "data/processed/adversarial/imagenet5/pgd_8_255.pt"
)
EMBEDDING_OUTPUT = Path(
    "data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt"
)
RESULT_PATH = Path(
    "results/metrics/16_imagenet_pgd.json"
)

EPSILON = 8 / 255
ALPHA = 2 / 255
STEPS = 20

# Conservative for a 24 GB L4 while backpropagating through ViT-S/14.
BATCH_SIZE = 16

CLASSES = {
    "n01440764": 0,
    "n02106662": 1,
    "n02391049": 2,
    "n02504458": 3,
    "n02814533": 4,
}

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageNetAttackDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
    ):
        self.samples = []

        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue

            wnid, image_id = line.split(",", 1)

            path = image_root / wnid / f"{image_id}.JPEG"

            if not path.exists():
                raise FileNotFoundError(path)

            self.samples.append(
                (
                    path,
                    CLASSES[wnid],
                    image_id,
                )
            )

        # IMPORTANT:
        # No normalization here. PGD attacks raw [0,1] tensors.
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, image_id = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label, image_id


# ---------------------------------------------------------------------------
# Differentiable classifier
# ---------------------------------------------------------------------------

class DinoClassifier(nn.Module):
    def __init__(
        self,
        encoder_model: nn.Module,
        head: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder_model
        self.head = head

        mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, images):
        normalized = (
            images - self.mean
        ) / self.std

        embeddings = self.encoder(normalized)

        return self.head(embeddings)


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
):
    clean = images.detach()

    # Random start inside the epsilon ball.
    adversarial = clean + torch.empty_like(
        clean
    ).uniform_(
        -EPSILON,
        EPSILON,
    )

    adversarial = adversarial.clamp(
        0.0,
        1.0,
    )

    criterion = nn.CrossEntropyLoss()

    for _ in range(STEPS):
        adversarial.requires_grad_(True)

        logits = model(adversarial)

        loss = criterion(
            logits,
            labels,
        )

        gradient = torch.autograd.grad(
            loss,
            adversarial,
            only_inputs=True,
        )[0]

        adversarial = (
            adversarial.detach()
            + ALPHA * gradient.sign()
        )

        delta = torch.clamp(
            adversarial - clean,
            min=-EPSILON,
            max=EPSILON,
        )

        adversarial = (
            clean + delta
        ).clamp(
            0.0,
            1.0,
        ).detach()

    return adversarial


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("Experiment 16: ImageNet PGD")
    print("=" * 72)
    print()
    print(f"Device:  {device}")
    print(f"Epsilon: {EPSILON:.6f} ({EPSILON * 255:.1f}/255)")
    print(f"Alpha:   {ALPHA:.6f} ({ALPHA * 255:.1f}/255)")
    print(f"Steps:   {STEPS}")

    dataset = ImageNetAttackDataset(
        MANIFEST_PATH,
        VAL_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Images:  {len(dataset)}")

    # -----------------------------------------------------------------------
    # Load frozen DINOv2
    # -----------------------------------------------------------------------

    encoder_wrapper = DinoV2Encoder(
        model_name="dinov2_vits14",
        device=device,
    )

    encoder = encoder_wrapper.model

    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    encoder.eval()

    # -----------------------------------------------------------------------
    # Load classifier head
    # -----------------------------------------------------------------------

    checkpoint = torch.load(
        HEAD_PATH,
        map_location=device,
        weights_only=False,
    )

    head = nn.Linear(
        384,
        5,
    ).to(device)

    head.load_state_dict(
        checkpoint["state_dict"]
    )

    for parameter in head.parameters():
        parameter.requires_grad_(False)

    head.eval()

    model = DinoClassifier(
        encoder,
        head,
    ).to(device)

    model.eval()

    # -----------------------------------------------------------------------
    # Attack
    # -----------------------------------------------------------------------

    clean_images_all = []
    adversarial_images_all = []
    clean_embeddings_all = []
    adversarial_embeddings_all = []
    labels_all = []
    clean_predictions_all = []
    adversarial_predictions_all = []
    image_ids_all = []

    for images, labels, image_ids in tqdm(
        loader,
        desc="Generating PGD",
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():
            clean_logits = model(images)

            clean_predictions = (
                clean_logits.argmax(dim=1)
            )

            normalized_clean = (
                images - model.mean
            ) / model.std

            clean_embeddings = encoder(
                normalized_clean
            )

        adversarial = pgd_attack(
            model,
            images,
            labels,
        )

        with torch.no_grad():
            adversarial_logits = model(
                adversarial
            )

            adversarial_predictions = (
                adversarial_logits.argmax(dim=1)
            )

            normalized_adversarial = (
                adversarial - model.mean
            ) / model.std

            adversarial_embeddings = encoder(
                normalized_adversarial
            )

        clean_images_all.append(
            images.cpu()
        )

        adversarial_images_all.append(
            adversarial.cpu()
        )

        clean_embeddings_all.append(
            clean_embeddings.cpu()
        )

        adversarial_embeddings_all.append(
            adversarial_embeddings.cpu()
        )

        labels_all.append(
            labels.cpu()
        )

        clean_predictions_all.append(
            clean_predictions.cpu()
        )

        adversarial_predictions_all.append(
            adversarial_predictions.cpu()
        )

        image_ids_all.extend(
            list(image_ids)
        )

    clean_images = torch.cat(
        clean_images_all
    )

    adversarial_images = torch.cat(
        adversarial_images_all
    )

    clean_embeddings = torch.cat(
        clean_embeddings_all
    )

    adversarial_embeddings = torch.cat(
        adversarial_embeddings_all
    )

    labels = torch.cat(
        labels_all
    )

    clean_predictions = torch.cat(
        clean_predictions_all
    )

    adversarial_predictions = torch.cat(
        adversarial_predictions_all
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    clean_correct = (
        clean_predictions == labels
    )

    adversarial_correct = (
        adversarial_predictions == labels
    )

    originally_correct = int(
        clean_correct.sum().item()
    )

    successful_attacks = int(
        (
            clean_correct
            & (adversarial_predictions != labels)
        ).sum().item()
    )

    clean_accuracy = (
        clean_correct.float().mean().item()
    )

    adversarial_accuracy = (
        adversarial_correct.float().mean().item()
    )

    attack_success_rate = (
        successful_attacks / originally_correct
        if originally_correct > 0
        else 0.0
    )

    perturbation = (
        adversarial_images - clean_images
    )

    max_linf = (
        perturbation
        .abs()
        .flatten(1)
        .max(dim=1)
        .values
        .max()
        .item()
    )

    mean_linf = (
        perturbation
        .abs()
        .flatten(1)
        .max(dim=1)
        .values
        .mean()
        .item()
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    ADV_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EMBEDDING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.parent.mkdir(
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
            "image_ids": image_ids_all,
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "steps": STEPS,
        },
        ADV_OUTPUT,
    )

    torch.save(
        {
            "clean_embeddings": clean_embeddings,
            "adversarial_embeddings": adversarial_embeddings,
            "labels": labels,
            "image_ids": image_ids_all,
            "clean_predictions": clean_predictions,
            "adversarial_predictions": adversarial_predictions,
        },
        EMBEDDING_OUTPUT,
    )

    results = {
        "experiment": "16_imagenet_pgd",
        "num_images": len(labels),
        "epsilon": EPSILON,
        "epsilon_255": EPSILON * 255,
        "alpha": ALPHA,
        "alpha_255": ALPHA * 255,
        "steps": STEPS,
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "originally_correct": originally_correct,
        "successful_attacks": successful_attacks,
        "attack_success_rate": attack_success_rate,
        "max_linf": max_linf,
        "mean_linf": mean_linf,
        "embedding_shape": list(
            clean_embeddings.shape
        ),
    }

    with RESULT_PATH.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PGD COMPLETE")
    print("=" * 72)
    print()
    print(
        f"Clean accuracy:       "
        f"{clean_accuracy:.4f}"
    )
    print(
        f"Adversarial accuracy: "
        f"{adversarial_accuracy:.4f}"
    )
    print(
        f"Originally correct:   "
        f"{originally_correct}/{len(labels)}"
    )
    print(
        f"Successful attacks:   "
        f"{successful_attacks}/{originally_correct}"
    )
    print(
        f"Attack success rate:  "
        f"{attack_success_rate:.4f}"
    )
    print(
        f"Maximum L-inf:        "
        f"{max_linf:.6f} "
        f"({max_linf * 255:.3f}/255)"
    )
    print(
        f"Mean L-inf:           "
        f"{mean_linf:.6f} "
        f"({mean_linf * 255:.3f}/255)"
    )
    print()
    print(
        f"Embedding shape:      "
        f"{tuple(clean_embeddings.shape)}"
    )
    print()
    print(f"Adversarial data: {ADV_OUTPUT}")
    print(f"Embeddings:       {EMBEDDING_OUTPUT}")
    print(f"Metrics:          {RESULT_PATH}")


if __name__ == "__main__":
    main()
