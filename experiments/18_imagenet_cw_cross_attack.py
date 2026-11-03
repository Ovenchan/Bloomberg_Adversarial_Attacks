"""
Experiment 18: ImageNet CW Cross-Attack Evaluation

Objective
---------
Evaluate whether the PGD-trained adversarial detector from Experiment 17
generalizes to Carlini-Wagner (CW) adversarial examples on the same
five-class ImageNet subset.

Inputs
------
- data/raw/imagenet/meta/imagenet5_final_val_50_per_class.txt
- data/raw/imagenet/val/<WNID>/*.JPEG
- data/processed/models/dinov2_vits14_imagenet5_linear_head.pt
- data/processed/models/imagenet5_pgd_logistic_detector.pkl
- data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt

Method
------
Generate CW L2 adversarial examples against the differentiable
DINOv2 + linear-head classifier using:

    c = 1.0
    kappa = 0
    steps = 100
    learning rate = 0.01

The PGD detector is not retrained or recalibrated. Cross-attack
performance is measured on the exact 50 held-out identities selected
in Experiment 17.

Representation analysis compares clean-to-CW displacement with the
previously measured clean-to-PGD displacement.

Outputs
-------
- data/processed/adversarial/imagenet5/cw.pt
- data/processed/embeddings/imagenet5/cw_dinov2_vits14.pt
- results/metrics/18_imagenet_cw_cross_attack.json

Research Goal
-------------
Determine whether the poor PGD-to-CW detector transfer observed on
CIFAR-10 also appears on genuine ImageNet-source photographs, and
whether cross-attack behavior is associated with representation-space
geometry.

Important Limitation
--------------------
This experiment uses a five-class ImageNet subset and a non-adaptive CW
attack. CW attacks the classifier objective but does not explicitly
optimize against the adversarial detector.

Next Experiment
---------------
None. This is the final ImageNet experiment for the current phase.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torchattacks
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

PGD_EMBEDDING_PATH = Path(
    "data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt"
)

DETECTOR_PATH = Path(
    "data/processed/models/imagenet5_pgd_logistic_detector.pkl"
)

ADV_OUTPUT = Path(
    "data/processed/adversarial/imagenet5/cw.pt"
)

EMBEDDING_OUTPUT = Path(
    "data/processed/embeddings/imagenet5/cw_dinov2_vits14.pt"
)

RESULT_PATH = Path(
    "results/metrics/18_imagenet_cw_cross_attack.json"
)

CW_C = 1.0
CW_KAPPA = 0.0
CW_STEPS = 100
CW_LR = 0.01

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

            path = (
                image_root
                / wnid
                / f"{image_id}.JPEG"
            )

            if not path.exists():
                raise FileNotFoundError(path)

            self.samples.append(
                (
                    path,
                    CLASSES[wnid],
                    image_id,
                )
            )

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
# Differentiable DINOv2 classifier
# ---------------------------------------------------------------------------

class DinoClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.head = head

        mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "mean",
            mean,
        )

        self.register_buffer(
            "std",
            std,
        )

    def forward(self, images):
        normalized = (
            images - self.mean
        ) / self.std

        embeddings = self.encoder(
            normalized
        )

        return self.head(
            embeddings
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity_per_row(
    x: torch.Tensor,
    y: torch.Tensor,
):
    return torch.nn.functional.cosine_similarity(
        x,
        y,
        dim=1,
    )


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
    print("Experiment 18: ImageNet CW Cross-Attack Evaluation")
    print("=" * 72)
    print()
    print(f"Device: {device}")
    print(f"CW c: {CW_C}")
    print(f"CW kappa: {CW_KAPPA}")
    print(f"CW steps: {CW_STEPS}")
    print(f"CW learning rate: {CW_LR}")

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

    print(f"Images: {len(dataset)}")

    # -----------------------------------------------------------------------
    # Load DINOv2
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
    # CW attack
    # -----------------------------------------------------------------------

    attack = torchattacks.CW(
        model,
        c=CW_C,
        kappa=CW_KAPPA,
        steps=CW_STEPS,
        lr=CW_LR,
    )

    clean_images_all = []
    cw_images_all = []

    clean_embeddings_all = []
    cw_embeddings_all = []

    labels_all = []
    clean_predictions_all = []
    cw_predictions_all = []

    image_ids_all = []

    for images, labels, image_ids in tqdm(
        loader,
        desc="Generating CW",
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

        cw_images = attack(
            images,
            labels,
        )

        with torch.no_grad():
            cw_logits = model(
                cw_images
            )

            cw_predictions = (
                cw_logits.argmax(dim=1)
            )

            normalized_cw = (
                cw_images - model.mean
            ) / model.std

            cw_embeddings = encoder(
                normalized_cw
            )

        clean_images_all.append(
            images.cpu()
        )

        cw_images_all.append(
            cw_images.cpu()
        )

        clean_embeddings_all.append(
            clean_embeddings.cpu()
        )

        cw_embeddings_all.append(
            cw_embeddings.cpu()
        )

        labels_all.append(
            labels.cpu()
        )

        clean_predictions_all.append(
            clean_predictions.cpu()
        )

        cw_predictions_all.append(
            cw_predictions.cpu()
        )

        image_ids_all.extend(
            list(image_ids)
        )

    clean_images = torch.cat(
        clean_images_all
    )

    cw_images = torch.cat(
        cw_images_all
    )

    clean_embeddings = torch.cat(
        clean_embeddings_all
    )

    cw_embeddings = torch.cat(
        cw_embeddings_all
    )

    labels = torch.cat(
        labels_all
    )

    clean_predictions = torch.cat(
        clean_predictions_all
    )

    cw_predictions = torch.cat(
        cw_predictions_all
    )

    # -----------------------------------------------------------------------
    # Attack metrics
    # -----------------------------------------------------------------------

    clean_correct = (
        clean_predictions == labels
    )

    cw_correct = (
        cw_predictions == labels
    )

    originally_correct = int(
        clean_correct.sum().item()
    )

    successful_mask = (
        clean_correct
        & (cw_predictions != labels)
    )

    successful_attacks = int(
        successful_mask.sum().item()
    )

    clean_accuracy = float(
        clean_correct.float().mean().item()
    )

    cw_accuracy = float(
        cw_correct.float().mean().item()
    )

    attack_success_rate = (
        successful_attacks / originally_correct
        if originally_correct > 0
        else 0.0
    )

    perturbation = (
        cw_images - clean_images
    )

    l2_pixel = torch.linalg.vector_norm(
        perturbation.flatten(1),
        dim=1,
    )

    linf_pixel = (
        perturbation
        .abs()
        .flatten(1)
        .max(dim=1)
        .values
    )

    mean_abs_pixel = (
        perturbation
        .abs()
        .flatten(1)
        .mean(dim=1)
    )

    # -----------------------------------------------------------------------
    # Representation geometry
    # -----------------------------------------------------------------------

    cw_displacement = (
        cw_embeddings - clean_embeddings
    )

    cw_cosine = (
        cosine_similarity_per_row(
            clean_embeddings,
            cw_embeddings,
        )
    )

    cw_l2 = torch.linalg.vector_norm(
        cw_displacement,
        dim=1,
    )

    clean_norm = torch.linalg.vector_norm(
        clean_embeddings,
        dim=1,
    )

    cw_relative_l2 = (
        cw_l2
        / clean_norm.clamp_min(1e-12)
    )

    # -----------------------------------------------------------------------
    # Load previous PGD displacement
    # -----------------------------------------------------------------------

    pgd_data = torch.load(
        PGD_EMBEDDING_PATH,
        map_location="cpu",
        weights_only=False,
    )

    pgd_clean = (
        pgd_data["clean_embeddings"]
        .float()
    )

    pgd_embeddings = (
        pgd_data["adversarial_embeddings"]
        .float()
    )

    pgd_image_ids = list(
        pgd_data["image_ids"]
    )

    if pgd_image_ids != image_ids_all:
        raise RuntimeError(
            "PGD and CW image identity ordering differs."
        )

    pgd_displacement = (
        pgd_embeddings - pgd_clean
    )

    displacement_alignment = (
        cosine_similarity_per_row(
            pgd_displacement,
            cw_displacement,
        )
    )

    # -----------------------------------------------------------------------
    # Frozen PGD detector
    # -----------------------------------------------------------------------

    with DETECTOR_PATH.open("rb") as f:
        detector_data = pickle.load(f)

    scaler = detector_data["scaler"]
    detector = detector_data["detector"]

    test_indices = np.array(
        detector_data["test_indices"],
        dtype=int,
    )

    expected_test_ids = list(
        detector_data["test_image_ids"]
    )

    actual_test_ids = [
        image_ids_all[i]
        for i in test_indices
    ]

    if actual_test_ids != expected_test_ids:
        raise RuntimeError(
            "Held-out detector identity ordering mismatch."
        )

    clean_np = clean_embeddings.numpy()
    cw_np = cw_embeddings.numpy()

    clean_test_scaled = scaler.transform(
        clean_np[test_indices]
    )

    cw_test_scaled = scaler.transform(
        cw_np[test_indices]
    )

    clean_probabilities = (
        detector.predict_proba(
            clean_test_scaled
        )[:, 1]
    )

    cw_probabilities = (
        detector.predict_proba(
            cw_test_scaled
        )[:, 1]
    )

    clean_predictions_detector = (
        clean_probabilities >= 0.5
    ).astype(int)

    cw_predictions_detector = (
        cw_probabilities >= 0.5
    ).astype(int)

    detector_accuracy = (
        (
            np.sum(
                clean_predictions_detector == 0
            )
            + np.sum(
                cw_predictions_detector == 1
            )
        )
        / (2 * len(test_indices))
    )

    cw_recall = float(
        np.mean(
            cw_predictions_detector == 1
        )
    )

    # AUC across held-out clean and CW.
    detector_labels = np.concatenate(
        [
            np.zeros(len(test_indices)),
            np.ones(len(test_indices)),
        ]
    )

    detector_scores = np.concatenate(
        [
            clean_probabilities,
            cw_probabilities,
        ]
    )

    from sklearn.metrics import roc_auc_score

    detector_auc = roc_auc_score(
        detector_labels,
        detector_scores,
    )

    confusion = np.array(
        [
            [
                int(
                    np.sum(
                        clean_predictions_detector == 0
                    )
                ),
                int(
                    np.sum(
                        clean_predictions_detector == 1
                    )
                ),
            ],
            [
                int(
                    np.sum(
                        cw_predictions_detector == 0
                    )
                ),
                int(
                    np.sum(
                        cw_predictions_detector == 1
                    )
                ),
            ],
        ]
    )

    # -----------------------------------------------------------------------
    # Detector decision-direction movement
    # -----------------------------------------------------------------------

    clean_decision = detector.decision_function(
        clean_test_scaled
    )

    cw_decision = detector.decision_function(
        cw_test_scaled
    )

    detector_decision_shift = (
        cw_decision - clean_decision
    )

    # PGD decision movement on the same held-out identities.
    pgd_test_scaled = scaler.transform(
        pgd_embeddings.numpy()[test_indices]
    )

    pgd_decision = detector.decision_function(
        pgd_test_scaled
    )

    pgd_detector_decision_shift = (
        pgd_decision - clean_decision
    )

    # -----------------------------------------------------------------------
    # Successful-CW-only detector recall
    # -----------------------------------------------------------------------

    successful_test_mask = (
        successful_mask.numpy()[
            test_indices
        ]
    )

    num_successful_test = int(
        successful_test_mask.sum()
    )

    if num_successful_test > 0:
        successful_cw_recall = float(
            np.mean(
                cw_predictions_detector[
                    successful_test_mask
                ] == 1
            )
        )
    else:
        successful_cw_recall = None

    # -----------------------------------------------------------------------
    # Save outputs
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
            "cw_images": cw_images,
            "labels": labels,
            "clean_predictions":
                clean_predictions,
            "cw_predictions":
                cw_predictions,
            "image_ids": image_ids_all,
            "c": CW_C,
            "kappa": CW_KAPPA,
            "steps": CW_STEPS,
            "learning_rate": CW_LR,
        },
        ADV_OUTPUT,
    )

    torch.save(
        {
            "clean_embeddings":
                clean_embeddings,
            "cw_embeddings":
                cw_embeddings,
            "labels": labels,
            "image_ids": image_ids_all,
            "clean_predictions":
                clean_predictions,
            "cw_predictions":
                cw_predictions,
        },
        EMBEDDING_OUTPUT,
    )

    results = {
        "experiment":
            "18_imagenet_cw_cross_attack",
        "attack": {
            "c": CW_C,
            "kappa": CW_KAPPA,
            "steps": CW_STEPS,
            "learning_rate": CW_LR,
            "num_images": len(labels),
            "clean_accuracy":
                clean_accuracy,
            "cw_accuracy":
                cw_accuracy,
            "originally_correct":
                originally_correct,
            "successful_attacks":
                successful_attacks,
            "attack_success_rate":
                attack_success_rate,
            "mean_pixel_l2":
                float(l2_pixel.mean().item()),
            "mean_pixel_linf":
                float(linf_pixel.mean().item()),
            "max_pixel_linf":
                float(linf_pixel.max().item()),
            "mean_absolute_pixel_change":
                float(
                    mean_abs_pixel.mean().item()
                ),
        },
        "geometry": {
            "mean_clean_cw_cosine_similarity":
                float(cw_cosine.mean().item()),
            "mean_l2_displacement":
                float(cw_l2.mean().item()),
            "mean_relative_l2_displacement":
                float(
                    cw_relative_l2.mean().item()
                ),
            "mean_pgd_cw_displacement_alignment":
                float(
                    displacement_alignment
                    .mean()
                    .item()
                ),
            "median_pgd_cw_displacement_alignment":
                float(
                    displacement_alignment
                    .median()
                    .item()
                ),
        },
        "frozen_pgd_detector": {
            "num_heldout_identities":
                len(test_indices),
            "accuracy":
                float(detector_accuracy),
            "roc_auc":
                float(detector_auc),
            "cw_recall":
                cw_recall,
            "confusion_matrix":
                confusion.tolist(),
            "clean_score_mean":
                float(
                    clean_probabilities.mean()
                ),
            "clean_score_median":
                float(
                    np.median(
                        clean_probabilities
                    )
                ),
            "cw_score_mean":
                float(
                    cw_probabilities.mean()
                ),
            "cw_score_median":
                float(
                    np.median(
                        cw_probabilities
                    )
                ),
            "mean_cw_detector_decision_shift":
                float(
                    detector_decision_shift.mean()
                ),
            "mean_pgd_detector_decision_shift":
                float(
                    pgd_detector_decision_shift.mean()
                ),
            "successful_cw_examples_in_test":
                num_successful_test,
            "successful_cw_recall":
                successful_cw_recall,
        },
    }

    with RESULT_PATH.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("CW ATTACK")
    print("=" * 72)
    print(
        f"Clean accuracy:       "
        f"{clean_accuracy:.4f}"
    )
    print(
        f"CW accuracy:          "
        f"{cw_accuracy:.4f}"
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
        f"Mean pixel L2:        "
        f"{l2_pixel.mean().item():.4f}"
    )
    print(
        f"Mean pixel L-inf:     "
        f"{linf_pixel.mean().item():.6f}"
    )
    print(
        f"Max pixel L-inf:      "
        f"{linf_pixel.max().item():.6f}"
    )
    print(
        f"Mean absolute change: "
        f"{mean_abs_pixel.mean().item():.6f}"
    )

    print()
    print("=" * 72)
    print("CW REPRESENTATION GEOMETRY")
    print("=" * 72)
    print(
        f"Clean-CW cosine similarity: "
        f"{cw_cosine.mean().item():.4f}"
    )
    print(
        f"Mean L2 displacement:        "
        f"{cw_l2.mean().item():.4f}"
    )
    print(
        f"Mean relative L2:            "
        f"{cw_relative_l2.mean().item():.4f}"
    )
    print(
        f"Mean PGD-CW alignment:       "
        f"{displacement_alignment.mean().item():.4f}"
    )

    print()
    print("=" * 72)
    print("FROZEN PGD DETECTOR -> CW")
    print("=" * 72)
    print(
        f"Accuracy:            "
        f"{detector_accuracy:.4f}"
    )
    print(
        f"ROC AUC:             "
        f"{detector_auc:.4f}"
    )
    print(
        f"CW recall:           "
        f"{cw_recall:.4f}"
    )
    print("Confusion matrix:")
    print(confusion)

    print()
    print(
        f"Clean score mean/median: "
        f"{clean_probabilities.mean():.6f} / "
        f"{np.median(clean_probabilities):.6f}"
    )
    print(
        f"CW score mean/median:    "
        f"{cw_probabilities.mean():.6f} / "
        f"{np.median(cw_probabilities):.6f}"
    )

    print()
    print(
        f"Mean PGD detector-direction shift: "
        f"{pgd_detector_decision_shift.mean():.4f}"
    )
    print(
        f"Mean CW detector-direction shift:  "
        f"{detector_decision_shift.mean():.4f}"
    )

    if successful_cw_recall is not None:
        print(
            f"Detector recall on successful CW:  "
            f"{successful_cw_recall:.4f} "
            f"({num_successful_test} successful "
            f"held-out attacks)"
        )

    print()
    print(f"CW images:     {ADV_OUTPUT}")
    print(f"CW embeddings: {EMBEDDING_OUTPUT}")
    print(f"Metrics:       {RESULT_PATH}")


if __name__ == "__main__":
    main()
