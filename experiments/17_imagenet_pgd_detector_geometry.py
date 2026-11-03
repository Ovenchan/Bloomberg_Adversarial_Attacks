"""
Experiment 17: ImageNet PGD Detector and Representation Geometry

Objective
---------
Measure the clean-to-PGD representation shift on the five-class ImageNet
subset and test whether a lightweight linear detector can distinguish
clean from PGD DINOv2 representations.

Inputs
------
- data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt

Method
------
The 250 official ImageNet validation identities are split deterministically
by original image ID:

    - 200 identities -> detector training
    - 50 identities  -> held-out detector evaluation

Each clean/adversarial pair remains in the same partition.

A logistic regression detector is trained on frozen 384-dimensional
DINOv2 representations. Representation geometry is measured using paired
clean and PGD embeddings.

Outputs
-------
- data/processed/models/imagenet5_pgd_logistic_detector.pkl
- results/metrics/17_imagenet_pgd_detector_geometry.json

Research Goal
-------------
Test whether the strong PGD representation shift and linear detectability
observed on CIFAR-10 also appear on genuine ImageNet-source photographs.

Important Limitation
--------------------
The detector is evaluated on a five-class ImageNet subset and against the
same PGD attack family used to construct its training examples. Cross-attack
generalization is evaluated separately.

Next Experiment
---------------
Evaluate the frozen PGD-trained detector against CW adversarial examples
and compare CW representation displacement with the PGD direction.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path(
    "data/processed/embeddings/imagenet5/pgd_dinov2_vits14.pt"
)

MODEL_PATH = Path(
    "data/processed/models/imagenet5_pgd_logistic_detector.pkl"
)

RESULT_PATH = Path(
    "results/metrics/17_imagenet_pgd_detector_geometry.json"
)

TRAIN_IDENTITIES = 200
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    similarity = torch.nn.functional.cosine_similarity(
        x,
        y,
        dim=1,
    )
    return float(similarity.mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Experiment 17: ImageNet PGD Detector + Geometry")
    print("=" * 72)

    data = torch.load(
        INPUT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    clean = data["clean_embeddings"].float()
    pgd = data["adversarial_embeddings"].float()
    labels = data["labels"].long()
    image_ids = list(data["image_ids"])

    n = len(image_ids)

    if n != 250:
        raise RuntimeError(
            f"Expected 250 image identities, found {n}."
        )

    if clean.shape != pgd.shape:
        raise RuntimeError(
            "Clean and PGD embedding shapes differ."
        )

    if clean.shape != (250, 384):
        raise RuntimeError(
            f"Unexpected embedding shape: {tuple(clean.shape)}"
        )

    if len(set(image_ids)) != n:
        raise RuntimeError(
            "Duplicate image IDs detected."
        )

    # -----------------------------------------------------------------------
    # Identity-level split
    # -----------------------------------------------------------------------

    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_idx = indices[:TRAIN_IDENTITIES]
    test_idx = indices[TRAIN_IDENTITIES:]

    if set(train_idx).intersection(set(test_idx)):
        raise RuntimeError(
            "Detector train/test identity overlap."
        )

    print()
    print(f"Detector TRAIN identities: {len(train_idx)}")
    print(f"Held-out identities:       {len(test_idx)}")

    clean_np = clean.numpy()
    pgd_np = pgd.numpy()

    x_train = np.concatenate(
        [
            clean_np[train_idx],
            pgd_np[train_idx],
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            np.zeros(len(train_idx)),
            np.ones(len(train_idx)),
        ]
    )

    x_test = np.concatenate(
        [
            clean_np[test_idx],
            pgd_np[test_idx],
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(len(test_idx)),
            np.ones(len(test_idx)),
        ]
    )

    # -----------------------------------------------------------------------
    # Standardization + detector
    # -----------------------------------------------------------------------

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(
        x_train
    )

    x_test_scaled = scaler.transform(
        x_test
    )

    detector = LogisticRegression(
        max_iter=5000,
        random_state=SEED,
    )

    detector.fit(
        x_train_scaled,
        y_train,
    )

    probabilities = detector.predict_proba(
        x_test_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    clean_scores = probabilities[
        :len(test_idx)
    ]

    pgd_scores = probabilities[
        len(test_idx):
    ]

    # -----------------------------------------------------------------------
    # Representation geometry
    # -----------------------------------------------------------------------

    displacement = pgd - clean

    cosine_similarity = mean_cosine_similarity(
        clean,
        pgd,
    )

    l2_per_image = torch.linalg.vector_norm(
        displacement,
        dim=1,
    )

    clean_norm = torch.linalg.vector_norm(
        clean,
        dim=1,
    )

    pgd_norm = torch.linalg.vector_norm(
        pgd,
        dim=1,
    )

    relative_l2 = (
        l2_per_image
        / clean_norm.clamp_min(1e-12)
    )

    absolute_norm_change = (
        pgd_norm - clean_norm
    ).abs()

    # Mean PGD displacement direction.
    mean_displacement = displacement.mean(
        dim=0
    )

    mean_displacement_norm = (
        torch.linalg.vector_norm(
            mean_displacement
        ).item()
    )

    geometry = {
        "mean_clean_adversarial_cosine_similarity":
            cosine_similarity,
        "mean_l2_displacement":
            float(l2_per_image.mean().item()),
        "median_l2_displacement":
            float(l2_per_image.median().item()),
        "mean_relative_l2_displacement":
            float(relative_l2.mean().item()),
        "mean_clean_embedding_norm":
            float(clean_norm.mean().item()),
        "mean_pgd_embedding_norm":
            float(pgd_norm.mean().item()),
        "mean_absolute_embedding_norm_change":
            float(absolute_norm_change.mean().item()),
        "mean_displacement_vector_norm":
            float(mean_displacement_norm),
    }

    # -----------------------------------------------------------------------
    # Save detector
    # -----------------------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "detector": detector,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_image_ids": [
                    image_ids[i]
                    for i in train_idx
                ],
                "test_image_ids": [
                    image_ids[i]
                    for i in test_idx
                ],
            },
            f,
        )

    # -----------------------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------------------

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "experiment":
            "17_imagenet_pgd_detector_geometry",
        "num_identities": n,
        "detector_train_identities":
            len(train_idx),
        "detector_test_identities":
            len(test_idx),
        "identity_overlap": 0,
        "detector": {
            "accuracy": float(accuracy),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "clean_score_mean":
                float(clean_scores.mean()),
            "clean_score_median":
                float(np.median(clean_scores)),
            "clean_score_max":
                float(clean_scores.max()),
            "pgd_score_mean":
                float(pgd_scores.mean()),
            "pgd_score_median":
                float(np.median(pgd_scores)),
            "pgd_score_min":
                float(pgd_scores.min()),
            "adversarial_recall":
                float(cm[1, 1] / cm[1].sum()),
        },
        "geometry": geometry,
        "model_path": str(MODEL_PATH),
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
    print("Detector")
    print("-" * 72)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"ROC AUC:            {auc:.4f}")
    print(
        f"Adversarial recall: "
        f"{cm[1, 1] / cm[1].sum():.4f}"
    )
    print("Confusion matrix:")
    print(cm)

    print()
    print("Held-out detector scores")
    print("-" * 72)
    print(
        f"Clean mean / median / max: "
        f"{clean_scores.mean():.6f} / "
        f"{np.median(clean_scores):.6f} / "
        f"{clean_scores.max():.6f}"
    )
    print(
        f"PGD mean / median / min:   "
        f"{pgd_scores.mean():.6f} / "
        f"{np.median(pgd_scores):.6f} / "
        f"{pgd_scores.min():.6f}"
    )

    print()
    print("Representation geometry")
    print("-" * 72)
    print(
        f"Clean-PGD cosine similarity: "
        f"{cosine_similarity:.4f}"
    )
    print(
        f"Mean L2 displacement:         "
        f"{l2_per_image.mean().item():.4f}"
    )
    print(
        f"Mean relative L2:             "
        f"{relative_l2.mean().item():.4f}"
    )
    print(
        f"Mean clean embedding norm:    "
        f"{clean_norm.mean().item():.4f}"
    )
    print(
        f"Mean PGD embedding norm:      "
        f"{pgd_norm.mean().item():.4f}"
    )
    print(
        f"Mean absolute norm change:    "
        f"{absolute_norm_change.mean().item():.4f}"
    )

    print()
    print(f"Detector: {MODEL_PATH}")
    print(f"Metrics:  {RESULT_PATH}")


if __name__ == "__main__":
    main()
