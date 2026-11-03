"""
Experiment:
    04 Linear Probe

Objective:
    Evaluate how much CIFAR-10 class information is preserved in frozen
    DINOv2 embeddings using a simple logistic regression classifier.

Inputs:
    - Saved DINOv2 embeddings for the official CIFAR-10 training split
    - Saved DINOv2 embeddings for the official CIFAR-10 test split
    - CIFAR-10 labels

Method:
    - Train logistic regression on DINOv2 embeddings from the official
      CIFAR-10 training set.
    - Evaluate on DINOv2 embeddings from the completely separate official
      CIFAR-10 test set.
    - Do not perform any additional random train/test split.

Outputs:
    - Classification accuracy on the CIFAR-10 test set
    - Precision, recall, and F1-score by class
    - Saved metrics and classification report under results/metrics/

Research Goal:
    Quantify whether frozen DINOv2 representations contain linearly accessible
    CIFAR-10 class information under a clean train/test-separated protocol
    before studying adversarial perturbations.

Important Limitation:
    DINOv2 is used as a frozen pretrained encoder and is not fine-tuned on
    CIFAR-10.

Next Experiment:
    05A Train Differentiable Linear Head
"""

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from utils.results import (
    save_metrics,
    save_text_report,
)


TRAIN_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/train_clean_dinov2_vits14.pt"
)

TEST_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/test_clean_dinov2_vits14.pt"
)

METRICS_PATH = (
    "results/metrics/"
    "04_linear_probe_metrics.json"
)

REPORT_PATH = (
    "results/metrics/"
    "04_linear_probe_classification_report.txt"
)


def main():
    # ---------------------------------------------------------
    # 1. Load official CIFAR-10 training embeddings.
    # ---------------------------------------------------------
    train_data = torch.load(
        TRAIN_EMBEDDING_PATH,
        map_location="cpu",
    )

    x_train = (
        train_data["embeddings"]
        .float()
        .numpy()
    )

    y_train = (
        train_data["labels"]
        .long()
        .numpy()
    )

    # ---------------------------------------------------------
    # 2. Load official CIFAR-10 test embeddings.
    # ---------------------------------------------------------
    test_data = torch.load(
        TEST_EMBEDDING_PATH,
        map_location="cpu",
    )

    x_test = (
        test_data["embeddings"]
        .float()
        .numpy()
    )

    y_test = (
        test_data["labels"]
        .long()
        .numpy()
    )

    print()
    print("=" * 60)
    print("CIFAR-10 DINOv2 Linear Probe")
    print("=" * 60)

    print()
    print("Training embeddings:", x_train.shape)
    print("Training labels:    ", y_train.shape)

    print()
    print("Test embeddings:    ", x_test.shape)
    print("Test labels:         ", y_test.shape)

    # ---------------------------------------------------------
    # 3. Validate expected embedding dimensions.
    # ---------------------------------------------------------
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            "Training and test embedding dimensions do not match."
        )

    # ---------------------------------------------------------
    # 4. Train linear probe exclusively on CIFAR-10 TRAIN.
    # ---------------------------------------------------------
    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )

    print()
    print("Training linear probe on official CIFAR-10 TRAIN split...")

    classifier.fit(
        x_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 5. Evaluate exclusively on CIFAR-10 TEST.
    # ---------------------------------------------------------
    predictions = classifier.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
    )

    # ---------------------------------------------------------
    # 6. Display results.
    # ---------------------------------------------------------
    print()
    print("Linear probe results")
    print("--------------------")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print()
    print(report)

    # ---------------------------------------------------------
    # 7. Save metrics.
    # ---------------------------------------------------------
    save_metrics(
        {
            "experiment": "04_linear_probe",
            "encoder": "dinov2_vits14",
            "dataset": "cifar10",
            "training_split": (
                "official_cifar10_train"
            ),
            "evaluation_split": (
                "official_cifar10_test"
            ),
            "train_samples": int(
                x_train.shape[0]
            ),
            "test_samples": int(
                x_test.shape[0]
            ),
            "embedding_dimension": int(
                x_train.shape[1]
            ),
            "classifier": (
                "logistic_regression"
            ),
            "solver": "lbfgs",
            "max_iter": 2000,
            "accuracy": float(
                accuracy
            ),
        },
        METRICS_PATH,
    )

    # ---------------------------------------------------------
    # 8. Save classification report.
    # ---------------------------------------------------------
    save_text_report(
        report,
        REPORT_PATH,
    )

    print()
    print(
        f"Saved metrics to {METRICS_PATH}"
    )

    print(
        f"Saved classification report to "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()