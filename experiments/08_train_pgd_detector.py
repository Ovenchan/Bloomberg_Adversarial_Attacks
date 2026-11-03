"""
Experiment:
    08 Train PGD Representation-Space Detector

Objective:
    Test whether clean and PGD-adversarial CIFAR-10 inputs can be
    distinguished using only their frozen DINOv2 embeddings.

Inputs:
    - Paired clean/adversarial DINOv2 embeddings from Experiment 06
    - Number of image pairs passed through --num-samples

Method:
    - Construct a binary classification dataset:
          clean embedding -> 0
          PGD embedding   -> 1
    - Split by original image index so the clean and adversarial versions
      of the same image always remain in the same partition.
    - Standardize embeddings using training data only.
    - Train a logistic-regression detector.
    - Evaluate on held-out image pairs.

Outputs:
    - Detector accuracy
    - Precision, recall, and F1
    - ROC-AUC
    - Confusion matrix
    - ROC curve figure
    - Confusion matrix figure
    - Saved fitted detector
    - Saved experiment metrics

Research Goal:
    Determine whether the representation shift observed under PGD contains
    a simple, linearly accessible signal that separates clean inputs from
    adversarial inputs.

Important Limitation:
    This experiment evaluates detection of the same attack family used
    during training (PGD). It does not establish generalization to unseen
    adversarial attacks.

Next:
    Evaluate the frozen PGD-trained detector on APGD, CW, transfer attacks,
    and other attack families without retraining.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.results import save_metrics


RANDOM_STATE = 42
TEST_SIZE = 0.20


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of clean/PGD image pairs to use.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = args.num_samples

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    input_path = (
        "data/processed/embeddings/"
        f"cifar10/pgd_{num_samples}_dinov2_vits14.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"08_pgd_detector_{num_samples}_metrics.json"
    )

    model_path = Path(
        "data/processed/models/"
        f"pgd_detector_{num_samples}_logistic.pkl"
    )

    figure_dir = Path(
        "results/figures/"
        "08_pgd_detector"
    )

    roc_path = (
        figure_dir
        / f"roc_curve_{num_samples}.png"
    )

    confusion_matrix_path = (
        figure_dir
        / f"confusion_matrix_{num_samples}.png"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Load paired embeddings.
    # ---------------------------------------------------------
    data = torch.load(
        input_path,
        map_location="cpu",
    )

    clean_embeddings = (
        data["clean_embeddings"]
        .float()
        .numpy()
    )

    adversarial_embeddings = (
        data["adversarial_embeddings"]
        .float()
        .numpy()
    )

    actual_samples = clean_embeddings.shape[0]

    if actual_samples != num_samples:
        raise ValueError(
            f"Requested {num_samples} pairs, "
            f"but input file contains {actual_samples}."
        )

    if clean_embeddings.shape != adversarial_embeddings.shape:
        raise ValueError(
            "Clean and adversarial embedding shapes do not match."
        )

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "PGD embeddings:",
        adversarial_embeddings.shape,
    )

    # ---------------------------------------------------------
    # 2. Split ORIGINAL IMAGE INDICES.
    #
    # This prevents the clean embedding of an image from being
    # placed in training while its adversarial counterpart is
    # placed in testing.
    # ---------------------------------------------------------
    image_indices = np.arange(
        actual_samples
    )

    train_indices, test_indices = train_test_split(
        image_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # ---------------------------------------------------------
    # 3. Construct binary training dataset.
    #
    # Label:
    #     0 = clean
    #     1 = PGD
    # ---------------------------------------------------------
    X_train = np.concatenate(
        [
            clean_embeddings[train_indices],
            adversarial_embeddings[train_indices],
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            np.zeros(
                len(train_indices),
                dtype=np.int64,
            ),
            np.ones(
                len(train_indices),
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    # ---------------------------------------------------------
    # 4. Construct held-out test dataset.
    # ---------------------------------------------------------
    X_test = np.concatenate(
        [
            clean_embeddings[test_indices],
            adversarial_embeddings[test_indices],
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(
                len(test_indices),
                dtype=np.int64,
            ),
            np.ones(
                len(test_indices),
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    print()
    print(
        f"Training image pairs: {len(train_indices)}"
    )

    print(
        f"Test image pairs:     {len(test_indices)}"
    )

    print(
        f"Training embeddings:  {len(X_train)}"
    )

    print(
        f"Test embeddings:      {len(X_test)}"
    )

    # ---------------------------------------------------------
    # 5. Train logistic-regression detector.
    #
    # StandardScaler is fit only on the training partition because
    # it is part of the sklearn pipeline. This avoids test leakage.
    # ---------------------------------------------------------
    detector = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print()
    print(
        "Training logistic-regression detector..."
    )

    detector.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 6. Evaluate on held-out image pairs.
    # ---------------------------------------------------------
    predictions = detector.predict(
        X_test
    )

    probabilities = detector.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "PGD",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "PGD",
        ],
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 7. Extract useful detector statistics.
    # ---------------------------------------------------------
    clean_precision = report_dict[
        "clean"
    ]["precision"]

    clean_recall = report_dict[
        "clean"
    ]["recall"]

    clean_f1 = report_dict[
        "clean"
    ]["f1-score"]

    pgd_precision = report_dict[
        "PGD"
    ]["precision"]

    pgd_recall = report_dict[
        "PGD"
    ]["recall"]

    pgd_f1 = report_dict[
        "PGD"
    ]["f1-score"]

    # ---------------------------------------------------------
    # 8. Generate ROC curve.
    # ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    # display = RocCurveDisplay(
    #     fpr=fpr,
    #     tpr=tpr,
    #     roc_auc=roc_auc,
    #     estimator_name=(
    #         "DINOv2 + Logistic Detector"
    #     ),
    # )

    display = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
    )

    display.plot()

    plt.title(
        "PGD Detection from DINOv2 Embeddings"
    )

    plt.tight_layout()

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # 9. Generate confusion matrix figure.
    # ---------------------------------------------------------
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Clean",
            "PGD",
        ],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Clean vs PGD Detection"
    )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # 10. Save metrics.
    # ---------------------------------------------------------
    metrics = {
        "experiment": "08_train_pgd_detector",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "attack": "pgd",
        "num_image_pairs": int(
            actual_samples
        ),
        "train_image_pairs": int(
            len(train_indices)
        ),
        "test_image_pairs": int(
            len(test_indices)
        ),
        "train_embeddings": int(
            len(X_train)
        ),
        "test_embeddings": int(
            len(X_test)
        ),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": float(
            accuracy
        ),
        "roc_auc": float(
            roc_auc
        ),
        "clean_precision": float(
            clean_precision
        ),
        "clean_recall": float(
            clean_recall
        ),
        "clean_f1": float(
            clean_f1
        ),
        "pgd_precision": float(
            pgd_precision
        ),
        "pgd_recall": float(
            pgd_recall
        ),
        "pgd_f1": float(
            pgd_f1
        ),
        "confusion_matrix": (
            cm.tolist()
        ),
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ---------------------------------------------------------
    # 11. Save fitted detector.
    #
    # Saving the full sklearn pipeline preserves both the fitted
    # StandardScaler and logistic-regression classifier. This is
    # important for later unseen-attack evaluation.
    # ---------------------------------------------------------
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        model_path,
        "wb",
    ) as file:
        pickle.dump(
            detector,
            file,
        )

    # ---------------------------------------------------------
    # 12. Print results.
    # ---------------------------------------------------------
    print()
    print("PGD detector results")
    print("--------------------")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print()
    print("Confusion matrix")
    print("----------------")

    print(cm)

    print()
    print(report_text)

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved detector to "
        f"{model_path}"
    )

    print(
        f"Saved ROC curve to "
        f"{roc_path}"
    )

    print(
        f"Saved confusion matrix to "
        f"{confusion_matrix_path}"
    )


if __name__ == "__main__":
    main()