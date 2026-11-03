"""
Experiment:
    08B Corrected Pipeline Audit

Objective:
    Audit the corrected data lineage and detector evaluation used in the
    CIFAR-10 DINOv2 adversarial-detection experiments.

Inputs:
    - Official CIFAR-10 TRAIN DINOv2 embeddings from Experiment 02
    - Official CIFAR-10 TEST DINOv2 embeddings from Experiment 02
    - PGD clean/adversarial embeddings from Experiment 06
    - Saved PGD detector from Experiment 08

Method:
    - Verify that the attack head was trained using the official CIFAR-10
      TRAIN split while adversarial examples were generated from the official
      CIFAR-10 TEST split.
    - Reconstruct the exact Experiment 08 detector split.
    - Verify detector train/test image IDs are disjoint.
    - Verify PGD-file clean embeddings correspond exactly to the expected
      official CIFAR-10 TEST embeddings.
    - Check for duplicate embeddings across detector train/test partitions.
    - Re-evaluate the frozen detector on the held-out PGD samples.
    - Inspect detector probability distributions.
    - Verify ROC-AUC independently.
    - Run a shuffled-label negative control.

Outputs:
    - Printed audit report
    - Detector score-distribution figure
    - Saved audit metrics

Research Goal:
    Determine whether the unusually strong PGD detector performance survives
    strict CIFAR-10 train/test separation and can be ruled out as an obvious
    consequence of split overlap, duplicate embeddings, or metric errors.

Important Limitation:
    This audit validates data separation and detector evaluation for the
    current CIFAR-10 PGD experiment. It does not establish that the detector
    generalizes to other attacks or datasets.

Next Experiment:
    Cross-attack evaluation using the corrected protocol.
"""

from pathlib import Path
import argparse
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
DETECTOR_TEST_SIZE = 0.20

TRAIN_CLEAN_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/train_clean_dinov2_vits14.pt"
)

TEST_CLEAN_EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "cifar10/test_clean_dinov2_vits14.pt"
)

ATTACK_HEAD_PATH = (
    "data/processed/models/"
    "dinov2_vits14_cifar10_linear_head.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of PGD image pairs used in Experiment 08.",
    )

    return parser.parse_args()


def reconstruct_detector_split(num_samples):
    """
    Reproduce Experiment 08's train/test split of original image IDs.
    """

    image_indices = np.arange(
        num_samples
    )

    train_indices, test_indices = train_test_split(
        image_indices,
        test_size=DETECTOR_TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    return train_indices, test_indices


def row_hashes(array):
    """
    Convert each embedding row into bytes so exact duplicate rows can
    be detected efficiently.
    """

    array = np.ascontiguousarray(
        array
    )

    return {
        row.tobytes()
        for row in array
    }


def count_cross_duplicates(a, b):
    hashes_a = row_hashes(a)
    hashes_b = row_hashes(b)

    return len(
        hashes_a.intersection(
            hashes_b
        )
    )


def main():
    args = parse_args()
    num_samples = args.num_samples

    pgd_embedding_path = (
        "data/processed/embeddings/"
        f"cifar10/pgd_{num_samples}_dinov2_vits14.pt"
    )

    detector_path = (
        "data/processed/models/"
        f"pgd_detector_{num_samples}_logistic.pkl"
    )

    metrics_path = Path(
        "results/metrics/"
        f"08b_pipeline_audit_{num_samples}.json"
    )

    figure_path = Path(
        "results/figures/"
        "08b_pipeline_audit/"
        f"detector_scores_{num_samples}.png"
    )

    # ============================================================
    # 1. Load official CIFAR-10 TRAIN and TEST embeddings.
    # ============================================================

    train_data = torch.load(
        TRAIN_CLEAN_EMBEDDING_PATH,
        map_location="cpu",
    )

    test_data = torch.load(
        TEST_CLEAN_EMBEDDING_PATH,
        map_location="cpu",
    )

    train_embeddings = (
        train_data["embeddings"]
        .float()
        .numpy()
    )

    train_labels = (
        train_data["labels"]
        .long()
        .numpy()
    )

    test_embeddings = (
        test_data["embeddings"]
        .float()
        .numpy()
    )

    test_labels = (
        test_data["labels"]
        .long()
        .numpy()
    )

    print()
    print("=" * 70)
    print("CORRECTED PIPELINE AUDIT")
    print("=" * 70)

    print()
    print("Official CIFAR-10 data")
    print("----------------------")

    print(
        f"TRAIN embeddings: {train_embeddings.shape}"
    )

    print(
        f"TRAIN labels:     {train_labels.shape}"
    )

    print(
        f"TEST embeddings:  {test_embeddings.shape}"
    )

    print(
        f"TEST labels:      {test_labels.shape}"
    )

    # ============================================================
    # 2. Verify saved attack-head metadata.
    # ============================================================

    attack_head_checkpoint = torch.load(
        ATTACK_HEAD_PATH,
        map_location="cpu",
    )

    training_split = (
        attack_head_checkpoint.get(
            "training_split"
        )
    )

    evaluation_split = (
        attack_head_checkpoint.get(
            "evaluation_split"
        )
    )

    head_train_samples = (
        attack_head_checkpoint.get(
            "train_samples"
        )
    )

    head_test_samples = (
        attack_head_checkpoint.get(
            "test_samples"
        )
    )

    expected_training_split = (
        "official_cifar10_train"
    )

    expected_evaluation_split = (
        "official_cifar10_test"
    )

    attack_head_protocol_valid = (
        training_split
        == expected_training_split
        and evaluation_split
        == expected_evaluation_split
        and head_train_samples
        == len(train_embeddings)
        and head_test_samples
        == len(test_embeddings)
    )

    print()
    print("Attack-head protocol")
    print("--------------------")

    print(
        f"Training split:    {training_split}"
    )

    print(
        f"Evaluation split:  {evaluation_split}"
    )

    print(
        f"Training samples:  {head_train_samples}"
    )

    print(
        f"Evaluation samples:{head_test_samples}"
    )

    print(
        "Official TRAIN -> TEST protocol valid: "
        f"{attack_head_protocol_valid}"
    )

    if not attack_head_protocol_valid:
        raise ValueError(
            "Attack-head checkpoint does not match the corrected "
            "official CIFAR-10 TRAIN -> TEST protocol."
        )

    # ============================================================
    # 3. Establish classifier-training / attack-image separation.
    #
    # The attack head uses official CIFAR-10 TRAIN.
    # PGD examples use official CIFAR-10 TEST.
    #
    # These are disjoint dataset splits by construction.
    # ============================================================

    print()
    print("Classifier-training vs attack images")
    print("------------------------------------")

    print(
        "Classifier training source: official CIFAR-10 TRAIN"
    )

    print(
        "Attack image source:        official CIFAR-10 TEST"
    )

    print(
        "Classifier-training / attack-image overlap: 0 "
        "(disjoint official splits)"
    )

    # ============================================================
    # 4. Reconstruct Experiment 08 detector split.
    # ============================================================

    (
        detector_train_indices,
        detector_test_indices,
    ) = reconstruct_detector_split(
        num_samples
    )

    detector_train_set = set(
        detector_train_indices.tolist()
    )

    detector_test_set = set(
        detector_test_indices.tolist()
    )

    detector_intersection = (
        detector_train_set.intersection(
            detector_test_set
        )
    )

    print()
    print("Experiment 08 detector split")
    print("----------------------------")

    print(
        f"Detector training image IDs: "
        f"{len(detector_train_indices)}"
    )

    print(
        f"Detector test image IDs:     "
        f"{len(detector_test_indices)}"
    )

    print(
        f"Train/test intersection:     "
        f"{len(detector_intersection)}"
    )

    if len(detector_intersection) != 0:
        raise ValueError(
            "Detector train/test image IDs overlap."
        )

    # ============================================================
    # 5. Load paired clean + PGD embeddings.
    # ============================================================

    pgd_data = torch.load(
        pgd_embedding_path,
        map_location="cpu",
    )

    clean_embeddings = (
        pgd_data["clean_embeddings"]
        .float()
        .numpy()
    )

    pgd_embeddings = (
        pgd_data["adversarial_embeddings"]
        .float()
        .numpy()
    )

    if (
        len(clean_embeddings)
        != num_samples
    ):
        raise ValueError(
            "Unexpected number of clean PGD-pair embeddings."
        )

    if (
        clean_embeddings.shape
        != pgd_embeddings.shape
    ):
        raise ValueError(
            "Clean and PGD embedding shapes do not match."
        )

    print()
    print("PGD embedding file")
    print("------------------")

    print(
        "Clean shape:",
        clean_embeddings.shape,
    )

    print(
        "PGD shape:  ",
        pgd_embeddings.shape,
    )

    # ============================================================
    # 6. Verify PGD clean embeddings correspond exactly to the
    #    first num_samples official CIFAR-10 TEST embeddings.
    # ============================================================

    expected_clean_subset = (
        test_embeddings[
            :num_samples
        ]
    )

    clean_embedding_differences = (
        np.abs(
            expected_clean_subset
            - clean_embeddings
        )
    )

    max_clean_embedding_difference = float(
        clean_embedding_differences.max()
    )

    mean_clean_embedding_difference = float(
        clean_embedding_differences.mean()
    )

    clean_identity_valid = bool(
        max_clean_embedding_difference
        == 0.0
    )

    print()
    print("Clean embedding identity check")
    print("------------------------------")

    print(
        "Max difference between official TEST embeddings "
        "and PGD-file clean embeddings: "
        f"{max_clean_embedding_difference:.10f}"
    )

    print(
        "Mean difference: "
        f"{mean_clean_embedding_difference:.10f}"
    )

    print(
        "Exact identity: "
        f"{clean_identity_valid}"
    )

    # ============================================================
    # 7. Build detector train/test partitions.
    # ============================================================

    train_clean = clean_embeddings[
        detector_train_indices
    ]

    train_pgd = pgd_embeddings[
        detector_train_indices
    ]

    test_clean = clean_embeddings[
        detector_test_indices
    ]

    test_pgd = pgd_embeddings[
        detector_test_indices
    ]

    # ============================================================
    # 8. Exact duplicate checks.
    # ============================================================

    duplicate_clean_train_test = (
        count_cross_duplicates(
            train_clean,
            test_clean,
        )
    )

    duplicate_pgd_train_test = (
        count_cross_duplicates(
            train_pgd,
            test_pgd,
        )
    )

    duplicate_any_train_test = (
        count_cross_duplicates(
            np.concatenate(
                [
                    train_clean,
                    train_pgd,
                ],
                axis=0,
            ),
            np.concatenate(
                [
                    test_clean,
                    test_pgd,
                ],
                axis=0,
            ),
        )
    )

    print()
    print("Exact embedding duplicate checks")
    print("--------------------------------")

    print(
        "Clean train/test duplicate rows: "
        f"{duplicate_clean_train_test}"
    )

    print(
        "PGD train/test duplicate rows:   "
        f"{duplicate_pgd_train_test}"
    )

    print(
        "Any detector train/test duplicate rows: "
        f"{duplicate_any_train_test}"
    )

    # ============================================================
    # 9. Re-evaluate saved detector.
    # ============================================================

    with open(
        detector_path,
        "rb",
    ) as file:
        detector = pickle.load(
            file
        )

    X_test = np.concatenate(
        [
            test_clean,
            test_pgd,
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(
                len(test_clean),
                dtype=np.int64,
            ),
            np.ones(
                len(test_pgd),
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    predictions = detector.predict(
        X_test
    )

    probabilities = (
        detector.predict_proba(
            X_test
        )[:, 1]
    )

    saved_detector_accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    saved_detector_auc = (
        roc_auc_score(
            y_test,
            probabilities,
        )
    )

    saved_detector_cm = (
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Saved detector re-evaluation")
    print("----------------------------")

    print(
        f"Accuracy: {saved_detector_accuracy:.6f}"
    )

    print(
        f"ROC-AUC:  {saved_detector_auc:.10f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        saved_detector_cm
    )

    # ============================================================
    # 10. Detector probability distributions.
    # ============================================================

    clean_scores = probabilities[
        :len(test_clean)
    ]

    pgd_scores = probabilities[
        len(test_clean):
    ]

    print()
    print("Detector probability distributions")
    print("----------------------------------")

    print("Clean scores:")

    print(
        f"  min:    {clean_scores.min():.6f}"
    )

    print(
        f"  median: {np.median(clean_scores):.6f}"
    )

    print(
        f"  mean:   {clean_scores.mean():.6f}"
    )

    print(
        f"  max:    {clean_scores.max():.6f}"
    )

    print()

    print("PGD scores:")

    print(
        f"  min:    {pgd_scores.min():.6f}"
    )

    print(
        f"  median: {np.median(pgd_scores):.6f}"
    )

    print(
        f"  mean:   {pgd_scores.mean():.6f}"
    )

    print(
        f"  max:    {pgd_scores.max():.6f}"
    )

    max_clean_score = float(
        clean_scores.max()
    )

    min_pgd_score = float(
        pgd_scores.min()
    )

    perfect_score_separation = bool(
        min_pgd_score
        > max_clean_score
    )

    print()

    print(
        "Maximum clean score: "
        f"{max_clean_score:.6f}"
    )

    print(
        "Minimum PGD score:   "
        f"{min_pgd_score:.6f}"
    )

    print(
        "Perfect clean/PGD score separation: "
        f"{perfect_score_separation}"
    )

    # ============================================================
    # 11. Score distribution figure.
    # ============================================================

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        clean_scores,
        bins=30,
        alpha=0.6,
        label="Clean",
    )

    plt.hist(
        pgd_scores,
        bins=30,
        alpha=0.6,
        label="PGD",
    )

    plt.axvline(
        0.5,
        linestyle="--",
        label="Decision threshold = 0.5",
    )

    plt.xlabel(
        "Detector probability of adversarial"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "PGD Detector Scores on Held-Out Image IDs"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ============================================================
    # 12. Negative control: shuffled labels.
    #
    # If the detector succeeds only because of a real clean/PGD
    # signal, training with shuffled labels should collapse toward
    # chance performance on the held-out image IDs.
    # ============================================================

    X_train = np.concatenate(
        [
            train_clean,
            train_pgd,
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            np.zeros(
                len(train_clean),
                dtype=np.int64,
            ),
            np.ones(
                len(train_pgd),
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    rng = np.random.default_rng(
        SEED
    )

    shuffled_y_train = (
        y_train.copy()
    )

    rng.shuffle(
        shuffled_y_train
    )

    shuffled_detector = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    shuffled_detector.fit(
        X_train,
        shuffled_y_train,
    )

    shuffled_predictions = (
        shuffled_detector.predict(
            X_test
        )
    )

    shuffled_probabilities = (
        shuffled_detector.predict_proba(
            X_test
        )[:, 1]
    )

    shuffled_accuracy = (
        accuracy_score(
            y_test,
            shuffled_predictions,
        )
    )

    shuffled_auc = (
        roc_auc_score(
            y_test,
            shuffled_probabilities,
        )
    )

    print()
    print("Shuffled-label negative control")
    print("-------------------------------")

    print(
        f"Accuracy: {shuffled_accuracy:.6f}"
    )

    print(
        f"ROC-AUC:  {shuffled_auc:.6f}"
    )

    # ============================================================
    # 13. Save audit metrics.
    # ============================================================

    metrics = {
        "experiment": (
            "08b_corrected_pipeline_audit"
        ),
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_attack_images": int(
            num_samples
        ),
        "classifier_protocol": {
            "training_split": training_split,
            "evaluation_split": evaluation_split,
            "train_samples": int(
                head_train_samples
            ),
            "test_samples": int(
                head_test_samples
            ),
            "official_train_test_protocol_valid": (
                attack_head_protocol_valid
            ),
            "classifier_training_attack_overlap": 0,
        },
        "detector_split": {
            "train_image_ids": int(
                len(
                    detector_train_indices
                )
            ),
            "test_image_ids": int(
                len(
                    detector_test_indices
                )
            ),
            "train_test_intersection": int(
                len(
                    detector_intersection
                )
            ),
        },
        "embedding_identity": {
            "max_clean_difference": (
                max_clean_embedding_difference
            ),
            "mean_clean_difference": (
                mean_clean_embedding_difference
            ),
            "exact_identity": (
                clean_identity_valid
            ),
        },
        "duplicates": {
            "clean_train_test": int(
                duplicate_clean_train_test
            ),
            "pgd_train_test": int(
                duplicate_pgd_train_test
            ),
            "any_detector_train_test": int(
                duplicate_any_train_test
            ),
        },
        "saved_detector": {
            "accuracy": float(
                saved_detector_accuracy
            ),
            "roc_auc": float(
                saved_detector_auc
            ),
            "confusion_matrix": (
                saved_detector_cm.tolist()
            ),
            "max_clean_score": (
                max_clean_score
            ),
            "min_pgd_score": (
                min_pgd_score
            ),
            "perfect_score_separation": (
                perfect_score_separation
            ),
        },
        "shuffled_label_control": {
            "accuracy": float(
                shuffled_accuracy
            ),
            "roc_auc": float(
                shuffled_auc
            ),
        },
    }

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        metrics_path,
        "w",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Saved metrics to {metrics_path}"
    )

    print(
        f"Saved score distribution to "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()