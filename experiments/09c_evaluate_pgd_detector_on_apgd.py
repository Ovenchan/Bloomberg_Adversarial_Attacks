"""
Experiment:
    09C PGD-to-APGD Cross-Attack Evaluation

Objective:
    Evaluate whether the representation-space detector trained ONLY on
    PGD embeddings generalizes to APGD adversarial examples.

Inputs:
    - Frozen PGD-trained logistic-regression detector from Experiment 08
    - Clean/APGD DINOv2 embeddings from Experiment 09B
    - The exact held-out detector image IDs reconstructed from Experiment 08

Method:
    - Reconstruct the exact Experiment 08 detector train/test split.
    - Select only the detector-held-out image IDs.
    - Do not retrain or refit the detector or StandardScaler.
    - Construct a balanced clean/APGD evaluation set from held-out IDs only.
    - Apply the frozen PGD detector directly.
    - Measure accuracy, precision, recall, F1, and ROC-AUC.

Outputs:
    - Cross-attack detection metrics
    - Confusion matrix
    - Detector probability ranges
    - Saved metrics JSON

Research Goal:
    Determine whether the PGD detector has learned a PGD-specific feature
    signature or a representation-space pattern that transfers to APGD.

Important Limitation:
    APGD is closely related to PGD. Strong transfer therefore supports
    cross-attack generalization within related gradient-based attacks, but
    does not establish generalization to structurally different attacks.

Next Experiment:
    Evaluate the detector on a structurally different attack family.
"""

import argparse
import pickle

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils.results import save_metrics


SEED = 42
DETECTOR_TEST_SIZE = 0.20


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of clean/APGD image pairs available from Experiment 09B.",
    )

    parser.add_argument(
        "--detector-samples",
        type=int,
        default=1000,
        help="Number of image pairs used in Experiment 08.",
    )

    return parser.parse_args()


def reconstruct_detector_split(num_samples):
    """
    Reproduce Experiment 08's split of original image IDs.
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


def main():
    args = parse_args()

    num_samples = args.num_samples
    detector_samples = args.detector_samples

    embedding_path = (
        "data/processed/embeddings/"
        f"cifar10/apgd_{num_samples}_dinov2_vits14.pt"
    )

    detector_path = (
        "data/processed/models/"
        f"pgd_detector_{detector_samples}_logistic.pkl"
    )

    metrics_path = (
        "results/metrics/"
        f"09c_pgd_detector_on_apgd_{num_samples}.json"
    )

    # ============================================================
    # 1. Basic consistency check.
    # ============================================================

    if num_samples != detector_samples:
        raise ValueError(
            "For exact Experiment 08 split reconstruction, "
            "--num-samples and --detector-samples must match."
        )

    # ============================================================
    # 2. Load APGD embeddings.
    # ============================================================

    data = torch.load(
        embedding_path,
        map_location="cpu",
    )

    clean_embeddings = (
        data["clean_embeddings"]
        .float()
        .numpy()
    )

    apgd_embeddings = (
        data["adversarial_embeddings"]
        .float()
        .numpy()
    )

    if clean_embeddings.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} clean embeddings, "
            f"found {clean_embeddings.shape[0]}."
        )

    if apgd_embeddings.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} APGD embeddings, "
            f"found {apgd_embeddings.shape[0]}."
        )

    if clean_embeddings.shape != apgd_embeddings.shape:
        raise ValueError(
            "Clean and APGD embedding shapes do not match."
        )

    # ============================================================
    # 3. Verify data lineage from Experiment 09B.
    # ============================================================

    source_split = data.get(
        "source_split"
    )

    classifier_training_split = data.get(
        "classifier_training_split"
    )

    protocol_valid = (
        source_split == "official_cifar10_test"
        and classifier_training_split == "official_cifar10_train"
    )

    print()
    print("=" * 70)
    print("PGD -> APGD CROSS-ATTACK EVALUATION")
    print("=" * 70)

    print()
    print("Data protocol")
    print("-------------")

    print(
        f"APGD image source:          {source_split}"
    )

    print(
        f"Classifier training split:  {classifier_training_split}"
    )

    print(
        f"Corrected protocol valid:   {protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "APGD embedding file does not match the corrected "
            "CIFAR-10 TRAIN -> TEST protocol."
        )

    # ============================================================
    # 4. Reconstruct exact Experiment 08 held-out IDs.
    # ============================================================

    (
        detector_train_indices,
        detector_test_indices,
    ) = reconstruct_detector_split(
        detector_samples
    )

    train_set = set(
        detector_train_indices.tolist()
    )

    test_set = set(
        detector_test_indices.tolist()
    )

    intersection = (
        train_set.intersection(
            test_set
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
        f"Detector held-out image IDs: "
        f"{len(detector_test_indices)}"
    )

    print(
        f"Train/test intersection:     "
        f"{len(intersection)}"
    )

    if len(intersection) != 0:
        raise ValueError(
            "Detector train/test image IDs overlap."
        )

    # ============================================================
    # 5. Select ONLY detector-held-out clean/APGD embeddings.
    # ============================================================

    heldout_clean = clean_embeddings[
        detector_test_indices
    ]

    heldout_apgd = apgd_embeddings[
        detector_test_indices
    ]

    X_test = np.concatenate(
        [
            heldout_clean,
            heldout_apgd,
        ],
        axis=0,
    )

    y_test = np.concatenate(
        [
            np.zeros(
                len(heldout_clean),
                dtype=np.int64,
            ),
            np.ones(
                len(heldout_apgd),
                dtype=np.int64,
            ),
        ],
        axis=0,
    )

    print()
    print("Cross-attack evaluation set")
    print("---------------------------")

    print(
        f"Held-out clean embeddings: {len(heldout_clean)}"
    )

    print(
        f"Held-out APGD embeddings:  {len(heldout_apgd)}"
    )

    print(
        f"Total embeddings:          {len(X_test)}"
    )

    # ============================================================
    # 6. Load FROZEN PGD detector.
    #
    # No fit(), refit(), or scaler training occurs here.
    # ============================================================

    with open(
        detector_path,
        "rb",
    ) as file:
        detector = pickle.load(
            file
        )

    print()
    print(
        "Loaded frozen PGD detector:"
    )

    print(
        detector_path
    )

    # ============================================================
    # 7. Cross-attack evaluation.
    # ============================================================

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

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "APGD",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "APGD",
        ],
        zero_division=0,
    )

    # ============================================================
    # 8. Detector score diagnostics.
    # ============================================================

    clean_scores = probabilities[
        :len(heldout_clean)
    ]

    apgd_scores = probabilities[
        len(heldout_clean):
    ]

    max_clean_score = float(
        clean_scores.max()
    )

    min_apgd_score = float(
        apgd_scores.min()
    )

    perfect_score_separation = bool(
        min_apgd_score
        > max_clean_score
    )

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

    print("APGD scores:")

    print(
        f"  min:    {apgd_scores.min():.6f}"
    )

    print(
        f"  median: {np.median(apgd_scores):.6f}"
    )

    print(
        f"  mean:   {apgd_scores.mean():.6f}"
    )

    print(
        f"  max:    {apgd_scores.max():.6f}"
    )

    print()

    print(
        f"Maximum clean score: {max_clean_score:.6f}"
    )

    print(
        f"Minimum APGD score:  {min_apgd_score:.6f}"
    )

    print(
        "Perfect clean/APGD score separation: "
        f"{perfect_score_separation}"
    )

    # ============================================================
    # 9. Save metrics.
    # ============================================================

    metrics = {
        "experiment": (
            "09c_evaluate_pgd_detector_on_apgd"
        ),
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "detector_training_attack": "pgd",
        "evaluation_attack": "apgd",
        "source_split": source_split,
        "classifier_training_split": classifier_training_split,
        "num_available_image_pairs": int(
            num_samples
        ),
        "detector_training_image_pairs": int(
            len(detector_train_indices)
        ),
        "detector_heldout_image_pairs": int(
            len(detector_test_indices)
        ),
        "num_evaluation_embeddings": int(
            len(X_test)
        ),
        "detector_train_test_intersection": int(
            len(intersection)
        ),
        "accuracy": float(
            accuracy
        ),
        "roc_auc": float(
            roc_auc
        ),
        "clean_precision": float(
            report["clean"]["precision"]
        ),
        "clean_recall": float(
            report["clean"]["recall"]
        ),
        "clean_f1": float(
            report["clean"]["f1-score"]
        ),
        "apgd_precision": float(
            report["APGD"]["precision"]
        ),
        "apgd_recall": float(
            report["APGD"]["recall"]
        ),
        "apgd_f1": float(
            report["APGD"]["f1-score"]
        ),
        "confusion_matrix": (
            cm.tolist()
        ),
        "max_clean_score": (
            max_clean_score
        ),
        "min_apgd_score": (
            min_apgd_score
        ),
        "perfect_score_separation": (
            perfect_score_separation
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
    print("PGD -> APGD cross-attack results")
    print("--------------------------------")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print()

    print("Confusion matrix")
    print("----------------")

    print(
        cm
    )

    print()

    print(
        report_text
    )

    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()