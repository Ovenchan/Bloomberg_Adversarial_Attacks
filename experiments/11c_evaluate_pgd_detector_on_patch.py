"""
Experiment:
    11C PGD-to-Localized-Patch Cross-Attack Evaluation

Objective:
    Evaluate whether the representation-space detector trained ONLY on
    PGD embeddings generalizes to localized adversarial patch examples.

Inputs:
    - Frozen PGD-trained logistic-regression detector from Experiment 08
    - Clean/localized-patch DINOv2 embeddings from Experiment 11B
    - Exact held-out detector image IDs reconstructed from Experiment 08

Method:
    - Reconstruct the exact Experiment 08 detector train/test split.
    - Select only the detector-held-out image IDs.
    - Do not retrain or refit the detector or StandardScaler.
    - Construct a balanced clean/patch evaluation set from held-out IDs.
    - Apply the frozen PGD detector directly.
    - Measure accuracy, precision, recall, F1, and ROC-AUC.
    - Inspect detector score distributions for clean and patch examples.
    - As a secondary diagnostic, report performance on patch examples
      that actually fooled the classifier.

Outputs:
    - Primary held-out clean-vs-patch cross-attack metrics
    - Confusion matrix
    - Detector probability ranges
    - Secondary successful-attack-only diagnostics
    - Saved metrics JSON

Research Goal:
    Determine whether the PGD-trained detector captures a representation-
    space adversarial signature that transfers to a spatially localized
    attack with substantially different perturbation geometry.

Important Limitation:
    The primary result evaluates all held-out patch examples, regardless
    of whether each patch successfully changed the classifier prediction.
    Successful-attack-only results are secondary diagnostics and should
    not replace the primary cross-attack result.

Next Experiment:
    Compare representation displacement across PGD, APGD, CW, and
    localized patch attacks.
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
        help="Number of clean/patch image pairs available from Experiment 11B.",
    )

    parser.add_argument(
        "--detector-samples",
        type=int,
        default=1000,
        help="Number of image pairs used when training the saved PGD detector.",
    )

    return parser.parse_args()


def reconstruct_detector_split(num_samples):
    """
    Reproduce Experiment 08's exact split of original image IDs.
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


def summarize_scores(name, scores):
    print(f"{name} scores:")
    print(f"  min:    {scores.min():.6f}")
    print(f"  median: {np.median(scores):.6f}")
    print(f"  mean:   {scores.mean():.6f}")
    print(f"  max:    {scores.max():.6f}")


def main():
    args = parse_args()

    num_samples = args.num_samples
    detector_samples = args.detector_samples

    embedding_path = (
        "data/processed/embeddings/"
        f"cifar10/patch_{num_samples}_dinov2_vits14.pt"
    )

    detector_path = (
        "data/processed/models/"
        f"pgd_detector_{detector_samples}_logistic.pkl"
    )

    metrics_path = (
        "results/metrics/"
        f"11c_pgd_detector_on_patch_{num_samples}.json"
    )

    # ============================================================
    # 1. Exact split consistency.
    # ============================================================

    if num_samples != detector_samples:
        raise ValueError(
            "For exact Experiment 08 split reconstruction, "
            "--num-samples and --detector-samples must match."
        )

    # ============================================================
    # 2. Load localized-patch embeddings.
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

    patch_embeddings = (
        data["adversarial_embeddings"]
        .float()
        .numpy()
    )

    labels = (
        data["labels"]
        .long()
        .numpy()
    )

    clean_predictions = (
        data["clean_predictions"]
        .long()
        .numpy()
    )

    patch_predictions = (
        data["adversarial_predictions"]
        .long()
        .numpy()
    )

    if clean_embeddings.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} clean embeddings, "
            f"found {clean_embeddings.shape[0]}."
        )

    if patch_embeddings.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} patch embeddings, "
            f"found {patch_embeddings.shape[0]}."
        )

    if clean_embeddings.shape != patch_embeddings.shape:
        raise ValueError(
            "Clean and patch embedding shapes do not match."
        )

    # ============================================================
    # 3. Verify corrected data lineage.
    # ============================================================

    source_split = data.get(
        "source_split"
    )

    classifier_training_split = data.get(
        "classifier_training_split"
    )

    classifier_evaluation_split = data.get(
        "classifier_evaluation_split"
    )

    attack_name = data.get(
        "attack"
    )

    attack_type = data.get(
        "attack_type"
    )

    protocol_valid = (
        source_split == "official_cifar10_test"
        and classifier_training_split == "official_cifar10_train"
        and attack_name == "localized_patch"
        and attack_type == "per_image_optimized_patch"
    )

    print()
    print("=" * 70)
    print("PGD -> LOCALIZED PATCH CROSS-ATTACK EVALUATION")
    print("=" * 70)

    print()
    print("Data protocol")
    print("-------------")

    print(
        f"Patch image source:         {source_split}"
    )

    print(
        f"Classifier training split:  {classifier_training_split}"
    )

    print(
        f"Classifier eval split:      {classifier_evaluation_split}"
    )

    print(
        f"Evaluation attack:           {attack_name}"
    )

    print(
        f"Attack type:                 {attack_type}"
    )

    print(
        f"Corrected protocol valid:    {protocol_valid}"
    )

    if not protocol_valid:
        raise ValueError(
            "Patch embedding file does not match the corrected "
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

    intersection = train_set.intersection(
        test_set
    )

    print()
    print("Experiment 08 detector split")
    print("----------------------------")

    print(
        f"Detector training image IDs: {len(detector_train_indices)}"
    )

    print(
        f"Detector held-out image IDs: {len(detector_test_indices)}"
    )

    print(
        f"Train/test intersection:     {len(intersection)}"
    )

    if len(intersection) != 0:
        raise ValueError(
            "Detector train/test image IDs overlap."
        )

    # ============================================================
    # 5. Select only detector-held-out image IDs.
    # ============================================================

    heldout_clean = clean_embeddings[
        detector_test_indices
    ]

    heldout_patch = patch_embeddings[
        detector_test_indices
    ]

    heldout_labels = labels[
        detector_test_indices
    ]

    heldout_clean_predictions = clean_predictions[
        detector_test_indices
    ]

    heldout_patch_predictions = patch_predictions[
        detector_test_indices
    ]

    X_test = np.concatenate(
        [
            heldout_clean,
            heldout_patch,
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
                len(heldout_patch),
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
        f"Held-out patch embeddings: {len(heldout_patch)}"
    )

    print(
        f"Total embeddings:          {len(X_test)}"
    )

    # ============================================================
    # 6. Load frozen PGD detector.
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
    print("Loaded frozen PGD detector:")
    print(detector_path)

    # ============================================================
    # 7. Primary cross-attack evaluation.
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
            "patch",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        predictions,
        target_names=[
            "clean",
            "patch",
        ],
        zero_division=0,
    )

    # ============================================================
    # 8. Detector score diagnostics.
    # ============================================================

    clean_scores = probabilities[
        :len(heldout_clean)
    ]

    patch_scores = probabilities[
        len(heldout_clean):
    ]

    max_clean_score = float(
        clean_scores.max()
    )

    min_patch_score = float(
        patch_scores.min()
    )

    perfect_score_separation = bool(
        min_patch_score
        > max_clean_score
    )

    print()
    print("Detector probability distributions")
    print("----------------------------------")

    summarize_scores(
        "Clean",
        clean_scores,
    )

    print()

    summarize_scores(
        "Patch",
        patch_scores,
    )

    print()

    print(
        f"Maximum clean score: {max_clean_score:.6f}"
    )

    print(
        f"Minimum patch score: {min_patch_score:.6f}"
    )

    print(
        "Perfect clean/patch score separation: "
        f"{perfect_score_separation}"
    )

    # ============================================================
    # 9. Secondary diagnostic:
    #    detector behavior on successful patch attacks only.
    #
    # This is NOT the primary reported result.
    # ============================================================

    originally_correct_mask = (
        heldout_clean_predictions
        == heldout_labels
    )

    successful_patch_mask = (
        originally_correct_mask
        & (
            heldout_patch_predictions
            != heldout_labels
        )
    )

    num_originally_correct = int(
        originally_correct_mask.sum()
    )

    num_successful_patch_attacks = int(
        successful_patch_mask.sum()
    )

    heldout_attack_success_rate = (
        num_successful_patch_attacks
        / num_originally_correct
        if num_originally_correct > 0
        else 0.0
    )

    patch_predictions_by_detector = predictions[
        len(heldout_clean):
    ]

    patch_probabilities_by_detector = probabilities[
        len(heldout_clean):
    ]

    successful_detector_predictions = (
        patch_predictions_by_detector[
            successful_patch_mask
        ]
    )

    successful_detector_scores = (
        patch_probabilities_by_detector[
            successful_patch_mask
        ]
    )

    if num_successful_patch_attacks > 0:
        successful_patch_detection_rate = float(
            (
                successful_detector_predictions
                == 1
            ).mean()
        )

        successful_patch_mean_score = float(
            successful_detector_scores.mean()
        )

        successful_patch_median_score = float(
            np.median(
                successful_detector_scores
            )
        )
    else:
        successful_patch_detection_rate = None
        successful_patch_mean_score = None
        successful_patch_median_score = None

    print()
    print("Successful-patch-only diagnostic")
    print("--------------------------------")

    print(
        f"Held-out originally correct:      "
        f"{num_originally_correct}"
    )

    print(
        f"Successful held-out patch attacks:"
        f" {num_successful_patch_attacks}"
    )

    print(
        f"Held-out patch attack success rate:"
        f" {heldout_attack_success_rate:.4f}"
    )

    if successful_patch_detection_rate is not None:
        print(
            f"PGD-detector recall on successful "
            f"patch attacks: {successful_patch_detection_rate:.4f}"
        )

        print(
            f"Mean detector score on successful "
            f"patch attacks: {successful_patch_mean_score:.6f}"
        )

        print(
            f"Median detector score on successful "
            f"patch attacks: {successful_patch_median_score:.6f}"
        )

    # ============================================================
    # 10. Save metrics.
    # ============================================================

    metrics = {
        "experiment": (
            "11c_evaluate_pgd_detector_on_patch"
        ),
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "detector_training_attack": "pgd",
        "evaluation_attack": "localized_patch",
        "evaluation_attack_type": (
            "per_image_optimized_patch"
        ),
        "source_split": source_split,
        "classifier_training_split": (
            classifier_training_split
        ),
        "classifier_evaluation_split": (
            classifier_evaluation_split
        ),
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
        "patch_precision": float(
            report["patch"]["precision"]
        ),
        "patch_recall": float(
            report["patch"]["recall"]
        ),
        "patch_f1": float(
            report["patch"]["f1-score"]
        ),
        "confusion_matrix": (
            cm.tolist()
        ),
        "max_clean_score": (
            max_clean_score
        ),
        "min_patch_score": (
            min_patch_score
        ),
        "perfect_score_separation": (
            perfect_score_separation
        ),
        "heldout_originally_correct": (
            num_originally_correct
        ),
        "heldout_successful_patch_attacks": (
            num_successful_patch_attacks
        ),
        "heldout_patch_attack_success_rate": float(
            heldout_attack_success_rate
        ),
        "successful_patch_detection_rate": (
            successful_patch_detection_rate
        ),
        "successful_patch_mean_detector_score": (
            successful_patch_mean_score
        ),
        "successful_patch_median_detector_score": (
            successful_patch_median_score
        ),
        "patch_size": data.get(
            "patch_size"
        ),
        "image_size": data.get(
            "image_size"
        ),
        "patch_area_ratio": data.get(
            "patch_area_ratio"
        ),
        "steps": data.get(
            "steps"
        ),
        "learning_rate": data.get(
            "learning_rate"
        ),
        "seed": data.get(
            "seed"
        ),
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ============================================================
    # 11. Print primary summary.
    # ============================================================

    print()
    print("PGD -> localized patch cross-attack results")
    print("-------------------------------------------")

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