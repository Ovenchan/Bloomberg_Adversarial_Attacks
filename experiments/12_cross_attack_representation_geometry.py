"""
Experiment:
    12 Cross-Attack DINOv2 Representation Geometry

Objective:
    Explain why a linear detector trained only on PGD-induced DINOv2
    representations transfers strongly to APGD and localized patch attacks
    but poorly to CW.

Inputs:
    - Paired clean/PGD DINOv2 embeddings from Experiment 06
    - Paired clean/APGD DINOv2 embeddings from Experiment 09B
    - Paired clean/CW DINOv2 embeddings from Experiment 10B
    - Paired clean/localized-patch DINOv2 embeddings from Experiment 11B
    - Frozen PGD logistic-regression detector from Experiment 08

Method:
    1. Verify that all attack files contain the exact same clean DINOv2
       embeddings for the same CIFAR-10 TEST image IDs.

    2. For each attack, measure:
         - clean/adversarial cosine similarity
         - Euclidean representation displacement
         - clean/adversarial embedding norms
         - absolute norm change
         - relative representation displacement

    3. Compare attack displacement directions image-by-image:
         cosine(delta_PGD, delta_APGD)
         cosine(delta_PGD, delta_CW)
         cosine(delta_PGD, delta_Patch)

    4. Apply the frozen PGD detector to all representations and measure:
         - detector adversarial probabilities
         - detector decision-function values
         - clean-to-adversarial decision-function shift

    5. Evaluate detector behavior on the exact 200 held-out image IDs from
       Experiment 08.

    6. Measure image-level association between:
         PGD displacement alignment
    and:
         PGD detector decision-function shift

       for APGD, CW, and localized patch attacks.

    7. Fit one common PCA basis across clean, PGD, APGD, CW, and Patch
       embeddings for visualization only.

Outputs:
    - Cross-attack geometry metrics JSON
    - Detector score distribution figure
    - Detector decision-shift figure
    - Common PCA figure
    - PGD-alignment vs detector-shift scatter figure

Research Goal:
    Determine whether differences in cross-attack detector transfer can be
    explained by attack-dependent geometry in frozen DINOv2 representation
    space.

Important Limitation:
    Correlation between representation geometry and detector behavior does
    not establish causality. PCA is used only as a supporting visualization
    because two components capture only a limited fraction of the full
    384-dimensional representation variance.

Next Experiment:
    Evaluate whether the observed representation-space behavior persists
    on a natural higher-resolution dataset such as an ImageNet subset.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


SEED = 42
DETECTOR_TEST_SIZE = 0.20

ATTACK_ORDER = [
    "pgd",
    "apgd",
    "cw",
    "patch",
]

TRANSFER_ATTACKS = [
    "apgd",
    "cw",
    "patch",
]

DISPLAY_NAMES = {
    "pgd": "PGD",
    "apgd": "APGD",
    "cw": "CW",
    "patch": "Patch",
}


# ============================================================
# Argument parsing
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of paired image embeddings to analyze.",
    )

    return parser.parse_args()


# ============================================================
# Utility functions
# ============================================================


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def load_embedding_file(path):
    data = torch.load(
        path,
        map_location="cpu",
    )

    clean_embeddings = to_numpy(
        data["clean_embeddings"]
    ).astype(np.float64)

    adversarial_embeddings = to_numpy(
        data["adversarial_embeddings"]
    ).astype(np.float64)

    return (
        data,
        clean_embeddings,
        adversarial_embeddings,
    )


def reconstruct_detector_split(num_samples):
    image_indices = np.arange(
        num_samples
    )

    train_indices, test_indices = train_test_split(
        image_indices,
        test_size=DETECTOR_TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    return (
        train_indices,
        test_indices,
    )


def cosine_similarity_rows(x, y):
    numerator = np.sum(
        x * y,
        axis=1,
    )

    x_norm = np.linalg.norm(
        x,
        axis=1,
    )

    y_norm = np.linalg.norm(
        y,
        axis=1,
    )

    denominator = (
        x_norm
        * y_norm
    )

    denominator = np.maximum(
        denominator,
        1e-12,
    )

    return (
        numerator
        / denominator
    )


def summarize_array(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "median": float(
            np.median(values)
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def compute_geometry(
    clean_embeddings,
    adversarial_embeddings,
):
    displacement = (
        adversarial_embeddings
        - clean_embeddings
    )

    cosine_similarity = cosine_similarity_rows(
        clean_embeddings,
        adversarial_embeddings,
    )

    euclidean_distance = np.linalg.norm(
        displacement,
        axis=1,
    )

    clean_norm = np.linalg.norm(
        clean_embeddings,
        axis=1,
    )

    adversarial_norm = np.linalg.norm(
        adversarial_embeddings,
        axis=1,
    )

    absolute_norm_change = np.abs(
        adversarial_norm
        - clean_norm
    )

    relative_displacement = (
        euclidean_distance
        / np.maximum(
            clean_norm,
            1e-12,
        )
    )

    return {
        "displacement": displacement,
        "cosine_similarity": cosine_similarity,
        "euclidean_distance": euclidean_distance,
        "clean_norm": clean_norm,
        "adversarial_norm": adversarial_norm,
        "absolute_norm_change": absolute_norm_change,
        "relative_displacement": relative_displacement,
    }


def detector_outputs(
    detector,
    embeddings,
):
    probabilities = detector.predict_proba(
        embeddings
    )[:, 1]

    decision_values = detector.decision_function(
        embeddings
    )

    return (
        probabilities,
        decision_values,
    )


def pearson_correlation(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if len(x) != len(y):
        raise ValueError(
            "Correlation arrays must have equal length."
        )

    if len(x) < 2:
        return float("nan")

    if np.std(x) < 1e-12:
        return float("nan")

    if np.std(y) < 1e-12:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def save_json(
    data,
    path,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print(
        f"Saved metrics to {path}"
    )


# ============================================================
# Visualization
# ============================================================


def plot_detector_scores(
    clean_scores,
    attack_scores,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        clean_scores,
        bins=40,
        alpha=0.45,
        label="Clean",
    )

    for attack in ATTACK_ORDER:
        plt.hist(
            attack_scores[attack],
            bins=40,
            alpha=0.35,
            label=DISPLAY_NAMES[attack],
        )

    plt.xlabel(
        "PGD detector adversarial probability"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Frozen PGD Detector Scores Across Attacks"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )

    print(
        f"Saved detector score figure to {output_path}"
    )


def plot_detector_decision_shift(
    decision_shifts,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = [
        decision_shifts[attack]
        for attack in ATTACK_ORDER
    ]

    labels = [
        DISPLAY_NAMES[attack]
        for attack in ATTACK_ORDER
    ]

    figure = plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        values,
        tick_labels=labels,
        showfliers=False,
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Attack"
    )

    plt.ylabel(
        "Change in PGD detector decision function"
    )

    plt.title(
        "Attack-Induced Shift Along PGD Detector Direction"
    )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )

    print(
        f"Saved detector-direction figure to {output_path}"
    )


def plot_common_pca(
    clean_embeddings,
    adversarial_embeddings,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = [
        clean_embeddings
    ]

    group_names = [
        "Clean"
    ]

    for attack in ATTACK_ORDER:
        combined.append(
            adversarial_embeddings[attack]
        )

        group_names.append(
            DISPLAY_NAMES[attack]
        )

    combined_matrix = np.concatenate(
        combined,
        axis=0,
    )

    pca = PCA(
        n_components=2,
        random_state=SEED,
    )

    projected = pca.fit_transform(
        combined_matrix
    )

    figure = plt.figure(
        figsize=(10, 7)
    )

    start = 0

    group_size = (
        clean_embeddings.shape[0]
    )

    for group_name in group_names:
        end = (
            start
            + group_size
        )

        group_projection = projected[
            start:end
        ]

        plt.scatter(
            group_projection[:, 0],
            group_projection[:, 1],
            s=8,
            alpha=0.35,
            label=group_name,
        )

        start = end

    explained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    plt.xlabel(
        "PC1"
    )

    plt.ylabel(
        "PC2"
    )

    plt.title(
        "Common PCA of Clean and Adversarial DINOv2 Representations\n"
        f"PC1 + PC2 variance explained: "
        f"{explained_variance * 100:.2f}%"
    )

    plt.legend(
        markerscale=2
    )

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )

    print(
        f"Saved common PCA figure to {output_path}"
    )

    return explained_variance


def plot_alignment_vs_detector_shift(
    displacement_alignments,
    decision_shifts,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(10, 7)
    )

    for attack in TRANSFER_ATTACKS:
        x = displacement_alignments[
            attack
        ]

        y = decision_shifts[
            attack
        ]

        plt.scatter(
            x,
            y,
            s=10,
            alpha=0.35,
            label=DISPLAY_NAMES[attack],
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Cosine similarity to PGD displacement direction"
    )

    plt.ylabel(
        "Change in PGD detector decision function"
    )

    plt.title(
        "PGD Displacement Alignment vs Detector Response"
    )

    plt.legend()

    plt.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        figure
    )

    print(
        f"Saved alignment-vs-detector figure to {output_path}"
    )


# ============================================================
# Main experiment
# ============================================================


def main():
    args = parse_args()

    num_samples = args.num_samples

    embedding_paths = {
        "pgd": (
            "data/processed/embeddings/"
            f"cifar10/pgd_{num_samples}_dinov2_vits14.pt"
        ),
        "apgd": (
            "data/processed/embeddings/"
            f"cifar10/apgd_{num_samples}_dinov2_vits14.pt"
        ),
        "cw": (
            "data/processed/embeddings/"
            f"cifar10/cw_{num_samples}_dinov2_vits14.pt"
        ),
        "patch": (
            "data/processed/embeddings/"
            f"cifar10/patch_{num_samples}_dinov2_vits14.pt"
        ),
    }

    detector_path = (
        "data/processed/models/"
        f"pgd_detector_{num_samples}_logistic.pkl"
    )

    metrics_path = (
        "results/metrics/"
        "12_cross_attack_representation_geometry.json"
    )

    figure_directory = Path(
        "results/figures/"
        "12_cross_attack_geometry"
    )

    # ============================================================
    # 1. Load paired embeddings
    # ============================================================

    raw_data = {}

    clean_embeddings_by_attack = {}

    adversarial_embeddings = {}

    print()
    print("=" * 70)
    print("CROSS-ATTACK REPRESENTATION GEOMETRY")
    print("=" * 70)

    for attack in ATTACK_ORDER:
        (
            raw_data[attack],
            clean_embeddings_by_attack[attack],
            adversarial_embeddings[attack],
        ) = load_embedding_file(
            embedding_paths[attack]
        )

        print()
        print(
            f"{DISPLAY_NAMES[attack]} embeddings"
        )

        print(
            "  clean:       ",
            clean_embeddings_by_attack[
                attack
            ].shape,
        )

        print(
            "  adversarial: ",
            adversarial_embeddings[
                attack
            ].shape,
        )

    # ============================================================
    # 2. Validate shapes and clean embedding identity
    # ============================================================

    reference_clean = (
        clean_embeddings_by_attack[
            "pgd"
        ]
    )

    expected_shape = (
        num_samples,
        384,
    )

    if reference_clean.shape != expected_shape:
        raise ValueError(
            f"Expected PGD clean embeddings with shape "
            f"{expected_shape}, found {reference_clean.shape}."
        )

    clean_identity_checks = {}

    print()
    print("Clean embedding identity checks")
    print("-------------------------------")

    for attack in ATTACK_ORDER:
        clean_embeddings = (
            clean_embeddings_by_attack[
                attack
            ]
        )

        attack_embeddings = (
            adversarial_embeddings[
                attack
            ]
        )

        if clean_embeddings.shape != expected_shape:
            raise ValueError(
                f"{DISPLAY_NAMES[attack]} clean embeddings "
                f"have unexpected shape "
                f"{clean_embeddings.shape}."
            )

        if attack_embeddings.shape != expected_shape:
            raise ValueError(
                f"{DISPLAY_NAMES[attack]} adversarial embeddings "
                f"have unexpected shape "
                f"{attack_embeddings.shape}."
            )

        difference = np.abs(
            clean_embeddings
            - reference_clean
        )

        max_difference = float(
            difference.max()
        )

        mean_difference = float(
            difference.mean()
        )

        exact_identity = bool(
            np.array_equal(
                clean_embeddings,
                reference_clean,
            )
        )

        clean_identity_checks[
            attack
        ] = {
            "max_difference": (
                max_difference
            ),
            "mean_difference": (
                mean_difference
            ),
            "exact_identity": (
                exact_identity
            ),
        }

        print(
            f"{DISPLAY_NAMES[attack]:<8} "
            f"max diff={max_difference:.10f} | "
            f"exact={exact_identity}"
        )

        if not exact_identity:
            raise ValueError(
                f"Clean embeddings for "
                f"{DISPLAY_NAMES[attack]} "
                "do not exactly match PGD clean embeddings."
            )

    # ============================================================
    # 3. Representation geometry
    # ============================================================

    geometry = {}

    geometry_metrics = {}

    print()
    print("Representation displacement")
    print("---------------------------")

    print(
        f"{'Attack':<8}"
        f"{'Cos Sim':>12}"
        f"{'L2 Dist':>12}"
        f"{'Rel L2':>12}"
        f"{'Norm Δ':>12}"
    )

    print(
        "-" * 56
    )

    for attack in ATTACK_ORDER:
        geometry[attack] = compute_geometry(
            reference_clean,
            adversarial_embeddings[
                attack
            ],
        )

        geometry_metrics[
            attack
        ] = {
            "cosine_similarity": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "cosine_similarity"
                    ]
                )
            ),
            "euclidean_distance": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "euclidean_distance"
                    ]
                )
            ),
            "clean_norm": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "clean_norm"
                    ]
                )
            ),
            "adversarial_norm": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "adversarial_norm"
                    ]
                )
            ),
            "absolute_norm_change": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "absolute_norm_change"
                    ]
                )
            ),
            "relative_displacement": (
                summarize_array(
                    geometry[
                        attack
                    ][
                        "relative_displacement"
                    ]
                )
            ),
        }

        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{geometry_metrics[attack]['cosine_similarity']['mean']:>12.4f}"
            f"{geometry_metrics[attack]['euclidean_distance']['mean']:>12.4f}"
            f"{geometry_metrics[attack]['relative_displacement']['mean']:>12.4f}"
            f"{geometry_metrics[attack]['absolute_norm_change']['mean']:>12.4f}"
        )

    # ============================================================
    # 4. Displacement-direction alignment
    # ============================================================

    pgd_displacement = (
        geometry[
            "pgd"
        ][
            "displacement"
        ]
    )

    displacement_alignment_arrays = {}

    displacement_alignment_metrics = {}

    print()
    print("Displacement-direction similarity")
    print("---------------------------------")

    for attack in TRANSFER_ATTACKS:
        similarities = (
            cosine_similarity_rows(
                pgd_displacement,
                geometry[
                    attack
                ][
                    "displacement"
                ],
            )
        )

        displacement_alignment_arrays[
            attack
        ] = similarities

        displacement_alignment_metrics[
            f"pgd_vs_{attack}"
        ] = summarize_array(
            similarities
        )

        print(
            f"PGD vs "
            f"{DISPLAY_NAMES[attack]:<5}: "
            f"mean="
            f"{similarities.mean():.4f} | "
            f"median="
            f"{np.median(similarities):.4f}"
        )

    # ============================================================
    # 5. Load frozen PGD detector
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
    # 6. Detector behavior across all samples
    # ============================================================

    (
        clean_probabilities,
        clean_decisions,
    ) = detector_outputs(
        detector,
        reference_clean,
    )

    attack_probabilities = {}

    attack_decisions = {}

    decision_shifts = {}

    detector_metrics = {}

    print()
    print("Frozen PGD detector behavior")
    print("----------------------------")

    print(
        f"{'Attack':<8}"
        f"{'Median Prob':>14}"
        f"{'Mean Prob':>14}"
        f"{'Mean ΔDecision':>18}"
    )

    print(
        "-" * 54
    )

    for attack in ATTACK_ORDER:
        (
            attack_probabilities[
                attack
            ],
            attack_decisions[
                attack
            ],
        ) = detector_outputs(
            detector,
            adversarial_embeddings[
                attack
            ],
        )

        decision_shifts[
            attack
        ] = (
            attack_decisions[
                attack
            ]
            - clean_decisions
        )

        detector_metrics[
            attack
        ] = {
            "adversarial_probability": (
                summarize_array(
                    attack_probabilities[
                        attack
                    ]
                )
            ),
            "decision_function": (
                summarize_array(
                    attack_decisions[
                        attack
                    ]
                )
            ),
            "decision_shift_from_clean": (
                summarize_array(
                    decision_shifts[
                        attack
                    ]
                )
            ),
            "fraction_above_default_threshold": float(
                np.mean(
                    attack_probabilities[
                        attack
                    ]
                    >= 0.5
                )
            ),
        }

        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{np.median(attack_probabilities[attack]):>14.6f}"
            f"{np.mean(attack_probabilities[attack]):>14.6f}"
            f"{np.mean(decision_shifts[attack]):>18.4f}"
        )

    clean_detector_metrics = {
        "probability": (
            summarize_array(
                clean_probabilities
            )
        ),
        "decision_function": (
            summarize_array(
                clean_decisions
            )
        ),
        "fraction_above_default_threshold": float(
            np.mean(
                clean_probabilities
                >= 0.5
            )
        ),
    }

    # ============================================================
    # 7. Exact Experiment 08 detector-held-out IDs
    # ============================================================

    (
        detector_train_indices,
        detector_test_indices,
    ) = reconstruct_detector_split(
        num_samples
    )

    intersection = np.intersect1d(
        detector_train_indices,
        detector_test_indices,
    )

    if len(intersection) != 0:
        raise ValueError(
            "Reconstructed detector train/test IDs overlap."
        )

    print()
    print("Exact Experiment 08 held-out detector IDs")
    print("-----------------------------------------")

    print(
        f"Training image IDs: "
        f"{len(detector_train_indices)}"
    )

    print(
        f"Held-out image IDs: "
        f"{len(detector_test_indices)}"
    )

    print(
        f"Intersection:       "
        f"{len(intersection)}"
    )

    heldout_clean_probabilities = (
        clean_probabilities[
            detector_test_indices
        ]
    )

    heldout_clean_decisions = (
        clean_decisions[
            detector_test_indices
        ]
    )

    heldout_metrics = {}

    print()

    print(
        f"{'Attack':<8}"
        f"{'Median Prob':>14}"
        f"{'Mean Prob':>14}"
        f"{'Mean ΔDecision':>18}"
        f"{'Detected':>12}"
    )

    print(
        "-" * 66
    )

    for attack in ATTACK_ORDER:
        heldout_attack_probabilities = (
            attack_probabilities[
                attack
            ][
                detector_test_indices
            ]
        )

        heldout_attack_decisions = (
            attack_decisions[
                attack
            ][
                detector_test_indices
            ]
        )

        heldout_decision_shift = (
            heldout_attack_decisions
            - heldout_clean_decisions
        )

        heldout_detected_fraction = float(
            np.mean(
                heldout_attack_probabilities
                >= 0.5
            )
        )

        heldout_metrics[
            attack
        ] = {
            "adversarial_probability": (
                summarize_array(
                    heldout_attack_probabilities
                )
            ),
            "decision_shift_from_clean": (
                summarize_array(
                    heldout_decision_shift
                )
            ),
            "fraction_detected": (
                heldout_detected_fraction
            ),
        }

        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{np.median(heldout_attack_probabilities):>14.6f}"
            f"{np.mean(heldout_attack_probabilities):>14.6f}"
            f"{np.mean(heldout_decision_shift):>18.4f}"
            f"{heldout_detected_fraction:>12.4f}"
        )

    # ============================================================
    # 8. Image-level correlation:
    #
    #    Does an attack displacement that is more PGD-like produce
    #    a larger PGD-detector decision shift?
    # ============================================================

    alignment_detector_correlations = {}

    print()
    print("PGD alignment vs detector decision shift")
    print("----------------------------------------")

    print(
        f"{'Attack':<8}"
        f"{'All r':>12}"
        f"{'Held-out r':>16}"
    )

    print(
        "-" * 36
    )

    for attack in TRANSFER_ATTACKS:
        alignment_all = (
            displacement_alignment_arrays[
                attack
            ]
        )

        shift_all = (
            decision_shifts[
                attack
            ]
        )

        all_correlation = (
            pearson_correlation(
                alignment_all,
                shift_all,
            )
        )

        alignment_heldout = (
            alignment_all[
                detector_test_indices
            ]
        )

        shift_heldout = (
            shift_all[
                detector_test_indices
            ]
        )

        heldout_correlation = (
            pearson_correlation(
                alignment_heldout,
                shift_heldout,
            )
        )

        alignment_detector_correlations[
            attack
        ] = {
            "all_samples_pearson_r": (
                all_correlation
            ),
            "heldout_samples_pearson_r": (
                heldout_correlation
            ),
            "num_all_samples": int(
                len(alignment_all)
            ),
            "num_heldout_samples": int(
                len(alignment_heldout)
            ),
        }

        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{all_correlation:>12.4f}"
            f"{heldout_correlation:>16.4f}"
        )

    # ============================================================
    # 9. Common PCA
    # ============================================================

    pca_variance_explained = (
        plot_common_pca(
            clean_embeddings=(
                reference_clean
            ),
            adversarial_embeddings=(
                adversarial_embeddings
            ),
            output_path=(
                figure_directory
                / "common_pca.png"
            ),
        )
    )

    # ============================================================
    # 10. Detector score distributions
    # ============================================================

    plot_detector_scores(
        clean_scores=(
            heldout_clean_probabilities
        ),
        attack_scores={
            attack: (
                attack_probabilities[
                    attack
                ][
                    detector_test_indices
                ]
            )
            for attack in ATTACK_ORDER
        },
        output_path=(
            figure_directory
            / "detector_score_distributions.png"
        ),
    )

    # ============================================================
    # 11. Detector decision-shift plot
    # ============================================================

    plot_detector_decision_shift(
        decision_shifts={
            attack: (
                decision_shifts[
                    attack
                ][
                    detector_test_indices
                ]
            )
            for attack in ATTACK_ORDER
        },
        output_path=(
            figure_directory
            / "detector_direction_shift.png"
        ),
    )

    # ============================================================
    # 12. Alignment vs detector shift plot
    # ============================================================

    plot_alignment_vs_detector_shift(
        displacement_alignments={
            attack: (
                displacement_alignment_arrays[
                    attack
                ][
                    detector_test_indices
                ]
            )
            for attack in TRANSFER_ATTACKS
        },
        decision_shifts={
            attack: (
                decision_shifts[
                    attack
                ][
                    detector_test_indices
                ]
            )
            for attack in TRANSFER_ATTACKS
        },
        output_path=(
            figure_directory
            / "alignment_vs_detector_shift.png"
        ),
    )

    # ============================================================
    # 13. Save metrics
    # ============================================================

    metrics = {
        "experiment": (
            "12_cross_attack_representation_geometry"
        ),
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_image_pairs": int(
            num_samples
        ),
        "embedding_dimension": int(
            reference_clean.shape[1]
        ),
        "attacks": (
            ATTACK_ORDER
        ),
        "clean_embedding_identity_checks": (
            clean_identity_checks
        ),
        "geometry": (
            geometry_metrics
        ),
        "displacement_direction_alignment": (
            displacement_alignment_metrics
        ),
        "alignment_vs_detector_shift_correlation": (
            alignment_detector_correlations
        ),
        "clean_detector_behavior": (
            clean_detector_metrics
        ),
        "attack_detector_behavior_all_samples": (
            detector_metrics
        ),
        "detector_heldout_image_count": int(
            len(detector_test_indices)
        ),
        "detector_heldout_behavior": (
            heldout_metrics
        ),
        "common_pca_variance_explained": float(
            pca_variance_explained
        ),
    }

    save_json(
        metrics,
        metrics_path,
    )

    # ============================================================
    # 14. Final compact summary
    # ============================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 12 SUMMARY")
    print("=" * 70)

    print()
    print("Representation geometry")
    print("-----------------------")

    print(
        f"{'Attack':<8}"
        f"{'Cos Sim':>12}"
        f"{'L2 Dist':>12}"
        f"{'Rel L2':>12}"
        f"{'PGD Δ Align':>14}"
    )

    print(
        "-" * 58
    )

    for attack in ATTACK_ORDER:
        if attack == "pgd":
            alignment = 1.0
        else:
            alignment = (
                displacement_alignment_metrics[
                    f"pgd_vs_{attack}"
                ][
                    "mean"
                ]
            )

        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{geometry_metrics[attack]['cosine_similarity']['mean']:>12.4f}"
            f"{geometry_metrics[attack]['euclidean_distance']['mean']:>12.4f}"
            f"{geometry_metrics[attack]['relative_displacement']['mean']:>12.4f}"
            f"{alignment:>14.4f}"
        )

    print()
    print("PGD-detector held-out behavior")
    print("------------------------------")

    print(
        f"{'Attack':<8}"
        f"{'Median Score':>16}"
        f"{'Mean ΔDecision':>18}"
        f"{'Detected':>12}"
    )

    print(
        "-" * 54
    )

    for attack in ATTACK_ORDER:
        print(
            f"{DISPLAY_NAMES[attack]:<8}"
            f"{heldout_metrics[attack]['adversarial_probability']['median']:>16.6f}"
            f"{heldout_metrics[attack]['decision_shift_from_clean']['mean']:>18.4f}"
            f"{heldout_metrics[attack]['fraction_detected']:>12.4f}"
        )

    print()
    print("PGD-alignment / detector-shift correlation")
    print("------------------------------------------")

    for attack in TRANSFER_ATTACKS:
        print(
            f"{DISPLAY_NAMES[attack]:<8} "
            f"all r="
            f"{alignment_detector_correlations[attack]['all_samples_pearson_r']:.4f} | "
            f"held-out r="
            f"{alignment_detector_correlations[attack]['heldout_samples_pearson_r']:.4f}"
        )

    print()

    print(
        "Common PCA PC1 + PC2 variance explained: "
        f"{pca_variance_explained * 100:.2f}%"
    )

    print()

    print(
        f"Saved metrics to {metrics_path}"
    )

    print(
        f"Saved figures to {figure_directory}"
    )


if __name__ == "__main__":
    main()