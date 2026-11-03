"""
Experiment:
    07 Compare Clean vs PGD Embeddings

Objective:
    Quantify how PGD adversarial attacks alter DINOv2 representation space
    by comparing clean and adversarial embeddings for the same CIFAR-10 images.

Inputs:
    - Paired clean/adversarial DINOv2 embeddings from Experiment 06
    - Number of samples passed through --num-samples

Outputs:
    - Mean and standard deviation of cosine similarity
    - Mean and standard deviation of Euclidean distance
    - Mean clean and adversarial embedding norms
    - Mean absolute norm change
    - Saved metrics under results/metrics/
    - PCA figure comparing clean and PGD embeddings

Research Goal:
    Determine whether successful adversarial attacks produce a measurable
    and structured shift in DINOv2 representation space, and verify that
    the pattern persists as the sample size increases.

Next Experiment:
    08_train_adversarial_detector.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from utils.results import save_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of paired clean/PGD embeddings to analyze.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    num_samples = args.num_samples

    input_path = (
        "data/processed/embeddings/"
        f"cifar10/pgd_{num_samples}_dinov2_vits14.pt"
    )

    metrics_path = (
        "results/metrics/"
        f"07_pgd_{num_samples}_embedding_comparison.json"
    )

    figure_path = Path(
        "results/figures/"
        f"07_clean_vs_pgd_pca_{num_samples}.png"
    )

    # ---------------------------------------------------------
    # 1. Load paired clean/adversarial embeddings.
    # ---------------------------------------------------------
    data = torch.load(
        input_path,
        map_location="cpu",
    )

    clean_embeddings = (
        data["clean_embeddings"]
        .float()
    )

    adversarial_embeddings = (
        data["adversarial_embeddings"]
        .float()
    )

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "Adversarial embeddings:",
        adversarial_embeddings.shape,
    )

    actual_samples = clean_embeddings.shape[0]

    if actual_samples != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, "
            f"but input file contains {actual_samples}."
        )

    # ---------------------------------------------------------
    # 2. Cosine similarity.
    # ---------------------------------------------------------
    cosine_similarity = F.cosine_similarity(
        clean_embeddings,
        adversarial_embeddings,
        dim=1,
    )

    mean_cosine_similarity = (
        cosine_similarity.mean().item()
    )

    std_cosine_similarity = (
        cosine_similarity.std().item()
    )

    # ---------------------------------------------------------
    # 3. Euclidean distance.
    # ---------------------------------------------------------
    euclidean_distance = torch.norm(
        adversarial_embeddings
        - clean_embeddings,
        p=2,
        dim=1,
    )

    mean_euclidean_distance = (
        euclidean_distance.mean().item()
    )

    std_euclidean_distance = (
        euclidean_distance.std().item()
    )

    # ---------------------------------------------------------
    # 4. Embedding norms.
    # ---------------------------------------------------------
    clean_norms = torch.norm(
        clean_embeddings,
        p=2,
        dim=1,
    )

    adversarial_norms = torch.norm(
        adversarial_embeddings,
        p=2,
        dim=1,
    )

    norm_change = (
        adversarial_norms
        - clean_norms
    ).abs()

    mean_clean_norm = (
        clean_norms.mean().item()
    )

    mean_adversarial_norm = (
        adversarial_norms.mean().item()
    )

    mean_absolute_norm_change = (
        norm_change.mean().item()
    )

    # ---------------------------------------------------------
    # 5. Save numerical metrics.
    # ---------------------------------------------------------
    metrics = {
        "experiment": "07_compare_pgd_embeddings",
        "dataset": "cifar10",
        "encoder": "dinov2_vits14",
        "num_samples": actual_samples,
        "embedding_dimension": clean_embeddings.shape[1],
        "mean_cosine_similarity": mean_cosine_similarity,
        "std_cosine_similarity": std_cosine_similarity,
        "mean_euclidean_distance": mean_euclidean_distance,
        "std_euclidean_distance": std_euclidean_distance,
        "mean_clean_embedding_norm": mean_clean_norm,
        "mean_adversarial_embedding_norm": mean_adversarial_norm,
        "mean_absolute_norm_change": mean_absolute_norm_change,
    }

    save_metrics(
        metrics,
        metrics_path,
    )

    # ---------------------------------------------------------
    # 6. Joint PCA.
    # ---------------------------------------------------------
    combined_embeddings = torch.cat(
        [
            clean_embeddings,
            adversarial_embeddings,
        ],
        dim=0,
    ).numpy()

    pca = PCA(
        n_components=2,
    )

    reduced = pca.fit_transform(
        combined_embeddings
    )

    clean_2d = reduced[
        :actual_samples
    ]

    adversarial_2d = reduced[
        actual_samples:
    ]

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    # ---------------------------------------------------------
    # 7. Save PCA plot.
    #
    # With 1,000 samples, drawing every pairwise connecting line
    # would make the plot unreadable. Only draw pair lines for
    # smaller exploratory runs.
    # ---------------------------------------------------------
    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(9, 7),
    )

    plt.scatter(
        clean_2d[:, 0],
        clean_2d[:, 1],
        label="Clean",
        alpha=0.55,
        s=18,
    )

    plt.scatter(
        adversarial_2d[:, 0],
        adversarial_2d[:, 1],
        label="PGD",
        alpha=0.55,
        s=18,
    )

    if actual_samples <= 200:
        for i in range(actual_samples):
            plt.plot(
                [
                    clean_2d[i, 0],
                    adversarial_2d[i, 0],
                ],
                [
                    clean_2d[i, 1],
                    adversarial_2d[i, 1],
                ],
                alpha=0.12,
                linewidth=0.6,
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        f"DINOv2 Representation Shift: "
        f"Clean vs PGD (n={actual_samples})"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    # ---------------------------------------------------------
    # 8. Print summary.
    # ---------------------------------------------------------
    print()
    print("Embedding comparison results")
    print("----------------------------")

    print(
        f"Samples:                    "
        f"{actual_samples}"
    )

    print(
        f"Mean cosine similarity:     "
        f"{mean_cosine_similarity:.4f}"
    )

    print(
        f"Std cosine similarity:      "
        f"{std_cosine_similarity:.4f}"
    )

    print(
        f"Mean Euclidean distance:    "
        f"{mean_euclidean_distance:.4f}"
    )

    print(
        f"Std Euclidean distance:     "
        f"{std_euclidean_distance:.4f}"
    )

    print(
        f"Mean clean norm:            "
        f"{mean_clean_norm:.4f}"
    )

    print(
        f"Mean adversarial norm:      "
        f"{mean_adversarial_norm:.4f}"
    )

    print(
        f"Mean absolute norm change:  "
        f"{mean_absolute_norm_change:.4f}"
    )

    print(
        f"PCA variance explained:     "
        f"{explained_variance:.4f}"
    )

    print()
    print(
        f"Saved metrics to "
        f"{metrics_path}"
    )

    print(
        f"Saved figure to "
        f"{figure_path}"
    )


if __name__ == "__main__":
    main()