"""
Experiment:
    03 Clean Embedding PCA

Objective:
    Project the saved clean DINOv2 embeddings into two dimensions using PCA
    and visually inspect whether the representation space contains meaningful
    class structure.

Inputs:
    - Saved clean DINOv2 embeddings
    - CIFAR-10 labels

Outputs:
    - PCA visualization
    - Saved figure under results/figures/

Research Goal:
    Sanity-check the geometry of the DINOv2 representation space before
    introducing adversarial attacks.

Next Experiment:
    04_linear_probe.py
"""

import torch

from analysis.pca import plot_pca


def main():
    data = torch.load(
        "data/processed/embeddings/cifar10/clean_dinov2_vits14.pt"
    )

    embeddings = data["embeddings"]
    labels = data["labels"]

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    plot_pca(
        embeddings=embeddings,
        labels=labels,
        output_path="results/figures/cifar10_dinov2_clean_pca.png",
    )


if __name__ == "__main__":
    main()