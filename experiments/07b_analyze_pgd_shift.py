"""
Experiment:
    07B Analyze PGD Representation Shift

Objective:
    Investigate whether the representation shifts produced by PGD are
    associated with the original CIFAR-10 class or the adversarial class
    predicted after the attack.

Inputs:
    - Paired clean/adversarial DINOv2 embeddings from Experiment 06
    - True CIFAR-10 labels
    - Clean and adversarial predictions

Outputs:
    - PCA of clean embeddings colored by true class
    - PCA of adversarial embeddings colored by adversarial predicted class
    - Paired clean-to-adversarial PCA movement colored by true class

Research Goal:
    Determine whether PGD-induced representation shifts exhibit class-specific
    or adversarial-target-specific structure.

Next Experiment:
    07C Scale PGD experiment to 1,000 images
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


INPUT_PATH = (
    "data/processed/embeddings/"
    "cifar10/pgd_100_dinov2_vits14.pt"
)

FIGURE_DIR = Path(
    "results/figures/07b_pgd_shift"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def save_class_scatter(
    points,
    classes,
    title,
    output_path,
):
    """
    Plot 2D PCA coordinates grouped by CIFAR-10 class.
    """
    plt.figure(figsize=(10, 8))

    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = classes == class_id

        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            label=class_name,
            alpha=0.7,
            s=35,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main():
    # ---------------------------------------------------------
    # 1. Load paired embeddings and labels.
    # ---------------------------------------------------------
    data = torch.load(
        INPUT_PATH,
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

    labels = data["labels"].numpy()

    adversarial_predictions = (
        data["adversarial_predictions"]
        .numpy()
    )

    num_samples = clean_embeddings.shape[0]

    print(
        "Clean embeddings:",
        clean_embeddings.shape,
    )

    print(
        "Adversarial embeddings:",
        adversarial_embeddings.shape,
    )

    # ---------------------------------------------------------
    # 2. Fit a single PCA to both sets.
    #
    # This is important: clean and adversarial points must share
    # exactly the same PCA coordinate system.
    # ---------------------------------------------------------
    combined_embeddings = torch.cat(
        [
            clean_embeddings,
            adversarial_embeddings,
        ],
        dim=0,
    ).numpy()

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        combined_embeddings
    )

    clean_2d = reduced[:num_samples]
    adversarial_2d = reduced[num_samples:]

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"Variance explained by PC1 + PC2: "
        f"{explained_variance:.4f}"
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. Clean embeddings colored by true CIFAR class.
    # ---------------------------------------------------------
    save_class_scatter(
        points=clean_2d,
        classes=labels,
        title=(
            "Clean DINOv2 Embeddings "
            "by True CIFAR-10 Class"
        ),
        output_path=(
            FIGURE_DIR
            / "clean_by_true_class.png"
        ),
    )

    # ---------------------------------------------------------
    # 4. Adversarial embeddings colored by the class predicted
    #    after PGD.
    # ---------------------------------------------------------
    save_class_scatter(
        points=adversarial_2d,
        classes=adversarial_predictions,
        title=(
            "PGD DINOv2 Embeddings "
            "by Adversarial Predicted Class"
        ),
        output_path=(
            FIGURE_DIR
            / "pgd_by_predicted_class.png"
        ),
    )

    # ---------------------------------------------------------
    # 5. Paired movement colored by original true class.
    # ---------------------------------------------------------
    plt.figure(
        figsize=(10, 8)
    )

    cmap = plt.get_cmap(
        "tab10"
    )

    for i in range(num_samples):
        class_id = labels[i]

        plt.plot(
            [
                clean_2d[i, 0],
                adversarial_2d[i, 0],
            ],
            [
                clean_2d[i, 1],
                adversarial_2d[i, 1],
            ],
            alpha=0.25,
            linewidth=0.8,
            color=cmap(class_id),
        )

        plt.scatter(
            clean_2d[i, 0],
            clean_2d[i, 1],
            color=cmap(class_id),
            s=25,
            alpha=0.7,
        )

        plt.scatter(
            adversarial_2d[i, 0],
            adversarial_2d[i, 1],
            color=cmap(class_id),
            marker="x",
            s=35,
            alpha=0.8,
        )

    # Dummy points for class legend.
    for class_id, class_name in enumerate(CLASS_NAMES):
        plt.scatter(
            [],
            [],
            color=cmap(class_id),
            label=class_name,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(
        "PGD Representation Movement "
        "by Original CIFAR-10 Class"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    movement_path = (
        FIGURE_DIR
        / "paired_shift_by_true_class.png"
    )

    plt.savefig(
        movement_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # 6. Print output summary.
    # ---------------------------------------------------------
    print()
    print("Saved figures:")

    print(
        FIGURE_DIR
        / "clean_by_true_class.png"
    )

    print(
        FIGURE_DIR
        / "pgd_by_predicted_class.png"
    )

    print(
        movement_path
    )


if __name__ == "__main__":
    main()