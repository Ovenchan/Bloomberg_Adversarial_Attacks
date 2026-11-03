import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(
    embeddings,
    labels,
    output_path=None,
):
    embeddings = embeddings.numpy()
    labels = labels.numpy()

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        s=8,
        alpha=0.6,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("DINOv2 Embeddings - CIFAR-10")
    plt.colorbar(
        scatter,
        label="CIFAR-10 class",
    )

    if output_path is not None:
        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

    plt.show()