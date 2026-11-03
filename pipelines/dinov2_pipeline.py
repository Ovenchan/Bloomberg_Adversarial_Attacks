from pathlib import Path

import torch
from tqdm import tqdm


def extract_embeddings(
    encoder,
    loader,
):
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(
        loader,
        desc="Extracting embeddings",
    ):
        embeddings = encoder.encode(images)

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels


def save_embeddings(
    embeddings,
    labels,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        output_path,
    )

    print(f"Saved embeddings to {output_path}")