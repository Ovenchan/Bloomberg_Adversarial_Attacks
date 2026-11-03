"""
Experiment:
    01 Dataset Sanity Check

Objective:
    Verify that the CIFAR-10 dataset loads correctly and that image
    preprocessing produces tensors with the expected shape for downstream
    visual encoder experiments.

Inputs:
    - CIFAR-10 test dataset
    - Image resizing and normalization transforms

Outputs:
    - Image batch shape
    - Label batch shape

Research Goal:
    Confirm the data pipeline is valid before extracting model representations.

Next Experiment:
    02_extract_embeddings.py
"""

from utils.datasets import get_cifar10_loader


def main():
    loader = get_cifar10_loader(
        batch_size=64,
        train=False,
    )

    images, labels = next(iter(loader))

    print("Images:", images.shape)
    print("Labels:", labels.shape)


if __name__ == "__main__":
    main()