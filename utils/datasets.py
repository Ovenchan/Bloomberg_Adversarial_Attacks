"""
Dataset loading utilities.

Provides reusable CIFAR-10 loaders for both standard DINOv2 feature
extraction and adversarial attack experiments.

Clean embedding experiments use ImageNet normalization during loading.
Adversarial experiments can request raw [0, 1] tensors so perturbations
such as epsilon=8/255 are applied in pixel space.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar10_loader(
    root="data/raw",
    batch_size=64,
    train=False,
    image_size=224,
    num_workers=0,
    normalize=True,
):
    transform_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize:
        transform_steps.append(
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
        )

    transform = transforms.Compose(transform_steps)

    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
    )

    return loader