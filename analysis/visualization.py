import matplotlib.pyplot as plt
import torch
import torchvision


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def denormalize(images):
    return images * IMAGENET_STD + IMAGENET_MEAN


def show_batch(
    images,
    n=16,
):
    images = denormalize(images[:n])
    images = images.clamp(0, 1)

    grid = torchvision.utils.make_grid(
        images,
        nrow=4,
    )

    grid = grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()