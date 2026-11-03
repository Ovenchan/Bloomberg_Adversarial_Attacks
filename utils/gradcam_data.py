"""Config-driven raw-pixel datasets preserving each established resize convention."""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from utils.datasets import get_cifar10_loader

IMAGENET_LABELS = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}
DEFAULT_SEED = 42


class SourceDataset(Dataset):
    """Wrap a Dataset to return image, label and original integer source ID."""

    def __init__(self, dataset):
        """Input: Dataset, including nested Subsets. Output: None."""
        self.dataset = dataset

    def __len__(self):
        """Return the wrapped dataset length as int."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Input: int row index. Output: (Tensor image, label, int source ID)."""
        image, label = self.dataset[index]
        source_index = index
        dataset = self.dataset
        while isinstance(dataset, Subset):
            source_index = int(dataset.indices[source_index])
            dataset = dataset.dataset
        return image, label, source_index


def make_loader(config, train=False, seed=DEFAULT_SEED):
    """Input: dataset dict, bool train, int seed. Output: deterministic DataLoader."""
    if config["name"] == "cifar10":
        dataset = get_cifar10_loader(
            root=config["root"],
            batch_size=config["batch_size"],
            train=train,
            image_size=config["image_size"],
            num_workers=config["num_workers"],
            normalize=False,
        ).dataset
    elif config["name"] == "imagenette":
        dataset = datasets.Imagenette(
            root=config["root"],
            split="train" if train else "val",
            size=config["size"],
            download=config["download"],
            transform=transforms.Compose(
                [
                    transforms.Resize(config["resize_size"]),
                    transforms.CenterCrop(config["image_size"]),
                    transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise ValueError("Supported datasets: cifar10, imagenette")
    indices = config.get("train_indices" if train else "indices")
    if indices is not None:
        if any(
            not isinstance(index, int) or not 0 <= index < len(dataset)
            for index in indices
        ):
            raise ValueError("Source indices must be valid dataset indices")
        dataset = Subset(dataset, indices)
    count = config.get("train_samples" if train else "num_samples")
    if count is not None:
        if not 0 < count <= len(dataset):
            raise ValueError("Sample count outside dataset bounds")
        dataset = Subset(dataset, range(count))
    return DataLoader(
        SourceDataset(dataset),
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=config["num_workers"],
        generator=torch.Generator().manual_seed(seed),
    )


def classifier_labels(labels, loader, dataset_name):
    """Input: Tensor[N], DataLoader, str name. Output: Tensor[N] in classifier label space."""
    if dataset_name == "cifar10":
        return labels
    dataset = loader.dataset
    while isinstance(dataset, (SourceDataset, Subset)):
        dataset = dataset.dataset
    mapping = torch.tensor(
        [IMAGENET_LABELS[wnid] for wnid in dataset.wnids], device=labels.device
    )
    return mapping[labels]
