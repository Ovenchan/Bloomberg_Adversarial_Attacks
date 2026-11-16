"""Frozen ResNet-50 classifier shared by the controlled (fixed-1,000) pipelines.

Imagenette: the original 1,000-class ImageNet head, labels mapped to ImageNet
indices. CIFAR-10: the same frozen backbone with the trained 10-class linear
head (trained only on the official CIFAR-10 TRAIN split; see experiments/gradcam.py).
Both accept raw [0, 1] pixels and normalize once inside ResNetEncoder.
"""

from pathlib import Path

import torch
from torch import nn

from encoders.cnn import ResNetEncoder
from utils.gradcam_data import IMAGENET_LABELS

CIFAR10_CLASSES = 10


def dataset_name(config):
    """Input: config dict. Output: str dataset name (defaults to imagenette)."""
    return config["dataset"].get("name", "imagenette")


def in_dataset_labels(config):
    """Classifier labels belonging to the dataset's own classes.

    Imagenette: the ten ImageNet indices (attacks can predict any of 1,000).
    CIFAR-10: all ten head outputs, so every prediction is in-dataset.
    """
    if dataset_name(config) == "cifar10":
        return frozenset(range(CIFAR10_CLASSES))
    return frozenset(IMAGENET_LABELS.values())


def load_cifar10_head(config):
    """Input: config dict. Output: validated head checkpoint dict."""
    path = Path(config["model"]["head_checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIFAR-10 head checkpoint {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("training_split") != "official_cifar10_train":
        raise ValueError("CIFAR-10 head was not trained on the official training split")
    if checkpoint.get("weights") != config["model"]["weights"]:
        raise ValueError("CIFAR-10 head backbone weights differ from the configured weights")
    if checkpoint.get("image_size") != config["dataset"]["image_size"]:
        raise ValueError("CIFAR-10 head preprocessing size differs from the configured image size")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
        raise ValueError("Invalid CIFAR-10 head state")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Non-finite CIFAR-10 head parameters")
    return checkpoint


def build_classifier(config):
    """Input: config dict. Output: eval-mode ResNetEncoder (frozen params, CPU)."""
    model = ResNetEncoder(weights=config["model"]["weights"])
    if dataset_name(config) == "cifar10":
        model.model.fc = nn.Linear(model.model.fc.in_features, CIFAR10_CLASSES)
        model.model.fc.load_state_dict(load_cifar10_head(config)["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
