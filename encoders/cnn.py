"""ResNet classifier accepting the shared raw-pixel attack interface."""

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from encoders.base import BaseEncoder
from utils.datasets import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_WEIGHTS = "IMAGENET1K_V2"
DEFAULT_TARGET_LAYER = "layer4.2"


class ResNetEncoder(nn.Module, BaseEncoder):
    """Input: weight name or injected nn.Module. Output: normalized classifier."""

    def __init__(self, weights=DEFAULT_WEIGHTS, model=None):
        """Input: str|None weights, optional nn.Module. Output: None."""
        super().__init__()
        self.model = (
            model
            if model is not None
            else resnet50(
                weights=None if weights is None else ResNet50_Weights[weights]
            )
        )
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.eval()

    def forward(self, images):
        """Input: float Tensor[N,3,H,W] in [0,1]. Output: Tensor[N,K] logits."""
        return self.model((images - self.mean) / self.std)

    def encode(self, images):
        """Input: raw Tensor[N,3,H,W]. Output: Tensor[N,K] classifier features."""
        return self(images)

    def target_layer(self, name=DEFAULT_TARGET_LAYER):
        """Input: str backbone module path. Output: nn.Module for explanation."""
        return self.model.get_submodule(name)

    def pooled_features(self, images):
        """Return pre-fc global-average-pooled features; preserve classifier APIs.

        Accepts the same raw [0,1] pixels as forward(), normalizes once, and
        retains input gradients. Standard ResNet-50 returns [N, 2048].
        """
        x = (images - self.mean) / self.std
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
                     "layer3", "layer4", "avgpool"):
            x = getattr(self.model, name)(x)
        return torch.flatten(x, 1)
