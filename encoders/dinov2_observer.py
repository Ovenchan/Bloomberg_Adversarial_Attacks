"""DINOv2 observer of the unchanged ResNet attack corpus (no class head)."""

import hashlib

import torch
from torch import nn

from encoders.dinov2 import DinoV2Encoder
from utils.datasets import IMAGENET_MEAN, IMAGENET_STD


class DinoV2Observer(nn.Module):
    """Raw [0,1] input -> frozen ViT-S/14 384-dimensional CLS embedding."""

    def __init__(self, device="cpu", backbone=None):
        super().__init__()
        self.backbone = (backbone if backbone is not None else
                         DinoV2Encoder(model_name="dinov2_vits14", device=device).model)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images):
        # Do not resize, reattack, or pass through the ResNet normalizer.
        return self.backbone((images - self.mean) / self.std)

    def weights_sha256(self):
        """Fingerprint the loaded weights because the existing Hub loader is unpinned."""
        digest = hashlib.sha256()
        for name, value in sorted(self.backbone.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()
