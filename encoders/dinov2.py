"""
DINOv2 visual encoder.

Loads a pretrained DINOv2 backbone and provides both gradient-free embedding
extraction for normal experiments and a differentiable forward pass for
adversarial attack experiments.
"""

import torch

from encoders.base import BaseEncoder


class DinoV2Encoder(BaseEncoder):
    def __init__(
        self,
        model_name="dinov2_vits14",
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print(
            f"Loading {model_name} "
            f"on {self.device}..."
        )

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )

        self.model.eval()
        self.model.to(self.device)

    def forward(self, images):
        """
        Differentiable forward pass.

        Used when gradients with respect to the input image are required,
        such as PGD adversarial attack generation.
        """
        images = images.to(self.device)

        return self.model(images)

    @torch.no_grad()
    def encode(self, images):
        """
        Gradient-free embedding extraction.

        Used for normal feature extraction and analysis.
        """
        embeddings = self.forward(images)

        return embeddings.cpu()