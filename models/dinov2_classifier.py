"""
Differentiable DINOv2 CIFAR-10 classifier.

Combines ImageNet normalization, the frozen DINOv2 visual encoder,
and a trained linear classification head into a single model.

The model accepts raw image tensors in [0, 1] pixel space so gradient-based
adversarial attacks can operate with interpretable pixel-space budgets.
"""

import torch
import torch.nn as nn


class DinoV2Classifier(nn.Module):
    def __init__(
        self,
        encoder,
        classifier,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "mean",
            mean,
        )

        self.register_buffer(
            "std",
            std,
        )

    def forward(self, images):
        normalized_images = (
            images - self.mean
        ) / self.std

        embeddings = self.encoder(
            normalized_images
        )

        logits = self.classifier(
            embeddings
        )

        return logits