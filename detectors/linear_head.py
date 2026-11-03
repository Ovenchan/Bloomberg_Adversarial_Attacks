"""
Linear classification head for frozen visual encoder embeddings.

This module provides a small differentiable classifier that maps DINOv2
embeddings to CIFAR-10 class logits. Unlike the sklearn linear probe used
for evaluation, this PyTorch head allows gradients to propagate through
the complete DINOv2 classification pipeline during adversarial attacks.
"""

import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(
        self,
        input_dim=384,
        num_classes=10,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)