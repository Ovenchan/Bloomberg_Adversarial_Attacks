"""Predicted-class Grad-CAM; no ground-truth labels enter explanation."""

import torch
from torch.nn import functional as F

CAM_EPSILON = 1e-7


class GradCAMExplainer:
    """Input: nn.Module classifier and target layer. Output: explanation object."""

    def __init__(self, model, target_layer):
        """Input: nn.Module model, nn.Module target_layer. Output: None."""
        self.model = model
        self.target_layer = target_layer

    def explain(self, images):
        """Input: raw Tensor[N,3,H,W]. Output: (maps[N,H,W], predictions[N])."""
        activations = []

        def capture(module, inputs, output):
            """Input: module, tuple inputs, Tensor output. Output: None."""
            activations.append(output)

        if self.model.training:
            raise ValueError("Grad-CAM requires an eval-mode classifier")
        handle = self.target_layer.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                logits = self.model(images.detach().requires_grad_(True))
                predictions = logits.argmax(dim=1)
                selected = logits.gather(1, predictions[:, None]).sum()
                activation = activations[-1]
                gradient = torch.autograd.grad(selected, activation)[0]
                maps = (
                    (gradient.mean(dim=(-2, -1), keepdim=True) * activation)
                    .sum(1)
                    .relu()
                )
                maps = maps - maps.amin(dim=(-2, -1), keepdim=True)
                maps = maps / (
                    maps.amax(dim=(-2, -1), keepdim=True) + CAM_EPSILON
                )
                maps = F.interpolate(
                    maps[:, None],
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                return maps.detach(), predictions.detach()
        finally:
            handle.remove()
