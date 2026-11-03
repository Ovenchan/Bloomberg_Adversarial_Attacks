"""Notebook-compatible bilinear heatmap vectorization."""

import torch
from torch.nn import functional as F

DEFAULT_GRID_SIZE = 14


class GradCAMVectorizer:
    """Input: int grid size. Output: stateless heatmap-to-vector transformer."""

    def __init__(self, grid_size=DEFAULT_GRID_SIZE):
        """Input: positive int. Output: None."""
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        self.grid_size = grid_size

    def transform(self, maps):
        """Input: Tensor/array[N,H,W] or [N,1,H,W]. Output: ndarray[N,grid²]."""
        maps = torch.as_tensor(maps, dtype=torch.float32)
        if maps.ndim == 3:
            maps = maps.unsqueeze(1)
        if (
            maps.ndim != 4
            or maps.shape[1] != 1
            or not torch.isfinite(maps).all()
        ):
            raise ValueError("Expected finite single-channel heatmaps")
        return (
            F.interpolate(
                maps,
                size=(self.grid_size, self.grid_size),
                mode="bilinear",
                align_corners=False,
            )
            .flatten(1)
            .detach()
            .cpu()
            .numpy()
        )
