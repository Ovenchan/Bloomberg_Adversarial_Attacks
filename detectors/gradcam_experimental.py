"""Optional notebook architectures; excluded from the primary logistic experiment."""

import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

CHANNELS = (1, 16, 32, 64)
KERNEL_SIZE = 3
PADDING = 1
STRIDE = 2
DECONV_KERNEL = 4
STAT_EPSILON = 1e-12
ACTIVATION_THRESHOLDS = (0.25, 0.50, 0.75)
TOP_FRACTION = 0.10


class GradCAMStatistics:
    """Input: heatmaps. Output: notebook ten-feature statistical representation."""

    def transform(self, maps):
        """Input: ndarray[N,H,W], H/W > 1. Output: ndarray[N,10]."""
        rows = []
        for cam in np.asarray(maps):
            if (
                cam.ndim != 2
                or min(cam.shape) <= 1
                or not np.isfinite(cam).all()
            ):
                raise ValueError(
                    "Expected finite 2D maps with spatial dimensions > 1"
                )
            cam = np.clip(cam, 0, 1)
            height, width = cam.shape
            total = cam.sum() + STAT_EPSILON
            probability = cam / total
            yy, xx = np.mgrid[:height, :width]
            xx, yy = xx / (width - 1), yy / (height - 1)
            center_x, center_y = (probability * xx).sum(), (
                probability * yy
            ).sum()
            entropy = -(
                probability * np.log(probability + STAT_EPSILON)
            ).sum() / np.log(cam.size)
            spread = np.sqrt(
                (
                    probability * ((xx - center_x) ** 2 + (yy - center_y) ** 2)
                ).sum()
            )
            count = max(1, int(TOP_FRACTION * cam.size))
            concentration = (
                np.partition(cam.flatten(), -count)[-count:].sum() / total
            )
            rows.append(
                [
                    cam.mean(),
                    cam.std(),
                    *[
                        (cam >= threshold).mean()
                        for threshold in ACTIVATION_THRESHOLDS
                    ],
                    entropy,
                    center_x,
                    center_y,
                    spread,
                    concentration,
                ]
            )
        return np.asarray(rows)


class GradCAMCNN(nn.Module):
    """Input: None. Output: notebook supervised CNN architecture (untrained)."""

    def __init__(self):
        """Input: None. Output: None."""
        super().__init__()
        layers = []
        for incoming, outgoing in zip(CHANNELS[:-1], CHANNELS[1:]):
            layers.extend(
                [
                    nn.Conv2d(
                        incoming, outgoing, KERNEL_SIZE, padding=PADDING
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(STRIDE),
                ]
            )
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(CHANNELS[-1], 1)

    def forward(self, maps):
        """Input: Tensor[N,1,H,W], H/W >= 8. Output: Tensor[N] adversarial logits."""
        return self.classifier(self.features(maps).flatten(1)).squeeze(1)


class GradCAMAutoencoder(nn.Module):
    """Input: None. Output: clean-only notebook reconstruction architecture (untrained)."""

    def __init__(self):
        """Input: None. Output: None."""
        super().__init__()
        encoder, decoder = [], []
        for incoming, outgoing in zip(CHANNELS[:-1], CHANNELS[1:]):
            encoder.extend(
                [
                    nn.Conv2d(
                        incoming,
                        outgoing,
                        KERNEL_SIZE,
                        stride=STRIDE,
                        padding=PADDING,
                    ),
                    nn.ReLU(),
                ]
            )
        reverse = CHANNELS[::-1]
        for incoming, outgoing in zip(reverse[:-1], reverse[1:]):
            decoder.extend(
                [
                    nn.ConvTranspose2d(
                        incoming,
                        outgoing,
                        DECONV_KERNEL,
                        stride=STRIDE,
                        padding=PADDING,
                    ),
                    nn.Sigmoid() if outgoing == CHANNELS[0] else nn.ReLU(),
                ]
            )
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(
            *decoder
        )

    def forward(self, maps):
        """Input: Tensor[N,1,H,W], spatial sizes divisible by 8. Output: reconstructed Tensor."""
        if any(
            size % (STRIDE ** (len(CHANNELS) - 1)) for size in maps.shape[-2:]
        ):
            raise ValueError(
                "Autoencoder spatial sizes must be divisible by 8; use full heatmaps"
            )
        return self.decoder(self.encoder(maps))

    def score_samples(self, maps):
        """Input: Tensor[N,1,H,W]. Output: Tensor[N] reconstruction MSE, higher = anomalous."""
        with torch.no_grad():
            return (self(maps) - maps).square().mean((1, 2, 3))


class GradCAMMahalanobisDetector:
    """Input: None. Output: clean-only shrinkage Mahalanobis vector ablation."""

    def __init__(self):
        """Input: None. Output: None."""
        self.covariance = LedoitWolf()
        self.scaler = StandardScaler()

    def fit(self, clean_features):
        """Input: ndarray[N,D] clean training features only. Output: fitted self."""
        self.covariance.fit(
            self.scaler.fit_transform(np.asarray(clean_features))
        )
        return self

    def score_samples(self, features):
        """Input: ndarray[N,D]. Output: ndarray[N] squared distances, higher = anomalous."""
        return self.covariance.mahalanobis(self.scaler.transform(features))
