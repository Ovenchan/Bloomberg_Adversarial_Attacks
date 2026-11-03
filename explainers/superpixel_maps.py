"""Common spatial conversion of explanations into detector vectors.

Superpixel coefficients are rasterised to pixels (sign and superpixel ID
preserved), converted to the primary positive-evidence map, scaled by the
positive maximum and passed through the *same* `GradCAMVectorizer`
(bilinear 14x14 -> 196 features) used for Grad-CAM heatmaps, so all three
explainers share one capacity-matched feature transform. A signed ablation
concatenates positive and negative maps (392 features), both scaled by the
largest absolute coefficient.
"""

import numpy as np
import torch

from explainers.vectorizer import DEFAULT_GRID_SIZE, GradCAMVectorizer


def rasterize(coefficients, segments):
    """Input: ndarray[K] coefficients, ndarray[H,W] labels 0..K-1.

    Output: ndarray[H,W] float32 signed pixel map.
    """
    return np.asarray(coefficients, dtype=np.float32)[np.asarray(segments)]


def superpixel_means(pixel_map, segments, num_segments=None):
    """Average a dense map inside each superpixel (Grad-CAM -> superpixel ranks).

    Input: ndarray[H,W] map, ndarray[H,W] labels, optional int K.
    Output: ndarray[K] float64 mean value per superpixel.
    """
    labels = np.asarray(segments).ravel()
    K = int(labels.max()) + 1 if num_segments is None else num_segments
    sums = np.bincount(labels, weights=np.asarray(pixel_map, np.float64).ravel(), minlength=K)
    counts = np.bincount(labels, minlength=K)
    return sums / np.maximum(counts, 1)


class SuperpixelMapVectorizer:
    """Input (constructor): int grid size. Output: coefficient-to-vector transformer."""

    def __init__(self, grid_size=DEFAULT_GRID_SIZE):
        """Input: positive int. Output: None."""
        self.vectorizer = GradCAMVectorizer(grid_size)

    def positive_maps(self, coefficient_list, segment_list):
        """Input: lists of ndarray[K_i] and ndarray[H,W]. Output: ndarray[N,H,W].

        Positive part divided by its maximum; inputs without positive
        evidence give an all-zero map (counted by `zero_map_mask`).
        """
        maps = []
        for coefficients, segments in zip(coefficient_list, segment_list):
            positive = np.clip(rasterize(coefficients, segments), 0, None)
            peak = positive.max()
            maps.append(positive / peak if peak > 0 else positive)
        return np.stack(maps)

    def transform(self, coefficient_list, segment_list):
        """Input: lists of coefficients and segments. Output: ndarray[N,grid^2]."""
        return self.vectorizer.transform(
            torch.from_numpy(self.positive_maps(coefficient_list, segment_list))
        )

    def transform_signed(self, coefficient_list, segment_list):
        """Input: lists of coefficients and segments. Output: ndarray[N,2*grid^2]."""
        positive, negative = [], []
        for coefficients, segments in zip(coefficient_list, segment_list):
            signed = rasterize(coefficients, segments)
            peak = np.abs(signed).max()
            signed = signed / peak if peak > 0 else signed
            positive.append(np.clip(signed, 0, None))
            negative.append(np.clip(-signed, 0, None))
        return np.concatenate(
            [
                self.vectorizer.transform(torch.from_numpy(np.stack(positive))),
                self.vectorizer.transform(torch.from_numpy(np.stack(negative))),
            ],
            axis=1,
        )

    @staticmethod
    def zero_map_mask(coefficient_list):
        """Input: list of ndarray[K]. Output: ndarray[N] bool, no positive evidence."""
        return np.array([not (np.asarray(c) > 0).any() for c in coefficient_list])
