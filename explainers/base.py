"""Shared explainer interfaces for the controlled SLICE/BELIEF/Grad-CAM study.

Grad-CAM keeps its original class (`explainers/gradcam.py`); the superpixel
surrogate explainers (SLICE, BELIEF and the LIME fallback they share) derive
from `BaseSurrogateExplainer` in `explainers/perturbation.py`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

STATUS_OK = "ok"
STATUS_LIME_FALLBACK = "lime_fallback"
STATUS_ALL_ELIMINATED = "all_eliminated"
STATUS_NONFINITE = "nonfinite"


@dataclass
class SurrogateExplanation:
    """Result of one superpixel-surrogate explanation.

    Attributes:
        coefficients: ndarray[K] float64, one signed weight per superpixel
            label 0..K-1 (eliminated superpixels carry 0).
        eliminated: ndarray[K] bool, superpixels removed as sign-unstable.
        status: str, one of the STATUS_* constants.
        target_class: int, the fixed explained class.
        prob0: float, probability of target_class on the unperturbed input.
        sigma: float, perturbation blur sigma (0 = replace with baseline).
        num_queries: int, classifier forward passes used (incl. prob0).
        num_steps: int, sampling rounds performed.
        runtime_seconds: float, wall time of the explanation call.
        extra: dict, method-specific diagnostics (JSON-serialisable values).
    """

    coefficients: np.ndarray
    eliminated: np.ndarray
    status: str
    target_class: int
    prob0: float
    sigma: float
    num_queries: int
    num_steps: int
    runtime_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseExplainer(ABC):
    """Abstract explainer. Input: one image context. Output: an explanation."""

    name = "base"

    @abstractmethod
    def explain(self, context, sampler):
        """Input: PerturbationContext, sampler. Output: SurrogateExplanation."""
        raise NotImplementedError
