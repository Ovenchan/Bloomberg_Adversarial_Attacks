"""Scaled logistic detector; the scaler is fitted only with detector training data."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from analysis.metrics import LABEL_CLEAN, LABEL_ADVERSARIAL

DEFAULT_SEED = 42
DEFAULT_MAX_ITER = 5000
DEFAULT_C = 1.0


class GradCAMLogisticDetector:
    """Input: logistic settings. Output: detector with adversarial probability scores."""

    def __init__(
        self, seed=DEFAULT_SEED, max_iter=DEFAULT_MAX_ITER, C=DEFAULT_C
    ):
        """Input: int seed/max_iter, float C. Output: None."""
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=seed, max_iter=max_iter, C=C),
        )

    def fit(self, features, labels):
        """Input: ndarray[N,D], binary ndarray[N]. Output: fitted self."""
        if set(np.unique(labels)) != {LABEL_CLEAN, LABEL_ADVERSARIAL}:
            raise ValueError(
                "Training requires both clean=0 and adversarial=1"
            )
        self.model.fit(features, labels)
        return self

    def score_samples(self, features):
        """Input: ndarray[N,D]. Output: ndarray[N], higher = more adversarial."""
        column = list(self.model.classes_).index(LABEL_ADVERSARIAL)
        return self.model.predict_proba(features)[:, column]
