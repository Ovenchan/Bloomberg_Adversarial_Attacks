"""
Shared detection evaluation metrics.

This module is the single implementation of the project's evaluation
standard. Both the explanation-based (CNN + Grad-CAM) and the
representation-based (DINOv2) pipelines must score their detectors through
this module — if the two pipelines compute metrics differently, the
comparison between them is not valid.

The protocol follows ViTGuard (Sun et al., ACSAC 2024):

    - AUC is the primary headline metric.
    - TPR at a fixed low FPR (1% and 5%) is reported alongside it, because
      raw accuracy is misleading for a detector that will be deployed at a
      low false-alarm budget.
    - Detection is scored ONLY on successful attacks: the clean image must
      be classified correctly AND the adversarial image must be classified
      incorrectly. Failed attack attempts are excluded, since an "attack"
      that never changed the prediction is not something a detector should
      be credited or penalised for finding.

Score direction convention (important):

    Detector scores must be oriented so that a HIGHER score means MORE
    likely adversarial, and labels must use 1 = adversarial, 0 = clean.
    A detector wired up backwards will report AUC = 1 - (true AUC), which
    looks like a broken detector rather than a flipped sign. Use
    `check_score_orientation` if unsure.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# FPR budgets at which TPR is reported. CLAUDE.md fixes these at 1% and 5%.
DEFAULT_FPR_TARGETS = (0.01, 0.05)

# Label convention for detection scoring.
LABEL_CLEAN = 0
LABEL_ADVERSARIAL = 1

# Returned when a metric is mathematically undefined for the given input
# (for example AUC when only one class is present).
UNDEFINED_METRIC = float("nan")


def _as_1d_array(values, name):
    """
    Coerce input to a flat float NumPy array.

    Input:
        values: array-like (list, np.ndarray, or torch.Tensor).
        name: str, used in error messages.
    Output:
        np.ndarray of shape (N,), dtype float64.
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    array = np.asarray(values).reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} is empty")

    return array.astype(np.float64)


def filter_successful_attacks(
    clean_predictions,
    adversarial_predictions,
    true_labels,
):
    """
    Identify attacks that actually succeeded, per the ViTGuard protocol.

    An attack counts as successful only if the model classified the clean
    image correctly and then classified the adversarial image incorrectly.

    Input:
        clean_predictions: array-like (N,), predicted class for clean images.
        adversarial_predictions: array-like (N,), predicted class for
            adversarial images.
        true_labels: array-like (N,), ground-truth class labels.
    Output:
        np.ndarray of shape (N,), dtype bool. True where the attack succeeded.
    """
    clean = _as_1d_array(clean_predictions, "clean_predictions")
    adversarial = _as_1d_array(
        adversarial_predictions,
        "adversarial_predictions",
    )
    labels = _as_1d_array(true_labels, "true_labels")

    if not (clean.shape == adversarial.shape == labels.shape):
        raise ValueError(
            "clean_predictions, adversarial_predictions and true_labels "
            f"must have the same length, got {clean.shape}, "
            f"{adversarial.shape}, {labels.shape}"
        )

    clean_correct = clean == labels
    adversarial_wrong = adversarial != labels

    return clean_correct & adversarial_wrong


def compute_auc(scores, labels):
    """
    Area under the ROC curve for detection scores.

    Input:
        scores: array-like (N,), higher = more likely adversarial.
        labels: array-like (N,), 1 = adversarial, 0 = clean.
    Output:
        float. AUC in [0, 1], or NaN if only one class is present.
    """
    score_array = _as_1d_array(scores, "scores")
    label_array = _as_1d_array(labels, "labels")

    if score_array.shape != label_array.shape:
        raise ValueError(
            "scores and labels must have the same length, got "
            f"{score_array.shape} and {label_array.shape}"
        )

    if np.unique(label_array).size < 2:
        return UNDEFINED_METRIC

    return float(roc_auc_score(label_array, score_array))


def compute_tpr_at_fpr(scores, labels, target_fpr):
    """
    True positive rate at a fixed false positive rate budget.

    Uses the conservative convention: the highest TPR achievable at any
    threshold whose FPR does not EXCEED `target_fpr`. No interpolation
    between ROC points, so the reported number is always attainable by a
    real threshold rather than implied by one.

    Input:
        scores: array-like (N,), higher = more likely adversarial.
        labels: array-like (N,), 1 = adversarial, 0 = clean.
        target_fpr: float in [0, 1], the false-alarm budget.
    Output:
        float. TPR in [0, 1], or NaN if only one class is present.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(
            f"target_fpr must be in [0, 1], got {target_fpr}"
        )

    score_array = _as_1d_array(scores, "scores")
    label_array = _as_1d_array(labels, "labels")

    if np.unique(label_array).size < 2:
        return UNDEFINED_METRIC

    false_positive_rate, true_positive_rate, _ = roc_curve(
        label_array,
        score_array,
    )

    within_budget = false_positive_rate <= target_fpr

    if not within_budget.any():
        return 0.0

    return float(true_positive_rate[within_budget].max())


def check_score_orientation(scores, labels):
    """
    Sanity check that detector scores are not wired up backwards.

    Input:
        scores: array-like (N,), higher = more likely adversarial.
        labels: array-like (N,), 1 = adversarial, 0 = clean.
    Output:
        bool. True if the orientation looks correct (AUC >= 0.5), False if
        the scores appear inverted. NaN AUC returns True (nothing to check).
    """
    auc = compute_auc(scores, labels)

    if np.isnan(auc):
        return True

    return auc >= 0.5


def paired_cosine_similarity(features_a, features_b):
    """
    Row-wise cosine similarity between two aligned feature matrices.

    Input:
        features_a: array-like (N, D).
        features_b: array-like (N, D), row i corresponds to row i of a.
    Output:
        np.ndarray of shape (N,), cosine similarity per row.
    """
    if hasattr(features_a, "detach"):
        features_a = features_a.detach().cpu().numpy()

    if hasattr(features_b, "detach"):
        features_b = features_b.detach().cpu().numpy()

    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"feature matrices must have the same shape, got {a.shape} "
            f"and {b.shape}"
        )

    numerator = (a * b).sum(axis=1)

    denominator = (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )

    # Guard against zero-norm rows rather than emitting divide-by-zero NaNs.
    denominator = np.where(denominator == 0.0, 1e-12, denominator)

    return numerator / denominator


def clean_vs_clean_baseline(clean_features, offset=None):
    """
    Background cosine-similarity level between UNRELATED clean features.

    Answers "how similar are two different clean images anyway?", which is
    what makes a clean-vs-adversarial similarity interpretable. A
    clean/adversarial similarity of 0.7 means very different things
    depending on whether unrelated images sit at 0.2 or at 0.65.

    Each row is paired with a different row via a fixed cyclic offset, so
    no feature vector is ever compared against itself.

    Input:
        clean_features: array-like (N, D), N >= 2.
        offset: int or None. Cyclic shift used to build pairs. Defaults to
            N // 2. Must not be a multiple of N (that would self-pair).
    Output:
        np.ndarray of shape (N,), cosine similarity per pair.
    """
    if hasattr(clean_features, "detach"):
        clean_features = clean_features.detach().cpu().numpy()

    features = np.asarray(clean_features, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError(
            f"clean_features must be 2D (N, D), got shape {features.shape}"
        )

    num_samples = features.shape[0]

    if num_samples < 2:
        raise ValueError(
            "clean_vs_clean_baseline needs at least 2 samples, got "
            f"{num_samples}"
        )

    if offset is None:
        offset = num_samples // 2

    if offset % num_samples == 0:
        raise ValueError(
            f"offset {offset} would pair rows with themselves for "
            f"{num_samples} samples"
        )

    shifted_index = (
        np.arange(num_samples) + offset
    ) % num_samples

    return paired_cosine_similarity(
        features,
        features[shifted_index],
    )


def summarize(values):
    """
    Mean/std/min/max summary of a 1D array of values.

    Uses the SAMPLE standard deviation (ddof=1), matching torch.std()'s
    default. NumPy defaults to the population std (ddof=0) instead, so
    reporting a mix of the two across scripts would put two subtly
    different "std" columns in the same results table.

    Input:
        values: array-like (N,).
    Output:
        dict with keys "mean", "std", "min", "max", "count". "std" is NaN
        for a single value, which has no sample standard deviation.
    """
    array = _as_1d_array(values, "values")

    standard_deviation = (
        float(array.std(ddof=1)) if array.size > 1 else UNDEFINED_METRIC
    )

    return {
        "mean": float(array.mean()),
        "std": standard_deviation,
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


class DetectionEvaluator:
    """
    The project's shared detection evaluation protocol.

    Deliberately the single implementation used by BOTH pipelines. Pipeline
    1 passes Grad-CAM heatmap-derived detector scores; Pipeline 2 passes
    embedding-distance detector scores. Everything downstream — the
    successful-attack filter, AUC, and TPR at fixed FPR — is then identical
    by construction.

    Input (constructor):
        fpr_targets: tuple of float, FPR budgets at which to report TPR.
            Defaults to DEFAULT_FPR_TARGETS (1% and 5%).
    """

    def __init__(self, fpr_targets=DEFAULT_FPR_TARGETS):
        self.fpr_targets = tuple(fpr_targets)

    def evaluate(
        self,
        clean_scores,
        adversarial_scores,
        success_mask=None,
    ):
        """
        Score a detector on clean vs adversarial examples.

        Input:
            clean_scores: array-like (N,), detector score per clean image.
                Higher = more likely adversarial.
            adversarial_scores: array-like (N,), detector score per
                adversarial image. Row i must correspond to row i of
                clean_scores (same source image).
            success_mask: array-like (N,) of bool, or None. Which attacks
                succeeded, from `filter_successful_attacks`. Adversarial
                examples where the attack failed are dropped. Clean scores
                are kept in full, since they define the false-positive rate
                and are unaffected by whether an attack worked.
        Output:
            dict with keys:
                "auc": float
                "tpr_at_fpr": dict mapping str(fpr) -> float
                "num_clean": int
                "num_adversarial_evaluated": int
                "num_adversarial_total": int
                "attack_success_rate": float
        """
        clean = _as_1d_array(clean_scores, "clean_scores")
        adversarial = _as_1d_array(
            adversarial_scores,
            "adversarial_scores",
        )

        num_adversarial_total = adversarial.size

        if success_mask is not None:
            mask = np.asarray(success_mask).reshape(-1).astype(bool)

            if mask.size != num_adversarial_total:
                raise ValueError(
                    "success_mask length must match adversarial_scores, "
                    f"got {mask.size} and {num_adversarial_total}"
                )

            adversarial = adversarial[mask]

        scores = np.concatenate([clean, adversarial])

        labels = np.concatenate(
            [
                np.full(clean.size, LABEL_CLEAN),
                np.full(adversarial.size, LABEL_ADVERSARIAL),
            ]
        )

        tpr_at_fpr = {
            str(target): compute_tpr_at_fpr(scores, labels, target)
            for target in self.fpr_targets
        }

        return {
            "auc": compute_auc(scores, labels),
            "tpr_at_fpr": tpr_at_fpr,
            "num_clean": int(clean.size),
            "num_adversarial_evaluated": int(adversarial.size),
            "num_adversarial_total": int(num_adversarial_total),
            "attack_success_rate": (
                float(adversarial.size / num_adversarial_total)
                if num_adversarial_total > 0
                else UNDEFINED_METRIC
            ),
        }
