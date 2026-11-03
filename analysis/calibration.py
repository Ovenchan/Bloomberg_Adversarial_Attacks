"""Operating thresholds frozen on independent clean calibration scores.

`analysis.metrics.compute_tpr_at_fpr` picks the best attainable point of
the *evaluation* ROC curve (a descriptive statistic). This module instead
implements the deployable protocol: thresholds come only from clean
calibration scores and are then applied unchanged to every test condition.

Rule (version CALIBRATION_RULE): with n calibration scores sorted in
descending order s_1 >= ... >= s_n and k = floor(alpha * n), the threshold
is s_{k+1} (the maximum s_1 when k = 0) and decisions are strict,
score > threshold. At most k
calibration scores exceed it, so the calibration FPR never exceeds alpha;
tied scores at the threshold are never flagged.
"""

import numpy as np
from scipy.stats import binomtest

from analysis.metrics import compute_auc

CALIBRATION_RULE = "conservative_upper_quantile_strict_v1"
CONFIDENCE_LEVEL = 0.95


def calibrate_threshold(clean_scores, alpha):
    """Input: array[n] clean calibration scores, float target FPR alpha.

    Output: dict with threshold, achieved calibration FPR, n and rule name.
    """
    scores = np.sort(np.asarray(clean_scores, dtype=np.float64))[::-1]
    if scores.size == 0:
        raise ValueError("Calibration requires clean scores")
    k = int(np.floor(alpha * scores.size))
    threshold = float(scores[min(k, scores.size - 1)])
    return {
        "alpha": float(alpha),
        "threshold": threshold,
        "calibration_fpr": float(np.mean(scores > threshold)),
        "num_calibration": int(scores.size),
        "rule": CALIBRATION_RULE,
    }


def clopper_pearson(successes, trials, level=CONFIDENCE_LEVEL):
    """Input: int successes, int trials. Output: (low, high) exact binomial CI or NaNs."""
    if trials == 0:
        return float("nan"), float("nan")
    interval = binomtest(int(successes), int(trials)).proportion_ci(level, method="exact")
    return float(interval.low), float(interval.high)


def rate_at_threshold(scores, threshold):
    """Input: array scores, float threshold. Output: dict rate, count, n, 95% CI."""
    scores = np.asarray(scores, dtype=np.float64)
    flagged = int(np.sum(scores > threshold))
    low, high = clopper_pearson(flagged, scores.size)
    return {
        "rate": flagged / scores.size if scores.size else float("nan"),
        "count": flagged,
        "n": int(scores.size),
        "ci_low": low,
        "ci_high": high,
    }


def frozen_detection_metrics(clean_test_scores, adversarial_scores, thresholds, num_eligible=None):
    """Detection of successful attacks at frozen calibration thresholds.

    Input: array clean test scores, array scores of *successful* attacks,
        list of calibrate_threshold dicts, optional int eligible attempts.
    Output: dict with AUC, per-alpha TPR / realised clean FPR / undetected
        successful-attack rate among eligible sources, and counts. AUC and
        TPR are NaN (with counts) when no successful attack exists.
    """
    clean = np.asarray(clean_test_scores, dtype=np.float64)
    adversarial = np.asarray(adversarial_scores, dtype=np.float64)
    result = {
        "num_clean": int(clean.size),
        "num_successful": int(adversarial.size),
        "auc": (
            compute_auc(
                np.r_[clean, adversarial], np.r_[np.zeros(clean.size), np.ones(adversarial.size)]
            )
            if adversarial.size
            else float("nan")
        ),
        "operating_points": {},
    }
    for calibration in thresholds:
        threshold = calibration["threshold"]
        tpr = rate_at_threshold(adversarial, threshold) if adversarial.size else None
        point = {
            "threshold": threshold,
            "tpr": tpr["rate"] if tpr else float("nan"),
            "tpr_ci": (tpr["ci_low"], tpr["ci_high"]) if tpr else (float("nan"),) * 2,
            "clean_fpr": rate_at_threshold(clean, threshold),
        }
        if num_eligible:
            missed = int(np.sum(adversarial <= threshold))
            point["undetected_success_rate"] = missed / num_eligible
        result["operating_points"][str(calibration["alpha"])] = point
    return result
