"""Paired source-identity bootstrap for detector comparisons.

Each replicate resamples test source identities with replacement (all
variants of a source move together) and, independently, calibration
identities; thresholds are re-derived inside every replicate so the
operating-point uncertainty includes calibration noise.
"""

import numpy as np
from scipy.stats import rankdata

from analysis.calibration import calibrate_threshold

DEFAULT_REPLICATES = 2000
DEFAULT_ALPHA = 0.05
INTERVAL_PERCENTILES = (2.5, 97.5)


def fast_auc(negatives, positives):
    """Mann-Whitney ROC-AUC with average ranks for ties.

    Input: array[n0] clean scores, array[n1] adversarial scores.
    Output: float AUC (NaN if either side is empty).
    """
    n0, n1 = len(negatives), len(positives)
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = rankdata(np.r_[negatives, positives])
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def paired_bootstrap(methods, reference, test_sources, positive_sources,
                     replicates=DEFAULT_REPLICATES, alpha=DEFAULT_ALPHA, seed=0):
    """Bootstrap AUC and frozen-threshold TPR, with differences to a reference.

    Input:
        methods: dict name -> {"clean": array[n_test] (aligned with
            test_sources), "positive": array[m] (aligned with positive_sources),
            "calibration": array[n_cal]}.
        reference: str method name used for paired differences.
        test_sources: array[n_test] source ids of clean test rows.
        positive_sources: array[m] source ids of the successful attacks.
        replicates: int, alpha: float target FPR, seed: int.
    Output: dict name -> {"auc": (low, high), "tpr": (low, high),
        "auc_diff": (mean, low, high), "tpr_diff": (mean, low, high)}.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(test_sources)
    clean_index = {s: np.flatnonzero(test_sources == s) for s in unique}
    positive_index = {s: np.flatnonzero(positive_sources == s) for s in unique}
    n_cal = len(next(iter(methods.values()))["calibration"])
    draws = {name: {"auc": [], "tpr": []} for name in methods}
    for _ in range(replicates):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        clean_rows = np.concatenate([clean_index[s] for s in chosen])
        positive_rows = np.concatenate([positive_index[s] for s in chosen])
        cal_rows = rng.integers(0, n_cal, n_cal)
        for name, scores in methods.items():
            clean = scores["clean"][clean_rows]
            positive = scores["positive"][positive_rows]
            threshold = calibrate_threshold(scores["calibration"][cal_rows], alpha)["threshold"]
            draws[name]["auc"].append(fast_auc(clean, positive))
            draws[name]["tpr"].append(np.mean(positive > threshold) if positive.size else np.nan)
    summary = {}
    for name in methods:
        auc = np.asarray(draws[name]["auc"])
        tpr = np.asarray(draws[name]["tpr"])
        entry = {
            "auc": tuple(np.nanpercentile(auc, INTERVAL_PERCENTILES)),
            "tpr": tuple(np.nanpercentile(tpr, INTERVAL_PERCENTILES)),
        }
        for metric, values in (("auc", auc), ("tpr", tpr)):
            diff = values - np.asarray(draws[reference][metric])
            entry[f"{metric}_diff"] = (
                float(np.nanmean(diff)), *np.nanpercentile(diff, INTERVAL_PERCENTILES)
            )
        summary[name] = entry
    return summary
