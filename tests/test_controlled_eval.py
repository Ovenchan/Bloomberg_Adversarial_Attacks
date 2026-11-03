"""Tests for frozen calibration, paired bootstrap and the controlled split."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.bootstrap import fast_auc, paired_bootstrap
from analysis.calibration import calibrate_threshold, frozen_detection_metrics, rate_at_threshold
from utils.controlled_data import PARTITION_ORDER, build_manifest, derive_seed, partition_counts

IMAGENETTE_ROOT = Path("data/raw/imagenette2/val")


def test_calibration_rule_bounds_fpr_and_uses_strict_decisions():
    """Threshold is the (k+1)-th largest score; at most k scores exceed it."""
    scores = np.arange(100, dtype=float)
    calibration = calibrate_threshold(scores, 0.05)
    assert calibration["threshold"] == 94.0
    assert calibration["calibration_fpr"] == 0.05
    assert calibrate_threshold(np.arange(50.0), 0.01)["threshold"] == 49.0  # k = 0 -> maximum


def test_calibration_with_ties_never_flags_tied_scores():
    """All-equal calibration scores give zero calibration false alarms."""
    calibration = calibrate_threshold(np.full(40, 0.3), 0.05)
    assert calibration["calibration_fpr"] == 0.0
    assert rate_at_threshold([0.3, 0.3, 0.31], calibration["threshold"])["count"] == 1


def test_frozen_metrics_report_undefined_without_successful_attacks():
    """No successful attacks -> NaN AUC/TPR with explicit counts, not 0 or 1."""
    thresholds = [calibrate_threshold(np.linspace(0, 1, 20), 0.05)]
    metrics = frozen_detection_metrics(np.linspace(0, 1, 10), np.zeros(0), thresholds, num_eligible=8)
    assert np.isnan(metrics["auc"]) and metrics["num_successful"] == 0
    assert np.isnan(metrics["operating_points"]["0.05"]["tpr"])


def test_frozen_metrics_undetected_rate_uses_eligible_denominator():
    """Undetected successful attacks are divided by eligible attempts."""
    thresholds = [calibrate_threshold(np.zeros(20), 0.05)]
    metrics = frozen_detection_metrics(np.zeros(10), np.array([0.0, 1.0, 1.0]), thresholds, num_eligible=6)
    point = metrics["operating_points"]["0.05"]
    assert point["tpr"] == pytest.approx(2 / 3) and point["undetected_success_rate"] == pytest.approx(1 / 6)


def test_fast_auc_matches_sklearn_with_ties():
    """Rank AUC equals sklearn including tied scores."""
    rng = np.random.default_rng(0)
    negatives = rng.integers(0, 5, 40).astype(float)
    positives = rng.integers(2, 7, 30).astype(float)
    expected = roc_auc_score(np.r_[np.zeros(40), np.ones(30)], np.r_[negatives, positives])
    assert fast_auc(negatives, positives) == pytest.approx(expected)


def test_paired_bootstrap_identical_methods_have_zero_difference():
    """Two identical score sets differ by exactly 0 in every replicate."""
    rng = np.random.default_rng(1)
    sources = np.array([f"s{i}" for i in range(30)])
    scores = {"clean": rng.normal(size=30), "positive": rng.normal(1, 1, 20), "calibration": rng.normal(size=25)}
    result = paired_bootstrap({"a": scores, "b": dict(scores)}, "a", sources, sources[:20], replicates=50)
    assert result["b"]["auc_diff"] == (0.0, 0.0, 0.0)
    assert result["b"]["auc"][0] <= result["b"]["auc"][1]


def test_derive_seed_is_deterministic_and_purpose_specific():
    """Seeds depend only on their parts."""
    assert derive_seed(42, "src", "slice") == derive_seed(42, "src", "slice")
    assert derive_seed(42, "src", "slice") != derive_seed(42, "src", "belief")


@pytest.mark.skipif(not IMAGENETTE_ROOT.is_dir(), reason="Imagenette not available")
def test_manifest_partitions_are_disjoint_stratified_and_seeded():
    """Each source is in one partition, partitions are class-balanced, panels lie in test."""
    sizes = {"train": 3, "dev": 1, "cal": 2, "test": 3}
    rows = build_manifest("data/raw", sizes, 2, 1, seed=42)
    assert len({r["source_id"] for r in rows}) == len(rows)
    for name in PARTITION_ORDER:
        counts = Counter(r["class_index"] for r in rows if r["partition"] == name)
        assert set(counts.values()) == {sizes[name]} and len(counts) == 10
        assert sorted(r["partition_row"] for r in rows if r["partition"] == name) == list(range(10 * sizes[name]))
    assert all(r["partition"] == "test" for r in rows if r["noise_panel"] or r["stability_panel"])
    again = build_manifest("data/raw", sizes, 2, 1, seed=42)
    assert [r["partition"] for r in again] == [r["partition"] for r in rows]


def test_partition_counts_accept_counts_and_fractions():
    """Integer counts pass through; fractions summing to 1 use the whole pool."""
    counts = {"train": 35, "dev": 10, "cal": 20, "test": 35}
    assert partition_counts(100, counts) == counts
    fractions = partition_counts(387, {"train": 0.35, "dev": 0.10, "cal": 0.20, "test": 0.35})
    assert sum(fractions.values()) == 387 and fractions["dev"] == 38
    with pytest.raises(ValueError):
        partition_counts(50, counts)


@pytest.mark.skipif(not IMAGENETTE_ROOT.is_dir(), reason="Imagenette not available")
def test_fixed_subset_has_1000_files_from_all_classes():
    """subset_per_class=100 -> fixed 1,000 sources, 100 per class, no reserve, reproducible."""
    sizes = {"train": 35, "dev": 10, "cal": 20, "test": 35}
    rows = [r for r in build_manifest("data/raw", sizes, 10, 2, seed=42, subset_per_class=100)
            if r["partition"] != "reserve"]
    assert len(rows) == 1000 and set(Counter(r["class_index"] for r in rows).values()) == {100}
    again = build_manifest("data/raw", sizes, 10, 2, seed=42, subset_per_class=100)
    assert {r["source_id"] for r in again if r["partition"] != "reserve"} == {r["source_id"] for r in rows}


@pytest.mark.skipif(not IMAGENETTE_ROOT.is_dir(), reason="Imagenette not available")
def test_full_val_fractions_assign_every_file():
    """Fractional partitions over the whole val split leave no reserve images."""
    rows = build_manifest("data/raw", {"train": 0.35, "dev": 0.10, "cal": 0.20, "test": 0.35}, 1, 1, seed=42)
    assert len(rows) == 3925 and not any(r["partition"] == "reserve" for r in rows)
