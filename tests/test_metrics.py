"""
Tests for the shared detection evaluation metrics.

Expected values here are hand-computed rather than taken from a previous
run of the code, so these tests verify the metric definitions themselves,
not merely that the implementation is self-consistent.
"""

import numpy as np
import pytest

from analysis.metrics import (
    DetectionEvaluator,
    check_score_orientation,
    clean_vs_clean_baseline,
    compute_auc,
    compute_tpr_at_fpr,
    filter_successful_attacks,
    paired_cosine_similarity,
    summarize,
)


# Fixture used across several tests, with hand-computed metrics.
#
# Negatives (clean, label 0): nine at 0.0, one at 0.95
# Positives (adversarial, label 1): five at 1.0, five at 0.5
#
# ROC points, sweeping the threshold downward:
#   threshold 1.00 -> TPR 0.5, FPR 0.0
#   threshold 0.95 -> TPR 0.5, FPR 0.1
#   threshold 0.50 -> TPR 1.0, FPR 0.1
#
# AUC by pairwise counting over 10 x 10 = 100 (positive, negative) pairs:
#   1.0 vs 0.0  -> 5 * 9 = 45 wins
#   1.0 vs 0.95 -> 5 * 1 =  5 wins
#   0.5 vs 0.0  -> 5 * 9 = 45 wins
#   0.5 vs 0.95 -> 0 wins (0.5 < 0.95)
#   total 95 wins, no ties -> AUC = 95 / 100 = 0.95
PARTIAL_SCORES = np.array(
    [0.0] * 9 + [0.95] + [1.0] * 5 + [0.5] * 5
)
PARTIAL_LABELS = np.array([0] * 10 + [1] * 10)


def test_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])

    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_no_signal():
    """Identical scores for both classes must give AUC 0.5, not 0 or 1."""
    scores = np.array([0.5] * 6)
    labels = np.array([0, 0, 0, 1, 1, 1])

    assert compute_auc(scores, labels) == pytest.approx(0.5)


def test_auc_hand_computed():
    assert compute_auc(PARTIAL_SCORES, PARTIAL_LABELS) == pytest.approx(0.95)


def test_auc_inverted_scores():
    """A backwards detector should report 1 - AUC, not silently pass."""
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([1, 1, 1, 0, 0, 0])

    assert compute_auc(scores, labels) == pytest.approx(0.0)


def test_auc_single_class_is_undefined_not_a_crash():
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([1, 1, 1])

    assert np.isnan(compute_auc(scores, labels))


def test_tpr_at_fpr_conservative_convention():
    """
    TPR must be the best value attainable WITHOUT exceeding the budget.

    At a 5% budget the only threshold with FPR <= 0.05 is the one giving
    TPR 0.5; the threshold that reaches TPR 1.0 costs FPR 0.1 and must not
    be credited. Interpolating between ROC points would wrongly report
    something above 0.5 here.
    """
    assert compute_tpr_at_fpr(
        PARTIAL_SCORES, PARTIAL_LABELS, 0.05
    ) == pytest.approx(0.5)

    assert compute_tpr_at_fpr(
        PARTIAL_SCORES, PARTIAL_LABELS, 0.01
    ) == pytest.approx(0.5)

    # At a 10% budget the better threshold becomes affordable.
    assert compute_tpr_at_fpr(
        PARTIAL_SCORES, PARTIAL_LABELS, 0.1
    ) == pytest.approx(1.0)


def test_tpr_at_fpr_perfect_detector():
    scores = np.array([0.0] * 10 + [1.0] * 10)
    labels = np.array([0] * 10 + [1] * 10)

    assert compute_tpr_at_fpr(scores, labels, 0.01) == pytest.approx(1.0)


def test_tpr_at_fpr_rejects_out_of_range_budget():
    with pytest.raises(ValueError):
        compute_tpr_at_fpr(PARTIAL_SCORES, PARTIAL_LABELS, 1.5)


def test_filter_successful_attacks():
    true_labels = np.array([0, 1, 2, 3])
    clean_predictions = np.array([0, 1, 9, 3])  # index 2 clean-wrong
    adversarial_predictions = np.array([5, 1, 5, 7])  # index 1 unchanged

    mask = filter_successful_attacks(
        clean_predictions,
        adversarial_predictions,
        true_labels,
    )

    # index 0: clean right, adv wrong  -> success
    # index 1: clean right, adv right  -> attack failed
    # index 2: clean wrong             -> excluded regardless
    # index 3: clean right, adv wrong  -> success
    assert mask.tolist() == [True, False, False, True]


def test_filter_successful_attacks_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        filter_successful_attacks(
            np.array([0, 1]),
            np.array([0, 1, 2]),
            np.array([0, 1]),
        )


def test_check_score_orientation():
    scores = np.array([0.1, 0.2, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])

    assert check_score_orientation(scores, labels) is True
    assert check_score_orientation(-scores, labels) is False


def test_paired_cosine_similarity_known_values():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = paired_cosine_similarity(a, b)

    assert result == pytest.approx([1.0, 0.0, -1.0])


def test_paired_cosine_similarity_handles_zero_vector():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])

    assert paired_cosine_similarity(a, b) == pytest.approx([0.0])


def test_clean_vs_clean_baseline_never_self_pairs():
    """
    Identical rows would give similarity 1.0 only if a row were compared
    against itself, so an orthogonal-pair construction catches self-pairing.
    """
    features = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )

    # Offset defaults to 4 // 2 = 2, pairing each row with the identical
    # row two positions along -> similarity 1.0 but NOT a self-comparison.
    result = clean_vs_clean_baseline(features)
    assert result == pytest.approx([1.0, 1.0, 1.0, 1.0])

    # Offset 1 pairs orthogonal rows.
    result = clean_vs_clean_baseline(features, offset=1)
    assert result == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_clean_vs_clean_baseline_rejects_self_pairing_offset():
    features = np.random.default_rng(0).normal(size=(6, 3))

    with pytest.raises(ValueError):
        clean_vs_clean_baseline(features, offset=6)

    with pytest.raises(ValueError):
        clean_vs_clean_baseline(features, offset=0)


def test_clean_vs_clean_baseline_requires_two_samples():
    with pytest.raises(ValueError):
        clean_vs_clean_baseline(np.array([[1.0, 2.0]]))


def test_summarize():
    result = summarize([1.0, 2.0, 3.0])

    assert result["mean"] == pytest.approx(2.0)
    assert result["min"] == pytest.approx(1.0)
    assert result["max"] == pytest.approx(3.0)
    assert result["count"] == 3


def test_summarize_uses_sample_std_matching_torch():
    """
    Must agree with torch.std() (ddof=1), not np.std() (ddof=0), so the
    existing torch-based experiment scripts and this module report the same
    number. For [1, 2, 3]: sample std = 1.0, population std = 0.8165.
    """
    result = summarize([1.0, 2.0, 3.0])

    assert result["std"] == pytest.approx(1.0)

    torch = pytest.importorskip("torch")
    expected = torch.tensor([1.0, 2.0, 3.0]).std().item()
    assert result["std"] == pytest.approx(expected)


def test_summarize_single_value_has_undefined_std():
    result = summarize([42.0])

    assert result["mean"] == pytest.approx(42.0)
    assert np.isnan(result["std"])
    assert result["count"] == 1


def test_evaluator_excludes_failed_attacks():
    """
    The two adversarial examples the attack failed on carry clean-looking
    scores. Including them would drag AUC down; the protocol says they must
    be dropped, leaving perfect separation.
    """
    clean_scores = np.array([0.0, 0.1, 0.2, 0.1])
    adversarial_scores = np.array([0.9, 0.05, 1.0, 0.0])
    success_mask = np.array([True, False, True, False])

    evaluator = DetectionEvaluator()
    result = evaluator.evaluate(
        clean_scores,
        adversarial_scores,
        success_mask,
    )

    assert result["auc"] == pytest.approx(1.0)
    assert result["num_clean"] == 4
    assert result["num_adversarial_evaluated"] == 2
    assert result["num_adversarial_total"] == 4
    assert result["attack_success_rate"] == pytest.approx(0.5)


def test_evaluator_without_mask_uses_everything():
    clean_scores = np.array([0.0, 0.1])
    adversarial_scores = np.array([0.9, 1.0])

    result = DetectionEvaluator().evaluate(clean_scores, adversarial_scores)

    assert result["auc"] == pytest.approx(1.0)
    assert result["num_adversarial_evaluated"] == 2
    assert result["attack_success_rate"] == pytest.approx(1.0)


def test_evaluator_reports_configured_fpr_targets():
    clean_scores = np.array([0.0] * 10)
    adversarial_scores = np.array([1.0] * 10)

    result = DetectionEvaluator().evaluate(clean_scores, adversarial_scores)

    assert set(result["tpr_at_fpr"]) == {"0.01", "0.05"}
    assert result["tpr_at_fpr"]["0.01"] == pytest.approx(1.0)


def test_evaluator_rejects_mismatched_mask():
    with pytest.raises(ValueError):
        DetectionEvaluator().evaluate(
            np.array([0.0, 0.1]),
            np.array([0.9, 1.0]),
            np.array([True]),
        )


def test_evaluator_accepts_torch_tensors():
    """Both pipelines produce torch tensors, so these must not need .numpy()."""
    torch = pytest.importorskip("torch")

    result = DetectionEvaluator().evaluate(
        torch.tensor([0.0, 0.1]),
        torch.tensor([0.9, 1.0]),
        torch.tensor([True, True]),
    )

    assert result["auc"] == pytest.approx(1.0)
