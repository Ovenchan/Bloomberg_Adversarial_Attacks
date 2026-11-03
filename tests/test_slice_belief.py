"""Parity and unit tests for the PyTorch SLICE/BELIEF ports.

Numerical parity is checked against the pinned upstream reference code
(tests/reference_oracle.py) on float64 fixtures; tests that need the
reference files are skipped when `tmp/research_references` is absent.
"""

import random

import numpy as np
import pytest
import torch
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from torch import nn

from explainers.base import STATUS_LIME_FALLBACK, STATUS_OK
from explainers.belief import BeliefExplainer, SignEntropyBayesianRidge
from explainers.perturbation import (
    ProbabilityAdapter,
    PerturbationQueryEngine,
    ReferenceReplaySampler,
    TorchSampler,
    batched_weighted_ridge,
    cosine_locality_weights,
    gaussian_blur,
    select_sigma,
    weighted_ridge,
)
from explainers.slice import SliceExplainer, kde_sign_entropy
from explainers.superpixel_maps import SuperpixelMapVectorizer, superpixel_means
import reference_oracle as oracle

needs_reference = pytest.mark.skipif(
    not oracle.references_available(), reason="pinned reference code not present"
)
ATOL = 1e-6
RTOL = 1e-5
IMAGE_SIZE = 16
TEST_PERTURB = 100
TEST_BOOTSTRAP = 200


class TorchTemplateModel(nn.Module):
    """Float64 torch twin of oracle.NumpyTemplateModel. Input: Tensor[N,3,H,W]."""

    def __init__(self, templates):
        """Input: ndarray[C,3,H,W]. Output: None."""
        super().__init__()
        self.register_buffer("templates", torch.as_tensor(templates, dtype=torch.float64))

    def forward(self, images):
        """Input: Tensor[N,3,H,W]. Output: Tensor[N,C] logits."""
        return torch.einsum("nchw,kchw->nk", images.double(), self.templates)


def make_fixture(seed=0, weak_blocks=True, zero_model=False):
    """Input: int seed, bools. Output: (image CHW, segments HW, templates)."""
    rng = np.random.default_rng(seed)
    image = rng.normal(size=(3, IMAGE_SIZE, IMAGE_SIZE))
    rows = np.arange(IMAGE_SIZE) // (IMAGE_SIZE // 2)
    cols = np.arange(IMAGE_SIZE) // (IMAGE_SIZE // 4)
    segments = (rows[:, None] * 4 + cols[None, :]).astype(np.int64)
    templates = rng.normal(size=(5, 3, IMAGE_SIZE, IMAGE_SIZE)) * 0.05
    if weak_blocks:
        templates[:, :, segments >= 4] *= 0.02
    if zero_model:
        templates[:] = 0.0
    return image, segments, templates


def ours(image, segments, templates, sigma, batch_size=64):
    """Input: fixture arrays, float sigma. Output: (engine, context)."""
    adapter = ProbabilityAdapter(TorchTemplateModel(templates), batch_size=batch_size)
    engine = PerturbationQueryEngine(adapter, "cpu", fill_rgb=(0.0, 0.0, 0.0), dtype=torch.float64)
    probs = adapter.probabilities(torch.as_tensor(image)[None])[0]
    target = int(probs.argmax())
    return engine, engine.context(torch.as_tensor(image), segments, target, sigma)


def seed_reference_rng(seed):
    """Input: int. Output: None; seeds numpy global and python random."""
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------- algebra


def test_weighted_ridge_matches_sklearn():
    """Weighted ridge with intercept equals sklearn Ridge(alpha=1)."""
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, (60, 7)).astype(float)
    y = rng.normal(size=60)
    w = rng.uniform(0.1, 1.0, 60)
    reference = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
    coef, intercept = weighted_ridge(torch.tensor(X), torch.tensor(y), torch.tensor(w))
    np.testing.assert_allclose(coef.numpy(), reference.coef_, atol=1e-10)
    assert abs(intercept - reference.intercept_) < 1e-10


def test_batched_bootstrap_equals_duplicated_rows():
    """Count-weighted batched ridge equals sklearn fits on resampled rows."""
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, (40, 5)).astype(float)
    y = rng.normal(size=40)
    w = rng.uniform(0.1, 1.0, 40)
    draws = [rng.integers(0, 40, 40) for _ in range(6)]
    counts = np.stack([np.bincount(d, minlength=40) for d in draws]).astype(float)
    batched = batched_weighted_ridge(
        torch.tensor(X), torch.tensor(y), torch.tensor(counts * w[None]), chunk=4
    ).numpy()
    for row, d in zip(batched, draws):
        expected = Ridge(alpha=1.0).fit(X[d], y[d], sample_weight=w[d]).coef_
        np.testing.assert_allclose(row, expected, atol=1e-10)


def test_cosine_weights_match_sklearn_including_zero_rows():
    """Cosine locality kernel equals the reference sklearn computation."""
    X = np.vstack([np.ones(6), np.zeros(6), np.random.default_rng(3).integers(0, 2, (20, 6))])
    distances = pairwise_distances(X, np.ones((1, 6)), metric="cosine").ravel()
    expected = np.sqrt(np.exp(-(distances ** 2) / 0.25 ** 2))
    np.testing.assert_allclose(cosine_locality_weights(torch.tensor(X)).numpy(), expected, atol=1e-12)


def test_gaussian_blur_matches_scipy_per_channel():
    """Torch blur equals scipy gaussian_filter (truncate 4, nearest) per channel."""
    image = np.random.default_rng(4).uniform(size=(3, 20, 17))
    for sigma in (0.1, 0.3, 0.5, 2.0):
        expected = ndimage.gaussian_filter(image, sigma=(0, sigma, sigma), mode="nearest", truncate=4.0)
        actual = gaussian_blur(torch.tensor(image), sigma).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_torch_sampler_bootstrap_counts_sum_to_n():
    """Each bootstrap replicate draws exactly n rows."""
    counts = TorchSampler(0).bootstrap_counts(torch.rand(50, dtype=torch.float64) + 0.1, 30)
    assert counts.shape == (30, 50) and torch.all(counts.sum(1) == 50)


@needs_reference
def test_kde_sign_entropy_matches_reference():
    """Vectorised KDE sign entropy equals the reference per-column scipy KDE."""
    module = oracle.load_reference(oracle.SLICE_REFERENCE)
    rng = np.random.default_rng(5)
    columns = [
        rng.normal(0.0, 1.0, 300),       # mixed signs
        rng.normal(3.0, 0.2, 300),       # all positive
        -np.abs(rng.normal(1.0, 1.0, 300)) - 1e-3,  # all negative
        np.zeros(300),                   # constant -> LinAlgError -> 0
        np.r_[rng.uniform(0, 1, 299), -1e-9],  # one tiny negative
    ]
    samples = np.stack(columns, axis=1)
    expected = np.array([module.SliceExplainer._calculate_entropy(None, c) for c in columns])
    actual = kde_sign_entropy(torch.tensor(samples)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert np.array_equal(actual == 0, expected == 0)


@needs_reference
@pytest.mark.parametrize("case", range(4))
def test_sem_bayesian_ridge_matches_reference(case):
    """SignEntropyBayesianRidge equals the reference SEMBayesianRidge."""
    module = oracle.load_reference(oracle.BELIEF_REFERENCE)
    rng = np.random.default_rng(10 + case)
    n, d = (200, 10) if case < 2 else (500, 25)
    X = np.vstack([np.ones(d), rng.integers(0, 2, (n - 1, d))]).astype(float)
    beta = rng.normal(size=d) * (rng.uniform(size=d) < 0.5)
    y = X @ beta * 0.05 + rng.normal(scale=0.05 if case % 2 else 0.01, size=n)
    w = rng.uniform(0.05, 1.0, n)
    reference = module.SEMBayesianRidge(slack=1e-6).fit(X, y, sample_weight=w)
    port = SignEntropyBayesianRidge(slack=1e-6).fit(torch.tensor(X), torch.tensor(y), torch.tensor(w))
    assert np.array_equal(port.eliminated_.numpy(), reference.train_mat_sel_idx.astype(bool))
    np.testing.assert_allclose(port.coef_.numpy(), reference.coef_, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(port.lambda_, reference.lambda_, rtol=RTOL)
    np.testing.assert_allclose(port.alpha_, reference.alpha_, rtol=RTOL)


# ------------------------------------------------------- end-to-end parity


@needs_reference
@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_slice_end_to_end_parity(sigma):
    """Same RNG draws -> same SLICE eliminations and coefficients."""
    module = oracle.load_reference(oracle.SLICE_REFERENCE, per_channel_blur=True)
    image, segments, templates = make_fixture()
    reference = module.SliceExplainer.from_arrays(
        np.transpose(image, (1, 2, 0)), segments, oracle.NumpyTemplateModel(templates)
    )
    seed_reference_rng(7)
    unstable, _, _, _, pos, neg = reference.get_slice_explanations(
        sigma, n_iterations=TEST_BOOTSTRAP, num_perturb=TEST_PERTURB
    )
    expected = oracle.reference_coefficients(segments.max() + 1, pos, neg)
    engine, context = ours(image, segments, templates, sigma)
    assert context.target_class == int(reference.top_pred_class[0])
    seed_reference_rng(7)
    result = SliceExplainer(engine, num_perturb=TEST_PERTURB, n_iterations=TEST_BOOTSTRAP).explain(
        context, ReferenceReplaySampler()
    )
    assert result.status == STATUS_OK
    assert np.array_equal(np.flatnonzero(result.eliminated), np.asarray(unstable))
    assert result.eliminated.any() and not result.eliminated.all()
    np.testing.assert_allclose(result.coefficients, expected, atol=ATOL, rtol=RTOL)


@needs_reference
def test_slice_lime_fallback_parity():
    """A constant classifier triggers the reference LIME fallback in both."""
    module = oracle.load_reference(oracle.SLICE_REFERENCE)
    image, segments, templates = make_fixture(zero_model=True)
    reference = module.SliceExplainer.from_arrays(
        np.transpose(image, (1, 2, 0)), segments, oracle.NumpyTemplateModel(templates)
    )
    seed_reference_rng(3)
    _, _, _, _, pos, neg = reference.get_slice_explanations(
        0.0, n_iterations=20, num_perturb=TEST_PERTURB
    )
    engine, context = ours(image, segments, templates, 0.0)
    seed_reference_rng(3)
    result = SliceExplainer(engine, num_perturb=TEST_PERTURB, n_iterations=20).explain(
        context, ReferenceReplaySampler()
    )
    assert result.status == STATUS_LIME_FALLBACK
    np.testing.assert_allclose(
        result.coefficients, oracle.reference_coefficients(8, pos, neg), atol=1e-12
    )


@needs_reference
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_belief_end_to_end_parity(seed):
    """Same masks -> same BELIEF coefficients and eliminations."""
    module = oracle.load_reference(oracle.BELIEF_REFERENCE)
    image, segments, templates = make_fixture(seed)
    reference = module.SliceExplainer.from_arrays(
        np.transpose(image, (1, 2, 0)), segments, oracle.NumpyTemplateModel(templates)
    )
    seed_reference_rng(11)
    _, _, pos, neg = reference.get_belief_explanations(num_perturb=TEST_PERTURB, sigma=0, zeta=1e-6)
    engine, context = ours(image, segments, templates, 0.0)
    seed_reference_rng(11)
    result = BeliefExplainer(engine, num_perturb=TEST_PERTURB).explain(context, ReferenceReplaySampler())
    np.testing.assert_allclose(
        result.coefficients, oracle.reference_coefficients(8, pos, neg), atol=ATOL, rtol=RTOL
    )


@needs_reference
def test_select_sigma_parity():
    """Adjusted-R^2 sigma scores match the reference select_sigma."""
    module = oracle.load_reference(oracle.SLICE_REFERENCE, per_channel_blur=True)
    image, segments, templates = make_fixture(weak_blocks=False)
    image = np.abs(image)  # blur must change the image for sigma > 0
    reference = module.SliceExplainer.from_arrays(
        np.transpose(image, (1, 2, 0)), segments, oracle.NumpyTemplateModel(templates)
    )
    scores = []
    seed_reference_rng(5)
    for sigma in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        scores.append(reference._explain(sigma, TEST_PERTURB, False, False)[1])
    engine, context = ours(image, segments, templates, 0.0)
    seed_reference_rng(5)
    selected, ours_scores = select_sigma(engine, context, ReferenceReplaySampler(), num_samples=TEST_PERTURB)
    # sigma 0.1 has a radius-0 kernel (identity blur): reference R^2 is noise.
    assert np.isnan(ours_scores[1])
    keep = [0, 2, 3, 4, 5]
    np.testing.assert_allclose(np.asarray(ours_scores)[keep], np.asarray(scores)[keep], atol=1e-8)
    assert selected == (0, 0.1, 0.2, 0.3, 0.4, 0.5)[keep[int(np.argmax(np.asarray(scores)[keep]))]]


# ------------------------------------------------------------ engine/maps


def test_query_chunking_and_mask_semantics():
    """Chunked queries equal unchunked ones; all-ones mask reproduces prob0."""
    image, segments, templates = make_fixture()
    engine_a, context_a = ours(image, segments, templates, 0.0, batch_size=7)
    engine_b, context_b = ours(image, segments, templates, 0.0, batch_size=1000)
    masks = TorchSampler(0).sample_masks(50, 8)
    masks[0] = 1
    a, b = engine_a.query(context_a, masks), engine_b.query(context_b, masks)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-14)
    assert abs(float(a[0]) - context_a.prob0) < 1e-14
    zero_image = engine_a.perturbed_images(context_a, torch.zeros(1, 8))
    assert torch.count_nonzero(zero_image) == 0


def test_explainers_use_fixed_target_class_and_count_queries():
    """Explanations keep the context class and count every forward pass."""
    image, segments, templates = make_fixture()
    engine, context = ours(image, segments, templates, 0.0)
    result = BeliefExplainer(engine, num_perturb=TEST_PERTURB).explain(context, TorchSampler(1))
    assert result.target_class == context.target_class
    assert result.num_queries == TEST_PERTURB - 1
    slice_result = SliceExplainer(engine, num_perturb=TEST_PERTURB, n_iterations=50).explain(
        context, TorchSampler(1)
    )
    assert slice_result.num_queries == slice_result.num_steps * (TEST_PERTURB - 1) or (
        slice_result.status == STATUS_LIME_FALLBACK
    )


def test_superpixel_vectorizer_positive_and_signed():
    """Positive maps peak at 1, zero maps stay zero, signed maps split signs."""
    segments = np.repeat(np.arange(4), 4 * 16).reshape(16, 16)
    coefficients = [np.array([2.0, -1.0, 0.5, 0.0]), np.array([-1.0, -2.0, 0.0, 0.0])]
    vectorizer = SuperpixelMapVectorizer(grid_size=4)
    maps = vectorizer.positive_maps(coefficients, [segments, segments])
    assert maps[0].max() == 1.0 and maps[0].min() == 0.0 and maps[1].max() == 0.0
    assert vectorizer.transform(coefficients, [segments] * 2).shape == (2, 16)
    signed = vectorizer.transform_signed(coefficients, [segments] * 2)
    assert signed.shape == (2, 32) and signed[1, :16].max() == 0 and signed[1, 16:].max() == 1.0
    assert list(vectorizer.zero_map_mask(coefficients)) == [False, True]
    np.testing.assert_allclose(superpixel_means(maps[0], segments), [1.0, 0.0, 0.25, 0.0])


def test_shared_initial_sample_matches_internal_sampling():
    """Injecting the first-step sample equals drawing it inside each explainer."""
    image, segments, templates = make_fixture()
    engine, context = ours(image, segments, templates, 0.0)
    shared_sampler = TorchSampler(9)
    shared = BeliefExplainer(engine, num_perturb=TEST_PERTURB).initial_sample(context, shared_sampler)
    with_shared = SliceExplainer(engine, num_perturb=TEST_PERTURB, n_iterations=50).explain(
        context, shared_sampler, initial_sample=shared
    )
    internal = SliceExplainer(engine, num_perturb=TEST_PERTURB, n_iterations=50).explain(
        context, TorchSampler(9)
    )
    np.testing.assert_allclose(with_shared.coefficients, internal.coefficients, atol=1e-12)
    belief_a = BeliefExplainer(engine, num_perturb=TEST_PERTURB).explain(context, None, initial_sample=shared)
    belief_b = BeliefExplainer(engine, num_perturb=TEST_PERTURB).explain(context, TorchSampler(9))
    np.testing.assert_allclose(belief_a.coefficients, belief_b.coefficients, atol=1e-12)
