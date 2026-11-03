"""PyTorch port of SLICE (Bora et al., CVPR 2024).

Reference: SliceExplainer.get_slice_explanations and _calculate_entropy in
slice/slice_explainer.py at SLICE revision 61d35c09b9f8bd9f770f5d48fae97a181a28f974,
with the driver defaults of slice_test.py (500 perturbations per step,
1,000 locality-weighted bootstrap ridge fits, at most 10 steps, tolerance 3,
rank stabilisation disabled).

Algorithm per sampling step:
  1. draw 499 random masks (+ the all-present row); superpixels already
     eliminated are held present;
  2. query the fixed explained class;
  3. 1,000 bootstrap replicates sampled with probability proportional to
     the cosine locality weight, each fitted with weighted ridge;
  4. Gaussian-KDE sign entropy of every retained coefficient; superpixels
     with non-zero entropy are eliminated and sampling restarts. After the
     first clean step, `tolerance` further clean steps confirm stability.
The final explanation is a weighted ridge fit on the last step's samples
restricted to the retained superpixels; eliminated superpixels get 0.
If no superpixel survives (or the last responses are constant), the
reference falls back to a fresh LIME fit, reported as STATUS_LIME_FALLBACK.
"""

import numpy as np
import torch

from explainers.base import (
    STATUS_LIME_FALLBACK,
    STATUS_NONFINITE,
    STATUS_OK,
    SurrogateExplanation,
)
from explainers.perturbation import (
    REFERENCE_KERNEL_WIDTH,
    REFERENCE_NUM_PERTURB,
    REFERENCE_RIDGE_ALPHA,
    REGRESSION_DTYPE,
    BaseSurrogateExplainer,
    batched_weighted_ridge,
    cosine_locality_weights,
    weighted_ridge,
)

SLICE_BOOTSTRAP_ITERATIONS = 1000
SLICE_MAX_STEPS = 10
SLICE_TOLERANCE = 3
KDE_GRID_POINTS = 1000
KDE_FEATURE_CHUNK = 8


def kde_sign_entropy(samples, grid_points=KDE_GRID_POINTS, chunk=KDE_FEATURE_CHUNK):
    """Reference `_calculate_entropy`, vectorised over features.

    For each column: scipy gaussian_kde whose scalar bandwidth factor is
    (Scott factor n^(-1/5)) * std(ddof=0), so the kernel variance is
    var(ddof=1) * factor^2; density on a 1,000-point grid over [min, max];
    P(+) and P(-) are Riemann sums over x >= 0 and x < 0. Entropy is 0 if
    either probability is exactly 0, or if the column is constant (the
    reference catches scipy's LinAlgError there).

    Input: Tensor[B,d] bootstrap coefficients. Output: Tensor[d] float64 bits.
    """
    samples = samples.to(REGRESSION_DTYPE)
    n, d = samples.shape
    entropies = torch.zeros(d, dtype=REGRESSION_DTYPE)
    scott = n ** (-1.0 / 5.0)
    for start in range(0, d, chunk):
        block = samples[:, start:start + chunk].T  # [c, n]
        std0 = block.std(dim=1, unbiased=False)
        var1 = block.var(dim=1, unbiased=True)
        lo, hi = block.min(dim=1).values, block.max(dim=1).values
        valid = std0 > 0
        if not valid.any():
            continue
        block, std0, var1, lo, hi = (
            block[valid], std0[valid], var1[valid], lo[valid], hi[valid]
        )
        bandwidth_var = var1 * (scott * std0) ** 2
        steps = torch.linspace(0, 1, grid_points, dtype=REGRESSION_DTYPE)
        grid = lo[:, None] + (hi - lo)[:, None] * steps[None]  # [c, G]
        dx = grid[:, 1] - grid[:, 0]
        diff = grid[:, :, None] - block[:, None, :]  # [c, G, n]
        density = torch.exp(-0.5 * diff ** 2 / bandwidth_var[:, None, None]).mean(2)
        density = density / torch.sqrt(2 * np.pi * bandwidth_var)[:, None]
        normalized = density / (density * dx[:, None]).sum(1, keepdim=True)
        positive = (normalized * (grid >= 0) * dx[:, None]).sum(1)
        negative = (normalized * (grid < 0) * dx[:, None]).sum(1)
        mixed = (positive != 0) & (negative != 0)
        safe_p = torch.where(mixed, positive, torch.ones_like(positive))
        safe_n = torch.where(mixed, negative, torch.ones_like(negative))
        value = -safe_p * torch.log2(safe_p) - safe_n * torch.log2(safe_n)
        block_entropy = torch.where(mixed, value, torch.zeros_like(value))
        indices = torch.arange(start, min(start + chunk, d))[valid]
        entropies[indices] = block_entropy
    return entropies


class SliceExplainer(BaseSurrogateExplainer):
    """SLICE: sign-entropy feature elimination with bootstrap ridge.

    Input (constructor): PerturbationQueryEngine and the reference
        hyper-parameters (see module constants).
    """

    name = "slice"

    def __init__(
        self,
        engine,
        num_perturb=REFERENCE_NUM_PERTURB,
        n_iterations=SLICE_BOOTSTRAP_ITERATIONS,
        max_steps=SLICE_MAX_STEPS,
        tolerance=SLICE_TOLERANCE,
        kernel_width=REFERENCE_KERNEL_WIDTH,
        ridge_alpha=REFERENCE_RIDGE_ALPHA,
    ):
        """Input: engine, ints num_perturb/n_iterations/max_steps/tolerance,
        floats kernel_width/ridge_alpha. Output: None."""
        super().__init__(engine, num_perturb, kernel_width, ridge_alpha)
        self.n_iterations = n_iterations
        self.max_steps = max_steps
        self.tolerance = tolerance

    def explain(self, context, sampler, initial_sample=None):
        """Input: PerturbationContext (prob0 set), sampler, optional (X, y)
            first-step training set (all-present row first) shared with
            BELIEF; step 1 has no eliminated superpixels, so a plain LIME
            sample is exactly what SLICE would draw there.

        Output: SurrogateExplanation; `extra` holds per-step eliminations
            and mean sign entropy.
        """
        start = self.timer()
        queries_before = context.num_queries
        K = context.num_segments
        eliminated = torch.zeros(K, dtype=torch.bool)
        tolerance_count = 0
        last = None
        history = []
        steps = 0
        for _ in range(self.max_steps):
            # Masks are drawn before the all-eliminated check, as in the
            # reference, so RNG streams stay aligned for replay tests.
            if steps == 0 and initial_sample is not None:
                X, y = initial_sample
            else:
                masks = sampler.sample_masks(self.num_perturb - 1, K)
                if bool(eliminated.all()):
                    last = None
                    break
                masks[:, eliminated] = 1
                X, y = self._training_set(context, masks)
            steps += 1
            last = (X, y)
            retained = torch.where(~eliminated)[0]
            X_retained = X[:, retained]
            weights = cosine_locality_weights(X_retained, self.kernel_width)
            counts = sampler.bootstrap_counts(weights, self.n_iterations)
            coefs = batched_weighted_ridge(
                X_retained, y, counts * weights[None], self.ridge_alpha
            )
            entropy = kde_sign_entropy(coefs)
            unstable = retained[entropy != 0]
            history.append(
                {
                    "num_unstable": int(unstable.numel()),
                    "mean_entropy": float(entropy.mean()) if entropy.numel() else 0.0,
                }
            )
            if unstable.numel():
                eliminated[unstable] = True
                tolerance_count = 0
            elif tolerance_count == 0 or tolerance_count < self.tolerance:
                tolerance_count += 1
            else:
                break
        coefficients, status = self._final_fit(context, sampler, eliminated, last)
        if not torch.isfinite(coefficients).all():
            status = STATUS_NONFINITE
        return SurrogateExplanation(
            coefficients=coefficients.numpy(),
            eliminated=eliminated.numpy(),
            status=status,
            target_class=context.target_class,
            prob0=context.prob0,
            sigma=context.sigma,
            num_queries=context.num_queries - queries_before,
            num_steps=steps,
            runtime_seconds=self.timer() - start,
            extra={"history": history},
        )

    def _final_fit(self, context, sampler, eliminated, last):
        """Reference post-loop fit (rank_stabilization=False).

        Input: context, sampler, Tensor[K] bool, (X, y) of the last step or None.
        Output: (Tensor[K] float64 coefficients, str status).
        """
        retained = torch.where(~eliminated)[0]
        if last is not None and retained.numel() > 0 and float(last[1].var(unbiased=False)) != 0:
            X, y = last
            X_retained = X[:, retained]
            weights = cosine_locality_weights(X_retained, self.kernel_width)
            coef, _ = weighted_ridge(X_retained, y, weights, self.ridge_alpha)
            coefficients = torch.zeros(context.num_segments, dtype=REGRESSION_DTYPE)
            coefficients[retained] = coef
            return coefficients, STATUS_OK
        return self.lime_coefficients(context, sampler), STATUS_LIME_FALLBACK
