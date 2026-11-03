"""PyTorch port of BELIEF (Bora et al., UAI 2025).

Reference: SEMBayesianRidge and SliceExplainer.get_belief_explanations in
slice/slice_explainer.py at BELIEF revision 70f373c0aab5aa6b7e679ed4ecb7571e7a79ba5b,
driver belief_20runs.py (500 perturbations, wrapper slack zeta = 1e-6).

SEMBayesianRidge is sklearn's BayesianRidge evidence maximisation
(alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6, lambda_init = 1, at most
300 iterations, no convergence stop) on locality-weighted, centred data,
extended with sign-entropy elimination: from iteration `skip_iters` on,
and then every skip_iters + 1 iterations, every retained coefficient whose
Gaussian-posterior sign entropy H(P(beta > 0)) exceeds `slack` is removed.
Eliminated coefficients are reported as 0 in the original superpixel order.
"""

import numpy as np
import torch

from explainers.base import (
    STATUS_ALL_ELIMINATED,
    STATUS_NONFINITE,
    STATUS_OK,
    SurrogateExplanation,
)
from explainers.perturbation import (
    REFERENCE_KERNEL_WIDTH,
    REFERENCE_NUM_PERTURB,
    REGRESSION_DTYPE,
    BaseSurrogateExplainer,
    cosine_locality_weights,
)

BELIEF_SLACK = 1e-6
BELIEF_SKIP_ITERS = 3
BELIEF_MAX_ITER = 300
BELIEF_LAMBDA_INIT = 1.0
BAYES_HYPER_PRIOR = 1e-6
FLOAT64_EPS = float(np.finfo(np.float64).eps)


def gaussian_sign_entropy(mean, std):
    """Reference `_calculate_sign_entropy` (binary entropy, bits).

    Input: Tensor[d] posterior means, Tensor[d] posterior std devs.
    Output: Tensor[d] float64; 0 where P(+) or P(-) is exactly 0. A zero
        std yields NaN, which the reference never eliminates (NaN > slack
        is False).
    """
    p_negative = torch.special.ndtr((0 - mean) / std)
    p_positive = 1 - p_negative
    degenerate = (p_positive == 0) | (p_negative == 0)
    safe_p = torch.where(degenerate, torch.ones_like(p_positive), p_positive)
    safe_n = torch.where(degenerate, torch.ones_like(p_negative), p_negative)
    value = -(safe_p * torch.log2(safe_p) + safe_n * torch.log2(safe_n))
    return torch.where(degenerate, torch.zeros_like(value), value)


class SignEntropyBayesianRidge:
    """Port of the reference SEMBayesianRidge.

    Input (constructor): float slack, int skip_iters, int max_iter, float
        lambda_init, float hyper-prior shape/rate (alpha_1..lambda_2).
    Fitted attributes: coef_ Tensor[d], eliminated_ Tensor[d] bool,
        alpha_, lambda_ floats, sigma_ Tensor[r,r], n_iter_ int.
    """

    def __init__(
        self,
        slack=BELIEF_SLACK,
        skip_iters=BELIEF_SKIP_ITERS,
        max_iter=BELIEF_MAX_ITER,
        lambda_init=BELIEF_LAMBDA_INIT,
        hyper_prior=BAYES_HYPER_PRIOR,
    ):
        """Input: floats/ints as documented. Output: None."""
        self.slack = slack
        self.skip_iters = skip_iters
        self.max_iter = max_iter
        self.lambda_init = lambda_init
        self.hyper_prior = hyper_prior

    @staticmethod
    def _posterior_mean(X, y, alpha, lam):
        """sklearn BayesianRidge._update_coef_.

        Input: Tensor[n,p], Tensor[n], floats alpha, lambda.
        Output: (coef Tensor[p], rmse float, eigenvalues Tensor[r], Vh Tensor[r,p]).
        """
        n, p = X.shape
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        eigen = S ** 2
        if n > p:
            coef = Vh.T @ ((Vh / (eigen + lam / alpha)[:, None]) @ (X.T @ y))
        else:
            coef = X.T @ ((U / (eigen + lam / alpha)[None, :]) @ (U.T @ y))
        rmse = float(((y - X @ coef) ** 2).sum())
        return coef, rmse, eigen, Vh

    def fit(self, X, y, sample_weight):
        """Input: Tensor[n,d] design, Tensor[n] targets, Tensor[n] weights.

        Output: self (fitted).
        """
        X = X.to(REGRESSION_DTYPE)
        y = y.to(REGRESSION_DTYPE)
        w = sample_weight.to(REGRESSION_DTYPE)
        x_offset = (w[:, None] * X).sum(0) / w.sum()
        y_offset = (w * y).sum() / w.sum()
        root = w.sqrt()
        X = (X - x_offset) * root[:, None]
        y = (y - y_offset) * root
        n, d = X.shape
        alpha = 1.0 / (float(y.var(unbiased=False)) + FLOAT64_EPS)
        lam = self.lambda_init
        eliminated = torch.zeros(d, dtype=torch.bool)
        full = torch.zeros(d, dtype=REGRESSION_DTYPE)
        skip_counter = self.skip_iters
        prior = self.hyper_prior
        eigen = torch.zeros(0, dtype=REGRESSION_DTYPE)
        Vh = torch.zeros(0, 0, dtype=REGRESSION_DTYPE)
        iteration = 0
        for iteration in range(self.max_iter):
            keep = torch.where(~eliminated)[0]
            if iteration >= skip_counter:
                if keep.numel() == 0:
                    break
                coef, _, eigen_e, Vh_e = self._posterior_mean(X[:, keep], y, alpha, lam)
                sigma = (Vh_e.T @ (Vh_e / (eigen_e + lam / alpha)[:, None])) / alpha
                entropy = gaussian_sign_entropy(coef, torch.sqrt(torch.diagonal(sigma)))
                eliminated[keep[entropy > self.slack]] = True
                skip_counter = iteration + self.skip_iters + 1
                keep = torch.where(~eliminated)[0]
            coef, rmse, eigen, Vh = self._posterior_mean(X[:, keep], y, alpha, lam)
            full[keep] = coef
            full[eliminated] = 0.0
            gamma = float((alpha * eigen / (lam + alpha * eigen)).sum())
            lam = (gamma + 2 * prior) / (float((coef ** 2).sum()) + 2 * prior)
            alpha = (n - gamma + 2 * prior) / (rmse + 2 * prior)
        self.n_iter_ = iteration + 1
        self.alpha_ = alpha
        self.lambda_ = lam
        self.coef_ = full
        self.eliminated_ = eliminated
        self.sigma_ = (Vh.T @ (Vh / (eigen + lam / alpha)[:, None])) / alpha
        self.intercept_ = float(y_offset - x_offset @ full)
        return self


class BeliefExplainer(BaseSurrogateExplainer):
    """BELIEF: one LIME sample set fitted by SignEntropyBayesianRidge.

    Input (constructor): PerturbationQueryEngine, int num_perturb, float
        kernel width, float slack (zeta).
    """

    name = "belief"

    def __init__(
        self,
        engine,
        num_perturb=REFERENCE_NUM_PERTURB,
        kernel_width=REFERENCE_KERNEL_WIDTH,
        slack=BELIEF_SLACK,
    ):
        """Input: engine, int, float, float. Output: None."""
        super().__init__(engine, num_perturb, kernel_width)
        self.slack = slack

    def explain(self, context, sampler, initial_sample=None):
        """Input: PerturbationContext (prob0 set), sampler, optional shared
            (X, y) LIME training set (all-present row first).

        Output: SurrogateExplanation (status all_eliminated -> all-zero map).
        """
        start = self.timer()
        queries_before = context.num_queries
        if initial_sample is None:
            masks = sampler.sample_masks(self.num_perturb - 1, context.num_segments)
            X, y = self._training_set(context, masks)
        else:
            X, y = initial_sample
        weights = cosine_locality_weights(X, self.kernel_width)
        model = SignEntropyBayesianRidge(slack=self.slack).fit(X, y, weights)
        coefficients = model.coef_.clone()
        status = STATUS_ALL_ELIMINATED if bool(model.eliminated_.all()) else STATUS_OK
        if not torch.isfinite(coefficients).all():
            status = STATUS_NONFINITE
        return SurrogateExplanation(
            coefficients=coefficients.numpy(),
            eliminated=model.eliminated_.numpy(),
            status=status,
            target_class=context.target_class,
            prob0=context.prob0,
            sigma=context.sigma,
            num_queries=context.num_queries - queries_before,
            num_steps=1,
            runtime_seconds=self.timer() - start,
            extra={"n_iter": model.n_iter_, "alpha": model.alpha_, "lambda": model.lambda_},
        )
