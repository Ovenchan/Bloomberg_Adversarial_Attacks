"""Shared superpixel-perturbation machinery for the PyTorch SLICE/BELIEF ports.

Ported from the authors' TensorFlow/NumPy reference implementations:
  SLICE  https://github.com/rebathip/SLICE-Stabilized-LIME-for-Consistent-Explanations-for-Image-Classification
         revision 61d35c09b9f8bd9f770f5d48fae97a181a28f974, slice/slice_explainer.py
  BELIEF https://github.com/rebathip/BELIEF
         revision 70f373c0aab5aa6b7e679ed4ecb7571e7a79ba5b, slice/slice_explainer.py

Pixel-space contract (documented deviation log: SLICE_BELIEF_result.md):
  * Images are raw RGB float tensors [3,H,W] in [0,1]; normalisation stays
    inside the classifier. The reference multiplies the *preprocessed* array
    by the mask, i.e. sets switched-off superpixels to the preprocessing
    mean. The equivalent raw-pixel operation is filling with the ImageNet
    mean colour, which is what `BASELINE_FILL_RGB` encodes.
  * sigma > 0 replaces switched-off superpixels by a Gaussian-blurred copy
    (scipy/skimage kernel: truncate 4, 'nearest' borders), applied per
    channel. Per-channel blur commutes with per-channel normalisation, so
    blurring raw pixels equals blurring the normalised array.
  * All regression / bootstrap / entropy algebra runs in float64 torch.
"""

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from explainers.base import BaseExplainer
from utils.datasets import IMAGENET_MEAN

REFERENCE_NUM_PERTURB = 500
REFERENCE_KERNEL_WIDTH = 0.25
REFERENCE_RIDGE_ALPHA = 1.0
REFERENCE_SIGMA_VALUES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
GAUSSIAN_TRUNCATE = 4.0
QUICKSHIFT_KERNEL_SIZE = 5
QUICKSHIFT_MAX_DIST = 200
QUICKSHIFT_RATIO = 0.2
QUICKSHIFT_SEED = 42
BASELINE_FILL_RGB = IMAGENET_MEAN
DEFAULT_QUERY_BATCH = 250
REGRESSION_DTYPE = torch.float64


# --------------------------------------------------------------------------
# Classifier adapter and image operations
# --------------------------------------------------------------------------


class ProbabilityAdapter:
    """Batched softmax queries of a raw-pixel classifier.

    Input (constructor): nn.Module mapping Tensor[N,3,H,W] in [0,1] to
        logits[N,C]; int batch_size; bool use_amp (float16 autocast on CUDA);
        bool channels_last (feed NHWC-strided batches; convert the model with
        `model.to(memory_format=torch.channels_last)` for the speed-up).
    """

    def __init__(self, model, batch_size=DEFAULT_QUERY_BATCH, use_amp=False, channels_last=False):
        """Input: nn.Module, int, bool, bool. Output: None."""
        self.model = model
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.channels_last = channels_last

    @torch.no_grad()
    def probabilities(self, images, target_class=None):
        """Input: Tensor[N,3,H,W] raw images, optional int class.

        Output: Tensor[N,C] softmax probabilities (float32, or float64 for a
            float64 model), or Tensor[N] probabilities of `target_class`.
        """
        outputs = []
        device_type = images.device.type
        for start in range(0, images.shape[0], self.batch_size):
            batch = images[start:start + self.batch_size]
            if self.channels_last:
                batch = batch.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=self.use_amp and device_type == "cuda",
            ):
                logits = self.model(batch)
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            probs = logits.softmax(dim=1)
            outputs.append(
                probs if target_class is None else probs[:, target_class]
            )
        return torch.cat(outputs)


def gaussian_kernel1d(sigma, truncate=GAUSSIAN_TRUNCATE):
    """Input: float sigma > 0, float truncate. Output: Tensor[2r+1] float64.

    Matches scipy.ndimage.gaussian_filter (used by skimage.filters.gaussian):
    radius r = int(truncate * sigma + 0.5), unnormalised exp(-x^2 / 2s^2)
    weights divided by their sum.
    """
    radius = int(truncate * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    weights = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return weights / weights.sum()


def gaussian_blur(image, sigma, truncate=GAUSSIAN_TRUNCATE):
    """Per-channel separable Gaussian blur with 'nearest' (replicate) borders.

    Input: Tensor[3,H,W] or [N,3,H,W], float sigma >= 0.
    Output: Tensor of the same shape and dtype.
    """
    if sigma <= 0:
        return image.clone()
    kernel = gaussian_kernel1d(sigma, truncate).to(image.device, image.dtype)
    radius = (kernel.numel() - 1) // 2
    if radius == 0:
        return image.clone()
    squeeze = image.ndim == 3
    x = image[None] if squeeze else image
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(
        F.pad(x, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    x = F.conv2d(
        F.pad(x, (0, 0, radius, radius), mode="replicate"),
        vertical,
        groups=channels,
    )
    return x[0] if squeeze else x


class SuperpixelSegmenter:
    """Quickshift segmentation with the SLICE reference driver parameters.

    Input (constructor): ints/floats for skimage.segmentation.quickshift.
    """

    def __init__(
        self,
        kernel_size=QUICKSHIFT_KERNEL_SIZE,
        max_dist=QUICKSHIFT_MAX_DIST,
        ratio=QUICKSHIFT_RATIO,
        seed=QUICKSHIFT_SEED,
    ):
        """Input: int kernel_size, float max_dist/ratio, int seed. Output: None."""
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.ratio = ratio
        self.seed = seed

    def segment(self, image):
        """Input: Tensor/ndarray [3,H,W] raw RGB in [0,1].

        Output: ndarray[H,W] int32 labels relabelled to 0..K-1.
        """
        from skimage.segmentation import quickshift

        array = image.detach().cpu().numpy() if hasattr(image, "detach") else image
        hwc = np.ascontiguousarray(np.transpose(array, (1, 2, 0)), dtype=np.float64)
        labels = quickshift(
            hwc,
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            ratio=self.ratio,
            rng=self.seed,
        )
        _, contiguous = np.unique(labels, return_inverse=True)
        return contiguous.reshape(labels.shape).astype(np.int32)


# --------------------------------------------------------------------------
# Perturbation contexts and queries
# --------------------------------------------------------------------------


@dataclass
class PerturbationContext:
    """One input prepared for perturbation queries.

    Attributes: image Tensor[3,H,W] (device), segments Tensor[H,W] long
    (device), num_segments int, target_class int, sigma float, fill
    Tensor[3,H,W] replacement image, prob0 float, num_queries int.
    """

    image: torch.Tensor
    segments: torch.Tensor
    num_segments: int
    target_class: int
    sigma: float
    fill: torch.Tensor
    prob0: float = float("nan")
    num_queries: int = 0


class PerturbationQueryEngine:
    """Builds superpixel-masked images and queries the fixed target class.

    Input (constructor): ProbabilityAdapter, torch.device, fill RGB tuple,
        torch dtype of perturbed images (float64 only for parity tests).
    """

    def __init__(self, adapter, device, fill_rgb=BASELINE_FILL_RGB, dtype=torch.float32):
        """Input: ProbabilityAdapter, torch.device, 3-tuple, dtype. Output: None."""
        self.adapter = adapter
        self.device = torch.device(device)
        self.dtype = dtype
        self.fill_rgb = torch.tensor(fill_rgb, dtype=dtype).view(3, 1, 1)

    def context(self, image, segments, target_class, sigma):
        """Input: Tensor[3,H,W] raw image, ndarray/Tensor[H,W] labels 0..K-1,
        int fixed class, float sigma.

        Output: PerturbationContext with prob0 queried on the unperturbed input.
        """
        image = image.to(self.device, self.dtype)
        segments = torch.as_tensor(np.asarray(segments), device=self.device).long()
        context = PerturbationContext(
            image=image,
            segments=segments,
            num_segments=int(segments.max().item()) + 1,
            target_class=int(target_class),
            sigma=float(sigma),
            fill=self._fill(image, sigma),
        )
        context.prob0 = float(
            self.adapter.probabilities(image[None], context.target_class)[0]
        )
        context.num_queries += 1
        return context

    def set_sigma(self, context, sigma):
        """Input: PerturbationContext, float sigma. Output: None (fill updated)."""
        context.sigma = float(sigma)
        context.fill = self._fill(context.image, sigma)

    def _fill(self, image, sigma):
        """Input: Tensor[3,H,W], float. Output: Tensor[3,H,W] replacement pixels."""
        if sigma == 0:
            return self.fill_rgb.to(image.device).expand_as(image).clone()
        return gaussian_blur(image, sigma)

    def perturbed_images(self, context, masks):
        """Input: context, Tensor[M,K] {0,1}. Output: Tensor[M,3,H,W] raw images."""
        keep = masks.to(self.device, self.dtype)[:, context.segments][:, None]
        return context.image[None] * keep + context.fill[None] * (1 - keep)

    def query(self, context, masks):
        """Input: context, Tensor[M,K] binary masks (1 = superpixel present).

        Output: Tensor[M] float64 (CPU) probabilities of context.target_class.
        """
        masks = torch.as_tensor(masks)
        outputs = []
        step = self.adapter.batch_size
        for start in range(0, masks.shape[0], step):
            images = self.perturbed_images(context, masks[start:start + step])
            outputs.append(
                self.adapter.probabilities(images, context.target_class)
            )
        context.num_queries += int(masks.shape[0])
        return torch.cat(outputs).to("cpu", REGRESSION_DTYPE)


# --------------------------------------------------------------------------
# Random samplers (perturbation masks and locality-weighted bootstraps)
# --------------------------------------------------------------------------


class TorchSampler:
    """Production sampler driven by an explicit torch.Generator.

    Input (constructor): int seed (derive per source/purpose by the caller).
    """

    def __init__(self, seed):
        """Input: int seed. Output: None."""
        self.generator = torch.Generator().manual_seed(int(seed))

    def sample_masks(self, rows, cols):
        """Input: int rows, int cols. Output: Tensor[rows,cols] int64 in {0,1}."""
        return torch.randint(0, 2, (rows, cols), generator=self.generator)

    def bootstrap_counts(self, weights, iterations):
        """Weighted resampling with replacement, `iterations` times.

        Input: Tensor[n] positive float64 weights, int iterations.
        Output: Tensor[iterations,n] float64 draw counts per original row.
        """
        n = weights.numel()
        probabilities = (weights / weights.sum()).to(torch.float64)
        draws = torch.multinomial(
            probabilities.expand(iterations, n), n, replacement=True,
            generator=self.generator,
        )
        counts = torch.zeros(iterations, n, dtype=torch.float64)
        counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        return counts


class ReferenceReplaySampler:
    """Replays the reference RNG calls (numpy global + python `random`).

    Used only by parity tests: with numpy/random seeded identically, it
    produces the same masks and bootstrap indices as the reference code.
    """

    def sample_masks(self, rows, cols):
        """Input: int rows, cols. Output: Tensor[rows,cols] int64 (np.random)."""
        values = np.random.randint(0, 2, rows * cols).reshape((rows, cols))
        return torch.from_numpy(values.astype(np.int64))

    def bootstrap_counts(self, weights, iterations):
        """Input: Tensor[n] weights, int iterations. Output: Tensor[B,n] counts."""
        w = weights.numpy()
        n = w.size
        counts = np.zeros((iterations, n))
        for i in range(iterations):
            indices = random.choices(range(n), weights=w, k=n)
            np.add.at(counts[i], indices, 1.0)
        return torch.from_numpy(counts)


# --------------------------------------------------------------------------
# Regression utilities (float64 torch)
# --------------------------------------------------------------------------


def cosine_locality_weights(X, kernel_width=REFERENCE_KERNEL_WIDTH):
    """Reference kernel sqrt(exp(-d^2 / w^2)), d = cosine distance to all-ones.

    Input: Tensor[n,d] binary design, float kernel width.
    Output: Tensor[n] float64 weights. Zero rows have distance 1 (sklearn).
    """
    X = X.to(REGRESSION_DTYPE)
    if X.shape[1] == 0:
        return torch.ones(X.shape[0], dtype=REGRESSION_DTYPE)
    norms = X.norm(dim=1)
    similarity = X.sum(dim=1) / (norms * np.sqrt(X.shape[1]))
    similarity = torch.where(norms > 0, similarity, torch.zeros_like(similarity))
    distance = (1.0 - similarity).clamp(0.0, 2.0)
    return torch.sqrt(torch.exp(-(distance ** 2) / kernel_width ** 2))


def weighted_ridge(X, y, weights=None, alpha=REFERENCE_RIDGE_ALPHA):
    """sklearn Ridge(alpha, fit_intercept=True) with sample weights.

    Input: Tensor[n,d], Tensor[n], optional Tensor[n] weights, float alpha.
    Output: (Tensor[d] coefficients, float intercept), float64.
    """
    X = X.to(REGRESSION_DTYPE)
    y = y.to(REGRESSION_DTYPE)
    w = torch.ones_like(y) if weights is None else weights.to(REGRESSION_DTYPE)
    x_offset = (w[:, None] * X).sum(0) / w.sum()
    y_offset = (w * y).sum() / w.sum()
    root = w.sqrt()
    Xc = (X - x_offset) * root[:, None]
    yc = (y - y_offset) * root
    gram = Xc.T @ Xc + alpha * torch.eye(X.shape[1], dtype=REGRESSION_DTYPE)
    coef = torch.linalg.solve(gram, Xc.T @ yc)
    return coef, float(y_offset - x_offset @ coef)


def batched_weighted_ridge(X, y, weights, alpha=REFERENCE_RIDGE_ALPHA, chunk=250):
    """Many weighted ridge fits sharing one design (bootstrap replicates).

    A bootstrap that draws row i c_i times, each carrying sample weight w_i,
    is the same weighted least-squares problem as one copy of row i with
    weight c_i * w_i; pass those products as `weights`.

    Input: Tensor[n,d], Tensor[n], Tensor[B,n] weights, float alpha, int chunk.
    Output: Tensor[B,d] float64 coefficients.
    """
    X = X.to(REGRESSION_DTYPE)
    y = y.to(REGRESSION_DTYPE)
    eye = alpha * torch.eye(X.shape[1], dtype=REGRESSION_DTYPE)
    outputs = []
    for start in range(0, weights.shape[0], chunk):
        W = weights[start:start + chunk].to(REGRESSION_DTYPE)
        total = W.sum(1, keepdim=True)
        x_offset = (W @ X) / total
        y_offset = (W @ y)[:, None] / total
        gram = (W[:, :, None] * X[None]).transpose(1, 2) @ X - total[..., None] * (
            x_offset[:, :, None] * x_offset[:, None, :]
        )
        rhs = (W * y) @ X - total * x_offset * y_offset
        outputs.append(torch.linalg.solve(gram + eye, rhs[..., None])[..., 0])
    return torch.cat(outputs)


def ridge_adjusted_r2(X, y, alpha=REFERENCE_RIDGE_ALPHA):
    """Adjusted R^2 of an unweighted ridge fit, as in reference select_sigma.

    Input: Tensor[n,d], Tensor[n]. Output: float (nan/-inf if y is constant).
    """
    coef, intercept = weighted_ridge(X, y, None, alpha)
    prediction = X.to(REGRESSION_DTYPE) @ coef + intercept
    residual = float(((y - prediction) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - np.float64(residual) / np.float64(total)
    n, p = X.shape
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


# --------------------------------------------------------------------------
# Shared explainer base
# --------------------------------------------------------------------------


class BaseSurrogateExplainer(BaseExplainer):
    """Common LIME-style machinery shared by SLICE and BELIEF.

    Input (constructor): PerturbationQueryEngine, int num_perturb (rows per
        sampling round including the all-present row), float kernel width,
        float ridge alpha.
    """

    name = "surrogate"

    def __init__(
        self,
        engine,
        num_perturb=REFERENCE_NUM_PERTURB,
        kernel_width=REFERENCE_KERNEL_WIDTH,
        ridge_alpha=REFERENCE_RIDGE_ALPHA,
    ):
        """Input: engine, int, float, float. Output: None."""
        self.engine = engine
        self.num_perturb = num_perturb
        self.kernel_width = kernel_width
        self.ridge_alpha = ridge_alpha

    def initial_sample(self, context, sampler):
        """Input: context, sampler. Output: (X, y) plain LIME training set."""
        masks = sampler.sample_masks(self.num_perturb - 1, context.num_segments)
        return self._training_set(context, masks)

    def _training_set(self, context, masks):
        """Prepend the all-present row and query the perturbations.

        Input: context, Tensor[num_perturb-1,K] masks.
        Output: (X Tensor[num_perturb,K] float64, y Tensor[num_perturb] float64).
        """
        y = self.engine.query(context, masks)
        ones = torch.ones(1, masks.shape[1], dtype=REGRESSION_DTYPE)
        X = torch.cat([ones, masks.to(REGRESSION_DTYPE)])
        y = torch.cat([torch.tensor([context.prob0], dtype=REGRESSION_DTYPE), y])
        return X, y

    def lime_coefficients(self, context, sampler):
        """Reference `get_lime_explanations`: one weighted ridge fit.

        Input: context, sampler. Output: Tensor[K] float64 coefficients.
        """
        masks = sampler.sample_masks(self.num_perturb - 1, context.num_segments)
        X, y = self._training_set(context, masks)
        weights = cosine_locality_weights(X, self.kernel_width)
        coef, _ = weighted_ridge(X, y, weights, self.ridge_alpha)
        return coef

    @staticmethod
    def timer():
        """Input: None. Output: float perf-counter seconds (CUDA-synchronised)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


def select_sigma(
    engine,
    context,
    sampler,
    sigma_values=REFERENCE_SIGMA_VALUES,
    num_samples=REFERENCE_NUM_PERTURB,
):
    """Reference `select_sigma`: one mask set re-queried under each sigma.

    Input: engine, context (sigma is restored afterwards), sampler, sequence
        of float sigmas, int rows including the all-present row.
    Output: (float selected sigma, list[float] adjusted R^2 per sigma;
        NaN for identity blurs). The largest (non-NaN) score wins; ties go to the later sigma,
        as with the reference's argsort()[-1].
    """
    original = context.sigma
    masks = sampler.sample_masks(num_samples - 1, context.num_segments)
    ones = torch.ones(1, masks.shape[1], dtype=REGRESSION_DTYPE)
    X = torch.cat([ones, masks.to(REGRESSION_DTYPE)])
    scores = []
    for sigma in sigma_values:
        engine.set_sigma(context, sigma)
        if sigma > 0 and torch.equal(context.fill, context.image):
            # Kernel radius 0 (e.g. sigma 0.1): the "blur" is the identity,
            # responses are constant and the reference R^2 is float noise.
            scores.append(float("nan"))
            continue
        y = torch.cat(
            [torch.tensor([context.prob0], dtype=REGRESSION_DTYPE),
             engine.query(context, masks)]
        )
        scores.append(ridge_adjusted_r2(X, y))
    engine.set_sigma(context, original)
    array = np.asarray(scores)
    finite = np.flatnonzero(~np.isnan(array))
    best = finite[np.argsort(array[finite], kind="stable")][-1]
    return float(sigma_values[best]), scores
