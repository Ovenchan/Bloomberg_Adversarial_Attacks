"""Load the pinned upstream SLICE/BELIEF reference code as a test oracle.

The upstream `slice/slice_explainer.py` files import TensorFlow, Keras and
OpenCV at module level although the explanation algebra is NumPy/SciPy/
scikit-learn. Those three imports are replaced by minimal stubs so the
unmodified source can run without TensorFlow. The only behavioural shim
is optional: `per_channel_blur=True` makes the reference blur act per
colour channel (the documented controlled-profile behaviour); the
unshimmed reference with scikit-image 0.23 also blurs across channels.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPOSITORY / "tmp" / "research_references"
SLICE_REFERENCE = REFERENCE_ROOT / "SLICE" / "slice" / "slice_explainer.py"
BELIEF_REFERENCE = REFERENCE_ROOT / "BELIEF" / "slice" / "slice_explainer.py"


def references_available():
    """Input: None. Output: bool, True when both pinned files exist."""
    return SLICE_REFERENCE.is_file() and BELIEF_REFERENCE.is_file()


def _stub_modules():
    """Input: None. Output: dict name -> stub module for TF/Keras/cv2."""
    cv2 = types.ModuleType("cv2")
    cv2.merge = lambda channels: np.stack(channels, axis=-1)
    keras = types.ModuleType("keras")
    applications = types.ModuleType("keras.applications")
    imagenet_utils = types.ModuleType("keras.applications.imagenet_utils")
    imagenet_utils.decode_predictions = None
    keras.applications = applications
    applications.imagenet_utils = imagenet_utils
    tensorflow = types.ModuleType("tensorflow")
    return {
        "cv2": cv2,
        "keras": keras,
        "keras.applications": applications,
        "keras.applications.imagenet_utils": imagenet_utils,
        "tensorflow": tensorflow,
    }


class _PerChannelFilters:
    """skimage.filters replacement whose gaussian blurs each channel separately."""

    @staticmethod
    def gaussian(image, sigma):
        """Input: ndarray[H,W,3], float. Output: ndarray[H,W,3] blurred per channel."""
        from skimage import filters

        return filters.gaussian(image, sigma, channel_axis=-1)


def load_reference(path, per_channel_blur=False):
    """Input: Path to a pinned slice_explainer.py, bool shim flag.

    Output: module object exposing SliceExplainer (and SEMBayesianRidge for BELIEF).
    """
    stubs = _stub_modules()
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            f"reference_{path.parent.parent.name.lower()}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    if per_channel_blur:
        module.filters = _PerChannelFilters()
    return module


class NumpyTemplateModel:
    """Deterministic float64 softmax template classifier for the reference side.

    Input (constructor): ndarray[C,3,H,W] templates. predict(): ndarray[N,H,W,3]
    -> ndarray[N,C] probabilities (Keras-like API used by the reference).
    """

    name = "stub"

    def __init__(self, templates):
        """Input: ndarray[C,3,H,W]. Output: None."""
        self.templates = np.asarray(templates, dtype=np.float64)

    def predict(self, images, verbose=0):
        """Input: ndarray[N,H,W,3], verbosity flag. Output: ndarray[N,C]."""
        x = np.transpose(np.asarray(images, dtype=np.float64), (0, 3, 1, 2))
        logits = np.einsum("nchw,kchw->nk", x, self.templates)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def reference_coefficients(num_segments, pos_dict, neg_dict):
    """Rebuild the full coefficient vector from the reference rank dictionaries.

    Input: int K, dicts with 'column_names' and 'column_means'.
    Output: ndarray[K] float64.
    """
    coefficients = np.zeros(num_segments)
    for table in (pos_dict, neg_dict):
        names = np.asarray(table["column_names"], dtype=int).reshape(-1)
        coefficients[names] = np.asarray(table["column_means"], dtype=np.float64).reshape(-1)
    return coefficients
