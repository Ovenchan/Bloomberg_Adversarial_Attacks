"""Pipeline 1 feature extraction; attacks and evaluation stay in shared modules."""

from explainers.gradcam import GradCAMExplainer
from explainers.vectorizer import GradCAMVectorizer


class GradCAMPipeline:
    """Input: encoder, target path and grid size. Output: composable extraction pipeline."""

    def __init__(self, encoder, target_layer, grid_size):
        """Input: ResNetEncoder, str target_layer, int grid_size. Output: None."""
        self.explainer = GradCAMExplainer(
            encoder, encoder.target_layer(target_layer)
        )
        self.vectorizer = GradCAMVectorizer(grid_size)

    def extract(self, images):
        """Input: raw Tensor[N,3,H,W]. Output: (ndarray[N,D], ndarray[N] predictions)."""
        maps, predictions = self.explainer.explain(images)
        return self.vectorizer.transform(maps), predictions.cpu().numpy()
