"""Offline Pipeline 1 tests; synthetic fixtures are never final research results."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from encoders.cnn import ResNetEncoder
from explainers.gradcam import GradCAMExplainer
from explainers.vectorizer import GradCAMVectorizer
from detectors.logistic import GradCAMLogisticDetector
from analysis.grouped_cv import GroupedDetectionCV
from analysis.metrics import DetectionEvaluator, filter_successful_attacks
from attacks.factory import build_attack


class TinyClassifier(nn.Module):
    """Input: images. Output: two spatially distinct class logits for targeting tests."""

    def __init__(self):
        """Input: None. Output: None."""
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0] = 1
            self.conv.weight[1, 1] = 1

    def forward(self, images):
        """Input: Tensor[N,3,H,W]. Output: Tensor[N,2]."""
        return self.conv(images).mean((-2, -1))


def test_normalization_and_encoder():
    """Input: injected model and raw pixels. Output: verifies single normalization."""
    backbone = TinyClassifier()
    encoder = ResNetEncoder(model=backbone)
    x = torch.rand(2, 3, 8, 8)
    assert torch.allclose(
        encoder(x), backbone((x - encoder.mean) / encoder.std)
    )
    assert torch.equal(encoder.encode(x), encoder(x))
    assert encoder.target_layer("conv") is backbone.conv


def test_real_resnet_wrapper():
    """Input: random-weight ResNet (no download). Output: validates shape and default block."""
    encoder = ResNetEncoder(weights=None)
    with torch.no_grad():
        assert encoder(torch.rand(1, 3, 32, 32)).shape == (1, 1000)
    assert encoder.target_layer() is encoder.model.layer4[-1]
    maps, predictions = GradCAMExplainer(
        encoder, encoder.target_layer()
    ).explain(torch.rand(1, 3, 32, 32))
    assert maps.shape == (1, 32, 32) and predictions.shape == (1,)
    assert torch.isfinite(maps).all()


def test_predicted_target_shape_and_hook_cleanup():
    """Input: two images with different predicted classes. Output: matching spatial CAMs."""
    model = TinyClassifier().eval()
    x = torch.zeros(2, 3, 8, 8)
    x[0, 0, :4] = 1
    x[0, 1, 4:] = 0.1
    x[1, 0, :4] = 0.1
    x[1, 1, 4:] = 1
    explainer = GradCAMExplainer(model, model.conv)
    with torch.no_grad():
        maps, predictions = explainer.explain(x)
    assert predictions.tolist() == [0, 1]
    assert maps.shape == (2, 8, 8)
    assert maps[0, :4].mean() > maps[0, 4:].mean()
    assert maps[1, 4:].mean() > maps[1, :4].mean()
    assert all(p.grad is None for p in model.parameters())
    assert not model.conv._forward_hooks
    zero, _ = explainer.explain(torch.zeros_like(x))
    assert torch.isfinite(zero).all() and not zero.any()


def test_vectorization():
    """Input: 14-square and larger heatmaps. Output: 196 finite features per row."""
    maps = torch.arange(196).reshape(1, 14, 14).float()
    vectorizer = GradCAMVectorizer()
    np.testing.assert_array_equal(
        vectorizer.transform(maps), maps.flatten(1).numpy()
    )
    assert vectorizer.transform(torch.ones(3, 1, 224, 224)).shape == (3, 196)
    with pytest.raises(ValueError):
        vectorizer.transform(torch.ones(3, 2, 14, 14))


def test_score_orientation():
    """Input: separable clean/adversarial features. Output: positive-class probabilities."""
    detector = GradCAMLogisticDetector().fit(
        np.array([[-2], [-1], [1], [2]]), [0, 0, 1, 1]
    )
    scores = detector.score_samples([[-2], [2]])
    assert scores.shape == (2,) and 0 <= scores[0] < scores[1] <= 1


def test_group_leakage_and_reproducibility():
    """Input: repeated original source IDs. Output: all variants stay in one held-out fold."""
    groups = np.repeat(np.arange(20), 4)
    cv = GroupedDetectionCV()
    seen = []
    for train, test in cv.split(groups):
        assert not set(groups[train]) & set(groups[test])
        seen.extend(test)
    assert sorted(seen) == list(range(len(groups)))
    for a, b in zip(cv.split(groups), cv.split(groups)):
        np.testing.assert_array_equal(a[1], b[1])


def test_cv_uses_shared_evaluator_and_frozen_training():
    """Input: successful/failed cross-attacks. Output: exact shared metrics and train-only scaler."""
    rng = np.random.default_rng(42)
    clean = rng.normal(size=(20, 4))
    labels = np.zeros(20, dtype=int)
    predictions = labels.copy()
    attacks = {
        "pgd": {"features": clean + 3, "predictions": np.ones(20)},
        "fgsm": {"features": clean + 2, "predictions": np.arange(20) % 2},
    }
    for attack in attacks.values():
        attack["source_ids"] = np.arange(20)
    cv = GroupedDetectionCV()
    results = cv.evaluate(clean, attacks, np.arange(20), labels, predictions)
    for index, (train, test) in enumerate(cv.split(np.arange(20))):
        detector = cv.detectors[index]
        expected = DetectionEvaluator().evaluate(
            detector.score_samples(clean[test]),
            detector.score_samples(attacks["fgsm"]["features"][test]),
            filter_successful_attacks(
                predictions[test],
                attacks["fgsm"]["predictions"][test],
                labels[test],
            ),
        )
        assert results["folds"][index]["metrics"]["fgsm"] == expected
        np.testing.assert_allclose(
            detector.model[0].mean_,
            np.concatenate([clean[train], (clean + 3)[train]]).mean(0),
        )
    before = copy.deepcopy(cv.detectors[0].model[1].coef_)
    attacks["fgsm"]["features"] = clean * 100
    cv.evaluate(clean, attacks, np.arange(20), labels, predictions)
    np.testing.assert_array_equal(before, cv.detectors[0].model[1].coef_)


def test_failed_attacks_and_alignment():
    """Input: failed attacks or misaligned records. Output: undefined metrics or explicit error."""
    clean = np.zeros((10, 2))
    attacks = {
        "pgd": {"features": clean + 1, "predictions": np.ones(10)},
        "fgsm": {"features": clean, "predictions": np.zeros(10)},
    }
    for attack in attacks.values():
        attack["source_ids"] = np.arange(10)
    cv = GroupedDetectionCV()
    result = cv.evaluate(
        clean, attacks, np.arange(10), np.zeros(10), np.zeros(10)
    )
    assert result["summary"]["fgsm"]["auc"]["valid_folds"] == 0
    assert np.isnan(result["summary"]["fgsm"]["auc"]["mean"])
    attacks["pgd"]["features"] = np.ones((9, 2))
    with pytest.raises(ValueError, match="align"):
        cv.evaluate(clean, attacks, np.arange(10), np.zeros(10), np.zeros(10))


@pytest.mark.parametrize(
    "name,params",
    [
        ("pgd", {"steps": 1}),
        ("fgsm", {"eps": 0.1}),
        ("apgd", {"steps": 2, "n_restarts": 1}),
        ("cw", {"steps": 2}),
        ("patch", {"steps": 2, "patch_size": 2}),
    ],
)
def test_shared_attacks(name, params):
    """Input: tiny classifier and raw images. Output: finite valid raw-pixel adversarial batch."""
    model = TinyClassifier().eval()
    images = torch.rand(2, 3, 8, 8)
    result = build_attack(name, model, params)(
        images, torch.zeros(2, dtype=torch.long)
    )
    assert result.shape == images.shape and torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1


def test_patch_scope():
    """Input: fixed patch coordinates. Output: pixels outside the patch remain identical."""
    from attacks.patch import generate_patch_attack

    images = torch.rand(2, 3, 8, 8)
    result = generate_patch_attack(
        TinyClassifier().eval(),
        images,
        torch.zeros(2, dtype=torch.long),
        [0, 0],
        [0, 0],
        patch_size=2,
        steps=2,
    )
    assert torch.equal(result[:, :, 2:], images[:, :, 2:])
    assert torch.equal(result[:, :, :, 2:], images[:, :, :, 2:])


def test_experiment_roundtrip(tmp_path, monkeypatch):
    """Input: synthetic loader/classifier/attack. Output: auditable archive, CSV, models and metrics."""
    import experiments.gradcam as experiment_module
    import csv
    import hashlib
    import json
    from torch.utils.data import DataLoader, Subset, TensorDataset

    images = torch.zeros(10, 3, 8, 8)
    images[:, 0] = 1
    labels = torch.zeros(10, dtype=torch.long)
    from utils.gradcam_data import SourceDataset

    indices = [8, 2, 6, 1, 9, 4, 3, 0, 7, 5]
    loader = DataLoader(
        SourceDataset(Subset(TensorDataset(images, labels), indices)),
        batch_size=2,
    )
    monkeypatch.setattr(
        experiment_module, "make_loader", lambda *args, **kwargs: loader
    )
    monkeypatch.setattr(
        experiment_module, "classifier_labels", lambda labels, *args: labels
    )
    monkeypatch.setattr(
        experiment_module.GradCAMExperiment,
        "classifier",
        lambda self: ResNetEncoder(model=TinyClassifier()),
    )
    monkeypatch.setattr(
        experiment_module, "build_attack", lambda *args: lambda x, y: x.flip(1)
    )
    cfg = {
        "seed": 42,
        "device": "cpu",
        "dataset": {"name": "imagenette"},
        "model": {"target_layer": "conv"},
        "gradcam": {"grid_size": 14},
        "attacks": {"pgd": {}},
        "features_path": str(tmp_path / "features.npz"),
        "results_dir": str(tmp_path / "results"),
        "evaluation": {"folds": 5, "min_correct_per_fold": 1},
        "detector": {},
    }
    experiment = experiment_module.GradCAMExperiment(cfg)
    experiment.extract()
    results = experiment.evaluate()
    assert len(results["folds"]) == 5
    assert (tmp_path / "features.csv").exists()
    assert (tmp_path / "results" / "detector_fold_4.joblib").exists()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "fold_metrics.csv").exists()
    with np.load(tmp_path / "features.npz") as arrays:
        np.testing.assert_array_equal(arrays["groups"], indices)
        np.testing.assert_array_equal(
            arrays["groups"], arrays["pgd_source_ids"]
        )
        np.testing.assert_array_equal(
            arrays["groups"], arrays["clean_source_ids"]
        )
        saved_arrays = dict(arrays)
    with (tmp_path / "results" / "fold_metrics.csv").open() as stream:
        fold_rows = list(csv.DictReader(stream))
    assert len(fold_rows) == 5
    assert set(fold_rows[0]) == {
        "fold",
        "attack",
        "auc",
        "tpr_at_0.01",
        "tpr_at_0.05",
        "attack_success_rate",
        "num_adversarial_evaluated",
        "num_clean",
    }
    for metric in (
        "auc",
        "attack_success_rate",
        "num_adversarial_evaluated",
        "num_clean",
    ):
        values = [fold["metrics"]["pgd"][metric] for fold in results["folds"]]
        assert results["summary"]["pgd"][metric]["std"] == pytest.approx(
            np.std(values, ddof=1)
        )
    cfg["gradcam"]["grid_size"] = 7
    with pytest.raises(ValueError, match="config mismatch"):
        experiment.evaluate()
    cfg["gradcam"]["grid_size"] = 14
    saved_arrays["pgd_source_ids"] = saved_arrays["pgd_source_ids"][::-1]
    path = tmp_path / "features.npz"
    np.savez_compressed(path, **saved_arrays)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source IDs"):
        experiment.evaluate()


def test_experimental_modules():
    """Input: synthetic heatmaps. Output: finite CNN, AE and clean-only statistical scores."""
    from detectors.gradcam_experimental import (
        GradCAMCNN,
        GradCAMAutoencoder,
        GradCAMStatistics,
        GradCAMMahalanobisDetector,
    )

    maps = torch.rand(4, 1, 16, 16)
    assert GradCAMCNN()(maps).shape == (4,)
    assert GradCAMAutoencoder().score_samples(maps).shape == (4,)
    features = GradCAMStatistics().transform(maps[:, 0].numpy())
    assert features.shape == (4, 10) and np.isfinite(features).all()
    detector = GradCAMMahalanobisDetector().fit(features)
    assert detector.score_samples(features).shape == (4,)


def test_cifar_head_train_load_and_provenance(tmp_path, monkeypatch):
    """Input: synthetic official-train substitute. Output: saved/loaded head and rejected bad provenance."""
    import experiments.gradcam as module
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import resnet18

    monkeypatch.setattr(
        module,
        "ResNetEncoder",
        lambda weights: ResNetEncoder(model=resnet18(weights=None)),
    )
    from utils.gradcam_data import SourceDataset

    loader = DataLoader(
        SourceDataset(
            TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
        ),
        batch_size=2,
    )
    monkeypatch.setattr(module, "make_loader", lambda *args, **kwargs: loader)
    checkpoint = tmp_path / "head.pt"
    config = {
        "seed": 42,
        "device": "cpu",
        "dataset": {"name": "cifar10", "image_size": 32},
        "model": {
            "weights": "IMAGENET1K_V2",
            "head_checkpoint": str(checkpoint),
        },
        "head_training": {"epochs": 1, "learning_rate": 0.001},
    }
    experiment = module.GradCAMExperiment(config)
    experiment.train_head()
    model = experiment.classifier()
    assert model.model.fc.out_features == 10 and not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["training_split"] == "official_cifar10_train"
    saved["training_split"] = "test"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="split"):
        experiment.classifier()


def test_source_ids_survive_nested_reordered_subsets():
    """Input: nested Subsets. Output: original IDs preserved, including duplicates."""
    from torch.utils.data import Subset, TensorDataset
    from utils.gradcam_data import SourceDataset

    base = TensorDataset(torch.arange(10), torch.arange(10))
    subset = Subset(Subset(base, [8, 2, 6, 1]), [2, 0, 2, 1])
    dataset = SourceDataset(subset)
    assert [dataset[index][2] for index in range(len(dataset))] == [6, 8, 6, 2]
    for image, label, source_id in dataset:
        assert int(image) == int(label) == source_id


def test_cv_rejects_reordered_source_ids():
    """Input: equal-sized misaligned attack rows. Output: explicit alignment failure."""
    clean = np.zeros((10, 2))
    groups = np.arange(10)
    attack = {
        "features": clean + 1,
        "predictions": np.ones(10),
        "source_ids": groups[::-1],
    }
    with pytest.raises(ValueError, match="source IDs"):
        GroupedDetectionCV().evaluate(
            clean, {"pgd": attack}, groups, np.zeros(10), np.zeros(10)
        )


def test_cross_attack_scoring_preserves_fitted_parameters(monkeypatch):
    """Input: all five attacks. Output: every fitted scaler/head parameter stays unchanged."""
    original_fit = GradCAMLogisticDetector.fit
    original_score = GradCAMLogisticDetector.score_samples
    snapshots = {}
    score_calls = []

    def fitted_state(detector):
        """Input: fitted detector. Output: dict of copied sklearn learned attributes."""
        return {
            f"{index}.{name}": copy.deepcopy(value)
            for index, step in enumerate(detector.model)
            for name, value in vars(step).items()
            if name.endswith("_")
        }

    def track_fit(detector, features, labels):
        """Input: detector and training arrays. Output: fitted detector with recorded state."""
        assert id(detector) not in snapshots
        result = original_fit(detector, features, labels)
        snapshots[id(detector)] = fitted_state(detector)
        return result

    def check_score(detector, features):
        """Input: detector and held-out array. Output: scores with unchanged fitted state."""
        before = fitted_state(detector)
        result = original_score(detector, features)
        after = fitted_state(detector)
        for name, expected in snapshots[id(detector)].items():
            np.testing.assert_array_equal(before[name], expected)
            np.testing.assert_array_equal(after[name], expected)
        score_calls.append(id(detector))
        return result

    monkeypatch.setattr(GradCAMLogisticDetector, "fit", track_fit)
    monkeypatch.setattr(GradCAMLogisticDetector, "score_samples", check_score)
    clean = np.random.default_rng(42).normal(size=(20, 4))
    attacks = {
        name: {
            "features": clean + index + 1,
            "predictions": np.ones(20),
            "source_ids": np.arange(20),
        }
        for index, name in enumerate(["pgd", "fgsm", "apgd", "cw", "patch"])
    }
    GroupedDetectionCV().evaluate(
        clean, attacks, np.arange(20), np.zeros(20), np.zeros(20)
    )
    assert len(snapshots) == 5
    assert len(score_calls) == 30


def test_head_step_updates_only_fc():
    """Input: one head training step. Output: unchanged backbone weights and BatchNorm buffers."""
    from experiments.gradcam import GradCAMExperiment
    from torchvision.models import resnet18

    model = ResNetEncoder(model=resnet18(weights=None))
    model.model.fc = nn.Linear(model.model.fc.in_features, 10)
    experiment = GradCAMExperiment(
        {
            "seed": 42,
            "device": "cpu",
            "head_training": {"learning_rate": 0.001},
        }
    )
    optimizer = experiment.head_optimizer(model)
    assert not model.training
    assert {
        id(p) for group in optimizer.param_groups for p in group["params"]
    } == {id(p) for p in model.model.fc.parameters()}
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith("model.fc.")
    before = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(
        model(torch.rand(2, 3, 32, 32)), torch.tensor([0, 1])
    )
    loss.backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith("model.fc."):
            assert torch.equal(before[name], value), name
    assert not torch.equal(before["model.fc.weight"], model.model.fc.weight)


def test_missing_head_fails_before_model_loading(tmp_path, monkeypatch):
    """Input: missing checkpoint. Output: failure before backbone or attack loading."""
    import experiments.gradcam as module

    def unexpected_model(*args, **kwargs):
        """Input: any arguments. Output: failure if backbone loading is attempted."""
        pytest.fail("Backbone must not load before missing-head validation")

    monkeypatch.setattr(module, "ResNetEncoder", unexpected_model)
    experiment = module.GradCAMExperiment(
        {
            "seed": 42,
            "device": "cpu",
            "dataset": {"name": "cifar10"},
            "model": {"head_checkpoint": str(tmp_path / "missing.pt")},
        }
    )
    with pytest.raises(FileNotFoundError, match="train-head"):
        experiment.extract()


@pytest.mark.parametrize(
    "field,value",
    [("epochs_completed", 0), ("optimizer_steps", 0), ("state_dict", {})],
)
def test_untrained_checkpoint_is_rejected(tmp_path, field, value):
    """Input: incomplete training provenance. Output: explicit checkpoint rejection."""
    from experiments.gradcam import GradCAMExperiment

    path = tmp_path / "head.pt"
    checkpoint = {
        "training_split": "official_cifar10_train",
        "weights": "IMAGENET1K_V2",
        "image_size": 224,
        "epochs_completed": 1,
        "optimizer_steps": 1,
        "state_dict": {"weight": torch.ones(10, 4), "bias": torch.zeros(10)},
    }
    checkpoint[field] = value
    torch.save(checkpoint, path)
    experiment = GradCAMExperiment(
        {
            "seed": 42,
            "device": "cpu",
            "dataset": {"name": "cifar10", "image_size": 224},
            "model": {
                "head_checkpoint": str(path),
                "weights": "IMAGENET1K_V2",
            },
        }
    )
    with pytest.raises(ValueError, match="train-head"):
        experiment.load_head_checkpoint()


def test_clean_quality_gate_runs_before_attacks(tmp_path, monkeypatch):
    """Input: insufficient clean correctness. Output: saved accuracy and no attack construction."""
    import json
    import experiments.gradcam as module
    from torch.utils.data import DataLoader, TensorDataset
    from utils.gradcam_data import SourceDataset

    images = torch.zeros(10, 3, 8, 8)
    images[:, 0] = 1
    loader = DataLoader(
        SourceDataset(TensorDataset(images, torch.ones(10, dtype=torch.long))),
        batch_size=2,
    )
    monkeypatch.setattr(module, "make_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(
        module, "classifier_labels", lambda labels, *args: labels
    )
    monkeypatch.setattr(
        module.GradCAMExperiment,
        "classifier",
        lambda self: ResNetEncoder(model=TinyClassifier()),
    )

    def unexpected_attack(*args):
        """Input: any arguments. Output: failure if an attack is built."""
        pytest.fail("Attacks must not start with insufficient clean accuracy")

    monkeypatch.setattr(module, "build_attack", unexpected_attack)
    config = {
        "seed": 42,
        "device": "cpu",
        "dataset": {"name": "imagenette"},
        "model": {"target_layer": "conv"},
        "gradcam": {"grid_size": 14},
        "evaluation": {"folds": 5, "min_correct_per_fold": 1},
        "results_dir": str(tmp_path),
    }
    with pytest.raises(ValueError, match="Too few"):
        module.GradCAMExperiment(config).extract()
    report = json.loads((tmp_path / "clean_classifier.json").read_text())
    assert report["num_correct"] == 0 and report["num_samples"] == 10
    assert report["clean_accuracy"] == 0
