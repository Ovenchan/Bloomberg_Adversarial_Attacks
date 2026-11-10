"""Offline tests: classifier equivalence, corpus integrity, and detector fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import resnet50

from encoders.cnn import ResNetEncoder
from experiments.resnet_representation import Corpus, extract, evaluate


def test_pooled_features_match_original_classifier_without_changing_encode():
    torch.manual_seed(0)
    model = ResNetEncoder(weights=None, model=resnet50(weights=None)).eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        features = model.pooled_features(images)
        assert features.shape == (2, 2048)
        torch.testing.assert_close(model.model.fc(features), model(images))
        torch.testing.assert_close(model.encode(images), model(images))


class TinyEncoder(nn.Module):
    """Fast injected encoder; no checkpoint/network required for integration tests."""
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.fc = nn.Linear(2048, 2, bias=False)
        with torch.no_grad():
            self.model.fc.weight.zero_()
            self.model.fc.weight[0, 0] = 1
            self.model.fc.weight[1, 0] = -1

    def pooled_features(self, x):
        z = torch.zeros(len(x), 2048, device=x.device)
        z[:, 0] = 0.5 - x.mean((1, 2, 3))
        z[:, 1] = x[:, 0, 0, 0]
        return z


def fixture_corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "images").mkdir(parents=True)
    config = {"paths": {"work_dir": str(root), "results_dir": str(tmp_path / "original_results")},
              "dataset": {"image_size": 4}, "model": {"weights": "IMAGENET1K_V2"},
              "partition_attacks": {"train": ["pgd"], "dev": ["pgd"], "test": ["pgd", "cw"]},
              "noise": {"noise": {}}, "seed": 42,
              "detector": {"C_grid": [0.01, 1.0], "max_iter": 500},
              "evaluation": {"fpr_targets": [0.01, 0.05]}}
    manifest, variants, hashes = [], [], {}
    for partition in ("train", "dev", "cal", "test"):
        sources = [f"{partition}_{i}" for i in range(8)]
        manifest.extend({"source_id": s, "partition": partition, "imagenet_label": 0} for s in sources)
        names = ["clean"] + config["partition_attacks"].get(partition, [])
        if partition == "test":
            names += ["noise"]
        for variant in names:
            attack = variant in ("pgd", "cw")
            x = torch.linspace(0.7 if attack else 0.1, 0.9 if attack else 0.3, 8)
            images = x[:, None, None, None].expand(8, 3, 4, 4).clone()
            if attack:
                images[-1] = 0.2  # failed attack must be excluded, not mislabeled positive
            if variant == "clean":
                images = (images * 255).round().to(torch.uint8)
            name = f"{partition}_{variant}"
            torch.save(images, root / "images" / f"{name}.pt")
            hashes[name] = hashlib.sha256(images.numpy().tobytes()).hexdigest()[:16]
            for i, source in enumerate(sources):
                success = attack and i < 7
                variants.append({"source_id": source, "partition": partition, "variant": variant,
                                 "imagenet_label": 0, "tensor_index": i, "clean_pred": 0,
                                 "pred": int(success), "eligible": True, "success": success,
                                 "explain": success if attack else True})
    for name, rows in (("manifest.csv", manifest), ("variants.csv", variants)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "corpus.json").write_text(json.dumps({"weights": config["model"]["weights"],
                                                "config": config, "tensors": hashes}))
    return config


def test_end_to_end_preserves_inputs_and_filters_failed_attacks(tmp_path):
    config = fixture_corpus(tmp_path)
    corpus = Corpus(config)
    before = {p: p.read_bytes() for p in corpus.root.rglob("*") if p.is_file()}
    output = tmp_path / "output"
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True)
    extract(corpus, feature_dir, torch.device("cpu"), 3, model=TinyEncoder())
    evaluate(corpus, feature_dir, output)
    assert all(p.read_bytes() == content for p, content in before.items())
    with (output / "stage1_detection.csv").open() as stream:
        results = list(csv.DictReader(stream))
    assert len(results) == 2
    assert all(int(r["num_successful"]) == 7 and int(r["num_eligible"]) == 8 for r in results)
    assert all(float(r["auc"]) == 1 for r in results)
    with np.load(feature_dir / "train_clean.npz") as data:
        assert data["features"].shape == (8, 2048)
        # uint8 -> raw float is applied exactly once
        assert data["features"][0, 0] == pytest.approx(0.5 - 26 / 255)
    assert (output / "detector.joblib").exists()
    thresholds = json.loads((output / "thresholds.json").read_text())
    assert all(t["num_calibration"] == 8 for t in thresholds)


def test_rejects_source_leakage(tmp_path):
    config = fixture_corpus(tmp_path)
    path = tmp_path / "corpus" / "variants.csv"
    path.write_text(path.read_text().replace("dev_0,dev", "train_0,dev"))
    with pytest.raises(ValueError, match="crosses partitions"):
        Corpus(config)


def test_rejects_changed_saved_tensor(tmp_path):
    corpus = Corpus(fixture_corpus(tmp_path))
    path = corpus.path(("train", "pgd"))
    x = torch.load(path, weights_only=True)
    x[0, 0, 0, 0] += 0.01
    torch.save(x, path)
    output = tmp_path / "features"
    output.mkdir()
    with pytest.raises(ValueError, match="hash mismatch"):
        extract(corpus, output, torch.device("cpu"), 8, model=TinyEncoder())


def test_rejects_classifier_mismatch(tmp_path):
    corpus = Corpus(fixture_corpus(tmp_path))
    output = tmp_path / "features"
    output.mkdir()
    model = TinyEncoder()
    with torch.no_grad():
        model.model.fc.weight.neg_()
    with pytest.raises(ValueError, match="predictions differ"):
        extract(corpus, output, torch.device("cpu"), 8, model=model)


def test_missing_corpus_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Copy the original saved corpus"):
        Corpus({"paths": {"work_dir": str(tmp_path / "missing")}})


class TinyDinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.seen = None

    def forward(self, images):
        self.seen = images.detach().clone()
        z = torch.zeros(len(images), 384, device=images.device)
        z[:, :3] = images.mean((2, 3)) * self.scale
        return z


def test_dino_observer_normalizes_once_and_freezes_backbone():
    from encoders.dinov2_observer import DinoV2Observer
    backbone = TinyDinoBackbone()
    observer = DinoV2Observer(backbone=backbone)
    raw = torch.rand(2, 3, 4, 4)
    original = raw.clone()
    assert observer(raw).shape == (2, 384)
    torch.testing.assert_close(backbone.seen, (raw - observer.mean) / observer.std)
    torch.testing.assert_close(raw, original)
    assert not observer.training and not backbone.scale.requires_grad
    assert len(observer.weights_sha256()) == 64


def test_dino_and_resnet_use_identical_corpus_and_evaluation_cohorts(tmp_path):
    from encoders.dinov2_observer import DinoV2Observer
    corpus = Corpus(fixture_corpus(tmp_path))
    runs = {}
    for name, observer in (("resnet", None), ("dinov2", DinoV2Observer(backbone=TinyDinoBackbone()))):
        output = tmp_path / name
        feature_dir = output / "features"
        feature_dir.mkdir(parents=True)
        extract(corpus, feature_dir, torch.device("cpu"), 3, model=TinyEncoder(), observer=observer)
        evaluate(corpus, feature_dir, output)
        with (output / "stage1_detection.csv").open() as stream:
            results = list(csv.DictReader(stream))
        assert all(int(r["num_successful"]) == 7 for r in results)
        assert all(float(r["auc"]) == 1 for r in results)
        expected_method = "resnet_pool" if observer is None else "dinov2_observer"
        assert all(r["method"] == expected_method for r in results)
        with (output / "scores.csv").open() as stream:
            scores = list(csv.DictReader(stream))
        provenance = json.loads((feature_dir / "provenance.json").read_text())
        runs[name] = (provenance, [(r["source_id"], r["partition"], r["variant"]) for r in scores])
    assert runs["resnet"][0]["corpus_files"] == runs["dinov2"][0]["corpus_files"]
    assert runs["resnet"][1] == runs["dinov2"][1]
    assert runs["dinov2"][0]["feature_dimension"] == 384
    for name, block in runs["resnet"][0]["blocks"].items():
        assert block["tensor_sha256"] == runs["dinov2"][0]["blocks"][name]["tensor_sha256"]


def test_dino_still_checks_resnet_predictions(tmp_path):
    from encoders.dinov2_observer import DinoV2Observer
    corpus = Corpus(fixture_corpus(tmp_path))
    output = tmp_path / "features"
    output.mkdir()
    target = TinyEncoder()
    with torch.no_grad():
        target.model.fc.weight.neg_()
    with pytest.raises(ValueError, match="predictions differ"):
        extract(corpus, output, torch.device("cpu"), 8, model=target,
                observer=DinoV2Observer(backbone=TinyDinoBackbone()))


def test_record_prediction_policy_reports_drift_without_changing_cohort(tmp_path):
    corpus = Corpus(fixture_corpus(tmp_path))
    output = tmp_path / "run"
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True)
    target = TinyEncoder()
    with torch.no_grad():
        target.model.fc.weight.neg_()
    original = [dict(r) for r in corpus.records]
    extract(corpus, feature_dir, torch.device("cpu"), 8, model=target, prediction_policy="record")
    assert corpus.records == original
    provenance = json.loads((feature_dir / "provenance.json").read_text())
    assert provenance["prediction_policy"] == "record"
    assert provenance["prediction_mismatch_count"] == len(corpus.records)
    with (feature_dir / "prediction_mismatches.csv").open() as stream:
        mismatches = list(csv.DictReader(stream))
    assert len(mismatches) == len(corpus.records)
    assert all(int(r["saved_prediction"]) != int(r["local_prediction"]) for r in mismatches)
    evaluate(corpus, feature_dir, output)
    with (output / "stage1_detection.csv").open() as stream:
        assert all(int(r["num_successful"]) == 7 for r in csv.DictReader(stream))


def test_stage1_metrics_match_teammate_evaluator(tmp_path):
    """Feed the same saved scores into the original evaluator; compare every field."""
    from types import SimpleNamespace
    from experiments.slice_belief_analysis import SliceBeliefAnalysis
    corpus = Corpus(fixture_corpus(tmp_path))
    output = tmp_path / 'run'
    feature_dir = output / 'features'
    feature_dir.mkdir(parents=True)
    extract(corpus, feature_dir, torch.device('cpu'), 8, model=TinyEncoder())
    evaluate(corpus, feature_dir, output)
    with (output / 'scores.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    def features(partition, variant, method):
        block = [r for r in scores if r['partition'] == partition and r['variant'] == variant]
        return (np.array([float(r['score']) for r in block]),
                np.array([r['source_id'] for r in block]), corpus.selected((partition, variant)))
    reference = object.__new__(SliceBeliefAnalysis)
    reference.bank = SimpleNamespace(features=features, variants=corpus.records)
    reference.noise = list(corpus.config['noise'])
    thresholds = json.loads((output / 'thresholds.json').read_text())
    detection, noise, _ = reference.evaluate(SimpleNamespace(score_samples=lambda x: x),
                                             'resnet_pool', thresholds, ['pgd', 'cw'])
    for filename, expected_rows in [('stage1_detection.csv', detection), ('stage1_noise.csv', noise)]:
        with (output / filename).open() as stream:
            actual = list(csv.DictReader(stream))
        for expected, found in zip(expected_rows, actual):
            for key, value in expected.items():
                if isinstance(value, str):
                    assert found[key] == value
                else:
                    assert float(found[key]) == pytest.approx(value, nan_ok=True), key


def test_stage2_matches_reference_and_excludes_holdout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import experiments.slice_belief_analysis as reference
    from experiments.resnet_representation_stage2 import ResNetStage2
    config = fixture_corpus(tmp_path)
    config.update(stage2_folds={'cw': ['pgd', 'apgd']}, experiment_seeds=[42, 123])
    config['evaluation'].update(max_matched_train=100, bootstrap_replicates=20)
    path = Path(config['paths']['work_dir']) / 'variants.csv'
    with path.open() as f:
        records = list(csv.DictReader(f))
    root = path.parent
    provenance = json.loads((root / 'corpus.json').read_text())
    for partition in ('train', 'dev'):
        config['partition_attacks'][partition].append('apgd')
        source = root / 'images' / f'{partition}_pgd.pt'
        (root / 'images' / f'{partition}_apgd.pt').write_bytes(source.read_bytes())
        provenance['tensors'][f'{partition}_apgd'] = provenance['tensors'][f'{partition}_pgd']
        records += [dict(r, variant='apgd') for r in list(records)
                    if r['partition'] == partition and r['variant'] == 'pgd']
    provenance['config'] = config
    (root / 'corpus.json').write_text(json.dumps(provenance))
    for r in records:
        r['class_index'] = int(r['source_id'].split('_')[-1]) % 2
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    corpus = Corpus(config)
    feature_dir = tmp_path / 'stage1' / 'features'
    feature_dir.mkdir(parents=True)
    extract(corpus, feature_dir, torch.device('cpu'), 8, model=TinyEncoder())
    runner = ResNetStage2(config, feature_dir, tmp_path / 'stage2')
    runner.run()
    ref = object.__new__(reference.SliceBeliefAnalysis)
    ref.config, ref.bank = config, runner.bank
    ref.alphas, ref.test_attacks, ref.noise = runner.alphas, runner.test_attacks, runner.noise
    ref.tables = tmp_path / 'reference'
    ref.tables.mkdir()
    monkeypatch.setattr(reference, 'METHODS', ('resnet_pool',))
    ref.stage2()
    for name in ('stage2_per_seed.csv', 'stage2_summary.csv', 'stage2_bootstrap.csv'):
        assert (runner.tables / name).read_bytes() == (ref.tables / name).read_bytes()
    train_assignments = [r for r in runner.assignments if r['partition'] == 'train']
    assert {r['mixture_attack'] for r in train_assignments} == {'pgd', 'apgd'}
    assert all(r['baseline_attack'] == 'pgd' for r in train_assignments)
    assert len(list((runner.tables / 'models').glob('*.joblib'))) == 4
    with (runner.tables / 'stage2_bootstrap.csv').open() as f:
        gain = next(csv.DictReader(f))
    assert float(gain['auc_gain']) == 0 and float(gain['tpr5_gain']) == 0
    # The consolidated entry point must run both stages with a single extraction,
    # and skip extraction completely when an existing cache is supplied.
    import sys
    import yaml
    import experiments.resnet_representation as combined
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    calls = []
    original_extract = combined.extract
    def offline_extract(corpus, destination, device, batch_size, **kwargs):
        calls.append(destination)
        return original_extract(corpus, destination, device, batch_size, model=TinyEncoder(), **kwargs)
    monkeypatch.setattr(combined, 'extract', offline_extract)
    for reuse in (False, True):
        directory = tmp_path / ('combined_cached' if reuse else 'combined_fresh')
        argv = ['resnet_representation', '--config', str(config_path), '--stage', 'both',
                '--output', str(directory)]
        if reuse:
            argv += ['--features', str(feature_dir)]
        monkeypatch.setattr(sys, 'argv', argv)
        combined.main()
        assert (directory / 'stage1_detection.csv').exists()
        assert (directory / 'stage2_summary.csv').exists()
        assert (directory / 'combined_results.csv').exists()
        assert (directory / 'SETUP.md').exists()
        assert not (directory / 'stage2').exists()
        assert len(calls) == 1
    config['stage2_folds'] = {'cw': ['pgd', 'cw']}
    with pytest.raises(ValueError, match='Held-out'):
        ResNetStage2(config, feature_dir, tmp_path / 'bad')
