"""ResNet pooled-feature baseline on the existing explanation attack corpus.

Run from the repository root with python -m experiments.resnet_representation.
Never generates images, attacks, or explanations; never writes to the corpus.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from analysis.calibration import calibrate_threshold, frozen_detection_metrics, rate_at_threshold
from analysis.metrics import compute_auc, compute_tpr_at_fpr
from analysis.bootstrap import fast_auc, paired_bootstrap
from experiments.slice_belief_analysis import SliceBeliefAnalysis, PRIMARY_ALPHA, CLEAN
from utils.controlled_data import derive_seed
from detectors.logistic import GradCAMLogisticDetector
from encoders.cnn import ResNetEncoder

METHOD = "resnet_pool"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


class Corpus:
    """Read-only corpus adapter, with identity and success-mask validation."""
    def __init__(self, config):
        self.config = config
        self.root = Path(config["paths"]["work_dir"])
        for name in ("manifest.csv", "variants.csv", "corpus.json"):
            if not (self.root / name).is_file():
                raise FileNotFoundError(
                    f"Missing explanation corpus artifact: {self.root / name}. "
                    "Copy the original saved corpus including images/; this command does not regenerate attacks.")
        self.provenance = json.loads((self.root / "corpus.json").read_text())
        if self.provenance["weights"] != config["model"]["weights"]:
            raise ValueError("Configured weights differ from the explanation corpus")
        saved = self.provenance["config"]
        for key in ("dataset", "model", "partition_attacks", "noise"):
            if saved[key] != config[key]:
                raise ValueError(f"Configuration differs from saved corpus: {key}")
        with (self.root / "manifest.csv").open(newline="") as stream:
            manifest = list(csv.DictReader(stream))
        identities = {r["source_id"]: r for r in manifest}
        if len(identities) != len(manifest):
            raise ValueError("Duplicate source IDs in manifest")
        with (self.root / "variants.csv").open(newline="") as stream:
            self.records = list(csv.DictReader(stream))
        self.blocks = {}
        for row in self.records:
            for key in ("tensor_index", "pred", "clean_pred", "imagenet_label"):
                row[key] = int(row[key])
            for key in ("eligible", "success", "explain"):
                if row[key] not in ("True", "False"):
                    raise ValueError(f"Invalid boolean: {key}")
                row[key] = row[key] == "True"
            source = identities.get(row["source_id"])
            if source is None or source["partition"] != row["partition"]:
                raise ValueError("Source identity missing or crosses partitions")
            if int(source["imagenet_label"]) != row["imagenet_label"]:
                raise ValueError("Source label mismatch")
            if row["eligible"] != (row["clean_pred"] == row["imagenet_label"]):
                raise ValueError("Incorrect clean eligibility mask")
            attack = row["variant"] in config["partition_attacks"].get(row["partition"], [])
            expected = row["eligible"] and row["pred"] != row["imagenet_label"]
            if attack and (row["success"] != expected or row["explain"] != expected):
                raise ValueError("Attack success mask differs from explanation cohort")
            if not attack and not row["explain"]:
                raise ValueError("Clean/noise rows must all be included")
            self.blocks.setdefault((row["partition"], row["variant"]), []).append(row)
        for key, rows in self.blocks.items():
            rows.sort(key=lambda r: r["tensor_index"])
            if [r["tensor_index"] for r in rows] != list(range(len(rows))):
                raise ValueError(f"Invalid tensor indices: {key}")
            if len({r["source_id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate source rows: {key}")
        required = [(p, "clean") for p in ("train", "dev", "cal", "test")]
        required += [(p, a) for p, attacks in config["partition_attacks"].items() for a in attacks]
        required += [("test", n) for n in config["noise"]]
        for key in required:
            if key not in self.blocks or not self.path(key).is_file():
                raise FileNotFoundError(f"Missing corpus block: {key}, {self.path(key)}")
        for partition in ("train", "dev", "cal", "test"):
            clean = {r["source_id"]: r for r in self.blocks[(partition, "clean")]}
            expected = {s for s, r in identities.items() if r["partition"] == partition}
            if set(clean) != expected:
                raise ValueError(f"Clean cohort differs from manifest: {partition}")
            for (part, variant), rows in self.blocks.items():
                if part != partition:
                    continue
                if variant in config["partition_attacks"].get(partition, []):
                    if {r["source_id"] for r in rows} != expected:
                        raise ValueError(f"Attack cohort differs from clean cohort: {part}/{variant}")
                for row in rows:
                    original = clean.get(row["source_id"])
                    if original is None or row["clean_pred"] != original["pred"]:
                        raise ValueError("Variant clean prediction differs from its original source")
        self.fingerprint = {name: sha(self.root / name) for name in ("manifest.csv", "variants.csv", "corpus.json")}

    def path(self, key):
        return self.root / "images" / f"{key[0]}_{key[1]}.pt"

    def selected(self, key):
        return [r for r in self.blocks[key] if r["explain"]]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def extract(corpus, destination, device, batch_size, model=None, observer=None,
            prediction_policy="strict"):
    """Extract all corpus rows, checking saved hashes and classifier predictions."""
    if prediction_policy not in ("strict", "record"):
        raise ValueError("prediction_policy must be strict or record")
    mismatches = []
    model = (model if model is not None else ResNetEncoder(
        weights=corpus.config["model"]["weights"])).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    method = METHOD if observer is None else "dinov2_observer"
    dimension = 2048 if observer is None else 384
    if observer is not None:
        observer.to(device).eval()
        for parameter in observer.parameters():
            parameter.requires_grad_(False)
    provenance = {"method": method, "feature_dimension": dimension, "weights": corpus.provenance["weights"],
                  "attack_target": "ResNet-50", "success_mask_source": "saved ResNet predictions",
                  "prediction_policy": prediction_policy,
                  "original_versions": corpus.provenance.get("versions", {}),
                  "corpus_files": corpus.fingerprint, "device": str(device), "torch": torch.__version__,
                  "normalization": "ImageNet normalization once per encoder on unchanged raw pixels", "blocks": {}}
    if observer is not None:
        provenance["observer"] = {"model": "dinov2_vits14", "weights_sha256": observer.weights_sha256(),
                                  "feature": "CLS embedding", "classification_head": None}
    for key, rows in sorted(corpus.blocks.items()):
        name = f"{key[0]}_{key[1]}"
        images = torch.load(corpus.path(key), map_location="cpu", weights_only=True)
        digest = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        if digest[:16] != corpus.provenance["tensors"][name]:
            raise ValueError(f"Saved tensor hash mismatch: {name}")
        size = corpus.config["dataset"]["image_size"]
        if images.shape != (len(rows), 3, size, size):
            raise ValueError(f"Unexpected image tensor shape: {name}")
        vectors, elapsed = [], 0.0
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            batch = batch.float() / 255 if batch.dtype == torch.uint8 else batch.float()
            if not torch.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
                raise ValueError(f"Images must be finite raw [0,1] pixels: {name}")
            batch = batch.to(device)
            synchronize(device)
            before = time.perf_counter()
            features = model.pooled_features(batch) if observer is None else observer(batch)
            synchronize(device)
            elapsed += time.perf_counter() - before
            if features.shape != (len(batch), dimension) or not torch.isfinite(features).all():
                raise ValueError(f"Expected finite {dimension}-dimensional features")
            resnet_features = features if observer is None else model.pooled_features(batch)
            logits = model.model.fc(resnet_features).cpu()
            predictions = logits.argmax(1).numpy()
            expected = np.array([r["pred"] for r in rows[start:start + len(batch)]])
            if not np.array_equal(predictions, expected):
                probabilities = logits.softmax(1)
                for j in np.flatnonzero(predictions != expected):
                    row = rows[start + j]
                    top2 = logits[j].topk(2).values
                    mismatches.append({"block": name, "source_id": row["source_id"],
                        "tensor_index": start + int(j), "saved_prediction": int(expected[j]),
                        "local_prediction": int(predictions[j]), "true_label": row["imagenet_label"],
                        "saved_success": row["success"], "saved_eligible": row["eligible"],
                        "saved_prediction_locally_correct": int(expected[j]) == row["imagenet_label"],
                        "local_prediction_correct": int(predictions[j]) == row["imagenet_label"],
                        "local_top2_logit_gap": float(top2[0] - top2[1]),
                        "local_winner_minus_saved_logit": float(logits[j, predictions[j]] - logits[j, expected[j]]),
                        "local_probability_saved_class": float(probabilities[j, expected[j]]),
                        "local_probability_winning_class": float(probabilities[j, predictions[j]])})
                report = destination / "prediction_mismatches.csv"
                write_csv(report, mismatches)
                if prediction_policy == "strict":
                    raise ValueError(f"Classifier predictions differ from saved corpus: {name}. "
                                     f"Details saved to {report}. Check weights/preprocessing and review "
                                     "the report. To explicitly retain the original corpus masks while "
                                     "recording local prediction differences, use --prediction-policy record "
                                     "with a NEW --output directory.")
            vectors.append(features.cpu().numpy())
        np.savez_compressed(destination / f"{name}.npz", features=np.concatenate(vectors),
                            source_ids=np.array([r["source_id"] for r in rows]))
        provenance["blocks"][name] = {"tensor_sha256": digest, "rows": len(rows),
                                      "feature_seconds": elapsed, "seconds_per_image": elapsed / len(rows)}
        print(f"{name}: {len(rows)} x {dimension} {method} features", flush=True)
    provenance["prediction_mismatch_count"] = len(mismatches)
    if mismatches:
        print(f"WARNING: {len(mismatches)} local predictions differ; original corpus masks retained. "
              "See features/prediction_mismatches.csv. This is not an exact prediction reproduction.", flush=True)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))


def evaluate(corpus, features_dir, output):
    """Stage 1: PGD-only detector, development tuning, frozen clean thresholds."""
    cfg = corpus.config
    bank = {}
    provenance = json.loads((features_dir / "provenance.json").read_text())
    method = provenance["method"]
    dimension = provenance["feature_dimension"]
    if (method, dimension) not in ((METHOD, 2048), ("dinov2_observer", 384)):
        raise ValueError("Unsupported representation method/dimension")
    if provenance["corpus_files"] != corpus.fingerprint:
        raise ValueError("Features were extracted from different corpus metadata")
    def features(partition, variant):
        key = (partition, variant)
        if key not in bank:
            with np.load(features_dir / f"{partition}_{variant}.npz", allow_pickle=False) as archive:
                rows = corpus.blocks[key]
                if not np.array_equal(archive["source_ids"], [r["source_id"] for r in rows]):
                    raise ValueError("Feature source order differs from corpus")
                X = archive["features"]
                if X.shape != (len(rows), dimension) or not np.isfinite(X).all():
                    raise ValueError("Invalid feature matrix")
                bank[key] = X[np.array([r["explain"] for r in rows])]
        return bank[key]
    neg, pos = features("train", "clean"), features("train", "pgd")
    dev_neg, dev_pos = features("dev", "clean"), features("dev", "pgd")
    if not all(len(x) for x in (neg, pos, dev_neg, dev_pos)):
        raise ValueError("Training/development require clean and successful PGD samples")
    best = None
    for C in cfg["detector"]["C_grid"]:
        detector = GradCAMLogisticDetector(C=C, seed=cfg["seed"], max_iter=cfg["detector"]["max_iter"])
        detector.fit(np.vstack([neg, pos]), np.r_[np.zeros(len(neg)), np.ones(len(pos))])
        auc = fast_auc(detector.score_samples(dev_neg), detector.score_samples(dev_pos))
        if best is None or auc > best[0]:
            best = (auc, C, detector)
    dev_auc, C, detector = best
    cal = detector.score_samples(features("cal", "clean"))
    thresholds = [calibrate_threshold(cal, alpha) for alpha in cfg["evaluation"]["fpr_targets"]]
    clean = detector.score_samples(features("test", "clean"))
    detection, noise, score_rows = [], [], []
    def record_scores(partition, variant, scores):
        score_rows.extend({"source_id": r["source_id"], "partition": partition, "variant": variant,
                           "score": float(s)} for r, s in zip(corpus.selected((partition, variant)), scores))
    record_scores("cal", "clean", cal)
    record_scores("test", "clean", clean)
    for attack in cfg["partition_attacks"]["test"]:
        X = features("test", attack)
        scores = detector.score_samples(X) if len(X) else np.zeros(0)
        record_scores("test", attack, scores)
        rows = corpus.blocks[("test", attack)]
        eligible = sum(r["eligible"] for r in rows)
        metrics = frozen_detection_metrics(clean, scores, thresholds, eligible)
        row = {"method": method, "C": C, "dev_auc": dev_auc, "attack": attack,
               "num_clean": len(clean), "num_successful": len(scores), "num_eligible": eligible,
               "asr": len(scores) / eligible if eligible else float("nan"), "auc": metrics["auc"]}
        for alpha, point in metrics["operating_points"].items():
            row.update({f"tpr@{alpha}": point["tpr"], f"tpr@{alpha}_ci_low": point["tpr_ci"][0],
                        f"tpr@{alpha}_ci_high": point["tpr_ci"][1],
                        f"clean_fpr@{alpha}": point["clean_fpr"]["rate"],
                        f"undetected@{alpha}": point.get("undetected_success_rate", float("nan"))})
            labels = np.r_[np.zeros(len(clean)), np.ones(len(scores))]
            row[f"test_roc_tpr@{alpha}"] = (compute_tpr_at_fpr(np.r_[clean, scores], labels, float(alpha))
                                            if len(scores) else float("nan"))
        detection.append(row)
    for name in cfg["noise"]:
        scores = detector.score_samples(features("test", name))
        record_scores("test", name, scores)
        records = corpus.selected(("test", name))
        correct = np.array([r["eligible"] for r in records])
        changed = np.array([r["pred"] != r["clean_pred"] for r in records])
        row = {"method": method, "noise": name, "n": len(scores),
               "prediction_changed": float(changed.mean())}
        for threshold in thresholds:
            rate = rate_at_threshold(scores, threshold["threshold"])
            alpha = threshold["alpha"]
            row.update({f"fpr@{alpha}": rate["rate"], f"fpr@{alpha}_ci_low": rate["ci_low"],
                        f"fpr@{alpha}_ci_high": rate["ci_high"]})
            row[f"fpr@{alpha}_clean_correct"] = rate_at_threshold(scores[correct], threshold["threshold"])["rate"]
            row[f"fpr@{alpha}_clean_incorrect"] = (rate_at_threshold(scores[~correct], threshold["threshold"])["rate"]
                                                   if (~correct).any() else float("nan"))
        noise.append(row)
    write_csv(output / "stage1_detection.csv", detection)
    write_csv(output / "stage1_noise.csv", noise)
    write_csv(output / "scores.csv", score_rows)
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (output / "evaluation.json").write_text(json.dumps({"config": cfg, "corpus_files": corpus.fingerprint,
                                                        "method": method, "feature_dimension": dimension,
                                                        "attack_target": "ResNet-50",
                                                        "C": C, "dev_auc": dev_auc}, indent=2))
    joblib.dump(detector, output / "detector.joblib")


class RepresentationBank:
    def __init__(self, corpus, directory):
        self.corpus = corpus
        self.directory = Path(directory)
        self.variants = [dict(r, class_index=int(r['class_index'])) for r in corpus.records]
        if not (self.directory / 'provenance.json').is_file():
            raise FileNotFoundError(f'Missing ResNet feature cache: {self.directory}. Restore the '
                                    'original features/ directory or run experiments.resnet_representation '
                                    'with --encoder resnet and a new output directory, then pass its '
                                    'features/ path via --features. No SLICE/BELIEF rerun is needed.')
        self.provenance = json.loads((self.directory / 'provenance.json').read_text())
        if self.provenance['method'] != 'resnet_pool' or self.provenance['feature_dimension'] != 2048:
            raise ValueError('Stage 2 requires cached ResNet pooled features')
        if self.provenance['corpus_files'] != corpus.fingerprint:
            raise ValueError('Feature corpus metadata differs from the supplied corpus')
        self.cache = {}
        self.feature_hashes = {}
        # Validate all needed archives before creating any output.
        for key, rows in corpus.blocks.items():
            path = self.directory / f'{key[0]}_{key[1]}.npz'
            with np.load(path, allow_pickle=False) as archive:
                features = archive['features']
                ids = archive['source_ids']
                if features.shape != (len(rows), 2048) or not np.isfinite(features).all():
                    raise ValueError(f'Invalid feature matrix: {path}')
                if not np.array_equal(ids, [r['source_id'] for r in rows]):
                    raise ValueError(f'Feature source order mismatch: {path}')
                mask = np.array([r['explain'] for r in rows])
                self.cache[key] = (features[mask].astype(np.float64), ids[mask],
                                   [r for r in rows if r['explain']])
            self.feature_hashes[path.name] = sha(path)

    def features(self, partition, variant, method):
        if method != 'resnet_pool':
            raise ValueError('Unknown representation method')
        return self.cache[(partition, variant)]

    def restricted(self, partition, variant, method, sources):
        X, ids, _ = self.features(partition, variant, method)
        return X[np.isin(ids, list(sources))]


class ResNetStage2(SliceBeliefAnalysis):
    """Reuse the teammate's sampling, detector, calibration and evaluation methods."""
    def __init__(self, config, feature_dir, output, existing_output=False):
        self.config = config
        self.corpus = Corpus(config)
        self.bank = RepresentationBank(self.corpus, feature_dir)
        self.methods = ('resnet_pool',)
        self.alphas = config['evaluation']['fpr_targets']
        self.test_attacks = config['partition_attacks']['test']
        self.noise = list(config['noise'])
        self.tables = Path(output)
        protected = [self.corpus.root.resolve(), Path(config['paths']['results_dir']).resolve(),
                     Path(feature_dir).resolve()]
        if any(self.tables.resolve() == p or p in self.tables.resolve().parents for p in protected):
            raise ValueError('Use a new output directory outside the existing corpus and results')
        for holdout, families in config['stage2_folds'].items():
            if holdout in families or holdout == 'pgd':
                raise ValueError('Held-out family must not enter either training arm')
            if not self.matched_pool('train', families) or not self.matched_pool('dev', families):
                raise ValueError(f'Empty matched training/development pool: {holdout}')
        if config['seed'] not in config['experiment_seeds']:
            raise ValueError('Primary seed must be included for bootstrap comparison')
        if existing_output and (any(self.tables.glob('stage2_*')) or (self.tables / 'models').exists()):
            raise FileExistsError('Stage 2 outputs already exist; refusing to overwrite')
        self.tables.mkdir(parents=True, exist_ok=existing_output)
        (self.tables / 'models').mkdir()
        self.assignments, self.score_rows, self.noise_rows, self.threshold_rows = [], [], [], []

    def save_arm(self, context, model, thresholds, cal_scores, raw, noise_rows):
        tag = f"{context['holdout']}_{context['seed']}_{context['arm']}"
        joblib.dump(model, self.tables / 'models' / f'{tag}.joblib')
        self.threshold_rows.extend({**context, **t} for t in thresholds)
        self.noise_rows.extend({**context, **row} for row in noise_rows)
        for partition, variant, scores in [('cal', 'clean', cal_scores)] + [
                ('test', variant, scores) for variant, scores in raw.items()]:
            _, ids, _ = self.bank.features(partition, variant, 'resnet_pool')
            self.score_rows.extend({**context, 'partition': partition, 'variant': variant,
                                   'source_id': str(source), 'score': float(score)}
                                  for source, score in zip(ids, scores))

    def run(self):
        self.stage2()
        write_csv(self.tables / 'stage2_assignments.csv', self.assignments)
        write_csv(self.tables / 'stage2_scores.csv', self.score_rows)
        write_csv(self.tables / 'stage2_noise.csv', self.noise_rows)
        (self.tables / 'stage2_thresholds.json').write_text(json.dumps(self.threshold_rows, indent=2))
        (self.tables / 'stage2_provenance.json').write_text(json.dumps({
            'config': self.config, 'method': 'resnet_pool', 'attack_target': 'ResNet-50',
            'feature_source': str(self.bank.directory.resolve()), 'feature_hashes': self.bank.feature_hashes,
            'corpus_files': self.corpus.fingerprint,
            'original_feature_provenance': self.bank.provenance,
            'mask_policy': 'Original corpus eligibility and successful-attack masks, unchanged',
            'bootstrap_scope': 'Mixture minus matched PGD-only, primary seed only; calibration resampled',
            'reference': 'experiments/slice_belief_analysis.py::SliceBeliefAnalysis.stage2',
            'reference_code_sha256': sha(Path(__file__).with_name('slice_belief_analysis.py')),
        }, indent=2))
        print(f'Stage 2 completed: {self.tables}', flush=True)

    def stage2(self):
        """Output: None; stage2_*.csv (per seed, seed summary, bootstrap of the mixture gain)."""
        cap = self.config["evaluation"]["max_matched_train"]
        per_seed, boot_rows = [], []
        for holdout, families in self.config["stage2_folds"].items():
            pool = self.matched_pool("train", families)[:cap]
            dev_pool = set(self.matched_pool("dev", families))
            for seed in self.config["experiment_seeds"]:
                assignment = self.allocate(pool, families, seed, holdout)
                print(f'Holdout={holdout}, seed={seed}, matched train N={len(pool)}, dev N={len(dev_pool)}', flush=True)
                self.assignments.extend({'holdout': holdout, 'seed': seed, 'partition': 'train',
                                         'source_id': s, 'mixture_attack': assignment[s], 'baseline_attack': 'pgd'}
                                        for s in pool)
                self.assignments.extend({'holdout': holdout, 'seed': seed, 'partition': 'dev',
                                         'source_id': s, 'mixture_attack': '+'.join(families), 'baseline_attack': 'pgd'}
                                        for s in sorted(dev_pool))
                for method in self.methods:
                    neg = self.bank.restricted("train", CLEAN, method, pool)
                    mix = np.vstack([self.bank.restricted("train", f, method,
                                                          [s for s in pool if assignment[s] == f]) for f in families])
                    arms = {
                        "pgd_only": (self.bank.restricted("train", "pgd", method, pool), ["pgd"]),
                        "mixture": (mix, families),
                    }
                    dev_neg = self.bank.features("dev", CLEAN, method)[0]
                    raw_arms = {}
                    for arm, (pos, dev_families) in arms.items():
                        dev = [(dev_neg, self.bank.restricted("dev", f, method, dev_pool)) for f in dev_families]
                        model, C, dev_auc = self.detector(pos, neg, dev, seed)
                        thresholds, cal_scores = self.thresholds(model, method)
                        rows, noise_rows, raw = self.evaluate(model, method, thresholds, self.test_attacks)
                        raw_arms[arm] = (raw, cal_scores)
                        self.save_arm({'holdout': holdout, 'seed': seed, 'arm': arm, 'method': method},
                                      model, thresholds, cal_scores, raw, noise_rows)
                        noise_fpr = np.mean([n[f"fpr@{PRIMARY_ALPHA}"] for n in noise_rows])
                        for row in rows:
                            per_seed.append({"holdout": holdout, "mixture": "+".join(families), "seed": seed,
                                             "arm": arm, "method": method, "N": len(pool), "C": C,
                                             "dev_auc": dev_auc, "is_holdout": row["attack"] == holdout,
                                             "mean_noise_fpr@0.05": noise_fpr, **row})
                    if seed == self.config["seed"]:
                        _, clean_ids, _ = self.bank.features("test", CLEAN, method)
                        _, positive_ids, _ = self.bank.features("test", holdout, method)
                        methods = {arm: {"clean": raw["clean"], "positive": raw[holdout], "calibration": cal}
                                   for arm, (raw, cal) in raw_arms.items()}
                        result = paired_bootstrap(methods, "pgd_only", clean_ids, positive_ids,
                                                  self.config["evaluation"]["bootstrap_replicates"], PRIMARY_ALPHA,
                                                  derive_seed(seed, "stage2", holdout, method) % 2 ** 32)
                        gain = result["mixture"]
                        boot_rows.append({"holdout": holdout, "method": method, "N": len(pool),
                                          "auc_gain": gain["auc_diff"][0], "auc_gain_ci_low": gain["auc_diff"][1],
                                          "auc_gain_ci_high": gain["auc_diff"][2],
                                          "tpr5_gain": gain["tpr_diff"][0], "tpr5_gain_ci_low": gain["tpr_diff"][1],
                                          "tpr5_gain_ci_high": gain["tpr_diff"][2]})
        write_csv(self.tables / "stage2_per_seed.csv", per_seed)
        write_csv(self.tables / "stage2_bootstrap.csv", boot_rows)
        summary = []
        keys = sorted({(r["holdout"], r["arm"], r["method"], r["attack"]) for r in per_seed})
        for holdout, arm, method, attack in keys:
            group = [r for r in per_seed if (r["holdout"], r["arm"], r["method"], r["attack"]) == (holdout, arm, method, attack)]
            entry = {"holdout": holdout, "arm": arm, "method": method, "attack": attack,
                     "is_holdout": attack == holdout, "N": group[0]["N"], "seeds": len(group)}
            for metric in ("auc", f"tpr@{PRIMARY_ALPHA}", f"clean_fpr@{PRIMARY_ALPHA}", "mean_noise_fpr@0.05"):
                values = np.array([g[metric] for g in group], dtype=float)
                entry[f"{metric}_mean"] = float(np.nanmean(values))
                entry[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan")
            summary.append(entry)
        write_csv(self.tables / "stage2_summary.csv", summary)


def summarize_stages(output, config):
    """Produce one comparable summary CSV and short, data-derived Markdown reports."""
    output = Path(output)
    def read(name):
        path = output / name
        if not path.exists():
            return []
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    corpus = Corpus(config)
    combined = []
    metrics = ['auc', 'tpr@0.01', 'tpr@0.05', 'clean_fpr@0.01', 'clean_fpr@0.05',
               'undetected@0.01', 'undetected@0.05', 'test_roc_tpr@0.01', 'test_roc_tpr@0.05']
    noise1 = read('stage1_noise.csv')
    for row in read('stage1_detection.csv'):
        entry = dict(stage='stage1', arm='pgd_only', holdout='', training_attacks='pgd',
                     method=row['method'], attack=row['attack'], is_holdout='', seeds=1,
                     num_train_clean=len(corpus.selected(('train', 'clean'))),
                     num_train_adversarial=len(corpus.selected(('train', 'pgd'))))
        for k in ('num_clean', 'num_successful', 'num_eligible'):
            entry[k] = int(row[k])
        entry['asr'] = entry['num_successful'] / entry['num_eligible'] if entry['num_eligible'] else float('nan')
        for metric in metrics:
            entry[metric + '_mean'] = float(row[metric])
            entry[metric + '_sd'] = ''
        for alpha in ('0.01', '0.05'):
            entry[f'mean_noise_fpr@{alpha}_mean'] = float(np.mean([float(n[f'fpr@{alpha}']) for n in noise1]))
            entry[f'mean_noise_fpr@{alpha}_sd'] = ''
        combined.append(entry)
    rows = read('stage2_per_seed.csv')
    noise2 = read('stage2_noise.csv')
    keys = sorted({(r['holdout'], r['arm'], r['method'], r['attack']) for r in rows})
    for holdout, arm, method, attack in keys:
        group = [r for r in rows if (r['holdout'], r['arm'], r['method'], r['attack']) == (holdout, arm, method, attack)]
        entry = dict(stage='stage2', arm=arm, holdout=holdout,
                     training_attacks=group[0]['mixture'] if arm == 'mixture' else 'pgd',
                     method=method, attack=attack, is_holdout=attack == holdout, seeds=len(group),
                     num_train_clean=int(group[0]['N']), num_train_adversarial=int(group[0]['N']))
        for k in ('num_clean', 'num_successful', 'num_eligible'):
            assert len({r[k] for r in group}) == 1, f'Inconsistent {k}'
            entry[k] = int(group[0][k])
        entry['asr'] = entry['num_successful'] / entry['num_eligible'] if entry['num_eligible'] else float('nan')
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            entry[metric + '_mean'] = float(np.mean(values))
            entry[metric + '_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else ''
        for alpha in ('0.01', '0.05'):
            values = [np.mean([float(n[f'fpr@{alpha}']) for n in noise2
                               if (n['holdout'], n['arm'], n['method'], n['seed']) == (holdout, arm, method, r['seed'])])
                      for r in group]
            entry[f'mean_noise_fpr@{alpha}_mean'] = float(np.mean(values))
            entry[f'mean_noise_fpr@{alpha}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else ''
        combined.append(entry)
    write_csv(output / 'combined_results.csv', combined)
    counts = {p: len(corpus.selected((p, 'clean'))) for p in ('train', 'dev', 'cal', 'test')}
    method_name = combined[0]['method'] if combined else 'resnet_pool'
    feature_description = ('Frozen DINOv2 ViT-S/14 supplies 384-dimensional embeddings while observing ResNet attacks.'
                           if method_name == 'dinov2_observer' else
                           f"Frozen ResNet-50 (`{config['model']['weights']}`) supplies 2,048-dimensional pooled features.")
    setup = f'''# Experiment setup

{feature_description}
Both stages reuse the explanation experiment's exact saved images and original
ResNet eligibility/success masks; local prediction differences are recorded in
the feature provenance. No attacks or SLICE/BELIEF explanations are regenerated.

- Source partitions: {counts['train']} train, {counts['dev']} development,
  {counts['cal']} clean calibration, {counts['test']} test; all variants stay with their source.
- Detector: training-only StandardScaler + logistic regression; C selected on development AUC.
- Stage 1: all clean training samples and successful PGD examples.
- Stage 2: fresh matched PGD-only and mixture detectors, each with N clean + N attacked
  samples from identical sources. The held-out family never enters fitting or tuning.
- Stage 2 seeds: {config.get('experiment_seeds', [])}. Thresholds are independently calibrated
  on clean calibration data at 1% and 5%, then frozen for all test conditions.
- FGSM is used only as an additional unseen-attack test in both stages. It is never
  included in detector training or hyperparameter selection, and is not a separate
  Stage 2 holdout setting. Thus, `attack=fgsm` with `is_holdout=False` does not
  mean FGSM was seen during training.
- Test detection includes only originally successful attacks. Clean FPR includes all clean
  test images; noise FPR is measured on the shared noise panel.

## Reading combined_results.csv

Each row is a stage/training strategy/held-out setting/test attack combination.
Stage 1 has one run; its `_mean` columns are the single-run results and `_sd` is blank.
Stage 2 averages over seeds; `_sd` is sample standard deviation, not a confidence interval.
Filter `stage=stage2` and `is_holdout=True` for the main generalization comparisons.
Stage 1 has no Stage 2 holdout designation. `num_train_clean` and
`num_train_adversarial` explicitly distinguish the full and matched training pools.

`tpr@0.05_mean` uses the clean-calibrated threshold; `test_roc_tpr@0.05_mean`
is a different, descriptive test-ROC statistic. Always compare like columns and
report actual test FPR. `undetected` divides missed successful attacks by eligible
attempts, whereas TPR divides detected attacks by successful attacks.
Noise means average the noise conditions within each run, then average runs.

The original detailed CSVs, confidence intervals, scores, thresholds and fitted
models are retained in this folder. Stage 2 paired-bootstrap gains are in
`stage2_bootstrap.csv` and apply to primary seed 42, not the three-seed mean.
'''
    (output / 'SETUP.md').write_text(setup)
    lines = ['# Main results', '', 'Results from the shared fixed-1,000-image corpus. All TPRs below use the',
             '5% clean-calibration target; actual test false-alarm rates can differ.', '',
             '## Stage 1: PGD-only training', '',
             '| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in combined:
        if r['stage'] == 'stage1':
            lines.append(f"| {r['attack']} | {r['auc_mean']:.4f} | {r['tpr@0.05_mean']:.2%} | {r['clean_fpr@0.05_mean']:.2%} | {r['mean_noise_fpr@0.05_mean']:.2%} |")
    lines += ['', '## Stage 2: held-out attack detection', '', 'Means over three seeds; compare arms within the same held-out attack.', '',
              '| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    held = [r for r in combined if r['stage'] == 'stage2' and r['is_holdout']]
    for r in held:
        lines.append(f"| {r['holdout']} | {r['arm']} | {r['num_train_clean']} + {r['num_train_adversarial']} | {r['auc_mean']:.4f} | {r['tpr@0.05_mean']:.2%} | {r['clean_fpr@0.05_mean']:.2%} | {r['mean_noise_fpr@0.05_mean']:.2%} |")
    lines += ['', '## Interpretation', '']
    for h in sorted({r['holdout'] for r in held}):
        arms = {r['arm']: r for r in held if r['holdout'] == h}
        mix, pgd = arms['mixture'], arms['pgd_only']
        lines.append(f"- {h}: mixture minus matched PGD-only changes AUC by {mix['auc_mean']-pgd['auc_mean']:+.4f}, "
                     f"detection by {100*(mix['tpr@0.05_mean']-pgd['tpr@0.05_mean']):+.2f} percentage points, "
                     f"and clean false alarms by {100*(mix['clean_fpr@0.05_mean']-pgd['clean_fpr@0.05_mean']):+.2f} percentage points.")
    lines += ['', 'Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.',
              'These means alone do not establish statistical significance. Review the primary-seed',
              'paired-bootstrap intervals separately. The recorded cross-machine prediction differences',
              'remain an audit limitation; original cohort membership is preserved.', '',
              'See [setup](SETUP.md) and [combined results](combined_results.csv).']
    (output / 'MAIN_RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/slice_belief_imagenette_fixed1000.yaml")
    parser.add_argument("--output", help="New output directory; defaults to a separate directory per encoder")
    parser.add_argument("--encoder", choices=("resnet", "dinov2"), default="resnet",
                        help="Feature encoder only; attacks and success masks remain ResNet-based")
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="stage1",
                        help="Run Stage 1, Stage 2, or both after a single feature extraction (Stage 2: ResNet only)")
    parser.add_argument("--features", help="Reuse an existing features/ directory; skips backbone inference")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--prediction-policy", choices=("strict", "record"), default="strict",
                        help="strict aborts on prediction differences; record audits them and keeps original masks")
    parser.add_argument("--check", action="store_true", help="Check required corpus artifacts without loading a model")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.encoder != "resnet" and args.stage != "stage1":
        parser.error("Stage 2 currently supports --encoder resnet; DINOv2 remains Stage 1")
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus = Corpus(cfg)
    if args.check:
        print(f"Corpus metadata and required files present: {corpus.root}; {len(corpus.blocks)} blocks")
        return
    output = Path(args.output or ("results/resnet_representation_fixed1000" if args.encoder == "resnet"
                                 else "results/dinov2_observer_fixed1000"))
    # New directories only: never overwrite an existing run or explanation work.
    protected = [corpus.root.resolve(), Path(cfg["paths"]["results_dir"]).resolve()]
    if any(output.resolve() == p or p in output.resolve().parents for p in protected):
        raise ValueError("Output must be separate from explanation corpus and results")
    if args.features:
        cached_provenance = json.loads((Path(args.features) / "provenance.json").read_text())
        expected_method = "resnet_pool" if args.encoder == "resnet" else "dinov2_observer"
        if cached_provenance["method"] != expected_method or cached_provenance["corpus_files"] != corpus.fingerprint:
            raise ValueError("Cached feature encoder or corpus does not match this run")
    output.mkdir(parents=True, exist_ok=False)
    feature_dir = Path(args.features) if args.features else output / "features"
    if not args.features:
        feature_dir.mkdir()
        observer = None
        if args.encoder == "dinov2":
            from encoders.dinov2_observer import DinoV2Observer
            observer = DinoV2Observer(device=args.device)
        extract(corpus, feature_dir, torch.device(args.device), args.batch_size, observer=observer,
                prediction_policy=args.prediction_policy)
    if args.stage in ("stage1", "both"):
        evaluate(corpus, feature_dir, output)
    if args.stage in ("stage2", "both"):
        ResNetStage2(cfg, feature_dir, output, existing_output=True).run()
    (output / "run.json").write_text(json.dumps({"stage": args.stage, "encoder": args.encoder,
        "features": str(feature_dir.resolve()), "reused_features": bool(args.features),
        "corpus_files": corpus.fingerprint}, indent=2))
    summarize_stages(output, cfg)
    print(f"Saved {args.encoder} representation baseline to {output}")


if __name__ == "__main__":
    main()
