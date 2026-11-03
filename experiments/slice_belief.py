"""Controlled Grad-CAM vs SLICE vs BELIEF study: corpus and explanation stages.

Stages (run in order; each is resumable / idempotent):
  manifest  class-stratified source manifest of Imagenette val
  corpus    clean tensors, ResNet-50 attacks and noise controls, predictions
  segment   Quickshift superpixels of every input to be explained
  sigma     global blur-sigma policy from reference select_sigma on dev images
  explain   Grad-CAM (all rows) + SLICE + BELIEF (rows entering detection)
  repeats   repeated SLICE/BELIEF runs on the stability panel
  fidelity  AOPC-deletion of every explainer (+ random ranking) on test rows
Detection, calibration and reporting live in experiments/slice_belief_analysis.py.

Usage: python experiments/slice_belief.py --config configs/experiments/slice_belief_imagenette.yaml --stage all
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from attacks.factory import build_attack
from encoders.cnn import ResNetEncoder
from explainers.belief import BeliefExplainer
from explainers.gradcam import GradCAMExplainer
from explainers.perturbation import (
    PerturbationQueryEngine,
    ProbabilityAdapter,
    SuperpixelSegmenter,
    TorchSampler,
    select_sigma,
)
from explainers.slice import SliceExplainer
from explainers.superpixel_maps import superpixel_means
from explainers.vectorizer import GradCAMVectorizer
from utils.controlled_data import (
    PARTITION_ORDER,
    build_manifest,
    derive_seed,
    load_images,
    partition_rows,
    read_manifest,
    write_manifest,
)
from utils.gradcam_data import IMAGENET_LABELS

CLEAN = "clean"
JSON_INDENT = 2
PREDICT_BATCH = 100
IMAGENETTE_CLASSES = sorted(IMAGENET_LABELS.values())
VARIANT_FIELDS = (
    "partition", "variant", "tensor_index", "source_id", "class_index", "imagenet_label",
    "clean_pred", "pred", "eligible", "success", "explain", "linf", "l2",
    "max_prob", "entropy", "in_imagenette", "noise_panel", "stability_panel",
)
STAGES = ("manifest", "corpus", "segment", "sigma", "explain", "repeats", "fidelity")


def resolve_device(requested):
    """Input: 'auto' or device str. Output: torch.device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def to_float(images):
    """Input: uint8 or float Tensor[N,3,H,W]. Output: float32 Tensor in [0,1]."""
    return images.float() / 255.0 if images.dtype == torch.uint8 else images.float()


def tensor_sha(tensor):
    """Input: Tensor. Output: str SHA-256 prefix of its bytes."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


def _segment_worker(arguments):
    """Input: (ndarray[3,H,W] float, dict segmenter params). Output: (ndarray[H,W] int16, float seconds)."""
    image, params = arguments
    start = time.perf_counter()
    labels = SuperpixelSegmenter(**params).segment(image)
    return labels.astype(np.int16), time.perf_counter() - start


class SliceBeliefStudy:
    """Input: resolved YAML config dict. Output: stage runner writing artifacts."""

    def __init__(self, config):
        """Input: dict config. Output: None."""
        self.config = config
        self.seed = config["seed"]
        self.device = resolve_device(config["device"])
        self.work = Path(config["paths"]["work_dir"])
        self.results = Path(config["paths"]["results_dir"])
        for sub in ("images", "segments", "explanations", "repeats"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        (self.results / "logs").mkdir(parents=True, exist_ok=True)
        self._model = None
        self._sigma = None
        torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------- utils

    def log(self, message):
        """Input: str. Output: None; prints and appends to the study log."""
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with (self.results / "logs" / "slice_belief.log").open("a") as stream:
            stream.write(line + "\n")

    @property
    def model(self):
        """Output: frozen eval-mode ResNetEncoder (raw-pixel input) on device."""
        if self._model is None:
            model = ResNetEncoder(weights=self.config["model"]["weights"]).to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model
        return self._model

    def manifest(self):
        """Output: list[dict] manifest rows (requires the manifest stage)."""
        return read_manifest(self.work / "manifest.csv")

    def variants(self):
        """Output: list[dict] variant records with typed fields."""
        with (self.work / "variants.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ("tensor_index", "class_index", "imagenet_label", "clean_pred", "pred"):
                row[key] = int(row[key])
            for key in ("linf", "l2", "max_prob", "entropy"):
                row[key] = float(row[key])
            for key in ("eligible", "success", "explain", "in_imagenette", "noise_panel", "stability_panel"):
                row[key] = row[key] == "True"
        return rows

    def shard_names(self):
        """Output: list[(partition, variant)] in explanation priority order."""
        cfg = self.config
        stage1 = [("train", CLEAN), ("train", "pgd"), ("dev", CLEAN), ("dev", "pgd"), ("cal", CLEAN),
                  ("test", CLEAN)]
        stage1 += [("test", a) for a in cfg["partition_attacks"]["test"]]
        stage1 += [("test", n) for n in cfg["noise"]]
        rest = [(p, a) for p in ("dev", "train") for a in cfg["partition_attacks"][p] if a != "pgd"]
        return stage1 + rest

    def tensor_path(self, partition, variant):
        """Input: str, str. Output: Path of the stored image tensor."""
        return self.work / "images" / f"{partition}_{variant}.pt"

    def load_tensor(self, partition, variant):
        """Input: str, str. Output: Tensor[N,3,H,W] (uint8 clean, float32 otherwise)."""
        return torch.load(self.tensor_path(partition, variant), weights_only=True)

    @torch.no_grad()
    def predict(self, images):
        """Input: Tensor[N,3,H,W] uint8/float. Output: (pred, max_prob, entropy) ndarrays (fp32 path)."""
        preds, maxima, entropies = [], [], []
        for start in range(0, len(images), PREDICT_BATCH):
            batch = to_float(images[start:start + PREDICT_BATCH]).to(self.device)
            probs = self.model(batch).softmax(1)
            preds.append(probs.argmax(1).cpu())
            maxima.append(probs.max(1).values.cpu())
            entropies.append(-(probs * probs.clamp_min(1e-30).log()).sum(1).cpu())
        return tuple(torch.cat(v).numpy() for v in (preds, maxima, entropies))

    # ------------------------------------------------------------- manifest

    def stage_manifest(self):
        """Output: None; writes manifest.csv (refuses to overwrite a different one)."""
        cfg = self.config
        rows = build_manifest(
            cfg["paths"]["root"], cfg["partitions_per_class"], cfg["panels_per_class"]["noise"],
            cfg["panels_per_class"]["stability"], self.seed, cfg["dataset"]["size"],
            cfg["dataset"].get("subset_per_class"),
        )
        self.check_budget(rows)
        path = self.work / "manifest.csv"
        if path.exists():
            existing = {r["source_id"]: r["partition"] for r in read_manifest(path)}
            if existing != {r["source_id"]: r["partition"] for r in rows}:
                raise ValueError("Existing manifest differs from the configured split; refusing to overwrite")
        write_manifest(rows, path)
        write_manifest([r for r in rows if r["partition"] != "reserve"], self.results / "source_files.csv")
        counts = Counter(r["partition"] for r in rows)
        ids = [r["source_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source IDs in manifest")
        self.log(f"manifest: {dict(counts)}")

    def estimate_hours(self, rows):
        """Projected wall time of all stages from measured per-input rates.

        Input: list[dict] manifest rows. Output: (float hours, int explained inputs).
        Uses config budget.explained_per_source (explained variants per source,
        noise excluded), budget.seconds_per_explained_input and
        budget.corpus_seconds_per_source, all measured on the first full run.
        """
        cfg, budget = self.config, self.config["budget"]
        counts = Counter(r["partition"] for r in rows)
        explained = sum(counts[p] * budget["explained_per_source"][p] for p in PARTITION_ORDER)
        explained += sum(r["noise_panel"] for r in rows) * len(cfg["noise"])
        repeated = sum(r["stability_panel"] for r in rows) * cfg["quality"]["repeats"] * 2
        sources = sum(counts[p] for p in PARTITION_ORDER)
        seconds = (explained + repeated) * budget["seconds_per_explained_input"]
        seconds += sources * budget["corpus_seconds_per_source"]
        return seconds / 3600.0, int(round(explained))

    def check_budget(self, rows):
        """Input: manifest rows. Output: None; raises if the projected run exceeds budget.max_hours."""
        budget = self.config.get("budget")
        if budget is None:
            return
        hours, explained = self.estimate_hours(rows)
        self.log(f"budget: ~{explained} explained inputs, projected {hours:.1f} h (limit {budget['max_hours']} h)")
        if hours > budget["max_hours"] and not budget.get("allow_over_budget", False):
            raise RuntimeError(
                f"Projected {hours:.1f} h exceeds budget.max_hours={budget['max_hours']}; use the fixed "
                "subset config (dataset.subset_per_class) or set budget.allow_over_budget: true"
            )

    # --------------------------------------------------------------- corpus

    def stage_corpus(self):
        """Output: None; writes image tensors, variants.csv and corpus.json."""
        cfg = self.config
        manifest = self.manifest()
        records, provenance = [], {"tensors": {}}
        for partition in PARTITION_ORDER:
            rows = partition_rows(manifest, partition)
            labels = torch.tensor([r["imagenet_label"] for r in rows])
            clean_path = self.tensor_path(partition, CLEAN)
            if clean_path.exists():
                clean = self.load_tensor(partition, CLEAN)
            else:
                clean = load_images(rows, cfg["dataset"]["resize_size"], cfg["dataset"]["image_size"])
                torch.save(clean, clean_path)
            provenance["tensors"][f"{partition}_{CLEAN}"] = tensor_sha(clean)
            clean_pred, clean_max, clean_entropy = self.predict(clean)
            eligible = clean_pred == labels.numpy()
            self.log(f"{partition}: clean accuracy {eligible.mean():.4f} ({eligible.sum()}/{len(rows)})")
            records += self._records(partition, CLEAN, rows, np.arange(len(rows)), clean_pred, clean_pred,
                                     clean_max, clean_entropy, eligible, np.zeros(len(rows), bool),
                                     np.ones(len(rows), bool), torch.zeros(len(rows)), torch.zeros(len(rows)))
            for attack in cfg["partition_attacks"].get(partition, []):
                adversarial = self._attack(partition, attack, rows, clean, labels)
                pred, maxima, entropy = self.predict(adversarial)
                success = eligible & (pred != labels.numpy())
                delta = (adversarial - to_float(clean)).flatten(1)
                provenance["tensors"][f"{partition}_{attack}"] = tensor_sha(adversarial)
                records += self._records(partition, attack, rows, np.arange(len(rows)), clean_pred, pred, maxima,
                                         entropy, eligible, success, success, delta.abs().amax(1), delta.norm(dim=1))
                self.log(f"{partition}/{attack}: ASR {success.sum()}/{eligible.sum()} = "
                         f"{success.sum() / max(eligible.sum(), 1):.4f}")
            if partition == "test":
                panel = np.array([i for i, r in enumerate(rows) if r["noise_panel"]])
                for name, spec in cfg["noise"].items():
                    noisy = self._noise(name, spec, [rows[i] for i in panel], clean[panel])
                    pred, maxima, entropy = self.predict(noisy)
                    delta = (noisy - to_float(clean[panel])).flatten(1)
                    provenance["tensors"][f"test_{name}"] = tensor_sha(noisy)
                    records += self._records(partition, name, [rows[i] for i in panel], np.arange(len(panel)),
                                             clean_pred[panel], pred, maxima, entropy, eligible[panel],
                                             np.zeros(len(panel), bool), np.ones(len(panel), bool),
                                             delta.abs().amax(1), delta.norm(dim=1))
                    self.log(f"test/{name}: prediction changed on {(pred != clean_pred[panel]).mean():.3f}")
        with (self.work / "variants.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=VARIANT_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        provenance.update(
            config=cfg,
            versions={p: importlib.metadata.version(p) for p in
                      ("torch", "torchvision", "torchattacks", "scikit-learn", "scikit-image", "numpy", "scipy")},
            weights=cfg["model"]["weights"],
        )
        (self.work / "corpus.json").write_text(json.dumps(provenance, indent=JSON_INDENT))

    def _records(self, partition, variant, rows, indices, clean_pred, pred, max_prob, entropy, eligible,
                 success, explain, linf, l2):
        """Input: aligned per-row arrays for one (partition, variant). Output: list[dict] VARIANT_FIELDS rows."""
        return [
            {
                "partition": partition, "variant": variant, "tensor_index": int(i),
                "source_id": row["source_id"], "class_index": row["class_index"],
                "imagenet_label": row["imagenet_label"], "clean_pred": int(clean_pred[k]),
                "pred": int(pred[k]), "eligible": bool(eligible[k]), "success": bool(success[k]),
                "explain": bool(explain[k]), "linf": float(linf[k]), "l2": float(l2[k]),
                "max_prob": float(max_prob[k]), "entropy": float(entropy[k]),
                "in_imagenette": int(pred[k]) in IMAGENETTE_CLASSES,
                "noise_panel": row["noise_panel"], "stability_panel": row["stability_panel"],
            }
            for k, (i, row) in enumerate(zip(indices, rows))
        ]

    def _attack(self, partition, attack, rows, clean, labels):
        """Input: partition/attack names, rows, uint8 clean, ImageNet labels.

        Output: Tensor[N,3,H,W] float32 adversarial images (cached on disk).
        """
        path = self.tensor_path(partition, attack)
        if path.exists():
            return self.load_tensor(partition, attack)
        cfg = self.config
        params = cfg["attacks"][attack]
        fn = build_attack(attack, self.model, params)
        batch = cfg["attack_batch_size"]
        outputs = []
        started = time.perf_counter()
        for start in range(0, len(rows), batch):
            torch.manual_seed(derive_seed(self.seed, partition, attack, start))
            images = to_float(clean[start:start + batch]).to(self.device)
            target = labels[start:start + batch].to(self.device)
            if attack == "patch":
                size = params["patch_size"]
                high = cfg["dataset"]["image_size"] - size + 1
                locations = np.array([
                    np.random.default_rng(derive_seed(self.seed, r["source_id"], "patch_location")).integers(0, high, 2)
                    for r in rows[start:start + batch]
                ])
                adversarial = fn(images, target, top_positions=locations[:, 0], left_positions=locations[:, 1])
            else:
                adversarial = fn(images, target)
            outputs.append(adversarial.detach().float().cpu())
        adversarial = torch.cat(outputs)
        torch.save(adversarial, path)
        self.log(f"{partition}/{attack}: generated in {time.perf_counter() - started:.0f}s")
        return adversarial

    def _noise(self, name, spec, rows, clean):
        """Input: noise name/spec, panel rows, uint8 clean. Output: Tensor float32 noisy images (cached)."""
        path = self.tensor_path("test", name)
        if path.exists():
            return self.load_tensor("test", name)
        images = to_float(clean)
        noisy = []
        for row, image in zip(rows, images):
            generator = torch.Generator().manual_seed(derive_seed(self.seed, row["source_id"], name))
            if spec["kind"] == "gaussian":
                noise = torch.randn(image.shape, generator=generator) * spec["scale"]
            else:
                noise = (torch.rand(image.shape, generator=generator) * 2 - 1) * spec["scale"]
            noisy.append((image + noise).clamp(0, 1))
        noisy = torch.stack(noisy)
        torch.save(noisy, path)
        return noisy

    # ---------------------------------------------------------- segmentation

    def segment_path(self, partition, variant):
        """Input: str, str. Output: Path of the shard's segmentation archive."""
        return self.work / "segments" / f"{partition}_{variant}.npz"

    def load_segments(self, partition, variant):
        """Input: str, str. Output: dict tensor_index -> ndarray[H,W] labels."""
        with np.load(self.segment_path(partition, variant)) as data:
            return {int(i): s for i, s in zip(data["index"], data["segments"])}

    def stage_segment(self):
        """Output: None; Quickshift for every explained row of every shard."""
        params = self.config["explainers"]["segmentation"]
        variants = self.variants()
        workers = self.config["explainers"]["segmentation_workers"]
        with Pool(workers) as pool:
            for partition, variant in self.shard_names():
                path = self.segment_path(partition, variant)
                if path.exists():
                    continue
                index = np.array([r["tensor_index"] for r in variants
                                  if r["partition"] == partition and r["variant"] == variant and r["explain"]])
                images = to_float(self.load_tensor(partition, variant)[index]).numpy()
                results = pool.map(_segment_worker, [(image, params) for image in images], chunksize=4)
                segments = np.stack([r[0] for r in results])
                seconds = np.array([r[1] for r in results])
                np.savez_compressed(path, index=index, segments=segments, seconds=seconds)
                counts = segments.reshape(len(segments), -1).max(1) + 1
                self.log(f"segment {partition}/{variant}: {len(index)} inputs, K median {np.median(counts):.0f} "
                         f"[{counts.min()}-{counts.max()}], {seconds.mean():.2f}s/input")

    # ----------------------------------------------------------------- sigma

    def query_engine(self):
        """Output: PerturbationQueryEngine on a channels-last copy of the classifier."""
        cfg = self.config["explainers"]
        query_model = copy.deepcopy(self.model).to(memory_format=torch.channels_last)
        adapter = ProbabilityAdapter(query_model, cfg["query_batch_size"], cfg["use_amp"],
                                     channels_last=True)
        return PerturbationQueryEngine(adapter, self.device)

    def stage_sigma(self):
        """Output: None; sigma.json with per-image reference selections and the global mode."""
        path = self.work / "sigma.json"
        if path.exists():
            self.log(f"sigma policy: {json.loads(path.read_text())['sigma']}")
            return
        cfg = self.config
        engine = self.query_engine()
        rows = [r for r in self.variants() if r["partition"] == "dev" and r["variant"] == CLEAN]
        rows = rows[: cfg["sigma_selection_images"]]
        images = self.load_tensor("dev", CLEAN)
        segments = self.load_segments("dev", CLEAN)
        choices = []
        for row in rows:
            context = engine.context(to_float(images[row["tensor_index"]]), segments[row["tensor_index"]],
                                     row["pred"], 0.0)
            sampler = TorchSampler(derive_seed(self.seed, row["source_id"], "sigma_selection"))
            sigma, scores = select_sigma(engine, context, sampler, cfg["explainers"]["sigma_values"])
            choices.append({"source_id": row["source_id"], "sigma": sigma, "scores": scores})
        counts = Counter(c["sigma"] for c in choices)
        best = max(counts.values())
        sigma = min(s for s, c in counts.items() if c == best)
        policy = {
            "sigma": sigma,
            "rule": "mode of reference select_sigma over dev clean images; ties -> smaller sigma",
            "counts": {str(k): v for k, v in sorted(counts.items())},
            "per_image": choices,
        }
        path.write_text(json.dumps(policy, indent=JSON_INDENT, allow_nan=True))
        self.log(f"sigma policy: {sigma} from {dict(counts)}")

    def sigma(self):
        """Output: float frozen global sigma (read once from sigma.json)."""
        if self._sigma is None:
            self._sigma = float(json.loads((self.work / "sigma.json").read_text())["sigma"])
        return self._sigma

    # --------------------------------------------------------------- explain

    def explainers(self, engine):
        """Input: engine. Output: (SliceExplainer, BeliefExplainer) with configured settings."""
        cfg = self.config["explainers"]
        return SliceExplainer(engine, **cfg["slice"]), BeliefExplainer(engine, **cfg["belief"])

    def explain_one(self, engine, slice_explainer, belief_explainer, image, segments, row, variant, tag):
        """Explain one input with SLICE and BELIEF sharing the first LIME sample.

        Input: engine, explainers, Tensor[3,H,W] float, ndarray[H,W], variant record,
            str variant, str seed tag ("main" or "repeat<r>").
        Output: dict method -> record dict (coefficients, status, cost...).
        """
        context = engine.context(image, segments, row["pred"], self.sigma())
        start = slice_explainer.timer()
        shared_sampler = TorchSampler(derive_seed(self.seed, row["source_id"], variant, tag, "lime_sample"))
        shared = slice_explainer.initial_sample(context, shared_sampler)
        shared_seconds = slice_explainer.timer() - start
        shared_queries = context.num_queries
        slice_sampler = TorchSampler(derive_seed(self.seed, row["source_id"], variant, tag, "slice"))
        results = {
            "slice": slice_explainer.explain(context, slice_sampler, initial_sample=shared),
            "belief": belief_explainer.explain(context, None, initial_sample=shared),
        }
        records = {}
        for method, result in results.items():
            records[method] = {
                "coefficients": result.coefficients.astype(np.float64),
                "eliminated": result.eliminated,
                "status": result.status,
                "target_class": result.target_class,
                "prob0": result.prob0,
                "sigma": result.sigma,
                "num_steps": result.num_steps,
                "standalone_queries": shared_queries + result.num_queries,
                "standalone_seconds": shared_seconds + result.runtime_seconds,
                "extra": result.extra,
            }
        return records

    @torch.no_grad()
    def gradcam_features(self, images, batch=50):
        """Input: Tensor[N,3,H,W]. Output: (ndarray[N,196] features, ndarray[N] predictions, float s/image)."""
        explainer = GradCAMExplainer(self.model, self.model.target_layer(self.config["model"]["target_layer"]))
        vectorizer = GradCAMVectorizer(self.config["explainers"]["grid_size"])
        features, predictions = [], []
        start = time.perf_counter()
        for begin in range(0, len(images), batch):
            maps, pred = explainer.explain(to_float(images[begin:begin + batch]).to(self.device))
            features.append(vectorizer.transform(maps))
            predictions.append(pred.cpu().numpy())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / max(len(images), 1)
        return np.concatenate(features), np.concatenate(predictions), seconds

    def explanation_path(self, partition, variant):
        """Input: str, str. Output: Path of the shard's explanation archive."""
        return self.work / "explanations" / f"{partition}_{variant}.pt"

    def stage_explain(self):
        """Output: None; per-shard archives with Grad-CAM features and SLICE/BELIEF records."""
        engine = self.query_engine()
        slice_explainer, belief_explainer = self.explainers(engine)
        variants = self.variants()
        total = sum(r["explain"] for r in variants)
        done_before = 0
        started = time.perf_counter()
        processed = 0
        for partition, variant in self.shard_names():
            path = self.explanation_path(partition, variant)
            shard = torch.load(path, weights_only=False) if path.exists() else {}
            rows = [r for r in variants if r["partition"] == partition and r["variant"] == variant]
            images = self.load_tensor(partition, variant)
            if "gradcam" not in shard:
                features, predictions, seconds = self.gradcam_features(images)
                expected = np.array([r["pred"] for r in rows])
                shard["gradcam"] = {"features": features, "predictions": predictions,
                                    "seconds_per_image": seconds,
                                    "prediction_mismatches": int((predictions != expected).sum())}
                torch.save(shard, path)
            segments = self.load_segments(partition, variant)
            shard.setdefault("records", {})
            todo = [r for r in rows if r["explain"] and r["tensor_index"] not in shard["records"]]
            done_before += sum(r["explain"] for r in rows) - len(todo)
            for k, row in enumerate(todo):
                index = row["tensor_index"]
                shard["records"][index] = self.explain_one(
                    engine, slice_explainer, belief_explainer, to_float(images[index]), segments[index],
                    row, variant, "main",
                )
                processed += 1
                if (k + 1) % self.config["explainers"]["checkpoint_every"] == 0 or k + 1 == len(todo):
                    torch.save(shard, path)
                    rate = (time.perf_counter() - started) / processed
                    remaining = total - done_before - processed
                    self.log(f"explain {partition}/{variant}: {k + 1}/{len(todo)} | {rate:.1f}s/input | "
                             f"ETA {remaining * rate / 3600:.2f}h")
            torch.save(shard, path)
        if torch.cuda.is_available():
            self.log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")

    # --------------------------------------------------------------- repeats

    def stage_repeats(self):
        """Output: None; repeats/{variant}.pt with R repeated SLICE/BELIEF runs per panel input."""
        engine = self.query_engine()
        slice_explainer, belief_explainer = self.explainers(engine)
        repeats = self.config["quality"]["repeats"]
        variants = self.variants()
        for variant in (CLEAN, "pgd"):
            path = self.work / "repeats" / f"test_{variant}.pt"
            archive = torch.load(path, weights_only=False) if path.exists() else {}
            rows = [r for r in variants if r["partition"] == "test" and r["variant"] == variant
                    and r["stability_panel"] and r["explain"]]
            images = self.load_tensor("test", variant)
            segments = self.load_segments("test", variant)
            for row in rows:
                index = row["tensor_index"]
                runs = archive.setdefault(index, [])
                for r in range(len(runs), repeats):
                    runs.append(self.explain_one(engine, slice_explainer, belief_explainer,
                                                 to_float(images[index]), segments[index], row, variant,
                                                 f"repeat{r}"))
                torch.save(archive, path)
            self.log(f"repeats test/{variant}: {len(rows)} inputs x {repeats}")

    # -------------------------------------------------------------- fidelity

    @torch.no_grad()
    def stage_fidelity(self):
        """Output: None; fidelity.csv with AOPC-deletion per (row, explainer).

        Mirrors the reference compute_aopc_del (positive ranks, num_ranks=5):
        for k = 1..5 the top-k positively ranked superpixels are replaced
        by the explainers' own perturbation fill (global sigma) and the drop
        of the explained-class probability is averaged. Grad-CAM ranks
        superpixels by mean heatmap; a seeded random ranking is the control.
        """
        engine = self.query_engine()
        ranks_k = self.config["quality"]["aopc_ranks"]
        explainer_cam = GradCAMExplainer(self.model, self.model.target_layer(self.config["model"]["target_layer"]))
        output = []
        variants = self.variants()
        for variant in (CLEAN, "pgd"):
            rows = [r for r in variants if r["partition"] == "test" and r["variant"] == variant and r["explain"]]
            images = self.load_tensor("test", variant)
            segments = self.load_segments("test", variant)
            records = torch.load(self.explanation_path("test", variant), weights_only=False)["records"]
            for row in rows:
                index = row["tensor_index"]
                image = to_float(images[index])
                labels = segments[index]
                context = engine.context(image, labels, row["pred"], self.sigma())
                K = context.num_segments
                cam, _ = explainer_cam.explain(image[None].to(self.device))
                cam_scores = superpixel_means(cam[0].cpu().numpy(), labels, K)
                rng = np.random.default_rng(derive_seed(self.seed, row["source_id"], variant, "random_rank"))
                order = np.argsort(-cam_scores, kind="stable")
                rankings = {"gradcam": order[cam_scores[order] > 0], "random": rng.permutation(K)}
                for method in ("slice", "belief"):
                    c = records[index][method]["coefficients"]
                    positive = np.flatnonzero(c > 0)
                    rankings[method] = positive[np.argsort(-c[positive], kind="stable")]
                for method, ranking in rankings.items():
                    masks = torch.ones(ranks_k, K)
                    for k in range(ranks_k):
                        masks[k, ranking[: k + 1]] = 0
                    drops = context.prob0 - engine.query(context, masks).numpy()
                    output.append({"variant": variant, "source_id": row["source_id"], "method": method,
                                   "num_positive": int(len(ranking)), "aopc_del": float(drops.mean()),
                                   "prob0": context.prob0})
        with (self.results / "fidelity.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        self.log(f"fidelity: {len(output)} rows")


def main():
    """Input: CLI --config and --stage. Output: None; artifacts on disk."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    study = SliceBeliefStudy(yaml.safe_load(Path(args.config).read_text()))
    for stage in STAGES if args.stage == "all" else (args.stage,):
        study.log(f"=== stage {stage} ===")
        getattr(study, f"stage_{stage}")()


if __name__ == "__main__":
    main()
