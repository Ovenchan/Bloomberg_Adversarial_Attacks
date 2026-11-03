"""Notebook-independent ResNet/Grad-CAM extraction and grouped detection experiment."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from torch import nn

from analysis.grouped_cv import GroupedDetectionCV
from attacks.factory import build_attack
from encoders.cnn import ResNetEncoder
from pipelines.cnn_pipeline import GradCAMPipeline
from utils.gradcam_data import make_loader, classifier_labels

CIFAR_CLASSES = 10
JSON_INDENT = 2


def seed_everything(seed):
    """Input: int seed. Output: None; seeds CPU/CUDA and deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class GradCAMExperiment:
    """Input: resolved config dict. Output: reproducible extraction/CV runner."""

    def __init__(self, config):
        """Input: dict config. Output: None."""
        self.config = config
        seed_everything(config["seed"])
        requested = config["device"]
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if requested == "auto"
            else torch.device(requested)
        )

    def classifier(self, training=False):
        """Input: bool training head. Output: eval-mode ResNetEncoder on selected device."""
        cfg = self.config
        checkpoint = None
        if cfg["dataset"]["name"] == "cifar10" and not training:
            checkpoint = self.load_head_checkpoint()
        model = ResNetEncoder(weights=cfg["model"]["weights"])
        if cfg["dataset"]["name"] == "cifar10":
            model.model.fc = nn.Linear(
                model.model.fc.in_features, CIFAR_CLASSES
            )
            if not training:
                try:
                    model.model.fc.load_state_dict(checkpoint["state_dict"])
                except RuntimeError as error:
                    raise ValueError(
                        "Invalid CIFAR head shape; run --stage train-head first"
                    ) from error
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def load_head_checkpoint(self):
        """Input: None. Output: validated trained CIFAR head checkpoint dict."""
        cfg = self.config
        path = Path(cfg["model"]["head_checkpoint"])
        instruction = "run --stage train-head first"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing CIFAR head checkpoint {path}; {instruction}"
            )
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise ValueError(
                f"Unreadable CIFAR head checkpoint; {instruction}"
            ) from error
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid CIFAR head checkpoint; {instruction}")
        if checkpoint.get("training_split") != "official_cifar10_train":
            raise ValueError(f"Invalid head training split; {instruction}")
        if checkpoint.get("weights") != cfg["model"]["weights"]:
            raise ValueError(f"Head backbone weights mismatch; {instruction}")
        if checkpoint.get("image_size") != cfg["dataset"]["image_size"]:
            raise ValueError(f"Head preprocessing mismatch; {instruction}")
        for field in ("epochs_completed", "optimizer_steps"):
            if (
                not isinstance(checkpoint.get(field), int)
                or checkpoint[field] <= 0
            ):
                raise ValueError(f"Head has no valid {field}; {instruction}")
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
            raise ValueError(f"Invalid head state; {instruction}")
        if any(
            not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            for value in state.values()
        ):
            raise ValueError(
                f"Non-finite or invalid head parameters; {instruction}"
            )
        return checkpoint

    def head_optimizer(self, model):
        """Input: ResNetEncoder. Output: Adam for FC only; backbone stays in eval mode."""
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.model.fc.parameters():
            parameter.requires_grad_(True)
        return torch.optim.Adam(
            model.model.fc.parameters(),
            lr=self.config["head_training"]["learning_rate"],
        )

    def train_head(self):
        """Input: None. Output: None; saves CIFAR head trained exclusively on official TRAIN."""
        cfg = self.config
        if cfg["dataset"]["name"] != "cifar10":
            raise ValueError(
                "Imagenette uses the pretrained ImageNet classifier directly"
            )
        if cfg["head_training"]["epochs"] <= 0:
            raise ValueError("Head training requires at least one epoch")
        model = self.classifier(training=True)
        loader = make_loader(cfg["dataset"], train=True, seed=cfg["seed"])
        optimizer = self.head_optimizer(model)
        optimizer_steps = 0
        for epoch in range(cfg["head_training"]["epochs"]):
            for images, labels, source_ids in loader:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(
                    model(images.to(self.device)), labels.to(self.device)
                )
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite head training loss; checkpoint not saved"
                    )
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
            print(f"Head epoch {epoch + 1} complete", flush=True)
        if optimizer_steps == 0:
            raise ValueError(
                "No head training steps completed; checkpoint not saved"
            )
        path = Path(cfg["model"]["head_checkpoint"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.model.fc.state_dict(),
                "training_split": "official_cifar10_train",
                "weights": cfg["model"]["weights"],
                "image_size": cfg["dataset"]["image_size"],
                "epochs_completed": cfg["head_training"]["epochs"],
                "optimizer_steps": optimizer_steps,
                "config": cfg,
            },
            path,
        )

    def evaluate_clean(self, model=None, loader=None):
        """Input: optional classifier and loader. Output: saved clean accuracy/counts dict."""
        cfg = self.config
        model = self.classifier() if model is None else model
        loader = (
            make_loader(cfg["dataset"], seed=cfg["seed"])
            if loader is None
            else loader
        )
        model.eval()
        groups, correct_rows = [], []
        with torch.no_grad():
            for images, labels, source_ids in loader:
                labels = classifier_labels(
                    labels.to(self.device), loader, cfg["dataset"]["name"]
                )
                predictions = model(images.to(self.device)).argmax(dim=1)
                groups.append(np.asarray(source_ids))
                correct_rows.append((predictions == labels).cpu().numpy())
        if not groups:
            raise ValueError("Clean evaluation dataset is empty")
        groups = np.concatenate(groups)
        correct_rows = np.concatenate(correct_rows)
        total = len(correct_rows)
        report = {
            "clean_accuracy": float(correct_rows.mean()),
            "num_correct": int(correct_rows.sum()),
            "num_samples": total,
            "num_source_images": int(len(np.unique(groups))),
            "config": cfg,
        }
        minimum = cfg["evaluation"]["min_correct_per_fold"]
        if not isinstance(minimum, int) or minimum <= 0:
            raise ValueError("min_correct_per_fold must be a positive integer")
        fold_counts = []
        if len(np.unique(groups)) >= cfg["evaluation"]["folds"]:
            cv = GroupedDetectionCV(
                n_splits=cfg["evaluation"]["folds"], seed=cfg["seed"]
            )
            for train, test in cv.split(groups):
                fold_counts.append(
                    int(len(np.unique(groups[test][correct_rows[test]])))
                )
        report["correct_sources_per_fold"] = fold_counts
        report["sufficient_clean_samples"] = bool(
            fold_counts and min(fold_counts) >= minimum
        )
        output = Path(cfg["results_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "clean_classifier.json").write_text(
            json.dumps(report, indent=JSON_INDENT)
        )
        print(
            f"Clean accuracy: {report['clean_accuracy']:.6f} ({report['num_correct']}/{total})",
            flush=True,
        )
        if not report["sufficient_clean_samples"]:
            raise ValueError(
                f"Too few correctly classified source images: {fold_counts}; "
                f"require at least {minimum} per fold before attack generation"
            )
        return report

    def extract(self):
        """Input: None. Output: Path to aligned feature archive, also writes raw CSV and provenance."""
        cfg = self.config
        model = self.classifier()
        pipeline = GradCAMPipeline(
            model, cfg["model"]["target_layer"], cfg["gradcam"]["grid_size"]
        )
        loader = make_loader(cfg["dataset"], seed=cfg["seed"])
        clean_report = self.evaluate_clean(model, loader)
        attacks = {
            name: build_attack(name, model, params)
            for name, params in cfg["attacks"].items()
        }
        if "pgd" not in attacks:
            raise ValueError("PGD must be configured")
        arrays = {
            "clean": [],
            "labels": [],
            "clean_predictions": [],
            "groups": [],
        }
        for name in attacks:
            arrays[name] = []
            arrays[f"{name}_predictions"] = []
            arrays[f"{name}_source_ids"] = []
        patch_locations = None
        if "patch" in attacks:

            max_position = (
                cfg["dataset"]["image_size"]
                - cfg["attacks"]["patch"]["patch_size"]
            )
            patch_locations = np.random.default_rng(cfg["seed"]).integers(
                low=0, high=max_position + 1, size=(len(loader.dataset), 2)
            )
        offset = 0
        for batch, (images, labels, source_ids) in enumerate(loader):
            images = images.to(self.device)
            labels = classifier_labels(
                labels.to(self.device), loader, cfg["dataset"]["name"]
            )
            features, predictions = pipeline.extract(images)
            arrays["clean"].append(features)
            arrays["labels"].append(labels.cpu().numpy())
            arrays["clean_predictions"].append(predictions)
            arrays["groups"].append(np.asarray(source_ids))
            for name, attack in attacks.items():
                if name == "patch":
                    locations = patch_locations[offset : offset + len(images)]
                    adversarial = attack(
                        images,
                        labels,
                        top_positions=locations[:, 0],
                        left_positions=locations[:, 1],
                    )
                else:
                    adversarial = attack(images, labels)
                features, predictions = pipeline.extract(adversarial)
                arrays[name].append(features)
                arrays[f"{name}_predictions"].append(predictions)
                arrays[f"{name}_source_ids"].append(
                    np.asarray(source_ids).copy()
                )
            offset += len(images)
            print(f"Extracted batch {batch + 1}/{len(loader)}", flush=True)
        arrays = {key: np.concatenate(value) for key, value in arrays.items()}

        arrays["clean_source_ids"] = arrays["groups"].copy()
        if patch_locations is not None:
            arrays["patch_locations"] = patch_locations
        path = Path(cfg["features_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        metadata = {
            "config": cfg,
            "targeting": "predicted",
            "attack_names": list(attacks),
            "clean_classifier": clean_report,
            "source_split": (
                "official_cifar10_test"
                if cfg["dataset"]["name"] == "cifar10"
                else "imagenette_val"
            ),
            "versions": {
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "torchvision",
                    "scikit-learn",
                    "numpy",
                    "torchattacks",
                )
            },
        }
        if cfg["dataset"]["name"] == "cifar10":
            metadata["head_sha256"] = hashlib.sha256(
                Path(cfg["model"]["head_checkpoint"]).read_bytes()
            ).hexdigest()
        metadata["archive_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=JSON_INDENT)
        )
        with path.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["source_id", "variant", "true_label", "prediction"]
                + [f"feature_{i}" for i in range(arrays["clean"].shape[1])]
            )
            for variant in ["clean", *attacks]:
                for i, vector in enumerate(arrays[variant]):
                    writer.writerow(
                        [
                            int(arrays["groups"][i]),
                            variant,
                            int(arrays["labels"][i]),
                            int(arrays[f"{variant}_predictions"][i]),
                            *vector,
                        ]
                    )
        return path

    def evaluate(self):
        """Input: None. Output: dict results; saves fold models, CSV metrics and JSON provenance."""
        cfg = self.config
        path = Path(cfg["features_path"])
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["targeting"] != "predicted":
            raise ValueError("Only predicted-class archives are accepted")
        if (
            metadata["archive_sha256"]
            != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Feature archive checksum mismatch")
        for key in ("dataset", "model", "gradcam", "attacks", "seed"):
            if metadata["config"][key] != cfg[key]:
                raise ValueError(f"Feature extraction config mismatch: {key}")
        with np.load(path, allow_pickle=False) as arrays:
            required_ids = ["groups", "clean_source_ids"] + [
                f"{name}_source_ids" for name in metadata["attack_names"]
            ]
            if any(key not in arrays for key in required_ids):
                raise ValueError(
                    "Archive lacks explicit source IDs; rerun --stage extract"
                )
            if not np.array_equal(
                arrays["groups"], arrays["clean_source_ids"]
            ):
                raise ValueError("Clean source IDs do not align with groups")
            attacks = {
                name: {
                    "features": arrays[name],
                    "predictions": arrays[f"{name}_predictions"],
                    "source_ids": arrays[f"{name}_source_ids"],
                }
                for name in metadata["attack_names"]
            }
            cv = GroupedDetectionCV(
                n_splits=cfg["evaluation"]["folds"],
                seed=cfg["seed"],
                detector_config=cfg["detector"],
            )
            result = cv.evaluate(
                arrays["clean"],
                attacks,
                arrays["groups"],
                arrays["labels"],
                arrays["clean_predictions"],
            )
        result["provenance"] = metadata
        result["evaluation_config"] = cfg
        output = Path(cfg["results_dir"])
        output.mkdir(parents=True, exist_ok=True)

        serializable = json.loads(
            json.dumps(result), parse_constant=lambda value: None
        )
        (output / "grouped_cv.json").write_text(
            json.dumps(serializable, indent=JSON_INDENT, allow_nan=False)
        )
        with (output / "fold_metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "fold",
                    "attack",
                    "auc",
                    "tpr_at_0.01",
                    "tpr_at_0.05",
                    "attack_success_rate",
                    "num_adversarial_evaluated",
                    "num_clean",
                ]
            )
            for fold in result["folds"]:
                for name, metrics in fold["metrics"].items():
                    writer.writerow(
                        [
                            fold["fold"],
                            name,
                            metrics["auc"],
                            metrics["tpr_at_fpr"]["0.01"],
                            metrics["tpr_at_fpr"]["0.05"],
                            metrics["attack_success_rate"],
                            metrics["num_adversarial_evaluated"],
                            metrics["num_clean"],
                        ]
                    )
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["attack", "metric", "mean", "std", "valid_folds"])
            for attack, metrics in result["summary"].items():
                for metric, values in metrics.items():
                    writer.writerow(
                        [
                            attack,
                            metric,
                            values["mean"],
                            values["std"],
                            values["valid_folds"],
                        ]
                    )
        for fold, detector in enumerate(cv.detectors):
            joblib.dump(detector, output / f"detector_fold_{fold}.joblib")
        return result


def main():
    """Input: CLI --config and --stage. Output: None; requested artifacts on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("train-head", "evaluate-clean", "extract", "evaluate", "all"),
        default="all",
    )
    args = parser.parse_args()
    experiment = GradCAMExperiment(
        yaml.safe_load(Path(args.config).read_text())
    )
    if args.stage == "train-head":
        experiment.train_head()
    if args.stage == "evaluate-clean":
        experiment.evaluate_clean()
    if args.stage in ("extract", "all"):
        experiment.extract()
    if args.stage in ("evaluate", "all"):
        experiment.evaluate()


if __name__ == "__main__":
    main()
