"""Detection analysis for the controlled Grad-CAM vs SLICE vs BELIEF study.

Reads the artifacts of experiments/slice_belief.py (no GPU needed) and
writes CSV/JSON tables plus an auto-generated markdown summary:
  * attack success / perturbation size per partition;
  * explanation status (zero maps, BELIEF all-eliminated, SLICE fallback);
  * Stage 1: PGD-only training, frozen clean-calibration thresholds,
    evaluation on PGD/APGD/CW/patch/FGSM and benign-noise false alarms,
    paired source bootstrap of differences to Grad-CAM;
  * Stage 2: sample- and identity-matched mixture vs PGD-only training on
    withheld families (3 allocation seeds), paired bootstrap of the gain;
  * explanation stability (repeat panel), fidelity (AOPC-del) and cost.

Usage: python experiments/slice_belief_analysis.py --config configs/experiments/slice_belief_imagenette.yaml
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from analysis.bootstrap import fast_auc, paired_bootstrap
from analysis.calibration import calibrate_threshold, frozen_detection_metrics, rate_at_threshold
from analysis.metrics import compute_tpr_at_fpr
from detectors.logistic import GradCAMLogisticDetector
from experiments.slice_belief import CLEAN, SliceBeliefStudy
from explainers.base import STATUS_ALL_ELIMINATED, STATUS_LIME_FALLBACK
from explainers.superpixel_maps import SuperpixelMapVectorizer
from utils.controlled_data import derive_seed

PRIMARY_METHODS = ("gradcam", "slice", "belief")
ABLATION_METHODS = ("slice_signed", "belief_signed")
BASELINE_METHODS = ("confidence", "imagenette_indicator", "segments")
METHODS = PRIMARY_METHODS + ABLATION_METHODS + BASELINE_METHODS
REFERENCE_METHOD = "gradcam"
METHOD_LABELS = {
    "gradcam": "Grad-CAM", "slice": "SLICE", "belief": "BELIEF",
    "slice_signed": "SLICE signed (392-d)", "belief_signed": "BELIEF signed (392-d)",
    "confidence": "Confidence baseline (max-prob, entropy)",
    "imagenette_indicator": "Prediction-in-Imagenette indicator", "segments": "Superpixel-count baseline",
}
TOP_K_JACCARD = 3
PRIMARY_ALPHA = 0.05


def write_csv(path, rows):
    """Input: Path, list[dict]. Output: None; CSV with the union of keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FeatureBank:
    """Detector inputs per (partition, variant, method) over explained rows.

    Input (constructor): SliceBeliefStudy. Rows of one block are the
    variant records with explain=True, ordered by tensor_index.
    """

    def __init__(self, study):
        """Input: SliceBeliefStudy. Output: None."""
        self.study = study
        self.variants = study.variants()
        self.vectorizer = SuperpixelMapVectorizer(study.config["explainers"]["grid_size"])
        self._shards = {}
        self._cache = {}

    def rows(self, partition, variant):
        """Input: str, str. Output: list[dict] explained variant records."""
        return sorted(
            (r for r in self.variants if r["partition"] == partition and r["variant"] == variant and r["explain"]),
            key=lambda r: r["tensor_index"],
        )

    def shard(self, partition, variant):
        """Input: str, str. Output: (explanation archive dict, segments dict)."""
        key = (partition, variant)
        if key not in self._shards:
            archive = torch.load(self.study.explanation_path(partition, variant), weights_only=False)
            self._shards[key] = (archive, self.study.load_segments(partition, variant))
        return self._shards[key]

    def features(self, partition, variant, method):
        """Input: str, str, str method. Output: (ndarray[n,D], ndarray[n] source ids, list rows)."""
        key = (partition, variant, method)
        rows = self.rows(partition, variant)
        if key not in self._cache:
            archive, segments = self.shard(partition, variant)
            index = [r["tensor_index"] for r in rows]
            if method == "gradcam":
                X = archive["gradcam"]["features"][index]
            elif method == "confidence":
                X = np.array([[r["max_prob"], r["entropy"]] for r in rows])
            elif method == "imagenette_indicator":
                X = np.array([[float(r["in_imagenette"])] for r in rows])
            elif method == "segments":
                X = np.array([[segments[i].max() + 1.0] for i in index])
            else:
                base = method.replace("_signed", "")
                coefficients = [archive["records"][i][base]["coefficients"] for i in index]
                labels = [segments[i] for i in index]
                X = (self.vectorizer.transform_signed(coefficients, labels) if method.endswith("_signed")
                     else self.vectorizer.transform(coefficients, labels))
            self._cache[key] = np.asarray(X, dtype=np.float64)
        return self._cache[key], np.array([r["source_id"] for r in rows]), rows

    def restricted(self, partition, variant, method, sources):
        """Input: names, method, set of source ids. Output: ndarray features of those sources."""
        X, ids, _ = self.features(partition, variant, method)
        return X[np.isin(ids, list(sources))]


class SliceBeliefAnalysis:
    """Input: config dict. Output: runner writing all analysis tables."""

    def __init__(self, config):
        """Input: dict. Output: None."""
        self.config = config
        self.study = SliceBeliefStudy(config)
        self.bank = FeatureBank(self.study)
        self.tables = self.study.results / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.alphas = config["evaluation"]["fpr_targets"]
        self.test_attacks = config["partition_attacks"]["test"]
        self.noise = list(config["noise"])
        self.summary = {}

    # ------------------------------------------------------------ helpers

    def detector(self, train_pos, train_neg, dev_sets, seed):
        """Fit StandardScaler->LogisticRegression, selecting C on development AUC.

        Input: ndarray positives/negatives, list[(dev_neg, dev_pos)], int seed.
        Output: (fitted detector, float C, float mean dev AUC).
        """
        X = np.vstack([train_neg, train_pos])
        y = np.r_[np.zeros(len(train_neg)), np.ones(len(train_pos))]
        best = None
        for C in self.config["detector"]["C_grid"]:
            model = GradCAMLogisticDetector(seed=seed, max_iter=self.config["detector"]["max_iter"], C=C).fit(X, y)
            score = np.mean([fast_auc(model.score_samples(n), model.score_samples(p)) for n, p in dev_sets])
            if best is None or score > best[2]:
                best = (model, C, float(score))
        return best

    def thresholds(self, model, method):
        """Input: detector, method. Output: list of calibration dicts (clean cal only)."""
        scores = model.score_samples(self.bank.features("cal", CLEAN, method)[0])
        return [calibrate_threshold(scores, alpha) for alpha in self.alphas], scores

    def eligible_count(self, partition, attack):
        """Input: str, str. Output: int eligible (clean-correct) attempts."""
        return sum(r["eligible"] for r in self.bank.variants if r["partition"] == partition and r["variant"] == attack)

    def evaluate(self, model, method, thresholds, attacks):
        """Score test clean, attacks and noise at frozen thresholds.

        Input: detector, method, thresholds, list of attack names.
        Output: (list[dict] detection rows, list[dict] noise rows, dict raw scores).
        """
        clean_X, _, clean_rows = self.bank.features("test", CLEAN, method)
        clean_scores = model.score_samples(clean_X)
        detection, noise, raw = [], [], {"clean": clean_scores}
        for attack in attacks:
            X, _, _ = self.bank.features("test", attack, method)
            scores = model.score_samples(X) if len(X) else np.zeros(0)
            raw[attack] = scores
            metrics = frozen_detection_metrics(clean_scores, scores, thresholds, self.eligible_count("test", attack))
            row = {"attack": attack, "num_clean": metrics["num_clean"], "num_successful": metrics["num_successful"],
                   "num_eligible": self.eligible_count("test", attack), "auc": metrics["auc"]}
            labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(scores))]
            for alpha, point in metrics["operating_points"].items():
                row[f"tpr@{alpha}"] = point["tpr"]
                row[f"tpr@{alpha}_ci_low"], row[f"tpr@{alpha}_ci_high"] = point["tpr_ci"]
                row[f"clean_fpr@{alpha}"] = point["clean_fpr"]["rate"]
                row[f"undetected@{alpha}"] = point.get("undetected_success_rate", float("nan"))
                row[f"test_roc_tpr@{alpha}"] = (compute_tpr_at_fpr(np.r_[clean_scores, scores], labels, float(alpha))
                                                if len(scores) else float("nan"))
            detection.append(row)
        for name in self.noise:
            X, _, rows = self.bank.features("test", name, method)
            scores = model.score_samples(X)
            raw[name] = scores
            correct = np.array([r["eligible"] for r in rows])
            changed = np.array([r["pred"] != r["clean_pred"] for r in rows])
            row = {"noise": name, "n": len(scores), "prediction_changed": float(changed.mean())}
            for calibration in thresholds:
                alpha = calibration["alpha"]
                overall = rate_at_threshold(scores, calibration["threshold"])
                row[f"fpr@{alpha}"] = overall["rate"]
                row[f"fpr@{alpha}_ci_high"] = overall["ci_high"]
                row[f"fpr@{alpha}_clean_correct"] = rate_at_threshold(scores[correct], calibration["threshold"])["rate"]
                row[f"fpr@{alpha}_clean_incorrect"] = (
                    rate_at_threshold(scores[~correct], calibration["threshold"])["rate"] if (~correct).any()
                    else float("nan"))
            noise.append(row)
        return detection, noise, raw

    # ------------------------------------------------------------- tables

    def table_corpus(self):
        """Output: None; asr.csv and explanation_status.csv."""
        rows = []
        for partition in ("train", "dev", "test"):
            for attack in self.config["partition_attacks"][partition]:
                records = [r for r in self.bank.variants if r["partition"] == partition and r["variant"] == attack]
                success = [r for r in records if r["success"]]
                eligible = sum(r["eligible"] for r in records)
                rows.append({
                    "partition": partition, "attack": attack, "attempts": len(records), "eligible": eligible,
                    "successful": len(success), "asr": len(success) / max(eligible, 1),
                    "mean_linf_255": np.mean([r["linf"] for r in success]) * 255 if success else float("nan"),
                    "mean_l2": np.mean([r["l2"] for r in success]) if success else float("nan"),
                    "pred_outside_imagenette": np.mean([not r["in_imagenette"] for r in success]) if success else float("nan"),
                })
        clean = {p: [r for r in self.bank.variants if r["partition"] == p and r["variant"] == CLEAN]
                 for p in ("train", "dev", "cal", "test")}
        self.summary["clean_accuracy"] = {p: float(np.mean([r["eligible"] for r in v])) for p, v in clean.items()}
        self.summary["clean_counts"] = {p: len(v) for p, v in clean.items()}
        write_csv(self.tables / "asr.csv", rows)
        status_rows = []
        for partition, variant in self.study.shard_names():
            archive, _ = self.bank.shard(partition, variant)
            records = list(archive["records"].values())
            for method in ("slice", "belief"):
                statuses = [r[method]["status"] for r in records]
                status_rows.append({
                    "partition": partition, "variant": variant, "method": method, "n": len(records),
                    "zero_positive_map": float(np.mean([not (r[method]["coefficients"] > 0).any() for r in records])),
                    "all_eliminated": float(np.mean([s == STATUS_ALL_ELIMINATED for s in statuses])),
                    "lime_fallback": float(np.mean([s == STATUS_LIME_FALLBACK for s in statuses])),
                    "mean_eliminated_fraction": float(np.mean([r[method]["eliminated"].mean() for r in records])),
                    "mean_positive_superpixels": float(np.mean([(r[method]["coefficients"] > 0).sum() for r in records])),
                })
            status_rows.append({"partition": partition, "variant": variant, "method": "gradcam",
                                "n": len(records),
                                "prediction_mismatches": archive["gradcam"]["prediction_mismatches"]})
        write_csv(self.tables / "explanation_status.csv", status_rows)

    def stage1(self):
        """Output: dict method -> fitted detector; stage1_*.csv tables."""
        detection, noise, boot_inputs, detectors = [], [], {}, {}
        for method in METHODS:
            pos = self.bank.features("train", "pgd", method)[0]
            neg = self.bank.features("train", CLEAN, method)[0]
            dev = [(self.bank.features("dev", CLEAN, method)[0], self.bank.features("dev", "pgd", method)[0])]
            model, C, dev_auc = self.detector(pos, neg, dev, self.config["seed"])
            thresholds, cal_scores = self.thresholds(model, method)
            rows, noise_rows, raw = self.evaluate(model, method, thresholds, self.test_attacks)
            for row in rows:
                detection.append({"method": method, "C": C, "dev_auc": dev_auc, **row})
            for row in noise_rows:
                noise.append({"method": method, **row})
            boot_inputs[method] = (raw, cal_scores)
            detectors[method] = (model, thresholds)
        write_csv(self.tables / "stage1_detection.csv", detection)
        write_csv(self.tables / "stage1_noise.csv", noise)
        boot_rows = []
        _, clean_ids, _ = self.bank.features("test", CLEAN, REFERENCE_METHOD)
        for attack in self.test_attacks:
            _, positive_ids, _ = self.bank.features("test", attack, REFERENCE_METHOD)
            if not len(positive_ids):
                continue
            methods = {m: {"clean": raw["clean"], "positive": raw[attack], "calibration": cal}
                       for m, (raw, cal) in boot_inputs.items()}
            result = paired_bootstrap(methods, REFERENCE_METHOD, clean_ids, positive_ids,
                                      self.config["evaluation"]["bootstrap_replicates"], PRIMARY_ALPHA,
                                      derive_seed(self.config["seed"], "bootstrap", attack) % 2 ** 32)
            for method, entry in result.items():
                boot_rows.append({"attack": attack, "method": method,
                                  "auc_ci_low": entry["auc"][0], "auc_ci_high": entry["auc"][1],
                                  "tpr5_ci_low": entry["tpr"][0], "tpr5_ci_high": entry["tpr"][1],
                                  "auc_diff_vs_gradcam": entry["auc_diff"][0],
                                  "auc_diff_ci_low": entry["auc_diff"][1], "auc_diff_ci_high": entry["auc_diff"][2],
                                  "tpr5_diff_vs_gradcam": entry["tpr_diff"][0],
                                  "tpr5_diff_ci_low": entry["tpr_diff"][1], "tpr5_diff_ci_high": entry["tpr_diff"][2]})
        write_csv(self.tables / "stage1_bootstrap.csv", boot_rows)
        return detectors

    def matched_pool(self, partition, families):
        """Input: partition, list of attack families. Output: sorted list of sources successful for all + PGD."""
        sets = []
        for family in set(families) | {"pgd"}:
            sets.append({r["source_id"] for r in self.bank.variants
                         if r["partition"] == partition and r["variant"] == family and r["success"]})
        return sorted(set.intersection(*sets))

    def allocate(self, pool, families, seed, fold):
        """Class-stratified, as-even-as-possible family assignment.

        Input: list sources, list families, int seed, str fold.
        Output: dict source -> family.
        """
        by_class = {}
        class_of = {r["source_id"]: r["class_index"] for r in self.bank.variants if r["variant"] == CLEAN}
        for source in pool:
            by_class.setdefault(class_of[source], []).append(source)
        rng = np.random.default_rng(derive_seed(seed, fold, "allocation"))
        offset = int(rng.integers(len(families)))
        assignment, counter = {}, offset
        for label in sorted(by_class):
            for source in rng.permutation(by_class[label]):
                assignment[str(source)] = families[counter % len(families)]
                counter += 1
        return assignment

    def stage2(self):
        """Output: None; stage2_*.csv (per seed, seed summary, bootstrap of the mixture gain)."""
        cap = self.config["evaluation"]["max_matched_train"]
        per_seed, boot_rows = [], []
        for holdout, families in self.config["stage2_folds"].items():
            pool = self.matched_pool("train", families)[:cap]
            dev_pool = set(self.matched_pool("dev", families))
            for seed in self.config["experiment_seeds"]:
                assignment = self.allocate(pool, families, seed, holdout)
                for method in METHODS:
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

    def stability(self, detectors):
        """Input: Stage-1 detectors. Output: None; stability.csv."""
        rows = []
        for variant in (CLEAN, "pgd"):
            path = self.study.work / "repeats" / f"test_{variant}.pt"
            if not path.exists():
                continue
            archive = torch.load(path, weights_only=False)
            segments = self.study.load_segments("test", variant)
            for method in ("slice", "belief"):
                signs, jaccards, spearmans, score_sd, zero = [], [], [], [], []
                model = detectors[method][0]
                for index, runs in archive.items():
                    coefficients = [run[method]["coefficients"] for run in runs]
                    features = self.bank.vectorizer.transform(coefficients, [segments[index]] * len(runs))
                    score_sd.append(np.std(model.score_samples(features), ddof=1))
                    zero.append(np.mean([not (c > 0).any() for c in coefficients]))
                    for a in range(len(runs)):
                        for b in range(a + 1, len(runs)):
                            ca, cb = coefficients[a], coefficients[b]
                            signs.append(np.mean(np.sign(ca) == np.sign(cb)))
                            ta = set(np.argsort(-ca)[:TOP_K_JACCARD][ca[np.argsort(-ca)[:TOP_K_JACCARD]] > 0])
                            tb = set(np.argsort(-cb)[:TOP_K_JACCARD][cb[np.argsort(-cb)[:TOP_K_JACCARD]] > 0])
                            jaccards.append(1.0 if not ta and not tb else len(ta & tb) / len(ta | tb))
                            if np.std(ca) > 0 and np.std(cb) > 0:
                                spearmans.append(spearmanr(ca, cb).statistic)
                rows.append({"variant": variant, "method": method, "inputs": len(archive),
                             "repeats": len(next(iter(archive.values()))),
                             "sign_agreement": float(np.mean(signs)), "top3_jaccard": float(np.mean(jaccards)),
                             "spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
                             "spearman_pairs": len(spearmans), "zero_map_rate": float(np.mean(zero)),
                             "detector_score_sd": float(np.mean(score_sd))})
            rows.append({"variant": variant, "method": "gradcam", "inputs": len(archive), "sign_agreement": 1.0,
                         "top3_jaccard": 1.0, "spearman": 1.0, "detector_score_sd": 0.0,
                         "note": "deterministic given weights and input"})
        write_csv(self.tables / "stability.csv", rows)

    def fidelity_and_cost(self):
        """Output: None; fidelity_summary.csv and cost.csv."""
        path = self.study.results / "fidelity.csv"
        if path.exists():
            with path.open(newline="") as stream:
                records = list(csv.DictReader(stream))
            rows = []
            for variant in (CLEAN, "pgd"):
                for method in ("gradcam", "slice", "belief", "random"):
                    values = [float(r["aopc_del"]) for r in records if r["variant"] == variant and r["method"] == method]
                    positives = [int(r["num_positive"]) for r in records if r["variant"] == variant and r["method"] == method]
                    rows.append({"variant": variant, "method": method, "n": len(values),
                                 "aopc_del_mean": float(np.mean(values)),
                                 "aopc_del_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                                 "median_positive_superpixels": float(np.median(positives))})
            write_csv(self.tables / "fidelity_summary.csv", rows)
        records, gradcam_seconds, segment_seconds = {"slice": [], "belief": []}, [], []
        for partition, variant in self.study.shard_names():
            archive, _ = self.bank.shard(partition, variant)
            gradcam_seconds.append(archive["gradcam"]["seconds_per_image"])
            with np.load(self.study.segment_path(partition, variant)) as data:
                segment_seconds += list(data["seconds"])
            for record in archive["records"].values():
                for method in records:
                    records[method].append(record[method])
        rows = []
        for method, values in records.items():
            seconds = np.array([v["standalone_seconds"] for v in values])
            queries = np.array([v["standalone_queries"] for v in values])
            rows.append({"method": method, "inputs": len(values), "median_seconds": float(np.median(seconds)),
                         "p90_seconds": float(np.percentile(seconds, 90)), "mean_seconds": float(seconds.mean()),
                         "median_queries": float(np.median(queries)), "p90_queries": float(np.percentile(queries, 90)),
                         "mean_steps": float(np.mean([v["num_steps"] for v in values]))})
        rows.append({"method": "gradcam", "inputs": "all rows", "median_seconds": float(np.median(gradcam_seconds)),
                     "mean_seconds": float(np.mean(gradcam_seconds)), "median_queries": 1, "mean_steps": 1})
        rows.append({"method": "quickshift (shared by SLICE/BELIEF)", "inputs": len(segment_seconds),
                     "median_seconds": float(np.median(segment_seconds)), "mean_seconds": float(np.mean(segment_seconds))})
        write_csv(self.tables / "cost.csv", rows)

    def run(self):
        """Output: None; all tables and summary.json."""
        self.table_corpus()
        detectors = self.stage1()
        self.stage2()
        self.stability(detectors)
        self.fidelity_and_cost()
        self.summary["sigma"] = json.loads((self.study.work / "sigma.json").read_text())["counts"]
        self.summary["sigma_value"] = self.study.sigma()
        (self.tables / "summary.json").write_text(json.dumps(self.summary, indent=2))
        self.study.log(f"analysis tables written to {self.tables}")


def main():
    """Input: CLI --config. Output: None."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    SliceBeliefAnalysis(yaml.safe_load(Path(args.config).read_text())).run()


if __name__ == "__main__":
    main()
