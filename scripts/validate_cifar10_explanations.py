"""Validate the completed native-32 study against its saved corpus and tables.

Run from the repository root: python -m scripts.validate_cifar10_explanations
This reads every explanation/repeat checkpoint, independently checks source
matching and summaries, and writes validation.json only after all checks pass.
"""
import csv
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.cifar10_explanations import CONFIG, STAGES
from experiments.slice_belief import SliceBeliefStudy
from experiments.slice_belief_analysis import METHODS
from experiments.slice_belief_analysis import SliceBeliefAnalysis
from sklearn.metrics import roc_auc_score


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recompute_stage1(config, root):
    """Refit detectors; verify table metrics with sklearn and direct decisions.

    Fitting/feature extraction use the established pipeline. Metric calculation
    below is independent of analysis.calibration and analysis.bootstrap.
    Save scores and thresholds so every operating point can be checked later.
    """
    analysis = SliceBeliefAnalysis(config)
    output, thresholds, checks = [], {}, 0
    for train_attack, directory in (("pgd", root / "tables"), ("cw", root / "cw_train")):
        det = {(r["method"], r["attack"]): r for r in rows(directory / "stage1_detection.csv")}
        noise = {(r["method"], r["noise"]): r for r in rows(directory / "stage1_noise.csv")}
        for method in METHODS:
            model, C, dev_auc = analysis.detector(
                analysis.bank.features("train", train_attack, method)[0],
                analysis.bank.features("train", "clean", method)[0],
                [(analysis.bank.features("dev", "clean", method)[0],
                  analysis.bank.features("dev", train_attack, method)[0])], config["seed"])
            scores = {}
            for partition, variant in [("cal", "clean"), ("test", "clean")] + [
                    ("test", a) for a in config["partition_attacks"]["test"]] + [
                    ("test", n) for n in config["noise"]]:
                features, ids, _ = analysis.bank.features(partition, variant, method)
                values = model.score_samples(features)
                scores[partition, variant] = values
                output.extend({"train_attack": train_attack, "method": method, "partition": partition,
                               "variant": variant, "source_id": str(source), "score": float(score)}
                              for source, score in zip(ids, values))
            cal, clean = scores["cal", "clean"], scores["test", "clean"]
            for alpha in config["evaluation"]["fpr_targets"]:
                k = int(np.floor(len(cal) * alpha))
                threshold = float(np.sort(cal)[-(k + 1)])
                assert np.count_nonzero(cal > threshold) <= k
                thresholds[f"{train_attack}/{method}/{alpha}"] = {
                    "threshold": threshold, "calibration_n": len(cal),
                    "calibration_fpr": float(np.mean(cal > threshold)),
                    "clean_test_fpr": float(np.mean(clean > threshold))}
                for attack in config["partition_attacks"]["test"]:
                    positive = scores["test", attack]
                    r = det[method, attack]
                    actual_auc = roc_auc_score(np.r_[np.zeros(len(clean)), np.ones(len(positive))], np.r_[clean, positive])
                    assert np.isclose(float(r["auc"]), actual_auc, atol=1e-12, rtol=0), (train_attack, method, attack, "auc")
                    assert np.isclose(float(r[f"tpr@{alpha}"]), np.mean(positive > threshold), atol=1e-12, rtol=0)
                    assert np.isclose(float(r[f"clean_fpr@{alpha}"]), np.mean(clean > threshold), atol=1e-12, rtol=0)
                    assert C == float(r["C"]) and np.isclose(dev_auc, float(r["dev_auc"]), atol=1e-12, rtol=0)
                    eligible = int(r["num_eligible"])
                    assert np.isclose(float(r[f"undetected@{alpha}"]), np.count_nonzero(positive <= threshold) / eligible,
                                      atol=1e-12, rtol=0)
                    checks += 1
                for name in config["noise"]:
                    assert np.isclose(float(noise[method, name][f"fpr@{alpha}"]),
                                      np.mean(scores["test", name] > threshold), atol=1e-12, rtol=0)
                    checks += 1
            print(f"independent metric check: {train_attack}/{method} passed", flush=True)
    with (root / "validation_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (root / "validation_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    return {"fitted_detectors": 2 * len(METHODS), "attack_noise_operating_point_checks": checks,
            "score_rows": len(output), "metric_tolerance": 1e-12}


def main():
    study = SliceBeliefStudy(yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8")))
    root, work = study.results, study.work
    status = json.loads((root / "experiment_status.json").read_text())
    assert status["status"] == "complete", status
    assert status["completed_stages"] == ["audit", *STAGES], status
    test_xml = ET.parse(root / "pytest.xml").getroot()
    test_cases = test_xml.findall(".//testcase")
    assert test_cases and not test_xml.findall(".//failure") and not test_xml.findall(".//error")
    audit = json.loads((root / "prediction_audit.json").read_text())
    assert audit["status"] == "passed" and len(audit["blocks"]) == 21
    assert sum(b["prediction_mismatches"] for b in audit["blocks"]) == 0
    for name, expected in audit["corpus_files"].items():
        assert digest(work / name) == expected, name
    preserved = json.loads((root / "preserved_results_sha256.json").read_text())
    for name, expected in preserved.items():
        assert digest(Path(name)) == expected, f"Previously saved result changed: {name}"
    env = json.loads((root / "execution_environment.json").read_text())
    assert env["identity"]["device"] == status["device"]
    assert all(importlib.metadata.version(name) == version for name, version in env["packages"].items())
    assert digest(Path(CONFIG)) == env["identity"]["config_sha256"]
    for name, expected in env["identity"]["extraction_sources"].items():
        assert digest(Path(name)) == expected, name
    assert digest(Path(study.config["model"]["head_checkpoint"])) == audit["head_sha256"]
    assert digest(work / "sigma.json") == env["reused"]["sigma_sha256"]
    assert study.sigma() == 0.3
    for name, expected in env["reused"]["segments"].items():
        assert digest(work / "segments" / name) == expected, name
    for name, expected in env["archive_files"].items():
        assert digest(work / name) == expected, name
    manifest_all, variants = study.manifest(), study.variants()
    manifest = [r for r in manifest_all if r["partition"] in ("train", "dev", "cal", "test")]
    reserve = [r for r in manifest_all if r["partition"] == "reserve"]
    assert len(manifest_all) == len({r["source_id"] for r in manifest_all})
    assert len(manifest_all) == len(manifest) + len(reserve)
    assert len(manifest) == 1000 and len({r["source_id"] for r in manifest}) == 1000
    assert {r["source_id"] for r in variants} == {r["source_id"] for r in manifest}
    assert not {r["source_id"] for r in reserve}.intersection(r["source_id"] for r in variants)
    partition_sizes = Counter(r["partition"] for r in manifest)
    assert partition_sizes == {"train": 350, "dev": 100, "cal": 200, "test": 350}
    source_partition = {r["source_id"]: r["partition"] for r in manifest}
    assert all(source_partition[r["source_id"]] == r["partition"] for r in variants)
    assert Counter((r["partition"], int(r["class_index"])) for r in manifest) == {
        (p, c): n for p, n in (("train", 35), ("dev", 10), ("cal", 20), ("test", 35)) for c in range(10)}
    assert all(r["source_id"].startswith("cifar10:test:") for r in manifest)
    for panel, per_class in (("noise_panel", 10), ("stability_panel", 2)):
        selected = [r for r in manifest if r[panel]]
        assert all(r["partition"] == "test" for r in selected)
        assert Counter(r["class_index"] for r in selected) == {c: per_class for c in range(10)}
    noise_sources = {r["source_id"] for r in manifest if r["noise_panel"]}
    for name in study.config["noise"]:
        group = [r for r in variants if r["variant"] == name]
        assert len(group) == 100 and {r["source_id"] for r in group} == noise_sources
    for r in variants:
        assert r["eligible"] == (r["clean_pred"] == r["imagenet_label"])
        if r["variant"] in study.config["attacks"]:
            assert r["success"] == (r["eligible"] and r["pred"] != r["imagenet_label"])
            assert r["explain"] == r["success"]
        else:
            assert r["explain"]
    assert {r["partition"] for r in variants if r["variant"] == "fgsm"} == {"test"}
    transferred = torch.load(work / "archives/mps_transfer_20261123/explanations/train_clean.pt", map_location="cpu", weights_only=False)
    transferred_count = len(transferred["records"])
    assert transferred_count == 220 and set(transferred["records"]) == set(range(transferred_count))
    blocks, statuses = [], {m: Counter() for m in ("slice", "belief")}
    zeros = Counter()

    def check_record(record, row, segments, count=False):
        for method in statuses:
            r = record[method]
            coefficients = np.asarray(r["coefficients"])
            assert coefficients.dtype == np.float64
            assert coefficients.shape == (int(segments.max()) + 1,)
            assert np.isfinite(coefficients).all(), (row, method)
            assert np.asarray(r["eliminated"]).shape == coefficients.shape
            assert r["target_class"] == row["pred"] and r["sigma"] == 0.3
            assert 0 <= r["prob0"] <= 1 and np.isfinite(r["standalone_seconds"])
            assert r["standalone_seconds"] >= 0 and r["standalone_queries"] >= 500
            if method == "slice":
                budget = study.config["explainers"]["slice"]
                assert 1 <= r["num_steps"] <= budget["max_steps"]
                assert r["standalone_queries"] <= 1 + (budget["max_steps"] + 1) * (budget["num_perturb"] - 1)
            else:
                assert r["standalone_queries"] == study.config["explainers"]["belief"]["num_perturb"]
            assert r["status"] in ("ok", "lime_fallback", "all_eliminated", "nonfinite")
            if count:
                statuses[method][r["status"]] += 1
                zeros[method] += int(not (coefficients > 0).any())

    for partition, variant in study.shard_names():
        selected = [r for r in variants if (r["partition"], r["variant"]) == (partition, variant)]
        required = {r["tensor_index"]: r for r in selected if r["explain"]}
        archive = torch.load(study.explanation_path(partition, variant), map_location="cpu", weights_only=False)
        segments = study.load_segments(partition, variant)
        assert set(archive["records"]) == set(required) == set(segments), (partition, variant)
        cam = archive["gradcam"]
        assert np.asarray(cam["features"]).shape == (len(selected), 196)
        assert np.isfinite(cam["features"]).all()
        assert np.array_equal(cam["predictions"], [r["pred"] for r in selected])
        assert cam["prediction_mismatches"] == 0
        zeros["gradcam"] += int(np.count_nonzero(~(np.asarray(cam["features"])[list(required)] > 0).any(axis=1)))
        for i, row in required.items():
            check_record(archive["records"][i], row, segments[i], count=True)
        blocks.append({"block": f"{partition}_{variant}", "required": len(required), "readable": len(archive["records"])})
    assert sum(b["readable"] for b in blocks) == sum(r["explain"] for r in variants) == 4282
    repeat_counts = {}
    for variant in ("clean", "pgd"):
        selected = {r["tensor_index"]: r for r in variants if r["partition"] == "test" and r["variant"] == variant
                    and r["explain"] and r["stability_panel"]}
        archive = torch.load(work / "repeats" / f"test_{variant}.pt", map_location="cpu", weights_only=False)
        segments = study.load_segments("test", variant)
        assert set(archive) == set(selected)
        for i, runs in archive.items():
            assert len(runs) == study.config["quality"]["repeats"] == 5
            for run in runs:
                check_record(run, selected[i], segments[i])
        repeat_counts[variant] = {"inputs": len(selected), "repeats": 5}
    tables = root / "tables"
    expected_counts = {}
    for attack in study.config["partition_attacks"]["test"]:
        group = [r for r in variants if r["partition"] == "test" and r["variant"] == attack]
        expected_counts[attack] = {"eligible": sum(r["eligible"] for r in group),
                                   "successful": sum(r["success"] for r in group)}
    for directory in (tables, root / "cw_train"):
        det, noise = rows(directory / "stage1_detection.csv"), rows(directory / "stage1_noise.csv")
        assert len(det) == len(METHODS) * 5 and len(noise) == len(METHODS) * 4
        assert {(r["method"], r["attack"]) for r in det} == {(m, a) for m in METHODS for a in expected_counts}
        for r in det:
            assert int(r["num_clean"]) == 350
            assert int(r["num_eligible"]) == expected_counts[r["attack"]]["eligible"]
            assert int(r["num_successful"]) == expected_counts[r["attack"]]["successful"]
            assert all(0 <= float(r[k]) <= 1 for k in ("auc", "tpr@0.05", "clean_fpr@0.05"))
        for r in noise:
            assert int(r["n"]) == 100 and 0 <= float(r["fpr@0.05"]) <= 1
        assert len(rows(directory / "stage1_bootstrap.csv")) == len(METHODS) * 5
    asr = rows(tables / "asr.csv")
    for r in asr:
        group = [v for v in variants if v["partition"] == r["partition"] and v["variant"] == r["attack"]]
        assert int(r["eligible"]) == sum(v["eligible"] for v in group)
        assert int(r["successful"]) == sum(v["success"] for v in group)
    matching = json.loads((tables / "stage2_matching.json").read_text())
    assignments = rows(tables / "stage2_assignments.csv")
    matched_counts = {}
    for fold in matching:
        h, families = fold["holdout"], fold["families"]
        assert families == study.config["stage2_folds"][h] and h not in families and "fgsm" not in families
        for partition in ("train", "dev"):
            pools = [{r["source_id"] for r in variants if r["partition"] == partition and r["variant"] == a and r["success"]}
                     for a in set(families) | {"pgd"}]
            expected = sorted(set.intersection(*pools))
            if partition == "train":
                expected = expected[:study.config["evaluation"]["max_matched_train"]]
            assert fold[f"{partition}_sources"] == expected
        pool = set(fold["train_sources"])
        for seed in study.config["experiment_seeds"]:
            group = [r for r in assignments if r["holdout"] == h and int(r["seed"]) == seed]
            assert len(group) == len(pool) and {r["source_id"] for r in group} == pool
            assert all(r["mixture_attack"] in families and r["pgd_only_attack"] == "pgd" for r in group)
        matched_counts[h] = {"train_per_class": len(pool), "dev_sources": len(fold["dev_sources"])}
    seeds = rows(tables / "stage2_per_seed.csv")
    assert len(seeds) == 3 * 3 * 2 * len(METHODS) * 5
    summaries = rows(tables / "stage2_summary.csv")
    assert len(summaries) == 3 * 2 * len(METHODS) * 5
    for r in summaries:
        group = [s for s in seeds if all(s[k] == r[k] for k in ("holdout", "arm", "method", "attack"))]
        assert {int(s["seed"]) for s in group} == {42, 123, 2026} and int(r["seeds"]) == 3
        assert (r["is_holdout"] == "True") == (r["holdout"] == r["attack"])
        assert int(r["N"]) == matched_counts[r["holdout"]]["train_per_class"]
        for metric in ("auc", "tpr@0.05", "clean_fpr@0.05", "mean_noise_fpr@0.05"):
            values = [float(s[metric]) for s in group]
            assert np.isclose(float(r[f"{metric}_mean"]), np.mean(values), atol=1e-12)
            assert np.isclose(float(r[f"{metric}_sd"]), np.std(values, ddof=1), atol=1e-12)
    assert len(rows(tables / "stage2_bootstrap.csv")) == 3 * len(METHODS)
    stage2_noise = rows(tables / "stage2_noise.csv")
    assert len(stage2_noise) == 3 * 3 * 2 * len(METHODS) * 4
    for s in seeds:
        group = [n for n in stage2_noise if all(n[k] == s[k] for k in ("holdout", "seed", "arm", "method"))]
        assert len(group) == 4 and all(int(n["n"]) == 100 for n in group)
        assert np.isclose(float(s["mean_noise_fpr@0.05"]), np.mean([float(n["fpr@0.05"]) for n in group]), atol=1e-12)
    fidelity = rows(root / "fidelity.csv")
    assert len(fidelity) == 4 * (350 + expected_counts["pgd"]["successful"])
    assert all(np.isfinite(float(r["aopc_del"])) for r in fidelity)
    expected_fidelity = {(r["variant"], r["source_id"], m) for r in variants
                         if r["partition"] == "test" and r["variant"] in ("clean", "pgd") and r["explain"]
                         for m in ("gradcam", "slice", "belief", "random")}
    assert {(r["variant"], r["source_id"], r["method"]) for r in fidelity} == expected_fidelity
    for filename in ("explanation_status.csv", "cost.csv", "stability.csv", "fidelity_summary.csv"):
        assert rows(tables / filename)
    assert (root / "MAIN_RESULTS.md").is_file()
    independent_metrics = recompute_stage1(study.config, root)
    report_metadata = {"validation.json", "report_input_inventory.json", "report_validation.json"}
    artifact_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")
                      and p.name not in report_metadata]
    validation = {"status": "passed", "explanations": 4282, "blocks": blocks,
                  "manifest_rows": len(manifest_all), "selected_sources": len(manifest), "reserve_sources": len(reserve),
                  "transferred_archive_records": transferred_count, "handoff_reported_partial_records": 200,
                  "handoff_discrepancy": "The unchanged transferred checkpoint and original MPS log contain 220 records, versus 200 stated in the handoff; all are archived and excluded from the CUDA result.",
                  "prediction_mismatches": 0, "partition_sizes": dict(partition_sizes),
                  "test_attack_counts": expected_counts, "matched_sources": matched_counts,
                  "repeat_counts": repeat_counts, "status_counts": {m: dict(c) for m, c in statuses.items()},
                  "zero_positive_maps": dict(zeros), "fidelity_rows": len(fidelity),
                  "preserved_result_files_verified": len(preserved),
                  "independent_stage1_metrics": independent_metrics,
                  "stage2_seed_rows": len(seeds), "stage2_bootstrap_rows": 3 * len(METHODS),
                  "artifact_sha256": {p.relative_to(root).as_posix(): digest(p) for p in artifact_paths}}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in validation.items() if k not in ("artifact_sha256", "blocks")}, indent=2))


if __name__ == "__main__":
    main()
