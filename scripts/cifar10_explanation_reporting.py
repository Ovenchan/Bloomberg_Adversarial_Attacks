"""Artifact-backed native-32 CIFAR-10 sections shared by the overall reports."""
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

RESULT = Path("results/slice_belief_cifar10_native32_fixed1000")
METHODS = ("gradcam", "slice", "belief", "slice_signed", "belief_signed", "confidence", "segments")
ATTACKS = ("pgd", "apgd", "cw", "patch", "fgsm")
NOISE = ("gauss2", "gauss4", "gauss8", "unif8")
LABEL = {"gradcam": "Grad-CAM", "slice": "SLICE", "belief": "BELIEF", "slice_signed": "SLICE signed",
         "belief_signed": "BELIEF signed", "confidence": "Confidence", "segments": "Superpixel count"}


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, values):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in values])


def pct(value):
    return f"{100 * float(value):.1f}%"


def number(value):
    return f"{float(value):.3f}" if math.isfinite(float(value)) else "undefined"


def significant(value):
    return f"{float(value):.4g}" if math.isfinite(float(value)) else "undefined"


def cell(row):
    return f"{number(row['auc'])} / {pct(row['tpr@0.05'])}"


class Native32Report:
    def __init__(self, repo=Path(".")):
        self.repo = Path(repo)
        self.root = self.repo / RESULT
        self.tables = self.root / "tables"
        self.validation = json.loads((self.root / "validation.json").read_text(encoding="utf-8"))
        assert self.validation["status"] == "passed" and self.validation["explanations"] == 4282
        for name, expected in self.validation["artifact_sha256"].items():
            assert hashlib.sha256((self.root / name).read_bytes()).hexdigest() == expected, (
                f"Validated artifact changed: {name}; rerun scripts.validate_cifar10_explanations")
        self.environment = json.loads((self.root / "execution_environment.json").read_text(encoding="utf-8"))
        self.detection = {(r["method"], r["attack"]): r for r in read(self.tables / "stage1_detection.csv")}
        self.noise = {(r["method"], r["noise"]): r for r in read(self.tables / "stage1_noise.csv")}
        self.cw = {(r["method"], r["attack"]): r for r in read(self.root / "cw_train/stage1_detection.csv")}

    def link(self, title, path):
        return f"[{title}]({(RESULT / path).as_posix()})"

    def setup(self):
        counts = self.validation["test_attack_counts"]
        matching = self.validation["matched_sources"]
        independent = self.validation["independent_stage1_metrics"]
        config = self.environment["config"]
        explainers = config["explainers"]
        text = [
            "The primary CIFAR-10 experiment uses 1,000 class-balanced official test sources, partitioned "
            "350/100/200/350 into train/development/clean calibration/test. Attacks and noise modify the native "
            "32×32 pixels before bilinear resizing to the stored 224×224 tensors. The classifier is a frozen "
            "ImageNet-pretrained ResNet-50 with a linear head trained on the official CIFAR-10 training split. "
            "DINOv2 is an independent observer of the same ResNet-targeted attacks. The older 224-resolution "
            "attack experiment remains a separate diagnostic.",
            f"Native-space PGD/APGD/FGSM use an L∞ budget of {255 * config['attacks']['pgd']['epsilon']:.0f}/255. "
            f"PGD and APGD each use {config['attacks']['pgd']['steps']} steps; the patch is "
            f"{config['attacks']['patch']['patch_size']}×{config['attacks']['patch']['patch_size']} native pixels. "
            f"CW uses c={config['attacks']['cw']['c']}, kappa={config['attacks']['cw']['kappa']} and "
            f"{config['attacks']['cw']['steps']} steps with its own distortion objective; it is not constrained by the shared L∞ budget.",
            "All 21 tensor blocks and three corpus metadata files match the saved ResNet provenance. The local "
            "classifier audit has **zero prediction mismatches**. All **4,282** required explanation records, "
            "Grad-CAM features, stability repetitions and fidelity records were read and validated. "
            "The saved development-selected sigma **0.3** and completed segmentation are reused.",
            f"SLICE and BELIEF share segmentation and the first {explainers['slice']['num_perturb']}-sample perturbation draw. "
            f"SLICE uses {explainers['slice']['n_iterations']} bootstrap fits, at most {explainers['slice']['max_steps']} "
            f"elimination steps and tolerance {explainers['slice']['tolerance']}; BELIEF uses slack {explainers['belief']['slack']}. "
            f"Training-only scaling precedes logistic regression; C is selected from {config['detector']['C_grid']} using "
            f"development AUC. Independent clean calibration freezes thresholds at targets {config['evaluation']['fpr_targets']}. "
            "Settings are recorded in the execution environment and [native-32 configuration](configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml).",
            table(["Test attack", "Eligible sources", "Successful attacks", "ASR"],
                  [[a.upper(), counts[a]["eligible"], counts[a]["successful"],
                    pct(counts[a]["successful"] / counts[a]["eligible"])] for a in ATTACKS]),
            "Detection TPR divides by successful attacks; ASR and undetected-success rate divide by eligible "
            "clean-correct sources. Every attack is compared with all 350 clean test sources. Each noise control "
            "has 100 test sources. Failed attacks do not enter detector positives. FGSM is test-only.",
            f"Independent refitting of {independent['fitted_detectors']} PGD/CW-trained Stage 1 detectors verified "
            f"{independent['attack_noise_operating_point_checks']} attack/noise operating points. AUC is recomputed "
            "with sklearn, while thresholds, TPR, clean/noise FPR and undetected rates are checked with direct "
            "score comparisons. The validation retains " + self.link("all recomputed scores", "validation_scores.csv") +
            " and " + self.link("calibration thresholds", "validation_thresholds.json") + ".",
            table(["Stage 2 holdout", "Permitted mixture", "Matched N clean + N attacked", "Matched dev sources"],
                  [[h.upper(), "+".join(a.upper() for a in self.environment["config"]["stage2_folds"][h]),
                    f"{matching[h]['train_per_class']} + {matching[h]['train_per_class']}",
                    matching[h]["dev_sources"]] for h in ("cw", "patch", "apgd")]),
            "Stage 2 refits PGD-only and mixture controls on identical successful source pools. "
            "The source allocation varies over seeds 42, 123 and 2026; neither the held-out family nor FGSM "
            "enters fitting or tuning. Both arms retain the original development-negative and clean-calibration protocol.",
            "Matched development counts refer to successful positive-source pools; clean development negatives remain all 100 sources in each arm.",
            "Sources: " + self.link("validation", "validation.json") + ", " + self.link("prediction audit", "prediction_audit.json") + ", " +
            self.link("ASR counts", "tables/asr.csv") + ", " + self.link("matched source pools", "tables/stage2_matching.json") + ", " +
            self.link("per-seed assignments", "tables/stage2_assignments.csv") + ".",
        ]
        return "\n\n".join(text)

    def explanation(self):
        text = ["#### PGD-trained Stage 1 and false alarms",
                "Cells show **AUC / TPR at the 5% clean-calibration target**. Actual clean test FPR is a separate column; "
                "the target does not guarantee a 5% test FPR. Positive explanation maps have 196 features and signed variants have 392. "
                "Confidence features retain the original corpus's saved maximum probabilities and entropies."]
        values = [[LABEL[m], *[cell(self.detection[m, a]) for a in ATTACKS], pct(self.detection[m, "pgd"]["clean_fpr@0.05"])] for m in METHODS]
        for label, directory in (("ResNet pooled", "resnet_representation_cifar10_native32"), ("DINOv2", "dinov2_observer_cifar10_native32")):
            det = {r["attack"]: r for r in read(self.repo / "results" / directory / "stage1_detection.csv")}
            values.append([label, *[cell(det[a]) for a in ATTACKS], pct(det["pgd"]["clean_fpr@0.05"])])
        text.append(table(["Features", *[a.upper() for a in ATTACKS], "Actual clean FPR"], values))
        text.append(table(["PGD-trained features", *[n + " FPR" for n in NOISE]],
                          [[LABEL[m], *[pct(self.noise[m, n]["fpr@0.05"]) for n in NOISE]] for m in METHODS]))
        text.append("Sources: " + self.link("Stage 1", "tables/stage1_detection.csv") + ", " + self.link("noise controls", "tables/stage1_noise.csv") +
                    ", [ResNet](results/resnet_representation_cifar10_native32/stage1_detection.csv), "
                    "[DINOv2](results/dinov2_observer_cifar10_native32/stage1_detection.csv). "
                    "The constant ten-class membership indicator is retained in raw CSVs but omitted here because every CIFAR-10 prediction belongs to its label set.")
        if all(float(self.detection["belief", a]["tpr@0.05"]) == 0 for a in ATTACKS):
            text.append("Positive-only BELIEF has zero TPR on every attack as well as zero clean/noise FPR at the 5% target. "
                        "Its independent calibration check records zero calibration FPR under the strict decision rule. "
                        "The zero false-alarm rate therefore does not indicate useful detection at this operating point.")
        text.append("#### Stage 2: matched mixtures and uncertainty")
        summaries = read(self.tables / "stage2_summary.csv")
        values = [[LABEL[r["method"]], r["holdout"].upper(), r["arm"], r["N"],
                   f"{number(r['auc_mean'])} ± {number(r['auc_sd'])}",
                   f"{pct(r['tpr@0.05_mean'])} ± {pct(r['tpr@0.05_sd'])}",
                   pct(r["clean_fpr@0.05_mean"]), pct(r["mean_noise_fpr@0.05_mean"])]
                  for r in summaries if r["is_holdout"] == "True" and r["method"] in METHODS]
        text.append(table(["Features", "Holdout", "Arm", "N per class", "AUC mean ± SD", "TPR mean ± SD", "Clean FPR", "Mean noise FPR"], values))
        bootstrap = read(self.tables / "stage2_bootstrap.csv")
        text.append("The following paired source-bootstrap intervals compare mixture minus its refitted PGD-only control "
                    "at seed 42 (2,000 test-source resamples, with calibration resampled). They describe the primary seed, "
                    "not the uncertainty of the three-seed mean. The displayed gain estimates are bootstrap means.")
        text.append(table(["Features", "Holdout", "Bootstrap mean AUC gain [95% interval]", "Bootstrap mean TPR gain [95% interval]"],
                          [[LABEL[r["method"]], r["holdout"].upper(),
                            f"{number(r['auc_gain'])} [{number(r['auc_gain_ci_low'])}, {number(r['auc_gain_ci_high'])}]",
                            f"{pct(r['tpr5_gain'])} [{pct(r['tpr5_gain_ci_low'])}, {pct(r['tpr5_gain_ci_high'])}]"]
                           for r in bootstrap if r["method"] in METHODS]))
        text.append("Sources: " + ", ".join(self.link(label, "tables/" + name) for label, name in
                    (("three-seed summaries", "stage2_summary.csv"), ("per-seed metrics", "stage2_per_seed.csv"),
                     ("per-seed noise", "stage2_noise.csv"), ("training-arm bootstrap", "stage2_bootstrap.csv"),
                     ("Stage 1 comparisons against Grad-CAM", "stage1_bootstrap.csv"))) +
                    ". Bootstrap comparisons are within the explanation pipeline; no cross-pipeline significance test was computed.")
        text.append("#### CW-trained Stage 1")
        text.append(table(["Features", *[a.upper() for a in ATTACKS], "Actual clean FPR"],
                          [[LABEL[m], *[cell(self.cw[m, a]) for a in ATTACKS], pct(self.cw[m, "cw"]["clean_fpr@0.05"])] for m in METHODS]))
        cw_noise = {(r["method"], r["noise"]): r for r in read(self.root / "cw_train/stage1_noise.csv")}
        text.append(table(["CW-trained features", *[n + " FPR" for n in NOISE]],
                          [[LABEL[m], *[pct(cw_noise[m, n]["fpr@0.05"]) for n in NOISE]] for m in METHODS]))
        text.append("CW training uses successful CW training positives and CW development tuning. Sources: " +
                    self.link("CW-trained detection", "cw_train/stage1_detection.csv") + ", " +
                    self.link("CW-trained noise", "cw_train/stage1_noise.csv") + ", " + self.link("CW bootstrap", "cw_train/stage1_bootstrap.csv") + ".")
        text.append("#### Explanation quality, stability and cost")
        statuses = self.validation["status_counts"]
        zeros = self.validation["zero_positive_maps"]
        text.append(table(["Method", "Records", "Zero positive maps", "LIME fallback", "All eliminated", "Nonfinite status"],
                          [[LABEL[m], sum(statuses[m].values()), f"{zeros[m]}/4282 ({pct(zeros[m]/4282)})",
                            statuses[m].get("lime_fallback", 0), statuses[m].get("all_eliminated", 0),
                            statuses[m].get("nonfinite", 0)] for m in ("slice", "belief")]))
        text.append(f"Grad-CAM has {zeros['gradcam']}/4282 ({pct(zeros['gradcam']/4282)}) zero positive feature maps "
                    "on the required explanation inputs and zero prediction mismatches across all stored rows.")
        text.append("Every stored coefficient array is readable, dimension-checked and finite. Zero maps and all-eliminated "
                    "or fallback outputs are retained and reported; they are not silently dropped.")
        text.append(table(["Test variant", "Method", "N", "Zero positive maps", "LIME fallback", "All eliminated", "Mean positive superpixels"],
                          [[r["variant"], LABEL[r["method"]], r["n"], pct(r["zero_positive_map"]),
                            pct(r["lime_fallback"]), pct(r["all_eliminated"]), number(r["mean_positive_superpixels"])]
                           for r in read(self.tables / "explanation_status.csv")
                           if r["partition"] == "test" and r["method"] in ("slice", "belief")]))
        text.append("High zero-map, all-eliminated or fallback rates limit interpretation as consistently informative local "
                    "attributions. The detection results describe this fixed sigma and CUDA precision setting; "
                    "they do not establish how a full-precision rerun would compare.")
        text.append(table(["Panel", "Method", "Inputs", "Repeats", "Zero positive maps", "Sign agreement", "Top-3 Jaccard", "Spearman", "Valid Spearman pairs", "Detector-score SD"],
                          [[r["variant"], LABEL[r["method"]], r["inputs"], r.get("repeats") or "deterministic",
                            pct(r["zero_map_rate"]) if r.get("zero_map_rate") else "—",
                            number(r["sign_agreement"]), number(r["top3_jaccard"]), number(r["spearman"]), r.get("spearman_pairs") or "—", number(r["detector_score_sd"])]
                           for r in read(self.tables / "stability.csv")]))
        text.append("The Grad-CAM stability row is the pipeline's deterministic-control assumption, not a repeated CUDA "
                    "measurement. SLICE and BELIEF are the methods actually recomputed five times per selected input. "
                    "Two empty positive maps receive Jaccard 1; Spearman excludes constant coefficient arrays. "
                    "Interpret stability together with panel zero-map rates and valid Spearman-pair counts.")
        text.append(table(["Panel", "Method", "N", "AOPC deletion mean ± SE"],
                          [[r["variant"], LABEL.get(r["method"], r["method"]), r["n"],
                            f"{significant(r['aopc_del_mean'])} ± {significant(r['aopc_del_se'])}"] for r in read(self.tables / "fidelity_summary.csv")]))
        text.append(table(["Method / timing origin", "Median seconds/input", "Mean seconds/input", "P90 seconds/input", "Median queries", "P90 queries"],
                          [[r["method"] + (" (transferred CPU segmentation)" if r["method"].startswith("quickshift") else " (local CUDA)"),
                            significant(r["median_seconds"]), significant(r["mean_seconds"]),
                            significant(r["p90_seconds"]) if r.get("p90_seconds") else "—", r.get("median_queries") or "—", r.get("p90_queries") or "—"]
                           for r in read(self.tables / "cost.csv")]))
        text.append("SLICE and BELIEF standalone costs each include their shared initial sample, so their columns should not "
                    "be summed as the joint run time. These timers exclude shared context creation and segmentation. "
                    "Grad-CAM's displayed mean/median are unweighted aggregates of 21 shard-level per-image batch times, "
                    "rather than individual-image latency percentiles. Quickshift measurements come from the transferred segmentation "
                    "and are kept separate from the CUDA extraction benchmark. Fidelity uses the saved blur-fill policy "
                    "and five top-positive-superpixel deletion ranks (plus a random-ranking control); it does not "
                    "separately measure deletion of negative evidence and is not a detector robustness metric.")
        text.append("Sources: " + ", ".join(self.link(label, path) for label, path in
                    (("explanation status", "tables/explanation_status.csv"), ("stability", "tables/stability.csv"),
                     ("fidelity summary", "tables/fidelity_summary.csv"), ("per-input fidelity", "fidelity.csv"), ("runtime/query cost", "tables/cost.csv"))) + ".")
        text.append(self.environment_text())
        return "\n\n".join(text)

    def findings(self):
        resnet = {r["attack"]: r for r in read(self.repo / "results/resnet_representation_cifar10_native32/stage1_detection.csv")}
        descriptions = []
        for attack in ATTACKS:
            best = max(METHODS[:5], key=lambda m: float(self.detection[m, attack]["auc"]))
            descriptions.append(f"{attack.upper()}: {LABEL[best]} {number(self.detection[best, attack]['auc'])} "
                                f"versus ResNet pooled {number(resnet[attack]['auc'])}")
        summary = read(self.tables / "stage2_summary.csv")
        index = {(r["method"], r["holdout"], r["arm"]): r for r in summary if r["is_holdout"] == "True"}
        gains = sum(float(index[m, h, "mixture"]["auc_mean"]) > float(index[m, h, "pgd_only"]["auc_mean"])
                    for m in METHODS[:5] for h in ("cw", "patch", "apgd"))
        return ("**Native-32 CIFAR-10 findings.** The highest observed Stage 1 AUC among the five explanation-map feature sets "
                "is " + "; ".join(descriptions) + ". These are descriptive comparisons selected after evaluation; "
                "the detailed tables retain all methods, TPRs and actual clean/noise FPRs. "
                f"Mixture training raises the three-seed mean held-out AUC in {gains}/15 explanation-map × holdout comparisons. "
                "The " + self.link("primary-seed bootstrap intervals", "tables/stage2_bootstrap.csv") + " quantify individual within-pipeline arm comparisons; "
                "they are not cross-pipeline tests or a multiple-comparison-adjusted claim. "
                "The CIFAR-10 classifier has ten outputs, so this result does not inherit Imagenette's out-of-class membership confound.")

    def environment_text(self):
        env = self.environment
        packages = ", ".join(f"{k} {v}" for k, v in env["packages"].items())
        tests = ET.parse(self.root / "pytest.xml").getroot()
        assert not tests.findall(".//failure") and not tests.findall(".//error")
        cases = tests.findall(".//testcase")
        skipped = [c.find("skipped") for c in cases if c.find("skipped") is not None]
        passed = sum(c.find("skipped") is None and c.find("failure") is None and c.find("error") is None for c in cases)
        reference_skips = sum("pinned reference code" in s.get("message", "") for s in skipped)
        skip_detail = "; ".join(sorted({s.get("message", "").removeprefix("Skipped: ") for s in skipped})) or "none"
        reference_note = ("The pinned upstream reference cases also ran locally; " + self.link("reference provenance", "reference_validation.json") +
                          " records source commits, hashes, optional test dependencies and the separate parity run. "
                          if reference_skips == 0 and (self.root / "reference_validation.json").is_file() else "")
        return (f"**Execution and cache decision:** Python {env['python']}, {env['gpu']}, CUDA runtime {env['cuda_runtime']}; {packages}. "
                "All primary Grad-CAM, SLICE/BELIEF and repeat caches were regenerated locally. Perturbation-model queries "
                "use CUDA float16 autocast/channels-last, Grad-CAM and the audit use float32, and regression uses CPU float64. "
                "TF32 is disabled: the default cuDNN TF32 path changed one clean prediction, while full CUDA FP32 agrees with "
                "CPU FP32/FP64 and the saved baseline. The strict zero-mismatch audit was then rerun over every block. "
                f"The transferred {self.validation['transferred_archive_records']} MPS full-precision explanations and partial repeat directory are preserved under "
                "`data/processed/slice_belief_cifar10_native32_fixed1000/archives/mps_transfer_20261123/`; none enters the primary "
                "explanation results or CUDA timing. The handoff stated 200 records; the unchanged checkpoint and original MPS log show 220. "
                "Segmentation and the MPS development-selected sigma 0.3 are reused. "
                "Seeds, splits, attack tensors, explanation budgets, and detector/calibration protocol are unchanged. "
                f"The requested tests passed with {passed} passed and {len(skipped)} skipped ({skip_detail}); "
                "the added CUDA mixed-precision query test passed. "
                + reference_note +
                "Sources: " + self.link("environment and archive hashes", "execution_environment.json") + ", " +
                self.link("numerical diagnosis", "audit_attempts/numerical_diagnosis.json") + ", " +
                self.link("package lock", "requirements-windows-cuda.lock.txt") + ", " + self.link("test transcript", "pytest.txt") + ".")

    def representation(self):
        text = ["The native-32 representations use the identical ResNet-targeted corpus described in the CIFAR-10 setup. "
                "DINOv2 remains an observer, rather than the attacked classifier. The following existing representation "
                "results are preserved beside the newly completed explanation results above. ResNet pooled features "
                "have 2,048 dimensions and DINOv2 CLS features have 384, versus 196/392 for explanation maps; "
                "these are complete-pipeline comparisons rather than dimension-matched ablations."]
        values, noise, stage2, cw = [], [], [], []
        for label, directory, s2directory in (("ResNet pooled", "resnet_representation_cifar10_native32", "resnet_representation_cifar10_native32"),
                                              ("DINOv2", "dinov2_observer_cifar10_native32", "dinov2_observer_cifar10_native32_stage2")):
            det = {r["attack"]: r for r in read(self.repo / "results" / directory / "stage1_detection.csv")}
            values.append([label, *[cell(det[a]) for a in ATTACKS], pct(det["pgd"]["clean_fpr@0.05"])])
            ns = {r["noise"]: r for r in read(self.repo / "results" / directory / "stage1_noise.csv")}
            noise.append([label, *[pct(ns[n]["fpr@0.05"]) for n in NOISE]])
            for r in read(self.repo / "results" / s2directory / "stage2_summary.csv"):
                if r["is_holdout"] == "True":
                    stage2.append([label, r["holdout"].upper(), r["arm"], r["N"], f"{number(r['auc_mean'])} ± {number(r['auc_sd'])}",
                                   pct(r["tpr@0.05_mean"]), pct(r["clean_fpr@0.05_mean"]), pct(r["mean_noise_fpr@0.05_mean"])])
            cd = {r["attack"]: r for r in read(self.repo / "results" / (directory + "_cw_train") / "stage1_detection.csv")}
            cn = {r["noise"]: r for r in read(self.repo / "results" / (directory + "_cw_train") / "stage1_noise.csv")}
            cw.append([label, cell(det["cw"]), cell(cd["cw"]), pct(cd["cw"]["clean_fpr@0.05"]), *[pct(cn[n]["fpr@0.05"]) for n in NOISE]])
        text.extend([table(["Features", *[a.upper() for a in ATTACKS], "Clean FPR"], values),
                     table(["Features", *[n + " FPR" for n in NOISE]], noise),
                     table(["Features", "Holdout", "Arm", "N per class", "AUC mean ± SD", "TPR", "Clean FPR", "Mean noise FPR"], stage2),
                     self.representation_bootstrap(),
                     table(["Features", "PGD-trained CW", "CW-trained CW", "CW-trained clean FPR", *[n + " FPR" for n in NOISE]], cw),
                     "Cells show AUC / TPR at the 5% calibration target. Stage 2 uses three-seed means and sample AUC SD; "
                     "the available paired bootstrap concerns mixture versus matched PGD-only within each representation "
                     "pipeline at seed 42. No paired cross-pipeline significance was computed. Sources: "
                     "[ResNet native-32](results/resnet_representation_cifar10_native32/), "
                     "[DINOv2 native-32](results/dinov2_observer_cifar10_native32/), "
                     "[DINOv2 Stage 2](results/dinov2_observer_cifar10_native32_stage2/), "
                     "[ResNet CW training](results/resnet_representation_cifar10_native32_cw_train/), "
                     "[DINOv2 CW training](results/dinov2_observer_cifar10_native32_cw_train/)."])
        text.append(self.findings())
        return "\n\n".join(text)

    def representation_bootstrap(self):
        values = []
        for label, directory in (("ResNet pooled", "resnet_representation_cifar10_native32"),
                                  ("DINOv2", "dinov2_observer_cifar10_native32_stage2")):
            for r in read(self.repo / "results" / directory / "stage2_bootstrap.csv"):
                values.append([label, r["holdout"].upper(),
                               f"{number(r['auc_gain'])} [{number(r['auc_gain_ci_low'])}, {number(r['auc_gain_ci_high'])}]",
                               f"{pct(r['tpr5_gain'])} [{pct(r['tpr5_gain_ci_low'])}, {pct(r['tpr5_gain_ci_high'])}]"])
        return ("Existing representation comparisons use the primary seed 42 and compare mixture minus refitted "
                "PGD-only within each pipeline, with test-source and calibration bootstrap resampling. "
                "Displayed estimates are bootstrap means, not three-seed means.\n\n" +
                table(["Features", "Holdout", "Bootstrap mean AUC gain [95% interval]", "Bootstrap mean TPR gain [95% interval]"], values) +
                "\n\nSources: [ResNet Stage 2 bootstrap](results/resnet_representation_cifar10_native32/stage2_bootstrap.csv), "
                "[DINOv2 Stage 2 bootstrap](results/dinov2_observer_cifar10_native32_stage2/stage2_bootstrap.csv). "
                "These comparisons do not test representation features against explanation features.")


def update_combined(repo=Path(".")):
    """Preserve the three main sections and all existing Imagenette text."""
    report = Native32Report(repo)
    path = Path(repo) / "COMBINED_EXPERIMENT_REPORT.md"
    text = path.read_text(encoding="utf-8")
    sections = (("setup", "### 1.5 CIFAR-10 native-32: shared setup", report.setup(), "## 2. Explanation-based pipeline"),
                ("explanation", "### 2.6 CIFAR-10 native-32: completed explanations", report.explanation(), "## 3. Representation-based pipeline"),
                ("representation", "### 3.7 CIFAR-10 native-32: representation comparison", report.representation(), None))
    for name, heading, content, before in sections:
        start, end = f"<!-- cifar10-native32-{name}:start -->", f"<!-- cifar10-native32-{name}:end -->"
        block = f"{start}\n{heading}\n\n{content}\n{end}\n\n"
        if start in text:
            a, b = text.index(start), text.index(end) + len(end)
            text = text[:a] + block.rstrip() + text[b:]
        elif before:
            text = text.replace(before, block + before, 1)
        else:
            text = text.rstrip() + "\n\n" + block
    text = text.replace("On this corpus, ResNet representations achieve higher Stage-1 AUC", "On the Imagenette corpus, ResNet representations achieve higher Stage-1 AUC", 1)
    text = text.replace("The new experiment changes the detector's input features to frozen ResNet pooled representations.",
                        "The representation experiment changes the detector's input features to frozen ResNet pooled representations. "
                        "Clearly labeled CIFAR-10 native-32 subsections extend the comparison to the completed explanation study and independent DINOv2 observer.", 1)
    text = text.replace("1. On identical saved inputs, ResNet pooled features outperform", "1. On identical saved Imagenette inputs, ResNet pooled features outperform", 1)
    if "Items 1–4 below summarize Imagenette" not in text:
        text = text.replace("### 3.6 Main findings and next comparison\n\n",
                            "### 3.6 Main findings and next comparison\n\nItems 1–4 below summarize Imagenette; native-32 CIFAR-10 findings follow in §3.7.\n\n", 1)
    text = text.replace("5. A subsequent DINOv2 comparison should preserve these saved tensors, source partitions, original ResNet masks and calibration rules, changing only the observer representation. It should be reported as a new result after execution, rather than inferred from the ResNet results.",
                        "5. The completed native-32 CIFAR-10 study now compares explanations with ResNet pooled features and the independent DINOv2 observer on identical saved tensors, splits, masks and calibration rules (§§1.5, 2.6, 3.7). Imagenette findings above remain specific to that corpus.")
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def update_main(repo=Path(".")):
    report = Native32Report(repo)
    content = ("# CIFAR-10 native-32: completed explanation experiment\n\n"
               "## Shared setup and validation\n\n" + report.setup() +
               "\n\n## Explanation detection, quality and cost\n\n" + report.explanation() +
               "\n\n## Representation comparison\n\n" + report.representation() + "\n")
    content = content.replace("](results/slice_belief_cifar10_native32_fixed1000/", "](")
    content = content.replace("](results/", "](../")
    content = content.replace("](configs/", "](../../configs/")
    (report.root / "MAIN_RESULTS.md").write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    update_main()
    update_combined()
