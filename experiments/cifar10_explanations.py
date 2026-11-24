"""Resume the native-32 CIFAR-10 explanation study without regenerating attacks.

Run from the repository root:
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-corpus/bin/python -m experiments.cifar10_explanations

Every invocation audits the original corpus and classifier before continuing.
Existing segmentation, sigma and explanation checkpoints are reused by the
underlying pipeline. Analysis and the report are regenerated from saved outputs.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from experiments.resnet_representation import Corpus, sha
from experiments.slice_belief import SliceBeliefStudy
from experiments.slice_belief_analysis import SliceBeliefAnalysis

CONFIG = "configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml"
REFERENCE = Path("results/resnet_representation_cifar10_native32")
STAGES = ("segment", "sigma", "explain", "repeats", "fidelity", "analysis", "cw", "report")


def execution_environment(study, config_path):
    """Record extraction provenance and reject mixed-backend active caches."""
    path = study.results / "execution_environment.json"
    precision = "cuda_float16_autocast" if study.device.type == "cuda" and study.config["explainers"]["use_amp"] else "float32"
    identity = {"device": str(study.device), "query_precision": precision,
                "torch": torch.__version__, "config_sha256": sha(Path(config_path)),
                "gpu": torch.cuda.get_device_name(study.device) if study.device.type == "cuda" else None,
                "extraction_sources": {name: sha(Path(name)) for name in
                                       ("experiments/slice_belief.py", "explainers/slice.py",
                                        "explainers/belief.py", "explainers/perturbation.py")}}
    cached = any(any((study.work / name).glob("*.pt")) for name in ("explanations", "repeats"))
    if cached:
        if not path.exists() or json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Active explanation caches lack matching device/precision provenance; archive them before extraction")
        return
    archive = study.work / "archives" / "mps_transfer_20261123"
    metadata = {
        "identity": identity, "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image", "torchattacks")},
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(study.device) if study.device.type == "cuda" else None,
        "prediction_audit_precision": "float32; no autocast",
        "gradcam_precision": "float32; no autocast",
        "regression_precision": "float64 on CPU",
        "query_memory_format": "channels_last",
        "backbone_weights_sha256": sha(Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"),
        "config": study.config,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cache_policy": "All primary explanations, Grad-CAM caches, repeats and fidelity regenerated on this device",
        "transferred_partial_archive": archive.as_posix() if archive.exists() else None,
        "archive_files": {p.relative_to(study.work).as_posix(): sha(p) for p in archive.rglob("*") if p.is_file()},
        "reused": {"sigma_sha256": sha(study.work / "sigma.json"), "sigma": study.sigma(),
                   "segments": {p.name: sha(p) for p in sorted((study.work / "segments").glob("*.npz"))}},
        "timing_scope": "CUDA explanation/Grad-CAM times only; quickshift timings are transferred from the original machine and must be reported separately",
    }
    save_json(path, metadata)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def audit(study):
    """Require exact saved tensors, masks, and predictions before comparisons."""
    corpus = Corpus(study.config)
    expected = json.loads((REFERENCE / "run.json").read_text())["corpus_files"]
    if corpus.fingerprint != expected:
        raise ValueError("Corpus metadata differs from the published native-32 baseline")
    original = json.loads((REFERENCE / "stage2_provenance.json").read_text())["original_feature_provenance"]
    blocks = []
    for key, rows in sorted(corpus.blocks.items()):
        images = study.load_tensor(*key)
        name = corpus.path(key).stem
        digest = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        if digest != original["blocks"][name]["tensor_sha256"]:
            raise ValueError(f"Tensor differs from baseline: {name}")
        predictions, _, _ = study.predict(images)
        mismatches = int(np.count_nonzero(predictions != np.array([r["pred"] for r in rows])))
        blocks.append({"block": name, "rows": len(rows), "sha256": digest,
                       "prediction_mismatches": mismatches})
        study.log(f"audit {name}: {len(rows)} rows, {mismatches} prediction mismatches")
        if mismatches:
            save_json(study.results / "prediction_audit.json", {"status": "failed", "blocks": blocks})
            raise ValueError(f"Classifier differs from baseline on {name}; stopping before explanations")
    save_json(study.results / "prediction_audit.json", {
        "status": "passed", "corpus_files": corpus.fingerprint, "blocks": blocks,
        "head_sha256": sha(Path(study.config["model"]["head_checkpoint"])),
        "device": str(study.device), "torch": torch.__version__,
    })


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report(study):
    """Use only completed tables; no placeholder or inferred detection results."""
    tables = study.results / "tables"
    cw_tables = study.results / "cw_train"
    methods = ("gradcam", "slice", "belief", "slice_signed", "belief_signed", "confidence", "segments")
    attacks = ("pgd", "apgd", "cw", "patch", "fgsm")
    detection = read_rows(tables / "stage1_detection.csv")
    noise = read_rows(tables / "stage1_noise.csv")
    stage2 = read_rows(tables / "stage2_summary.csv")
    cw = read_rows(cw_tables / "stage1_detection.csv")
    percent = lambda v: f"{100 * float(v):.1f}%"
    cell = lambda r: f"{float(r['auc']):.3f} / {percent(r['tpr@0.05'])}"
    index = {(r["method"], r["attack"]): r for r in detection}
    rows = [[m, *[cell(index[m, a]) for a in attacks], percent(index[m, "pgd"]["clean_fpr@0.05"])] for m in methods]
    for name, directory in (("ResNet pooled", REFERENCE), ("DINOv2", Path("results/dinov2_observer_cifar10_native32"))):
        rs = {r["attack"]: r for r in read_rows(directory / "stage1_detection.csv")}
        rows.append([name, *[cell(rs[a]) for a in attacks], percent(rs["pgd"]["clean_fpr@0.05"])])
    text = ["# CIFAR-10 native-32: explanation and representation comparison", "",
            "## 1. Shared setup", "",
            "The 1,000-source corpus and all 21 tensor blocks match the saved ResNet baseline. "
            "Attacks modify native 32×32 pixels before resizing to 224×224. The ImageNet-pretrained "
            "ResNet-50 backbone is frozen and uses a CIFAR-10-trained linear head. "
            "Train/dev/calibration/test contain 350/100/200/350 sources; failed attacks are excluded "
            "from detector positives. FGSM is test-only. Classifier predictions are audited before extraction.", "",
            "SLICE/BELIEF share per-input segmentation and the first perturbation sample. "
            "Positive maps contain 196 features; signed variants contain 392. Logistic-regression C "
            "is selected on development data. Thresholds are frozen using independent clean calibration data. "
            "Stage 2 compares mixtures with refitted PGD-only controls on matched source identities, over three seeds.", "",
            "## 2. Detection results", "",
            "Cells show AUC / TPR at the **5% calibration target**; actual clean test FPR is reported separately.", "",
            markdown_table(["Features", *[a.upper() for a in attacks], "Clean FPR"], rows), "",
            "### Noise false alarms", "",
            markdown_table(["Features", "Noise", "FPR"], [[r["method"], r["noise"], percent(r["fpr@0.05"])] for r in noise if r["method"] in methods]), "",
            "### Stage 2: held-out attacks", "",
            markdown_table(["Features", "Holdout", "Arm", "N per class", "AUC mean", "AUC SD", "TPR", "Clean FPR", "Mean noise FPR"],
                           [[r["method"], r["holdout"], r["arm"], r["N"], f"{float(r['auc_mean']):.3f}", f"{float(r['auc_sd']):.3f}",
                             percent(r["tpr@0.05_mean"]), percent(r["clean_fpr@0.05_mean"]), percent(r["mean_noise_fpr@0.05_mean"])]
                            for r in stage2 if r["is_holdout"] == "True" and r["method"] in methods]), "",
            "### CW-trained detectors", "",
            markdown_table(["Features", "PGD-trained CW", "CW-trained CW", "CW-trained clean FPR"],
                           [[r["method"], cell(index[r["method"], "cw"]), cell(r), percent(r["clean_fpr@0.05"])]
                            for r in cw if r["attack"] == "cw" and r["method"] in methods]), "",
            "## 3. Quality, uncertainty and scope", "",
            "[Tables](tables/) include paired bootstrap comparisons, explanation status, stability, fidelity and runtime. "
            "[CW-trained tables](cw_train/) contain the additional known-attack evaluation. "
            "[Prediction audit](prediction_audit.json) records tensor hashes and local prediction agreement.", "",
            "All attacks target the classifier, not the detector. Stage 1 uses one fitted detector per method; "
            "Stage 2 varies attack allocation over three seeds. Noise conditions have only 100 images each. "
            "The constant class-membership indicator is omitted from these tables because every CIFAR classifier "
            "prediction is one of its ten classes. Feature dimensions differ across methods. "
            "Results alone do not establish significance between explanation and representation pipelines.", ""]
    (study.results / "MAIN_RESULTS.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    study = SliceBeliefStudy(yaml.safe_load(Path(args.config).read_text()))
    # TF32 convolution changed one saved clean prediction on the RTX 4070 Ti.
    # Use full FP32 for the strict classifier audit and Grad-CAM; configured
    # float16 autocast remains enabled for perturbation queries only.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if study.config["dataset"].get("attack_size") != 32 or study.config["dataset"].get("name") != "cifar10":
        raise ValueError("This runner requires the native-32 CIFAR-10 configuration")
    # The OS releases this lock on process exit, including an interrupted run.
    lock = (study.results / "runner.lock").open("a+b")
    if os.name == "nt":
        import ctypes
        import msvcrt
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Could not prevent Windows system sleep")
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = study.results / "experiment_status.json"
    state = {"status": "running", "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
             "device": str(study.device), "config": args.config, "completed_stages": [], "stage": "audit"}
    save_json(status_path, state)
    try:
        audit(study)
        execution_environment(study, args.config)
        state["completed_stages"].append("audit")
        for stage in args.stages:
            state["stage"] = stage
            save_json(status_path, state)
            study.log(f"=== controlled CIFAR-10 stage: {stage} ===")
            if stage == "analysis":
                SliceBeliefAnalysis(study.config).run()
            elif stage == "cw":
                analysis = SliceBeliefAnalysis(study.config, train_attack="cw", tables_dir=study.results / "cw_train")
                analysis.stage1()
                save_json(analysis.tables / "run.json", {"train_attack": "cw", "config": study.config})
            elif stage == "report":
                report(study)
            else:
                getattr(study, "stage_" + stage)()
            state["completed_stages"].append(stage)
        state["status"] = "complete"
        state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S%z")
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        raise
    finally:
        save_json(status_path, state)
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == "__main__":
    main()
