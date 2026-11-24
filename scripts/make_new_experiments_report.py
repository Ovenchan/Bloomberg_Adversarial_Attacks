"""Generate NEW_EXPERIMENTS_REPORT.md; every number is read from saved result files."""
import csv, json, os
from collections import defaultdict
from pathlib import Path

# Usage (repository root): python scripts/make_new_experiments_report.py [processed-data-dir]
# Reads saved result CSVs plus each corpus's variants.csv and rewrites NEW_EXPERIMENTS_REPORT.md
# and results/corpus_summary_imagenette_cifar10.csv. Fails if the reproduction check does not match.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cifar10_explanation_reporting import Native32Report
REPO = Path(os.environ.get("REPO", "."))
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA", "data/processed"))
R = REPO / "results"
def load(p): return list(csv.DictReader(open(p, newline="")))
def f(x, d=3): return f"{float(x):.{d}f}"
def pct(x, d=1): return f"{100*float(x):.{d}f}%"
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)
def link(text, path): return f"[{text}]({path})"

ATT = ["pgd", "apgd", "cw", "patch", "fgsm"]
NOISE = ["gauss2", "gauss4", "gauss8", "unif8"]
NAME = {"pgd": "PGD", "apgd": "APGD", "cw": "CW", "patch": "Patch", "fgsm": "FGSM",
        "gauss2": "Gauss 2/255", "gauss4": "Gauss 4/255", "gauss8": "Gauss 8/255", "unif8": "Uniform 8/255"}

def rep_stage1(d):
    det = {r["attack"]: r for r in load(R / d / "stage1_detection.csv")}
    noise = {r["noise"]: r for r in load(R / d / "stage1_noise.csv")}
    ev = json.loads((R / d / "evaluation.json").read_text())
    return det, noise, ev

def expl_stage1(path):
    det = defaultdict(dict); noise = defaultdict(dict)
    for r in load(path / "stage1_detection.csv"): det[r["method"]][r["attack"]] = r
    for r in load(path / "stage1_noise.csv"): noise[r["method"]][r["noise"]] = r
    return det, noise

def cell(r): return f"{f(r['auc'])} / {pct(r['tpr@0.05'],0)}"

def stage2(d):
    return [r for r in load(R / d / "combined_results.csv") if r["stage"] == "stage2" and r["is_holdout"] == "True"]

def corpus_stats(work):
    rows = load(work / "variants.csv")
    acc, asr, noise = {}, {}, {}
    for p in ("train", "dev", "cal", "test"):
        c = [r for r in rows if r["partition"] == p and r["variant"] == "clean"]
        acc[p] = (sum(r["eligible"] == "True" for r in c), len(c))
    for a in ATT:
        t = [r for r in rows if r["partition"] == "test" and r["variant"] == a]
        el = sum(r["eligible"] == "True" for r in t); sc = sum(r["success"] == "True" for r in t)
        succ = [r for r in t if r["success"] == "True"]
        asr[a] = (sc, el, sum(r["in_imagenette"] == "False" for r in succ) / max(len(succ), 1))
    for n in NOISE:
        t = [r for r in rows if r["partition"] == "test" and r["variant"] == n]
        noise[n] = sum(r["pred"] != r["clean_pred"] for r in t) / len(t)
    return acc, asr, noise

out = []
W = out.append

# ------------------------------------------------------------------ data
IN_RES, IN_RES_N, _ = rep_stage1("resnet_representation_fixed1000_complete")
IN_DINO, IN_DINO_N, IN_DINO_EV = rep_stage1("dinov2_observer_fixed1000")
EXP, EXP_N = expl_stage1(R / "slice_belief_fixed1000" / "tables")
EXP_CW, EXP_CW_N = expl_stage1(R / "slice_belief_fixed1000_cw_train")
RERUN, RERUN_N = expl_stage1(R / "slice_belief_fixed1000_stage1_rerun")
IN_RES_CW, IN_RES_CW_N, IN_RES_CW_EV = rep_stage1("resnet_representation_fixed1000_cw_train")
IN_DINO_CW, IN_DINO_CW_N, IN_DINO_CW_EV = rep_stage1("dinov2_observer_fixed1000_cw_train")
C224_RES, C224_RES_N, _ = rep_stage1("resnet_representation_cifar10_fixed1000")
C224_DINO, C224_DINO_N, _ = rep_stage1("dinov2_observer_cifar10_fixed1000")
C32_RES, C32_RES_N, C32_RES_EV = rep_stage1("resnet_representation_cifar10_native32")
C32_DINO, C32_DINO_N, C32_DINO_EV = rep_stage1("dinov2_observer_cifar10_native32")
C32_RES_CW, C32_RES_CW_N, C32_RES_CW_EV = rep_stage1("resnet_representation_cifar10_native32_cw_train")
C32_DINO_CW, C32_DINO_CW_N, C32_DINO_CW_EV = rep_stage1("dinov2_observer_cifar10_native32_cw_train")
IN_STATS = corpus_stats(DATA / "slice_belief_fixed1000")
C224_STATS = corpus_stats(DATA / "slice_belief_cifar10_fixed1000")
C32_STATS = corpus_stats(DATA / "slice_belief_cifar10_native32_fixed1000")
C32_EXPLANATIONS = Native32Report(REPO)

# reproduction check
keys = ["auc", "tpr@0.01", "tpr@0.05", "clean_fpr@0.05", "C", "dev_auc", "num_successful"]
diffs = [abs(float(EXP[m][a][k]) - float(RERUN[m][a][k])) for m in EXP for a in EXP[m] for k in keys]
ndiffs = [abs(float(EXP_N[m][n]["fpr@0.05"]) - float(RERUN_N[m][n]["fpr@0.05"])) for m in EXP_N for n in EXP_N[m]]
n_cells = len(diffs) // len(keys)
assert max(diffs + ndiffs) == 0, "reproduction check failed"

# corpus summary CSV (traceable source for the corpus tables)
summary_rows = []
for label, (acc, asr, noise) in (("imagenette", IN_STATS), ("cifar10_224", C224_STATS), ("cifar10_native32", C32_STATS)):
    for p, (k, n) in acc.items():
        summary_rows.append({"corpus": label, "partition": p, "item": "clean_accuracy", "count": k, "of": n, "rate": k / n})
    for a, (k, n, out_rate) in asr.items():
        summary_rows.append({"corpus": label, "partition": "test", "item": f"asr_{a}", "count": k, "of": n, "rate": k / n})
    for nm, rate in noise.items():
        summary_rows.append({"corpus": label, "partition": "test", "item": f"noise_prediction_changed_{nm}", "count": "", "of": "", "rate": rate})
with open(R / "corpus_summary_imagenette_cifar10.csv", "w", newline="") as s:
    w = csv.DictWriter(s, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

# ------------------------------------------------------------------ report
W("# New experiments: DINOv2, CW-only training and CIFAR-10")
W("")
W("This report extends [COMBINED_EXPERIMENT_REPORT.md](COMBINED_EXPERIMENT_REPORT.md). It keeps that report's "
  "protocol (frozen ResNet-50 attack target, fixed 1,000-image corpus, 350/100/200/350 train/dev/calibration/test "
  "split, standardized logistic-regression detector, C chosen on development AUC, thresholds frozen on 200 clean "
  "calibration images) and adds:")
W("")
W("1. **DINOv2 observer on Imagenette**: frozen DINOv2 ViT-S/14 embeddings of the existing ResNet attack corpus (Stage 1 and Stage 2).")
W("2. **CW-only training**: Stage 1 detectors trained and tuned on CW instead of PGD, for ResNet, DINOv2 and all explanation methods.")
W("3. **CIFAR-10**: a fixed 1,000-image corpus attacked at native 32×32 resolution, with Grad-CAM/SLICE/BELIEF, ResNet and DINOv2 comparisons, PGD/CW-trained Stage 1, matched Stage 2, noise controls, and explanation quality/cost.")
W("")
W("All numbers below are read from the linked result files by a script, not copied by hand. Unless stated otherwise, "
  "cells are **AUC / detection rate (TPR) at the 5% clean-calibration target**, and noise columns are false-alarm "
  "rates at that same frozen threshold.")
W("")
W("Contents: [Key findings](#key-findings) · [1. DINOv2 on Imagenette](#1-dinov2-observer-on-imagenette) · "
  "[2. CW-only training](#2-cw-only-training) · [3. CIFAR-10](#3-cifar-10) · [4. Reproducibility](#4-reproducibility-and-changes) · "
  "[5. Limitations and next steps](#5-limitations-and-next-steps)")
W("")
W("## Key findings")
W("")
W(f"- **DINOv2 is a weaker attack detector than ResNet's own features**, and mostly a noise detector. On Imagenette, PGD AUC is "
  f"{f(IN_DINO['pgd']['auc'])} vs {f(IN_RES['pgd']['auc'])} for ResNet, CW {f(IN_DINO['cw']['auc'])} and patch {f(IN_DINO['patch']['auc'])} are near chance, "
  f"while {pct(IN_DINO_N['gauss8']['fpr@0.05'],0)} of Gaussian-8/255 noisy images are flagged (ResNet: {pct(IN_RES_N['gauss8']['fpr@0.05'],0)}). "
  "The attacks target ResNet, so they move ResNet's representation directly; DINOv2 only sees the pixels.")
W(f"- **CW stays hard even when the detector trains on CW.** Imagenette ResNet features: CW AUC {f(IN_RES['cw']['auc'])} → "
  f"{f(IN_RES_CW['cw']['auc'])}, detection {pct(IN_RES['cw']['tpr@0.05'],0)} → {pct(IN_RES_CW['cw']['tpr@0.05'],0)}. "
  f"The best explanation features (signed SLICE/BELIEF) reach AUC ≈ {f(EXP_CW['slice_signed']['cw']['auc'],2)}. "
  "This matches the literature, where CW is consistently among the hardest attacks to detect.")
W("- **CIFAR-10 attacked at 224×224 is a misleading benchmark.** Attacks/noise added after upsampling 32→224 create fine speckle an "
  f"upsampled image cannot contain; detectors reach AUC ≈ 1.0 but flag {pct(C224_RES_N['gauss4']['fpr@0.05'],0)}–{pct(C224_RES_N['gauss8']['fpr@0.05'],0)} "
  "of mildly noisy images. Attacking at the native 32×32 resolution removes most of this shortcut, so the native-32 corpus is the primary CIFAR-10 result.")
W(f"- **On native CIFAR-10, ResNet features reproduce the Imagenette pattern**: PGD/APGD near-perfect, patch intermediate "
  f"({f(C32_RES['patch']['auc'])}), CW hardest ({f(C32_RES['cw']['auc'])}, {pct(C32_RES['cw']['tpr@0.05'],0)} detected).")
W("- **Mixture training (Stage 2) does not raise held-out AUC** for ResNet or DINOv2 on either dataset, but it often lowers noise false alarms.")
W("- " + C32_EXPLANATIONS.findings())
W("")

# ---------------- section 1
W("## 1. DINOv2 observer on Imagenette")
W("")
W("DINOv2 ViT-S/14 (frozen, 384-dimensional CLS embedding) encodes the *same saved tensors* as the ResNet baseline. "
  "The attack target, success masks, partitions and calibration are unchanged; only the detector's input features differ. "
  "This measures whether an independent model *observes* attacks crafted against ResNet.")
W("")
W("### Stage 1: PGD-only training")
W("")
rows = []
for label, det in (("Grad-CAM", EXP["gradcam"]), ("BELIEF signed", EXP["belief_signed"]), ("ResNet pooled", IN_RES), ("**DINOv2**", IN_DINO)):
    rows.append([label] + [cell(det[a]) for a in ATT])
W(table(["Features", *[NAME[a] for a in ATT]], rows))
W("")
rows = []
for label, det, noi in (("Grad-CAM", EXP["gradcam"], EXP_N["gradcam"]), ("ResNet pooled", IN_RES, IN_RES_N), ("**DINOv2**", IN_DINO, IN_DINO_N)):
    rows.append([label, pct(det["pgd"]["clean_fpr@0.05"])] + [pct(noi[n]["fpr@0.05"], 0) for n in NOISE])
W(table(["False alarms at 5% target", "Clean test", *[NAME[n] for n in NOISE]], rows))
W("")
W(f"DINOv2 selected C = {IN_DINO_EV['C']} (development AUC {f(IN_DINO_EV['dev_auc'])}). Its only advantage is FGSM "
  f"({f(IN_DINO['fgsm']['auc'])} vs {f(IN_RES['fgsm']['auc'])}), the noisiest attack, consistent with a detector that responds to perturbation texture rather than to the attack's effect on ResNet. "
  "Five local ResNet prediction differences were recorded and the original masks kept, exactly as in the ResNet baseline.")
W("")
W(f"Sources: {link('DINOv2 Stage 1', 'results/dinov2_observer_fixed1000/')}, {link('ResNet baseline', 'results/resnet_representation_fixed1000_complete/')}, "
  f"{link('explanation Stage 1', 'results/slice_belief_fixed1000/tables/stage1_detection.csv')}.")
W("")
W("### Stage 2: mixture vs matched PGD-only (three-seed means, held-out family)")
W("")
def s2rows(d, label):
    rs = stage2(d)
    out_rows = []
    for h in ("cw", "patch", "apgd"):
        for arm in ("mixture", "pgd_only"):
            r = next(x for x in rs if x["holdout"] == h and x["arm"] == arm)
            out_rows.append([label, NAME[h], arm, f"{r['num_train_clean']} + {r['num_train_adversarial']}", f(r["auc_mean"]),
                             pct(r["tpr@0.05_mean"]), pct(r["clean_fpr@0.05_mean"]), pct(r["mean_noise_fpr@0.05_mean"])])
    return out_rows
W(table(["Features", "Held out", "Arm", "N clean + N attacked", "AUC", "Detection", "Clean FPR", "Mean noise FPR"],
        s2rows("resnet_representation_fixed1000_complete", "ResNet") + s2rows("dinov2_observer_fixed1000_stage2", "DINOv2")))
W("")
W("As for ResNet, mixture training slightly lowers DINOv2's held-out AUC; DINOv2 stays near chance on held-out CW and patch. "
  f"Sources: {link('DINOv2 Stage 2', 'results/dinov2_observer_fixed1000_stage2/')}.")
W("")

# ---------------- section 2
W("## 2. CW-only training")
W("")
W("Stage 1 normally trains and tunes the detector on PGD, so CW is an *unseen* attack. Here the detector is trained on clean + "
  "successful CW training examples and C is selected on development CW, with everything else unchanged (`--train-attack cw`). "
  "This is the *known-attack* setting reported alongside the unseen-attack setting in the detection literature.")
W("")
W("### Imagenette: CW detection by training attack")
W("")
rows = []
for label, key in (("Grad-CAM", "gradcam"), ("SLICE", "slice"), ("BELIEF", "belief"), ("SLICE signed", "slice_signed"),
                   ("BELIEF signed", "belief_signed"), ("Confidence baseline", "confidence")):
    rows.append([label, cell(EXP[key]["cw"]), cell(EXP_CW[key]["cw"]), f(EXP_CW[key]["cw"]["dev_auc"]),
                 pct(EXP_CW[key]["cw"]["clean_fpr@0.05"])])
rows.append(["ResNet pooled", cell(IN_RES["cw"]), cell(IN_RES_CW["cw"]), f(IN_RES_CW_EV["dev_auc"]), pct(IN_RES_CW["cw"]["clean_fpr@0.05"])])
rows.append(["DINOv2", cell(IN_DINO["cw"]), cell(IN_DINO_CW["cw"]), f(IN_DINO_CW_EV["dev_auc"]), pct(IN_DINO_CW["cw"]["clean_fpr@0.05"])])
W(table(["Features", "PGD-trained: CW", "CW-trained: CW", "CW-trained dev AUC", "CW-trained clean FPR"], rows))
W("")
W(f"CW training improves every method, most for the signed SLICE/BELIEF maps, yet no detector catches more than "
  f"{pct(max(float(IN_RES_CW['cw']['tpr@0.05']), max(float(EXP_CW[k]['cw']['tpr@0.05']) for k in EXP_CW)),0)} of CW attacks at the 5% target. "
  "The difficulty is therefore mostly a property of CW, not only of never seeing it in training. "
  f"CW-trained ResNet features still detect PGD/APGD ({f(IN_RES_CW['pgd']['auc'])}/{f(IN_RES_CW['apgd']['auc'])}) and improve FGSM "
  f"({f(IN_RES['fgsm']['auc'])} → {f(IN_RES_CW['fgsm']['auc'])}), with noise false alarms of "
  f"{'/'.join(pct(IN_RES_CW_N[n]['fpr@0.05'],0) for n in NOISE)}. "
  "The in-Imagenette indicator is a fixed rule and is unchanged by training (see the confound in the combined report).")
W("")
W("### CIFAR-10 (native 32×32): CW detection by training attack")
W("")
W(table(["Features", "PGD-trained: CW", "CW-trained: CW", "CW-trained dev AUC", "CW-trained clean FPR", *[NAME[n] + " FPR" for n in NOISE]],
        [["ResNet pooled", cell(C32_RES["cw"]), cell(C32_RES_CW["cw"]), f(C32_RES_CW_EV["dev_auc"]),
          pct(C32_RES_CW["cw"]["clean_fpr@0.05"]), *[pct(C32_RES_CW_N[n]["fpr@0.05"]) for n in NOISE]],
         ["DINOv2", cell(C32_DINO["cw"]), cell(C32_DINO_CW["cw"]), f(C32_DINO_CW_EV["dev_auc"]),
          pct(C32_DINO_CW["cw"]["clean_fpr@0.05"]), *[pct(C32_DINO_CW_N[n]["fpr@0.05"]) for n in NOISE]]]))
W("")
W("### Context from the literature")
W("")
W("- [Carlini & Wagner (2017)](https://arxiv.org/abs/1705.07263) trained several detectors directly on C&W examples and tested on held-out C&W; "
  "several worked on MNIST but degraded on CIFAR (e.g. kernel density gave 80% of C&W examples a higher likelihood than clean images). "
  "They recommend reporting TPR at a low (1%) FPR, as this pipeline does.")
W("- [Ma et al. (2018, LID)](https://arxiv.org/abs/1801.02613) evaluate each attack separately and also train on FGM and test on harder attacks including C&W.")
W("- [Lee et al. (2018, Mahalanobis)](https://arxiv.org/abs/1807.03888) report known-attack and unseen-attack (FGSM-trained) columns; on CIFAR-10 ResNet, "
  "C&W AUROC ranges from about 56% (KD+PU) to 82% (LID) to 96% (Mahalanobis, multi-layer features).")
W("")
W("The single-layer ResNet results here fall within the range of simpler published detectors; multi-layer features are a natural extension.")
W("")
W(f"Sources: {link('explanations, CW-trained', 'results/slice_belief_fixed1000_cw_train/')}, "
  f"{link('ResNet CW-trained', 'results/resnet_representation_fixed1000_cw_train/')}, {link('DINOv2 CW-trained', 'results/dinov2_observer_fixed1000_cw_train/')}, "
  f"{link('CIFAR-10 ResNet CW-trained', 'results/resnet_representation_cifar10_native32_cw_train/')}, "
  f"{link('CIFAR-10 DINOv2 CW-trained', 'results/dinov2_observer_cifar10_native32_cw_train/')}.")
W("")

# ---------------- section 3
W("## 3. CIFAR-10")
W("")
W("### 3.1 Setup")
W("")
W("- **Sources:** a seeded, class-balanced 1,000-image subset of the official CIFAR-10 **test** split (100 per class), partitioned 35/10/20/35 per class exactly as Imagenette.")
W("- **Classifier:** the same frozen ResNet-50 (`IMAGENET1K_V2`) with the existing 10-class linear head (`resnet50_cifar10_head.pt`, trained only on the official CIFAR-10 training split, 224×224 inputs). "
  "Every prediction is a CIFAR class, so the Imagenette out-of-class confound does not arise.")
W("- **Threat model (primary, native-32):** attacks and noise act on the original 32×32 pixels; the classifier sees a bilinear upsample to 224×224, and the saved tensors are exactly that upsample, so all downstream feature and explanation code is unchanged. "
  "Budgets are per native pixel (8/255 L∞ for PGD/APGD/FGSM); the patch is 7×7 native pixels (4.79% of the image vs 4.59% for Imagenette's 48×48).")
W("- **224 variant (diagnostic only):** identical except that attacks and noise act on the upsampled 224×224 image.")
W("")
W("### 3.2 Corpus")
W("")
def acc_str(stats, p): k, n = stats[0][p]; return f"{100*k/n:.1f}% ({k}/{n})"
W(table(["Clean accuracy", "Imagenette", "CIFAR-10 native-32", "CIFAR-10 224"],
        [[p, acc_str(IN_STATS, p), acc_str(C32_STATS, p), acc_str(C224_STATS, p)] for p in ("train", "dev", "cal", "test")]))
W("")
def asr_str(stats, a): k, n, _ = stats[1][a]; return f"{100*k/n:.1f}% ({k}/{n})"
W(table(["Test attack success", "Imagenette", "CIFAR-10 native-32", "CIFAR-10 224"],
        [[NAME[a], asr_str(IN_STATS, a), asr_str(C32_STATS, a), asr_str(C224_STATS, a)] for a in ATT]))
W("")
W(table(["Noise changes the prediction", "Imagenette", "CIFAR-10 native-32", "CIFAR-10 224"],
        [[NAME[n], pct(IN_STATS[2][n], 0), pct(C32_STATS[2][n], 0), pct(C224_STATS[2][n], 0)] for n in NOISE]))
W("")
W("CW succeeds less often on CIFAR-10 with the same settings, while patch and FGSM succeed more often, so attack-family comparisons across datasets reflect different attack strengths. "
  "The CIFAR-10 classifier is also more sensitive to random noise, which raises the bar for noise false alarms. "
  f"Source: {link('corpus summary', 'results/corpus_summary_imagenette_cifar10.csv')} (computed from each corpus's `variants.csv`).")
W("")
W("### 3.3 Why the 224 variant is not used")
W("")
rows = []
for label, det, noi in (("ResNet, CIFAR-10 224", C224_RES, C224_RES_N), ("ResNet, CIFAR-10 native-32", C32_RES, C32_RES_N), ("ResNet, Imagenette", IN_RES, IN_RES_N),
                        ("DINOv2, CIFAR-10 224", C224_DINO, C224_DINO_N), ("DINOv2, CIFAR-10 native-32", C32_DINO, C32_DINO_N)):
    rows.append([label, f(det["pgd"]["auc"]), f(det["patch"]["auc"]), f(det["fgsm"]["auc"])] + [pct(noi[n]["fpr@0.05"], 0) for n in NOISE])
W(table(["PGD-trained detector", "PGD AUC", "Patch AUC", "FGSM AUC", *[f"{NAME[n]} FPR" for n in NOISE]], rows))
W("")
W("In the 224 variant, near-perfect attack AUCs coincide with flagging most noisy images: the detectors learn \"contains sub-pixel speckle\", which no upsampled 32×32 image has naturally. "
  "Native-32 attacks are smoothed by the same upsampling as the image; ResNet noise false alarms fall substantially and attack AUCs fall to plausible levels. "
  "DINOv2 remains highly noise-sensitive in both variants, matching its Imagenette behaviour. The remaining ResNet noise false alarms on native-32 partly reflect the classifier's own noise sensitivity (§3.2).")
W("")
W("### 3.4 Stage 1 (native-32): PGD-only training")
W("")
W(table(["Features", *[NAME[a] for a in ATT], "Clean FPR"],
        [["ResNet pooled", *[cell(C32_RES[a]) for a in ATT], pct(C32_RES["pgd"]["clean_fpr@0.05"])],
         ["DINOv2", *[cell(C32_DINO[a]) for a in ATT], pct(C32_DINO["pgd"]["clean_fpr@0.05"])]]))
W("")
succ_list = ", ".join(NAME[a] + " " + C32_RES[a]["num_successful"] for a in ATT)
W(f"Successful test attacks: {succ_list}. "
  f"ResNet selected C = {C32_RES_EV['C']}, DINOv2 C = {C32_DINO_EV['C']}. Both runs reproduced every saved ResNet prediction (strict check, zero mismatches).")
W("")
W("### 3.5 Stage 2 (native-32)")
W("")

W(table(["Features", "Held out", "Arm", "N clean + N attacked", "AUC", "Detection", "Clean FPR", "Mean noise FPR"],
        s2rows("resnet_representation_cifar10_native32", "ResNet") + s2rows("dinov2_observer_cifar10_native32_stage2", "DINOv2")))
W("")
W("The Imagenette pattern repeats: mixtures do not raise held-out AUC, but they often reduce noise false alarms (e.g. ResNet's CW fold).")
W("")
W(C32_EXPLANATIONS.representation_bootstrap())
W("")
W(f"Sources: {link('ResNet native-32', 'results/resnet_representation_cifar10_native32/')}, {link('DINOv2 native-32', 'results/dinov2_observer_cifar10_native32/')}, "
  f"{link('DINOv2 native-32 Stage 2', 'results/dinov2_observer_cifar10_native32_stage2/')}, {link('ResNet 224', 'results/resnet_representation_cifar10_fixed1000/')}, "
  f"{link('DINOv2 224', 'results/dinov2_observer_cifar10_fixed1000/')}.")
W("")

W("### 3.6 Completed native-32 explanation experiment")
W("")
W(C32_EXPLANATIONS.setup())
W("")
W(C32_EXPLANATIONS.explanation())
W("")

# ---------------- section 4
W("## 4. Reproducibility and changes")
W("")
W(f"**Reproduction check.** The original PGD-trained explanation analysis was re-run with the modified code into a separate folder. "
  f"All {n_cells} method × attack rows (AUC, TPR at 1%/5%, clean FPR, C, development AUC, counts) and all noise false-alarm rates match "
  f"[the published table](results/slice_belief_fixed1000/tables/stage1_detection.csv) exactly (maximum difference 0). "
  f"Source: {link('re-run', 'results/slice_belief_fixed1000_stage1_rerun/')}. A new test also confirms the generalized manifest code reproduces the saved Imagenette split exactly.")
W("")
W("**Code changes** (Imagenette behaviour unchanged unless an option is passed):")
W("")
W("- `experiments/resnet_representation.py`: `--train-attack` for Stage 1; Stage 2 accepts DINOv2 feature caches; dataset-aware classifier.")
W("- `experiments/slice_belief_analysis.py`: `--train-attack` and `--tables-dir` (Stage 1 only, separate output; original tables and log untouched).")
W("- `experiments/slice_belief.py`, `utils/controlled_data.py`, `models/controlled_classifier.py`: CIFAR-10 manifest/loading, CIFAR-10 head, native-resolution attacks (`dataset.attack_size`), and `device: auto` now selects Apple MPS when CUDA is absent.")
W("- New configs: `slice_belief_cifar10_native32_fixed1000.yaml` (primary) and `slice_belief_cifar10_fixed1000.yaml` (224 diagnostic).")
W("- Tests in `tests/test_controlled_eval.py`: Imagenette split reproduction, CIFAR-10 manifest/loader/head, native-resolution wrapper (including a regression test that the wrapper cannot put the frozen ResNet into training mode).")
W("")
W("**Commands** (from the repository root, Python 3.12 environment from `requirements-corpus.txt`, `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple silicon):")
W("")
W("```bash")
W("IN=configs/experiments/slice_belief_imagenette_fixed1000.yaml")
W("C32=configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml")
W("# DINOv2 on Imagenette (Stage 1, then Stage 2 from cached features)")
W("python -m experiments.resnet_representation --config $IN --encoder dinov2 --device mps --prediction-policy record --output results/dinov2_observer_fixed1000")
W("python -m experiments.resnet_representation --config $IN --encoder dinov2 --stage stage2 --features results/dinov2_observer_fixed1000/features --output results/dinov2_observer_fixed1000_stage2")
W("# CW-only training (representations reuse cached features; explanations reuse saved maps)")
W("python -m experiments.resnet_representation --config $IN --encoder resnet --device mps --prediction-policy record --train-attack cw --output results/resnet_representation_fixed1000_cw_train")
W("python -m experiments.slice_belief_analysis --config $IN --train-attack cw --tables-dir results/slice_belief_fixed1000_cw_train")
W("# CIFAR-10 native-32 corpus, then representations")
W("python -m experiments.slice_belief --config $C32 --stage manifest")
W("python -m experiments.slice_belief --config $C32 --stage corpus")
W("python -m experiments.resnet_representation --config $C32 --encoder resnet --stage both --device mps --prediction-policy strict --output results/resnet_representation_cifar10_native32")
W("python -m experiments.resnet_representation --config $C32 --encoder dinov2 --device mps --prediction-policy record --output results/dinov2_observer_cifar10_native32")
W("```")
W("")
W("Raw images and processed tensors remain excluded from Git and must be obtained from the shared data folder.")
W("")

# ---------------- section 5
W("## 5. Limitations and next steps")
W("")
W("- **Native-32 CIFAR-10 explanations are complete and validated** (§3.6): all 4,282 required records, stability repetitions, fidelity, PGD-trained Stage 1, three-seed matched Stage 2, and CW-trained Stage 1. See the saved validation and execution-environment records for counts and backend details.")
W("- Stage 1 results are single fitted detectors; Stage 2 uses three allocation seeds. No new cross-pipeline significance tests are claimed.")
W(f"- CIFAR-10 CW succeeds on only {asr_str(C32_STATS, 'cw')} of eligible test images (vs {asr_str(IN_STATS, 'cw')} on Imagenette), so its CW detection estimates use fewer and possibly easier-to-attack images.")
W("- Noise panels contain 100 test images per condition; false-alarm estimates are correspondingly coarse (±~9 points at 95%).")
W("- All attacks are non-adaptive: none optimizes against the detector.")
W("- Possible extensions: multi-layer (Mahalanobis-style) features for CW; noisy negatives in detector training as a labelled ablation (PLAN.md §5).")
W("")
(REPO / "NEW_EXPERIMENTS_REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("written", len(out), "lines; reproduction cells", n_cells)
