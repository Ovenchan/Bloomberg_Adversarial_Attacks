# New experiments: DINOv2, CW-only training and CIFAR-10

This report extends [COMBINED_EXPERIMENT_REPORT.md](COMBINED_EXPERIMENT_REPORT.md). It keeps that report's protocol (frozen ResNet-50 attack target, fixed 1,000-image corpus, 350/100/200/350 train/dev/calibration/test split, standardized logistic-regression detector, C chosen on development AUC, thresholds frozen on 200 clean calibration images) and adds:

1. **DINOv2 observer on Imagenette**: frozen DINOv2 ViT-S/14 embeddings of the existing ResNet attack corpus (Stage 1 and Stage 2).
2. **CW-only training**: Stage 1 detectors trained and tuned on CW instead of PGD, for ResNet, DINOv2 and all explanation methods.
3. **CIFAR-10**: a fixed 1,000-image CIFAR-10 corpus attacked at the native 32×32 resolution, with ResNet and DINOv2 Stage 1 and Stage 2.

All numbers below are read from the linked result files by a script, not copied by hand. Unless stated otherwise, cells are **AUC / detection rate (TPR) at the 5% clean-calibration target**, and noise columns are false-alarm rates at that same frozen threshold.

Contents: [Key findings](#key-findings) · [1. DINOv2 on Imagenette](#1-dinov2-observer-on-imagenette) · [2. CW-only training](#2-cw-only-training) · [3. CIFAR-10](#3-cifar-10) · [4. Reproducibility](#4-reproducibility-and-changes) · [5. Limitations and next steps](#5-limitations-and-next-steps)

## Key findings

- **DINOv2 is a weaker attack detector than ResNet's own features**, and mostly a noise detector. On Imagenette, PGD AUC is 0.927 vs 0.999 for ResNet, CW 0.585 and patch 0.554 are near chance, while 48% of Gaussian-8/255 noisy images are flagged (ResNet: 13%). The attacks target ResNet, so they move ResNet's representation directly; DINOv2 only sees the pixels.
- **CW stays hard even when the detector trains on CW.** Imagenette ResNet features: CW AUC 0.726 → 0.826, detection 24% → 28%. The best explanation features (signed SLICE/BELIEF) reach AUC ≈ 0.71. This matches the literature, where CW is consistently among the hardest attacks to detect.
- **CIFAR-10 attacked at 224×224 is a misleading benchmark.** Attacks/noise added after upsampling 32→224 create fine speckle an upsampled image cannot contain; detectors reach AUC ≈ 1.0 but flag 87%–99% of mildly noisy images. Attacking at the native 32×32 resolution removes most of this shortcut, so the native-32 corpus is the primary CIFAR-10 result.
- **On native CIFAR-10, ResNet features reproduce the Imagenette pattern**: PGD/APGD near-perfect, patch intermediate (0.872), CW hardest (0.645, 11% detected).
- **Mixture training (Stage 2) does not raise held-out AUC** for ResNet or DINOv2 on either dataset, but it often lowers noise false alarms.

## 1. DINOv2 observer on Imagenette

DINOv2 ViT-S/14 (frozen, 384-dimensional CLS embedding) encodes the *same saved tensors* as the ResNet baseline. The attack target, success masks, partitions and calibration are unchanged; only the detector's input features differ. This measures whether an independent model *observes* attacks crafted against ResNet.

### Stage 1: PGD-only training

| Features | PGD | APGD | CW | Patch | FGSM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 0.944 / 74% | 0.934 / 75% | 0.591 / 16% | 0.614 / 5% | 0.530 / 7% |
| BELIEF signed | 0.894 / 71% | 0.914 / 75% | 0.596 / 9% | 0.772 / 30% | 0.572 / 5% |
| ResNet pooled | 0.999 / 100% | 0.999 / 100% | 0.726 / 24% | 0.945 / 73% | 0.892 / 54% |
| **DINOv2** | 0.927 / 68% | 0.932 / 70% | 0.585 / 8% | 0.554 / 7% | 0.963 / 84% |

| False alarms at 5% target | Clean test | Gauss 2/255 | Gauss 4/255 | Gauss 8/255 | Uniform 8/255 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 5.1% | 6% | 7% | 6% | 7% |
| ResNet pooled | 6.6% | 7% | 9% | 13% | 10% |
| **DINOv2** | 5.1% | 14% | 27% | 48% | 29% |

DINOv2 selected C = 0.01 (development AUC 0.922). Its only advantage is FGSM (0.963 vs 0.892), the noisiest attack, consistent with a detector that responds to perturbation texture rather than to the attack's effect on ResNet. Five local ResNet prediction differences were recorded and the original masks kept, exactly as in the ResNet baseline.

Sources: [DINOv2 Stage 1](results/dinov2_observer_fixed1000/), [ResNet baseline](results/resnet_representation_fixed1000_complete/), [explanation Stage 1](results/slice_belief_fixed1000/tables/stage1_detection.csv).

### Stage 2: mixture vs matched PGD-only (three-seed means, held-out family)

| Features | Held out | Arm | N clean + N attacked | AUC | Detection | Clean FPR | Mean noise FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet | CW | mixture | 195 + 195 | 0.672 | 32.4% | 9.5% | 11.4% |
| ResNet | CW | pgd_only | 195 + 195 | 0.727 | 33.2% | 9.4% | 15.0% |
| ResNet | Patch | mixture | 320 + 320 | 0.901 | 64.9% | 6.3% | 13.4% |
| ResNet | Patch | pgd_only | 320 + 320 | 0.941 | 72.4% | 6.6% | 11.2% |
| ResNet | APGD | mixture | 195 + 195 | 0.996 | 98.8% | 5.5% | 10.1% |
| ResNet | APGD | pgd_only | 195 + 195 | 0.999 | 99.7% | 9.4% | 15.0% |
| DINOv2 | CW | mixture | 195 + 195 | 0.558 | 4.9% | 4.5% | 21.5% |
| DINOv2 | CW | pgd_only | 195 + 195 | 0.567 | 5.1% | 4.6% | 23.5% |
| DINOv2 | Patch | mixture | 320 + 320 | 0.545 | 8.2% | 6.6% | 30.4% |
| DINOv2 | Patch | pgd_only | 320 + 320 | 0.553 | 7.9% | 6.6% | 31.0% |
| DINOv2 | APGD | mixture | 195 + 195 | 0.856 | 42.6% | 4.5% | 20.0% |
| DINOv2 | APGD | pgd_only | 195 + 195 | 0.905 | 52.7% | 4.6% | 23.5% |

As for ResNet, mixture training slightly lowers DINOv2's held-out AUC; DINOv2 stays near chance on held-out CW and patch. Sources: [DINOv2 Stage 2](results/dinov2_observer_fixed1000_stage2/).

## 2. CW-only training

Stage 1 normally trains and tunes the detector on PGD, so CW is an *unseen* attack. Here the detector is trained on clean + successful CW training examples and C is selected on development CW, with everything else unchanged (`--train-attack cw`). This is the *known-attack* setting reported alongside the unseen-attack setting in the detection literature.

### Imagenette: CW detection by training attack

| Features | PGD-trained: CW | CW-trained: CW | CW-trained dev AUC | CW-trained clean FPR |
| --- | ---: | ---: | ---: | ---: |
| Grad-CAM | 0.591 / 16% | 0.615 / 21% | 0.604 | 6.9% |
| SLICE | 0.466 / 8% | 0.558 / 9% | 0.547 | 6.3% |
| BELIEF | 0.419 / 4% | 0.556 / 9% | 0.575 | 5.7% |
| SLICE signed | 0.607 / 13% | 0.714 / 20% | 0.727 | 4.9% |
| BELIEF signed | 0.596 / 9% | 0.708 / 17% | 0.726 | 4.0% |
| Confidence baseline | 0.582 / 2% | 0.742 / 9% | 0.737 | 4.9% |
| ResNet pooled | 0.726 / 24% | 0.826 / 28% | 0.820 | 5.4% |
| DINOv2 | 0.585 / 8% | 0.601 / 11% | 0.596 | 6.3% |

CW training improves every method, most for the signed SLICE/BELIEF maps, yet no detector catches more than 28% of CW attacks at the 5% target. The difficulty is therefore mostly a property of CW, not only of never seeing it in training. CW-trained ResNet features still detect PGD/APGD (0.995/0.997) and improve FGSM (0.892 → 0.922), with noise false alarms of 10%/10%/10%/10%. The in-Imagenette indicator is a fixed rule and is unchanged by training (see the confound in the combined report).

### CIFAR-10 (native 32×32): CW detection by training attack

| Features | PGD-trained: CW | CW-trained: CW | CW-trained dev AUC |
| --- | ---: | ---: | ---: |
| ResNet pooled | 0.645 / 11% | 0.698 / 15% | 0.786 |
| DINOv2 | 0.668 / 21% | 0.674 / 20% | 0.689 |

### Context from the literature

- [Carlini & Wagner (2017)](https://arxiv.org/abs/1705.07263) trained several detectors directly on C&W examples and tested on held-out C&W; several worked on MNIST but degraded on CIFAR (e.g. kernel density gave 80% of C&W examples a higher likelihood than clean images). They recommend reporting TPR at a low (1%) FPR, as this pipeline does.
- [Ma et al. (2018, LID)](https://arxiv.org/abs/1801.02613) evaluate each attack separately and also train on FGM and test on harder attacks including C&W.
- [Lee et al. (2018, Mahalanobis)](https://arxiv.org/abs/1807.03888) report known-attack and unseen-attack (FGSM-trained) columns; on CIFAR-10 ResNet, C&W AUROC ranges from about 56% (KD+PU) to 82% (LID) to 96% (Mahalanobis, multi-layer features).

The single-layer ResNet results here fall within the range of simpler published detectors; multi-layer features are a natural extension.

Sources: [explanations, CW-trained](results/slice_belief_fixed1000_cw_train/), [ResNet CW-trained](results/resnet_representation_fixed1000_cw_train/), [DINOv2 CW-trained](results/dinov2_observer_fixed1000_cw_train/), [CIFAR-10 ResNet CW-trained](results/resnet_representation_cifar10_native32_cw_train/), [CIFAR-10 DINOv2 CW-trained](results/dinov2_observer_cifar10_native32_cw_train/).

## 3. CIFAR-10

### 3.1 Setup

- **Sources:** a seeded, class-balanced 1,000-image subset of the official CIFAR-10 **test** split (100 per class), partitioned 35/10/20/35 per class exactly as Imagenette.
- **Classifier:** the same frozen ResNet-50 (`IMAGENET1K_V2`) with the existing 10-class linear head (`resnet50_cifar10_head.pt`, trained only on the official CIFAR-10 training split, 224×224 inputs). Every prediction is a CIFAR class, so the Imagenette out-of-class confound does not arise.
- **Threat model (primary, native-32):** attacks and noise act on the original 32×32 pixels; the classifier sees a bilinear upsample to 224×224, and the saved tensors are exactly that upsample, so all downstream feature and explanation code is unchanged. Budgets are per native pixel (8/255 L∞ for PGD/APGD/FGSM); the patch is 7×7 native pixels (4.79% of the image vs 4.59% for Imagenette's 48×48).
- **224 variant (diagnostic only):** identical except that attacks and noise act on the upsampled 224×224 image.

### 3.2 Corpus

| Clean accuracy | Imagenette | CIFAR-10 native-32 | CIFAR-10 224 |
| --- | ---: | ---: | ---: |
| train | 91.4% (320/350) | 87.4% (306/350) | 88.3% (309/350) |
| dev | 94.0% (94/100) | 89.0% (89/100) | 91.0% (91/100) |
| cal | 91.5% (183/200) | 94.0% (188/200) | 93.5% (187/200) |
| test | 90.6% (317/350) | 91.1% (319/350) | 90.6% (317/350) |

| Test attack success | Imagenette | CIFAR-10 native-32 | CIFAR-10 224 |
| --- | ---: | ---: | ---: |
| PGD | 100.0% (317/317) | 100.0% (319/319) | 100.0% (317/317) |
| APGD | 100.0% (317/317) | 100.0% (319/319) | 100.0% (317/317) |
| CW | 99.7% (316/317) | 77.7% (248/319) | 74.8% (237/317) |
| Patch | 64.0% (203/317) | 93.1% (297/319) | 98.1% (311/317) |
| FGSM | 33.8% (107/317) | 74.6% (238/319) | 65.9% (209/317) |

| Noise changes the prediction | Imagenette | CIFAR-10 native-32 | CIFAR-10 224 |
| --- | ---: | ---: | ---: |
| Gauss 2/255 | 1% | 5% | 11% |
| Gauss 4/255 | 3% | 8% | 21% |
| Gauss 8/255 | 6% | 17% | 26% |
| Uniform 8/255 | 3% | 10% | 19% |

CW succeeds less often on CIFAR-10 with the same settings, while patch and FGSM succeed more often, so attack-family comparisons across datasets reflect different attack strengths. The CIFAR-10 classifier is also more sensitive to random noise, which raises the bar for noise false alarms. Source: [corpus summary](results/corpus_summary_imagenette_cifar10.csv) (computed from each corpus's `variants.csv`).

### 3.3 Why the 224 variant is not used

| PGD-trained detector | PGD AUC | Patch AUC | FGSM AUC | Gauss 2/255 FPR | Gauss 4/255 FPR | Gauss 8/255 FPR | Uniform 8/255 FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet, CIFAR-10 224 | 1.000 | 1.000 | 1.000 | 41% | 87% | 99% | 90% |
| ResNet, CIFAR-10 native-32 | 1.000 | 0.872 | 0.864 | 6% | 32% | 42% | 37% |
| ResNet, Imagenette | 0.999 | 0.945 | 0.892 | 7% | 9% | 13% | 10% |
| DINOv2, CIFAR-10 224 | 1.000 | 0.752 | 1.000 | 22% | 95% | 100% | 96% |
| DINOv2, CIFAR-10 native-32 | 0.992 | 0.514 | 0.998 | 39% | 89% | 100% | 92% |

In the 224 variant, near-perfect attack AUCs coincide with flagging most noisy images: the detectors learn "contains sub-pixel speckle", which no upsampled 32×32 image has naturally. Native-32 attacks are smoothed by the same upsampling as the image; ResNet noise false alarms fall substantially and attack AUCs fall to plausible levels. DINOv2 remains highly noise-sensitive in both variants, matching its Imagenette behaviour. The remaining ResNet noise false alarms on native-32 partly reflect the classifier's own noise sensitivity (§3.2).

### 3.4 Stage 1 (native-32): PGD-only training

| Features | PGD | APGD | CW | Patch | FGSM | Clean FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet pooled | 1.000 / 100% | 1.000 / 100% | 0.645 / 11% | 0.872 / 51% | 0.864 / 38% | 2.6% |
| DINOv2 | 0.992 / 97% | 0.993 / 97% | 0.668 / 21% | 0.514 / 6% | 0.998 / 100% | 7.1% |

Successful test attacks: PGD 319, APGD 319, CW 248, Patch 297, FGSM 238. ResNet selected C = 10.0, DINOv2 C = 0.1. Both runs reproduced every saved ResNet prediction (strict check, zero mismatches).

### 3.5 Stage 2 (native-32)

| Features | Held out | Arm | N clean + N attacked | AUC | Detection | Clean FPR | Mean noise FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet | CW | mixture | 289 + 289 | 0.624 | 10.1% | 2.9% | 8.2% |
| ResNet | CW | pgd_only | 289 + 289 | 0.644 | 12.5% | 2.9% | 27.3% |
| ResNet | Patch | mixture | 235 + 235 | 0.878 | 56.0% | 5.0% | 22.6% |
| ResNet | Patch | pgd_only | 235 + 235 | 0.898 | 55.6% | 2.0% | 18.8% |
| ResNet | APGD | mixture | 221 + 221 | 0.985 | 93.4% | 4.0% | 18.0% |
| ResNet | APGD | pgd_only | 221 + 221 | 0.999 | 99.4% | 2.6% | 21.8% |
| DINOv2 | CW | mixture | 289 + 289 | 0.628 | 15.9% | 6.8% | 67.0% |
| DINOv2 | CW | pgd_only | 289 + 289 | 0.648 | 17.7% | 5.7% | 75.8% |
| DINOv2 | Patch | mixture | 235 + 235 | 0.492 | 3.7% | 5.6% | 71.2% |
| DINOv2 | Patch | pgd_only | 235 + 235 | 0.506 | 5.4% | 6.3% | 76.5% |
| DINOv2 | APGD | mixture | 221 + 221 | 0.971 | 85.4% | 5.8% | 59.2% |
| DINOv2 | APGD | pgd_only | 221 + 221 | 0.992 | 96.2% | 6.6% | 75.5% |

The Imagenette pattern repeats: mixtures do not raise held-out AUC, but they often reduce noise false alarms (e.g. ResNet's CW fold).

Sources: [ResNet native-32](results/resnet_representation_cifar10_native32/), [DINOv2 native-32](results/dinov2_observer_cifar10_native32/), [DINOv2 native-32 Stage 2](results/dinov2_observer_cifar10_native32_stage2/), [ResNet 224](results/resnet_representation_cifar10_fixed1000/), [DINOv2 224](results/dinov2_observer_cifar10_fixed1000/).

## 4. Reproducibility and changes

**Reproduction check.** The original PGD-trained explanation analysis was re-run with the modified code into a separate folder. All 40 method × attack rows (AUC, TPR at 1%/5%, clean FPR, C, development AUC, counts) and all noise false-alarm rates match [the published table](results/slice_belief_fixed1000/tables/stage1_detection.csv) exactly (maximum difference 0). Source: [re-run](results/slice_belief_fixed1000_stage1_rerun/). A new test also confirms the generalized manifest code reproduces the saved Imagenette split exactly.

**Code changes** (Imagenette behaviour unchanged unless an option is passed):

- `experiments/resnet_representation.py`: `--train-attack` for Stage 1; Stage 2 accepts DINOv2 feature caches; dataset-aware classifier.
- `experiments/slice_belief_analysis.py`: `--train-attack` and `--tables-dir` (Stage 1 only, separate output; original tables and log untouched).
- `experiments/slice_belief.py`, `utils/controlled_data.py`, `models/controlled_classifier.py`: CIFAR-10 manifest/loading, CIFAR-10 head, native-resolution attacks (`dataset.attack_size`), and `device: auto` now selects Apple MPS when CUDA is absent.
- New configs: `slice_belief_cifar10_native32_fixed1000.yaml` (primary) and `slice_belief_cifar10_fixed1000.yaml` (224 diagnostic).
- Tests in `tests/test_controlled_eval.py`: Imagenette split reproduction, CIFAR-10 manifest/loader/head, native-resolution wrapper (including a regression test that the wrapper cannot put the frozen ResNet into training mode).

**Commands** (from the repository root, Python 3.12 environment from `requirements-corpus.txt`, `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple silicon):

```bash
IN=configs/experiments/slice_belief_imagenette_fixed1000.yaml
C32=configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml
# DINOv2 on Imagenette (Stage 1, then Stage 2 from cached features)
python -m experiments.resnet_representation --config $IN --encoder dinov2 --device mps --prediction-policy record --output results/dinov2_observer_fixed1000
python -m experiments.resnet_representation --config $IN --encoder dinov2 --stage stage2 --features results/dinov2_observer_fixed1000/features --output results/dinov2_observer_fixed1000_stage2
# CW-only training (representations reuse cached features; explanations reuse saved maps)
python -m experiments.resnet_representation --config $IN --encoder resnet --device mps --prediction-policy record --train-attack cw --output results/resnet_representation_fixed1000_cw_train
python -m experiments.slice_belief_analysis --config $IN --train-attack cw --tables-dir results/slice_belief_fixed1000_cw_train
# CIFAR-10 native-32 corpus, then representations
python -m experiments.slice_belief --config $C32 --stage manifest
python -m experiments.slice_belief --config $C32 --stage corpus
python -m experiments.resnet_representation --config $C32 --encoder resnet --stage both --device mps --prediction-policy strict --output results/resnet_representation_cifar10_native32
python -m experiments.resnet_representation --config $C32 --encoder dinov2 --device mps --prediction-policy record --output results/dinov2_observer_cifar10_native32
```

Raw images and processed tensors remain excluded from Git and must be obtained from the shared data folder.

## 5. Limitations and next steps

- **Grad-CAM, SLICE and BELIEF have not yet been run on CIFAR-10.** The pipeline supports it (`--stage segment`, `sigma`, `explain`, then the analysis), but the projected cost is ~7 h on the RTX 4060 laptop and likely more on Apple silicon. CW-trained explanation detectors on CIFAR-10 follow in minutes once explanations exist.
- Stage 1 results are single fitted detectors; Stage 2 uses three allocation seeds. No new cross-pipeline significance tests are claimed.
- CIFAR-10 CW succeeds on only 77.7% (248/319) of eligible test images (vs 99.7% (316/317) on Imagenette), so its CW detection estimates use fewer and possibly easier-to-attack images.
- Noise panels contain 100 test images per condition; false-alarm estimates are correspondingly coarse (±~9 points at 95%).
- All attacks are non-adaptive: none optimizes against the detector.
- Possible extensions: multi-layer (Mahalanobis-style) features for CW; noisy negatives in detector training as a labelled ablation (PLAN.md §5).

