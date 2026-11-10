# Adversarial detection with explanations and representations

This report combines the fixed-1,000-image SLICE/BELIEF study with Experiment 2, the ResNet representation baseline. Both pipelines use the same saved clean and attacked images, source partitions, attack-success masks, detector family, and calibration protocol. The new experiment changes the detector's input features to frozen ResNet pooled representations.

On this corpus, ResNet representations achieve higher Stage-1 AUC than all tested explanation features on all five attacks. Mixture training does not improve ResNet's mean held-out AUC over its matched PGD-only control. These are results for this saved corpus; differences in actual false-positive rates and the limitations below matter when interpreting detection rates.

Contents: [1. Overall setup](#1-overall-setup) · [2. Explanation-based pipeline](#2-explanation-based-pipeline) · [3. Representation-based pipeline](#3-representation-based-pipeline)

## 1. Overall setup

### 1.1 Dataset, classifier and shared images

The experiment uses a seeded, class-balanced subset of Imagenette validation: 100 images from each of 10 classes, seed 42. It avoids the historical first-1,000-sorted-files subset, which covered only three classes. The full 3,925-image run was projected to exceed the explanation study's 10-hour budget, so the fixed subset is the primary comparison.

| Partition | Images | Purpose |
| --- | ---: | --- |
| Train | 350 | Fit feature scaling and the attack detector |
| Development | 100 | Select logistic-regression regularization and explanation blur |
| Clean calibration | 200 | Set detection thresholds independently of fitting and tuning |
| Test | 350 | Evaluate clean false alarms and attack detection |

Partitions contain 35/10/20/35 images per class. A source ID identifies an original image; its clean, attacked and noisy variants remain in the same partition. The noise panel has 100 test sources, and the stability panel has 20 test sources.

The attacked classifier is a frozen, 1,000-class torchvision ResNet-50 with `IMAGENET1K_V2` weights. Images use resize size 232 and crop size 224. Its original clean accuracies are 91.4% train, 94.0% development, 91.5% calibration and 90.6% test. The **classifier** predicts an object class; the separate **detector** predicts whether an input is adversarial.

The representation experiment loads the explanation study's saved tensors from `data/processed/slice_belief_fixed1000/`. It does not recreate attacks or rerun explanations. Tensor hashes check input identity; original saved ResNet predictions determine eligibility and success. Local prediction differences are audited separately (§3.5).

Sources: [shared configuration](configs/experiments/slice_belief_imagenette_fixed1000.yaml), [source manifest](results/slice_belief_fixed1000/source_files.csv).

### 1.2 Attacks and noise controls

All attacks target the shared ResNet classifier, not the logistic detector.

| Attack | Saved-corpus settings |
| --- | --- |
| PGD | L∞ budget 8/255, step size 2/255, 20 steps, random start |
| APGD | L∞ budget 8/255, 20 steps, cross-entropy loss, one restart, seed 42 |
| CW | c = 1, kappa = 0, 100 steps, learning rate 0.01 |
| Patch | 48 × 48 patch, 100 steps, learning rate 0.05 |
| FGSM | L∞ budget 8/255 |

CW uses its own distortion/loss objective; 8/255 is not a shared CW constraint. The patch attack changes a local region and does not share PGD's small global L∞ budget. Attack-family comparisons therefore describe these fixed settings, rather than equal perturbation strength.

Non-adversarial controls are Gaussian noise at 2/255, 4/255 and 8/255, and uniform noise at 8/255. Each condition uses the same 100-source test panel.

An image is **eligible** if the original clean prediction is correct. An attack is **successful** if that eligible image becomes misclassified. Detection evaluates successful attacks against all 350 clean test images.

| Test attack | Eligible | Successful | Attack success rate |
| --- | --- | --- | --- |
| PGD | 317 | 317 | 100.00% |
| APGD | 317 | 317 | 100.00% |
| CW | 317 | 316 | 99.68% |
| PATCH | 317 | 203 | 64.04% |
| FGSM | 317 | 107 | 33.75% |

Source: [attack corpus statistics](results/slice_belief_fixed1000/tables/asr.csv).

### 1.3 Shared detector, two stages and metrics

Every learned detector uses training-only `StandardScaler` followed by logistic regression. C is selected from {0.01, 0.1, 1, 10} by development AUC. Each fitted detector receives its own thresholds from the same 200 clean calibration images, targeting 1% and 5% FPR. Thresholds are then frozen across test attacks and noise conditions.

**Stage 1: PGD-only training.** Fit on all 350 clean training images and 320 successful PGD training examples, select C on development clean/PGD data, and test on PGD, APGD, CW, patch and FGSM.

**Stage 2: matched mixture versus PGD-only training.** Withhold one family from both fitting and tuning. Within each fold, retain sources for which all permitted attacks succeed. Both arms use the same sources and the same N clean + N adversarial examples. The mixture assigns one permitted attack to each source; PGD-only uses PGD for those sources. Allocation seeds are 42, 123 and 2026.

| Held-out family | Permitted mixture | N clean + N attacked (train) | Matched development sources |
| --- | --- | ---: | ---: |
| CW | PGD, APGD, patch | 195 + 195 | 62 |
| Patch | PGD, APGD, CW | 320 + 320 | 94 |
| APGD | PGD, CW, patch | 195 + 195 | 62 |

Stage-2 PGD-only is refitted on the matched pool, so it is a distinct control from Stage 1. FGSM is test-only in both stages: it never enters training or hyperparameter selection and has no separate held-out fold. A result row with `attack=fgsm, is_holdout=False` still describes an unseen attack.

Metric definitions used throughout:

- **AUC:** separation of successful adversarial examples from clean test images.
- **TPR at a 1%/5% calibration target:** detected successful attacks divided by successful attacks, using the frozen threshold. This does not guarantee a 1%/5% FPR on test data.
- **Clean/noise FPR:** false alarms divided by the relevant clean/noisy test count. Mean noise FPR averages the four conditions within a run, then averages runs where applicable.
- **ASR:** successful attacks divided by eligible sources.
- **Undetected rate:** missed successful attacks divided by eligible sources, a different denominator from TPR.
- **`test_roc_tpr`:** a descriptive threshold selected from the test ROC, distinct from the primary clean-calibrated TPR.

Stage-2 tables report means across three seeds; saved `_sd` columns are sample standard deviations. Paired bootstrap uses 2,000 source-image resamples with calibration resampled as well. Stage-2 gain intervals apply only to primary seed 42, not the three-seed mean. Existing intervals compare methods within the explanation pipeline or training arms within a pipeline; no new paired significance test between ResNet features and explanation features is claimed here.

### 1.4 Comparability and shared limitations

The comparison holds pixels, source partitions, original success masks, attack settings, logistic-detector selection and calibration rules fixed. Feature extraction changes: explanation maps have 196 or 392 dimensions, while ResNet pooled features have 2,048. Thus this is a comparison of complete feature pipelines, not a dimension-matched ablation. Hardware and extraction precision also differ.

The original explanation report flags substantial Imagenette/ImageNet pretraining overlap (reported as approximately 96.6% of Imagenette validation). That overlap estimate was not independently re-audited for this combined report. Detector train/test separation does not establish that the pretrained classifier has never seen these source images.

More than 99% of the successful attacks predict outside the ten Imagenette classes. A simple class-membership indicator achieves AUC near 0.95, including on CW and FGSM. High detector AUC may therefore reflect this dataset/classifier structure as well as adversarial artifacts. The indicator has tied binary scores: conservative calibration gives zero TPR at the 1%/5% targets, while its natural cutoff yields about 99% detection at about 10% clean FPR. It is a useful confound check, not a directly equivalent operating point.

Results concern successful attacks against this classifier under the listed settings. They do not establish general robustness against adaptive detector-aware attacks or performance on unrelated datasets.

## 2. Explanation-based pipeline

### 2.1 Additional setup

Grad-CAM uses ResNet layer `layer4.2` and a 14 × 14 map (196 features). SLICE and BELIEF use quickshift superpixels and rasterize positive coefficients onto the same grid. Signed variants concatenate positive and negative parts (392 features). Auxiliary baselines use maximum softmax probability and entropy, Imagenette class membership, or superpixel count.

SLICE uses 500 perturbations, 1,000 bootstrap ridge fits, up to 10 elimination steps and tolerance 3, with a LIME fallback. BELIEF uses Bayesian-ridge sign-entropy elimination with slack 1e-6 and shares SLICE's initial 500-sample draw. A global blur sigma of 0.4 is selected on 50 development clean images. Model queries use fp16 autocast/channels-last; regression uses float64. The implementation documents per-channel blur and ImageNet-mean masking conventions relative to the reference implementations.

The source report records reference-parity checks for regression, sign entropy, elimination and blur selection, with coefficient agreement to 1e-6. These historical checks are reported here without rerunning the costly explanation experiment.

Implementation: [study](experiments/slice_belief.py), [analysis](experiments/slice_belief_analysis.py), [SLICE](explainers/slice.py), [BELIEF](explainers/belief.py).

The original explanation study was run from the repository root with:

```bash
python -m experiments.slice_belief \
  --config configs/experiments/slice_belief_imagenette_fixed1000.yaml --stage all
python -m experiments.slice_belief_analysis \
  --config configs/experiments/slice_belief_imagenette_fixed1000.yaml
```

These commands document the original study. Existing saved explanation results suffice for the comparison below; running Experiment 2 does not require running these commands again.

### 2.2 Stage 1: PGD-only detection

Each cell is **AUC / TPR at 1% calibration target / TPR at 5% calibration target**, expressed as proportions.

| Method | PGD | APGD | CW | PATCH | FGSM |
| --- | --- | --- | --- | --- | --- |
| Grad-CAM | 0.9439 / 0.4479 / 0.7445 | 0.9339 / 0.4826 / 0.7476 | 0.5913 / 0.0633 / 0.1582 | 0.6143 / 0.0246 / 0.0542 | 0.5298 / 0.0187 / 0.0654 |
| SLICE | 0.6047 / 0.0284 / 0.1136 | 0.5882 / 0.0379 / 0.1262 | 0.4660 / 0.0032 / 0.0759 | 0.4463 / 0.0000 / 0.0394 | 0.4739 / 0.0093 / 0.0374 |
| BELIEF | 0.8527 / 0.5931 / 0.6593 | 0.8743 / 0.6845 / 0.7256 | 0.4190 / 0.0032 / 0.0411 | 0.5683 / 0.0148 / 0.1034 | 0.5504 / 0.0187 / 0.0654 |
| SLICE signed | 0.6900 / 0.0883 / 0.1577 | 0.7474 / 0.1009 / 0.1830 | 0.6071 / 0.0570 / 0.1297 | 0.6407 / 0.0246 / 0.0837 | 0.5135 / 0.0280 / 0.0374 |
| BELIEF signed | 0.8942 / 0.6341 / 0.7098 | 0.9141 / 0.7161 / 0.7539 | 0.5963 / 0.0285 / 0.0886 | 0.7724 / 0.1379 / 0.3005 | 0.5719 / 0.0187 / 0.0467 |
| Confidence | 0.9228 / 0.8454 / 0.8833 | 0.9401 / 0.8991 / 0.9117 | 0.5822 / 0.0127 / 0.0222 | 0.5233 / 0.0887 / 0.1970 | 0.4085 / 0.0000 / 0.0000 |
| Imagenette indicator | 0.9513 / 0.0000 / 0.0000 | 0.9497 / 0.0000 / 0.0000 | 0.9529 / 0.0000 / 0.0000 | 0.9504 / 0.0000 / 0.0000 | 0.9529 / 0.0000 / 0.0000 |
| Superpixel count K | 0.6233 / 0.0315 / 0.0662 | 0.6125 / 0.0221 / 0.0726 | 0.5059 / 0.0095 / 0.0348 | 0.4690 / 0.0099 / 0.0345 | 0.6588 / 0.0280 / 0.0654 |

Source: [Stage-1 detection](results/slice_belief_fixed1000/tables/stage1_detection.csv), [paired method comparisons](results/slice_belief_fixed1000/tables/stage1_bootstrap.csv).

Grad-CAM is a strong explanation baseline for PGD/APGD. Positive-only SLICE and BELIEF generally underperform it. Retaining signed BELIEF coefficients improves patch detection: AUC 0.772 versus 0.614 for Grad-CAM, with the original paired-bootstrap AUC difference +0.159 [0.108, 0.208]. Better deletion fidelity does not imply better attack detection.

### 2.3 Stage 2: held-out detection

The following are three-seed means on the designated held-out family. TPR uses the 5% calibration target. Both training arms use the matched pools in §1.3.

| Held out | Method | Mixture AUC | PGD-only AUC | Mixture TPR | PGD-only TPR |
| --- | --- | --- | --- | --- | --- |
| CW | Grad-CAM | 0.6089 | 0.5825 | 15.93% | 13.92% |
| CW | SLICE | 0.4689 | 0.4753 | 3.38% | 5.06% |
| CW | BELIEF | 0.4165 | 0.4159 | 2.95% | 3.48% |
| CW | SLICE signed | 0.6621 | 0.5869 | 17.83% | 10.13% |
| CW | BELIEF signed | 0.6440 | 0.5753 | 13.82% | 11.08% |
| CW | Confidence | 0.6541 | 0.5622 | 7.38% | 3.16% |
| PATCH | Grad-CAM | 0.5942 | 0.5938 | 7.55% | 4.93% |
| PATCH | SLICE | 0.4472 | 0.4527 | 2.63% | 4.93% |
| PATCH | BELIEF | 0.5673 | 0.5690 | 4.93% | 9.36% |
| PATCH | SLICE signed | 0.7287 | 0.6300 | 17.24% | 9.85% |
| PATCH | BELIEF signed | 0.8330 | 0.7695 | 43.68% | 28.57% |
| PATCH | Confidence | 0.6955 | 0.6370 | 2.96% | 11.33% |
| APGD | Grad-CAM | 0.8750 | 0.9300 | 62.46% | 72.87% |
| APGD | SLICE | 0.4933 | 0.5767 | 4.63% | 11.36% |
| APGD | BELIEF | 0.8388 | 0.8640 | 69.61% | 75.39% |
| APGD | SLICE signed | 0.7261 | 0.7008 | 21.35% | 12.30% |
| APGD | BELIEF signed | 0.9126 | 0.9083 | 74.97% | 76.34% |
| APGD | Confidence | 0.9236 | 0.9359 | 2.21% | 91.48% |

Sources: [Stage-2 means and FPRs](results/slice_belief_fixed1000/tables/stage2_summary.csv), [per-seed results](results/slice_belief_fixed1000/tables/stage2_per_seed.csv), [seed-42 bootstrap](results/slice_belief_fixed1000/tables/stage2_bootstrap.csv).

Signed maps benefit from mixture training on held-out CW and patch. BELIEF signed reaches patch AUC 0.833 and TPR 43.68%. Gains are not universal: Grad-CAM loses APGD performance, and its patch mean AUC is approximately unchanged. The latter masks allocation variability: mixture patch AUC is approximately 0.557, 0.591 and 0.635 across seeds. Its negative seed-42 bootstrap gain must not be presented as the three-seed mean gain.

### 2.4 False alarms, explanation behaviour and cost

Stage-1 false alarms at the 5% calibration target:

| Method | Clean test FPR | Gaussian 2/255 | Gaussian 4/255 | Gaussian 8/255 | Uniform 8/255 |
| --- | --- | --- | --- | --- | --- |
| Grad-CAM | 5.14% | 6.00% | 7.00% | 6.00% | 7.00% |
| SLICE | 5.71% | 8.00% | 4.00% | 9.00% | 5.00% |
| BELIEF | 5.14% | 7.00% | 3.00% | 8.00% | 2.00% |
| SLICE signed | 4.29% | 3.00% | 4.00% | 6.00% | 7.00% |
| BELIEF signed | 2.86% | 8.00% | 4.00% | 7.00% | 7.00% |
| Confidence | 1.71% | 4.00% | 4.00% | 3.00% | 4.00% |

Source: [noise controls](results/slice_belief_fixed1000/tables/stage1_noise.csv). Noise estimates use only 100 sources per condition; the point estimates alone do not establish equivalence to the calibration target.

BELIEF produces zero-positive maps on approximately 55% of PGD and 67% of APGD examples, but almost never on clean, CW or FGSM inputs. An empty-map flag alone gives AUC 0.776/0.834 on PGD/APGD. This supports an explanation-collapse signal, without proving a particular causal mechanism.

On five explanation reruns over the small stability panel, clean-map Spearman correlations are approximately 0.988 for SLICE and 0.986 for BELIEF. Top-five-positive-superpixel deletion gives clean AOPC 0.0186/0.0185 for SLICE/BELIEF versus 0.0014 for Grad-CAM; PGD values are 0.0040/0.0040 versus 0.0002. The deletion evaluation uses the same segmentation and fill as SLICE/BELIEF, so its advantage should not be generalized to all notions of explanation quality.

| Component | Median seconds/input | Model queries |
| --- | ---: | --- |
| Grad-CAM | 0.005 | One forward and backward pass |
| SLICE | 8.12 | 3,494 median |
| BELIEF | 1.37 | 500 |
| Shared quickshift segmentation | 1.37 | — |

These are original RTX 4060 laptop measurements over 4,159 explained inputs. They are not a controlled timing comparison with the Mac representation run.

Sources: [elimination status](results/slice_belief_fixed1000/tables/explanation_status.csv), [stability](results/slice_belief_fixed1000/tables/stability.csv), [fidelity](results/slice_belief_fixed1000/tables/fidelity_summary.csv), [cost](results/slice_belief_fixed1000/tables/cost.csv).

### 2.5 Replication and scope

The original report also describes an independent 900-image run, with 30/10/20/30 images per class across train/dev/cal/test. Its main explanation-method rankings broadly agree with the fixed-1,000 study, while elevated Grad-CAM noise false alarms did not replicate. Those results are supporting evidence, not pooled observations for the present comparison. Earlier sorted-subset and grouped-cross-validation results used different protocols and are excluded from the primary comparison tables.

## 3. Representation-based pipeline

### 3.1 Additional setup for Experiment 2

The frozen ResNet-50 supplies its 2,048-dimensional global-average-pooling output before the final classifier layer. Each input is encoded independently, with ImageNet normalization applied once to the unchanged saved pixels. The detector does not receive a clean counterpart, a clean-versus-attacked feature difference, or explanation maps. The backbone is not fine-tuned.

Stages 1 and 2 run from [one experiment module](experiments/resnet_representation.py), using [pooled feature extraction](encoders/cnn.py) and the shared detector/evaluation conventions. The saved run used cached features originally extracted on MPS with PyTorch 2.4.1 and torchvision 0.19.1. Original corpus masks are retained under `--prediction-policy record`.

DINOv2 support exists for Stage 1, but no completed DINOv2 run under this shared fixed-1,000 protocol is included in the current result folder. The results below are ResNet only.

### 3.2 Stage 1 results and direct pipeline comparison

The ResNet detector selected C = 0.01. All attacks use the same fitted detector and calibration thresholds.

| Attack | AUC | TPR at 1% target | TPR at 5% target | Actual clean FPR at 5% target |
| --- | --- | --- | --- | --- |
| PGD | 0.9987 | 98.42% | 100.00% | 6.57% |
| APGD | 0.9991 | 99.37% | 100.00% | 6.57% |
| CW | 0.7264 | 8.23% | 24.37% | 6.57% |
| PATCH | 0.9447 | 52.71% | 72.91% | 6.57% |
| FGSM | 0.8919 | 21.50% | 54.21% | 6.57% |

The actual clean-test FPR at the 1% target is 1.14% (4/350); at the 5% target it is 6.57% (23/350). Mean noise FPR at the 5% target is 9.75%.

For direct comparison, each cell below is **AUC / TPR at the 5% calibration target**. Actual clean FPR is method-specific.

| Method | Clean FPR | PGD | APGD | CW | PATCH | FGSM |
| --- | --- | --- | --- | --- | --- | --- |
| Grad-CAM | 5.14% | 0.9439 / 74.45% | 0.9339 / 74.76% | 0.5913 / 15.82% | 0.6143 / 5.42% | 0.5298 / 6.54% |
| BELIEF signed | 2.86% | 0.8942 / 70.98% | 0.9141 / 75.39% | 0.5963 / 8.86% | 0.7724 / 30.05% | 0.5719 / 4.67% |
| Confidence | 1.71% | 0.9228 / 88.33% | 0.9401 / 91.17% | 0.5822 / 2.22% | 0.5233 / 19.70% | 0.4085 / 0.00% |
| ResNet pooled | 6.57% | 0.9987 / 100.00% | 0.9991 / 100.00% | 0.7264 / 24.37% | 0.9447 / 72.91% | 0.8919 / 54.21% |

| Method | Clean test FPR | Gaussian 2/255 | Gaussian 4/255 | Gaussian 8/255 | Uniform 8/255 |
| --- | --- | --- | --- | --- | --- |
| Grad-CAM | 5.14% | 6.00% | 7.00% | 6.00% | 7.00% |
| BELIEF signed | 2.86% | 8.00% | 4.00% | 7.00% | 7.00% |
| ResNet pooled | 6.57% | 7.00% | 9.00% | 13.00% | 10.00% |

ResNet pooled features have higher observed AUC than every explanation feature set in §2.2 for each attack. PGD/APGD are nearly perfectly separated, and patch/FGSM detection improves substantially. CW remains difficult at calibrated thresholds: AUC is 0.7264 but TPR is only 24.37% at the 5% target.

The TPR gains accompany a higher clean FPR than Grad-CAM and BELIEF signed, and noise FPR reaches 13% for Gaussian 8/255. These results support stronger discrimination on this corpus, not a claim of equal false-alarm cost. The binary membership baseline still has higher AUC than ResNet on CW and FGSM, reinforcing the confound in §1.4.

Sources: [ResNet Stage-1 detection](results/resnet_representation_fixed1000_complete/stage1_detection.csv), [noise](results/resnet_representation_fixed1000_complete/stage1_noise.csv), [thresholds](results/resnet_representation_fixed1000_complete/thresholds.json).

### 3.3 Stage 2 results

Only designated held-out rows are shown below. Values are three-seed means; TPR and false alarms use the 5% calibration target.

| Held out | Arm | N clean + N attacked | AUC | TPR | Clean FPR | Mean noise FPR |
| --- | --- | --- | --- | --- | --- | --- |
| APGD | mixture | 195 + 195 | 0.9956 | 98.84% | 5.52% | 10.08% |
| APGD | pgd_only | 195 + 195 | 0.9991 | 99.68% | 9.43% | 15.00% |
| CW | mixture | 195 + 195 | 0.6724 | 32.38% | 9.52% | 11.42% |
| CW | pgd_only | 195 + 195 | 0.7270 | 33.23% | 9.43% | 15.00% |
| PATCH | mixture | 320 + 320 | 0.9011 | 64.86% | 6.29% | 13.42% |
| PATCH | pgd_only | 320 + 320 | 0.9407 | 72.41% | 6.57% | 11.25% |

For a direct Stage-2 pipeline comparison:

| Held out | Method | Mixture AUC / TPR | PGD-only AUC / TPR |
| --- | --- | --- | --- |
| CW | Grad-CAM | 0.6089 / 15.93% | 0.5825 / 13.92% |
| CW | BELIEF signed | 0.6440 / 13.82% | 0.5753 / 11.08% |
| CW | ResNet pooled | 0.6724 / 32.38% | 0.7270 / 33.23% |
| PATCH | Grad-CAM | 0.5942 / 7.55% | 0.5938 / 4.93% |
| PATCH | BELIEF signed | 0.8330 / 43.68% | 0.7695 / 28.57% |
| PATCH | ResNet pooled | 0.9011 / 64.86% | 0.9407 / 72.41% |
| APGD | Grad-CAM | 0.8750 / 62.46% | 0.9300 / 72.87% |
| APGD | BELIEF signed | 0.9126 / 74.97% | 0.9083 / 76.34% |
| APGD | ResNet pooled | 0.9956 / 98.84% | 0.9991 / 99.68% |

ResNet's mixture mean AUC is lower than matched PGD-only for all three held-out families: approximately −0.0547 on CW, −0.0396 on patch and −0.0034 on APGD. Mixture training therefore does not show a general benefit for this representation baseline, although it lowers APGD-fold clean FPR. Its effect differs from the gains observed for signed explanation maps on CW and patch.

Across the mixture arms, ResNet has higher observed held-out AUC than the explanation methods in §2.3, including only a small advantage over SLICE signed on CW. These are descriptive comparisons; inspect each detector's actual FPR in the linked summaries and do not infer cross-pipeline statistical significance from within-pipeline intervals.

Sources: [ResNet Stage-2 summary](results/resnet_representation_fixed1000_complete/stage2_summary.csv), [per-seed results](results/resnet_representation_fixed1000_complete/stage2_per_seed.csv), [combined Stage-1/2 results](results/resnet_representation_fixed1000_complete/combined_results.csv).

### 3.4 Stage-2 uncertainty

The following are saved **seed-42 bootstrap mean gains** and 95% intervals for mixture minus matched PGD-only. They are not differences between the three-seed means above. TPR differences are percentage points (pp).

| Held out | AUC gain [95% interval] | TPR gain, pp [95% interval] |
| --- | --- | --- |
| CW | -0.0359 [-0.0563, -0.0156] | -3.31 [-11.33, +6.11] |
| PATCH | -0.0363 [-0.0578, -0.0160] | -7.54 [-19.16, +4.23] |
| APGD | -0.0027 [-0.0086, +0.0004] | +0.01 [-0.94, +0.94] |

For seed 42, the AUC-gain intervals are negative for CW and patch and cross zero for APGD. All three TPR-gain intervals cross zero. Source: [ResNet bootstrap results](results/resnet_representation_fixed1000_complete/stage2_bootstrap.csv).

### 3.5 Input identity, prediction audit and reproducibility

The saved provenance records 4,950 encoded input variants, per-block tensor hashes and five local classifier prediction mismatches. Matching image hashes verifies the inputs, not bitwise-identical model outputs. The original clean-correct and attack-success masks remain authoritative, preserving the original cohorts despite local numerical differences. Matching package versions did not eliminate those differences, so their cause is not established by the version check alone.

The current run reused a feature cache at a backup directory that is no longer present in this checkout. Embedded feature provenance survives in [Stage-2 provenance](results/resnet_representation_fixed1000_complete/stage2_provenance.json), including input hashes, software versions, MPS device and mismatch count. A separate row-level mismatch CSV is not present in this result directory. The aggregate audit can be inspected now; a full fresh audit requires extracting features again from the saved corpus.

To rerun both representation stages from the repository root, install the [corpus-matched dependencies](requirements-corpus.txt), provide the original corpus including `images/`, and use a new output directory:

```bash
python -m experiments.resnet_representation --encoder resnet --check
python -m experiments.resnet_representation \
  --encoder resnet --stage both --device mps \
  --prediction-policy record \
  --output results/resnet_representation_fixed1000_rerun
```

Use `--device cpu` or `--device cuda` on other hardware. `record` explicitly preserves the original masks while logging local prediction differences; strict mode is the default and stops on mismatches. This command does not rerun SLICE/BELIEF or regenerate attacked images. If a valid feature cache is available, `--features path/to/features` skips backbone extraction.

The result directory contains Stage-1 tables, flat `stage2_*` tables, scores, thresholds and fitted models. `combined_results.csv` has 35 rows: five Stage-1 rows and 30 Stage-2 rows (three folds × two arms × five attacks). Filter `stage=stage2` and `is_holdout=True` for the six primary ResNet held-out rows.

This report uses repository-relative links and can be committed as a standalone Markdown report. Its explanation narrative is adapted from the locally supplied `SLICE_BELIEF_result.md`, which is ignored by Git; numerical tables here are drawn from the linked result CSVs. The new report does not depend on publishing that ignored source document. Include the linked ResNet result tables and experiment code when committing the new experiment. Raw images and processed tensors remain excluded by the existing `.gitignore` and must be obtained separately for reproduction.

### 3.6 Main findings and next comparison

1. On identical saved inputs, ResNet pooled features outperform the tested explanation features in Stage-1 AUC, with strong PGD/APGD performance and improved patch/FGSM detection. CW remains a weakness at calibrated thresholds.
2. Signed BELIEF is a useful explanation ablation, especially for patch. Explanation stability and deletion fidelity alone do not predict detector quality.
3. Mixture training helps some signed explanation detectors but lowers ResNet's mean held-out AUC relative to its matched PGD-only controls. Attack diversity is not uniformly beneficial under this training protocol.
4. Report actual clean/noise FPR alongside TPR, retain the membership/confidence controls, and keep the prediction-audit and pretraining-overlap limitations visible.
5. A subsequent DINOv2 comparison should preserve these saved tensors, source partitions, original ResNet masks and calibration rules, changing only the observer representation. It should be reported as a new result after execution, rather than inferred from the ResNet results.
