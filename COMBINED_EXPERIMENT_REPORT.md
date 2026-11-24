# Adversarial detection with explanations and representations

This report combines the fixed-1,000-image SLICE/BELIEF study with Experiment 2, the ResNet representation baseline. Both pipelines use the same saved clean and attacked images, source partitions, attack-success masks, detector family, and calibration protocol. The representation experiment changes the detector's input features to frozen ResNet pooled representations. Clearly labeled CIFAR-10 native-32 subsections extend the comparison to the completed explanation study and independent DINOv2 observer.

**Main result: detection depends strongly on the attack family used for training.** On both datasets, PGD-trained ResNet features separate PGD/APGD better than the tested explanation features. On native-32 CIFAR-10, however, representation detectors also flag substantially more ordinary noise. Explanation detectors offer fewer noise false alarms, but miss many other attacks; frequent empty SLICE/BELIEF maps further limit claims about explanation quality. These results do not establish a universally best detector.

| Question | What the completed native-32 CIFAR-10 results support |
| --- | --- |
| Do explanation features detect attacks? | SLICE signed detects 85.3% of successful PGD and 94.7% of APGD attacks, but only 3.2% of CW, 15.2% of patch and 2.1% of FGSM attacks at the 5% calibration target. |
| Are representation features better? | ResNet has higher Stage-1 AUC than all tested explanation maps on every attack, but flags 42% of Gaussian-8/255 inputs, versus 1% for Grad-CAM and 5% for SLICE signed. |
| Does more varied training solve generalization? | No consistent improvement: mixture training increases mean held-out AUC in only 5/15 explanation-method comparisons and in none of the ResNet/DINOv2 comparisons. |
| Does good detection imply good explanations? | No. SLICE often produces empty positive maps on PGD/APGD; this may provide a detection signal without informative attribution. BELIEF is often empty even on clean inputs. |
| Is CW intrinsically undetectable here? | No. CW-specific training improves several detectors; confidence features reach CW AUC 0.887 and TPR 55.6%, but lose PGD/APGD detection. |

The overview describes observed results, not cross-pipeline significance tests. Detailed numerical evidence remains below. Read [§2.6](#26-cifar-10-native-32-completed-explanations) for explanation-specific implications and [§3.8](#38-what-the-comparison-supports) for the comparison and its practical limits. Imagenette results remain separately labeled because its classifier has an additional class-membership confound (§1.4).

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

The detector is a binary classifier: clean inputs have label 0 and successful attacked inputs have label 1. Each feature method gets its own fitted detector. It flags suspicious inputs; it does not correct the ResNet class prediction. Every learned detector uses training-only `StandardScaler` followed by logistic regression. C is selected from {0.01, 0.1, 1, 10} by development AUC. Each fitted detector receives its own thresholds from the same 200 clean calibration images, targeting 1% and 5% FPR. Thresholds are then frozen across test attacks and noise conditions.

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

<!-- cifar10-native32-setup:start -->
### 1.5 CIFAR-10 native-32: shared setup

The primary CIFAR-10 experiment uses 1,000 class-balanced official test sources, partitioned 350/100/200/350 into train/development/clean calibration/test. Attacks and noise modify the native 32×32 pixels before bilinear resizing to the stored 224×224 tensors. The classifier is a frozen ImageNet-pretrained ResNet-50 with a linear head trained on the official CIFAR-10 training split. DINOv2 is an independent observer of the same ResNet-targeted attacks. The older 224-resolution attack experiment remains a separate diagnostic.

Native-space PGD/APGD/FGSM use an L∞ budget of 8/255. PGD and APGD each use 20 steps; the patch is 7×7 native pixels. CW uses c=1.0, kappa=0.0 and 100 steps with its own distortion objective; it is not constrained by the shared L∞ budget.

All 21 tensor blocks and three corpus metadata files match the saved ResNet provenance. The local classifier audit has **zero prediction mismatches**. All **4,282** required explanation records, Grad-CAM features, stability repetitions and fidelity records were read and validated. The saved development-selected sigma **0.3** and completed segmentation are reused.

SLICE and BELIEF share segmentation and the first 500-sample perturbation draw. SLICE uses 1000 bootstrap fits, at most 10 elimination steps and tolerance 3; BELIEF uses slack 1e-06. Training-only scaling precedes logistic regression; C is selected from [0.01, 0.1, 1.0, 10.0] using development AUC. Independent clean calibration freezes thresholds at targets [0.01, 0.05]. Settings are recorded in the execution environment and [native-32 configuration](configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml).

| Test attack | Eligible sources | Successful attacks | ASR |
| --- | --- | --- | --- |
| PGD | 319 | 319 | 100.0% |
| APGD | 319 | 319 | 100.0% |
| CW | 319 | 248 | 77.7% |
| PATCH | 319 | 297 | 93.1% |
| FGSM | 319 | 238 | 74.6% |

Detection TPR divides by successful attacks; ASR and undetected-success rate divide by eligible clean-correct sources. Every attack is compared with all 350 clean test sources. Each noise control has 100 test sources. Failed attacks do not enter detector positives. FGSM is test-only.

Independent refitting of 16 PGD/CW-trained Stage 1 detectors verified 288 attack/noise operating points. AUC is recomputed with sklearn, while thresholds, TPR, clean/noise FPR and undetected rates are checked with direct score comparisons. The validation retains [all recomputed scores](results/slice_belief_cifar10_native32_fixed1000/validation_scores.csv) and [calibration thresholds](results/slice_belief_cifar10_native32_fixed1000/validation_thresholds.json).

| Stage 2 holdout | Permitted mixture | Matched N clean + N attacked | Matched dev sources |
| --- | --- | --- | --- |
| CW | PGD+APGD+PATCH | 289 + 289 | 83 |
| PATCH | PGD+APGD+CW | 235 + 235 | 64 |
| APGD | PGD+CW+PATCH | 221 + 221 | 60 |

Stage 2 refits PGD-only and mixture controls on identical successful source pools. The source allocation varies over seeds 42, 123 and 2026; neither the held-out family nor FGSM enters fitting or tuning. Both arms retain the original development-negative and clean-calibration protocol.

Matched development counts refer to successful positive-source pools; clean development negatives remain all 100 sources in each arm.

Sources: [validation](results/slice_belief_cifar10_native32_fixed1000/validation.json), [prediction audit](results/slice_belief_cifar10_native32_fixed1000/prediction_audit.json), [ASR counts](results/slice_belief_cifar10_native32_fixed1000/tables/asr.csv), [matched source pools](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_matching.json), [per-seed assignments](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_assignments.csv).
<!-- cifar10-native32-setup:end -->

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

<!-- cifar10-native32-explanation:start -->
<!-- editorial-summary: preserve during report regeneration -->
### 2.6 CIFAR-10 native-32: completed explanations

This experiment compares **Grad-CAM, SLICE and BELIEF against ResNet and DINOv2 features on native-resolution CIFAR-10**.

The main finding is: **explanation-based detectors recognize PGD/APGD reasonably well, but generalize poorly to other attacks. ResNet features detect more attacks overall, while explanation features produce fewer false alarms on ordinary noise.** There is also a major explanation-quality issue: many SLICE/BELIEF outputs are empty.

The saved validation records mark the run complete, with **4,282 explanation inputs validated and zero classifier-prediction mismatches**. These inputs include clean, successful attacked and noisy versions of the 1,000 original images; stability repetitions are counted separately.

#### 2.6.1 What was tested?

The experiment uses 1,000 CIFAR-10 images, split into 350 training, 100 development, 200 calibration and 350 test images.

Attacks modify the original **32×32 pixels**, before resizing for the classifier. This matters because attacking after enlargement to 224×224 can introduce artificial patterns that make detection misleadingly easy.

Each detector is a separately fitted **logistic regression model** that distinguishes clean from attacked images using:

- **Explanation maps:** which image regions support the prediction.
- **ResNet features:** the attacked classifier’s internal representation.
- **DINOv2 features:** an independent model’s representation.

Stage 1 trains on PGD attacks and tests on several attack families.

#### 2.6.2 PGD/APGD detection works; transfer to other attacks is weak

Attack cells show **AUC / TPR at the 5% clean-calibration target**. AUC measures ranking (1.0 = perfect separation; 0.5 = chance-level ranking). TPR is the percentage of successful attacks detected at the frozen threshold. The final column reports the **actual clean-test FPR** at that same threshold; it need not equal the 5% calibration target.

| Detector features | PGD | APGD | CW | Patch | FGSM | Actual clean-test FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grad-CAM | 0.917 / 59.6% | 0.928 / 63.9% | 0.500 / 8.1% | 0.381 / 3.0% | 0.586 / 14.3% | 4.9% |
| SLICE signed | 0.910 / 85.3% | 0.961 / 94.7% | 0.528 / 3.2% | 0.583 / 15.2% | 0.548 / 2.1% | 4.6% |
| BELIEF signed | 0.715 / 0.6% | 0.716 / 0.6% | 0.235 / 0.0% | 0.561 / 0.3% | 0.311 / 0.4% | 0.3% |
| ResNet pooled | 1.000 / 99.7% | 1.000 / 99.7% | 0.645 / 11.3% | 0.872 / 50.5% | 0.864 / 38.2% | 2.6% |
| DINOv2 | 0.992 / 97.2% | 0.993 / 97.2% | 0.668 / 20.6% | 0.514 / 6.1% | 0.998 / 99.6% | 7.1% |

“Signed” features retain both positive and negative attribution information.

**Interpretation:** learning the signature of PGD transfers well to APGD, but does not establish a general detector of adversarial images. Scores below 0.5 indicate that the learned ranking often reverses on the new attack family.

AUC also differs from actual detection at a chosen threshold. With thresholds calibrated to target 5% clean false alarms:

- SLICE signed catches **85.3% of PGD** and **94.7% of APGD**.
- It catches only **3.2% of CW**, **15.2% of patch**, and **2.1% of FGSM**.
- ResNet catches **99.7% of PGD/APGD**, but still only **11.3% of CW**.

These detection rates count **successful attacks**, not all attempted attacks.

#### 2.6.3 Better detection comes with a noise tradeoff

On Gaussian noise with standard deviation 8/255:

| Detector | Noise incorrectly flagged |
|---|---:|
| Grad-CAM | 1% |
| SLICE signed | 5% |
| ResNet pooled | 42% |
| DINOv2 | 100% |

DINOv2’s strong FGSM performance therefore needs qualification: it also flags many randomly perturbed images. This is consistent with sensitivity to perturbations that are not specific to adversarial intent, without proving the underlying mechanism.

Explanation features are less sensitive to these noise controls, but that advantage must be weighed against their missed attacks. Actual clean-test false-alarm rates are **4.9% for Grad-CAM, 4.6% for SLICE signed, 2.6% for ResNet and 7.1% for DINOv2**. A 5% calibration target does not guarantee exactly 5% on new data, so these TPR comparisons are not at identical realized test FPR.

#### 2.6.4 Many explanations collapse to empty maps

Empty maps are a key diagnostic for interpreting the detection results.

| Test input | SLICE: empty positive map | BELIEF: empty positive map |
|---|---:|---:|
| Clean | 3.1% | 60.6% |
| PGD | 85.3% | 99.7% |
| APGD | 94.4% | 100% |

For SLICE, empty maps are much more common under PGD/APGD. **That pattern could itself help detection**, without demonstrating that the method produces informative explanations of attacked predictions.

BELIEF is frequently empty even on clean images. Its positive-only detector catches **zero attacks at the selected threshold**, despite nonzero AUC.

Likewise, repeatedly producing an empty map can look perfectly “stable.” Stability alone is therefore insufficient evidence of explanation quality. These findings apply to the tested settings and mixed-precision execution; a full-precision comparison remains unresolved.

#### 2.6.5 Training on more attack types does not consistently help

Stage 2 compares mixture training against a **refitted, matched PGD-only control**, using identical source pools and equal training counts within each holdout. The held-out attack is excluded from both fitting and tuning. This PGD-only control differs from the full Stage-1 detector in §2.6.2.

The table includes all five explanation feature sets and three held-out attacks. Values are **means over three allocation seeds**. TPR uses each fitted detector's threshold calibrated to target 5% clean false alarms; actual clean-test FPR is shown alongside it. **Bold positive AUC changes** identify setups with higher observed mean AUC under mixture training; they do not imply statistically established gains.

| Held-out attack | Features | PGD-only AUC / TPR | Mixture AUC / TPR | AUC change | Actual clean FPR: PGD-only → mixture |
| --- | --- | ---: | ---: | ---: | ---: |
| CW | Grad-CAM | 0.531 / 9.3% | 0.705 / 21.2% | **+0.174** | 4.3% → 5.9% |
| CW | SLICE | 0.494 / 4.4% | 0.499 / 4.0% | **+0.005** | 4.9% → 2.4% |
| CW | BELIEF | 0.214 / 0.8% | 0.247 / 6.5% | **+0.034** | 1.7% → 4.4% |
| CW | SLICE signed | 0.528 / 2.4% | 0.518 / 1.6% | -0.011 | 4.3% → 4.3% |
| CW | BELIEF signed | 0.219 / 0.0% | 0.313 / 10.5% | **+0.094** | 0.0% → 5.9% |
| PATCH | Grad-CAM | 0.478 / 6.4% | 0.663 / 10.7% | **+0.185** | 6.3% → 5.1% |
| PATCH | SLICE | 0.580 / 0.3% | 0.571 / 1.9% | -0.009 | 0.9% → 1.9% |
| PATCH | BELIEF | 0.549 / 0.7% | 0.476 / 3.8% | -0.073 | 1.1% → 7.1% |
| PATCH | SLICE signed | 0.601 / 16.2% | 0.590 / 16.3% | -0.011 | 4.0% → 5.0% |
| PATCH | BELIEF signed | 0.557 / 0.3% | 0.498 / 5.7% | -0.059 | 0.0% → 7.3% |
| APGD | Grad-CAM | 0.921 / 68.3% | 0.675 / 19.5% | -0.245 | 6.9% → 5.9% |
| APGD | SLICE | 0.947 / 94.4% | 0.840 / 0.1% | -0.107 | 4.0% → 6.6% |
| APGD | BELIEF | 0.680 / 0.6% | 0.423 / 0.0% | -0.257 | 2.0% → 6.7% |
| APGD | SLICE signed | 0.963 / 94.7% | 0.919 / 31.6% | -0.043 | 3.7% → 5.9% |
| APGD | BELIEF signed | 0.716 / 0.0% | 0.448 / 0.0% | -0.268 | 0.0% → 5.9% |

**Which setups benefit?** Mean AUC increases in **5 of 15 comparisons**: Grad-CAM, SLICE, BELIEF and BELIEF signed on held-out CW, and Grad-CAM on held-out patch. Every explanation method loses mean AUC on held-out APGD. Grad-CAM shows the clearest AUC tradeoff: CW improves from 0.531 to 0.705 and patch from 0.478 to 0.663, while APGD drops from 0.921 to 0.675.

**AUC gains do not always mean better detection at the chosen threshold.** SLICE's CW AUC rises slightly, but its TPR falls from 4.4% to 4.0%. Conversely, some patch TPRs rise despite lower AUC, sometimes alongside higher clean FPR. BELIEF and BELIEF signed remain below chance-level CW AUC even after their improvements. The table therefore identifies gains in ranking separately from operational detection and false-alarm costs.

Source: [three-seed matched Stage-2 results](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_summary.csv). The saved [bootstrap intervals](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_bootstrap.csv) describe the primary seed only, not uncertainty in the three-seed mean. These results suggest a tradeoff between learning different attack signatures, rather than a consistent benefit from greater training diversity.

#### 2.6.6 Training specifically on CW changes the result

For CW detection:

| Features | PGD-trained AUC | CW-trained AUC |
|---|---:|---:|
| Grad-CAM | 0.500 | 0.774 |
| BELIEF signed | 0.235 | 0.778 |
| Confidence baseline | 0.113 | **0.887** |
| ResNet pooled | 0.645 | 0.698 |

The simple confidence baseline catches **55.6% of successful CW attacks** after CW training, with 4.6% clean false alarms. However, its PGD/APGD detection collapses.

**The defensible project conclusion is that detection depends strongly on the attack family used for training.** These results do not establish a universally best detector or show that better explanation maps necessarily yield better detection. They instead reveal useful tradeoffs among attack coverage, noise sensitivity, and explanation reliability. All attacks here are non-adaptive—they were not optimized to evade the detector. Feature dimensions and extraction precision differ across pipelines, and no cross-pipeline significance test was performed.

Detailed results and provenance: [full experiment report](results/slice_belief_cifar10_native32_fixed1000/MAIN_RESULTS.md), [PGD-trained detection](results/slice_belief_cifar10_native32_fixed1000/tables/stage1_detection.csv), [noise controls](results/slice_belief_cifar10_native32_fixed1000/tables/stage1_noise.csv), [matched Stage 2](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_summary.csv), [CW-trained detection](results/slice_belief_cifar10_native32_fixed1000/cw_train/stage1_detection.csv), [explanation status](results/slice_belief_cifar10_native32_fixed1000/tables/explanation_status.csv), and [validation](results/slice_belief_cifar10_native32_fixed1000/validation.json). The full report retains all methods, uncertainty intervals, fidelity, stability, timing and execution details.

<!-- cifar10-native32-explanation:end -->


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

Items 1–4 below summarize Imagenette; native-32 CIFAR-10 findings follow in §3.7.

1. On identical saved Imagenette inputs, ResNet pooled features outperform the tested explanation features in Stage-1 AUC, with strong PGD/APGD performance and improved patch/FGSM detection. CW remains a weakness at calibrated thresholds.
2. Signed BELIEF is a useful explanation ablation, especially for patch. Explanation stability and deletion fidelity alone do not predict detector quality.
3. Mixture training helps some signed explanation detectors but lowers ResNet's mean held-out AUC relative to its matched PGD-only controls. Attack diversity is not uniformly beneficial under this training protocol.
4. Report actual clean/noise FPR alongside TPR, retain the membership/confidence controls, and keep the prediction-audit and pretraining-overlap limitations visible.
5. The completed native-32 CIFAR-10 study now compares explanations with ResNet pooled features and the independent DINOv2 observer on identical saved tensors, splits, masks and calibration rules (§§1.5, 2.6, 3.7). Imagenette findings above remain specific to that corpus.

<!-- cifar10-native32-representation:start -->
### 3.7 CIFAR-10 native-32: representation comparison

The native-32 representations use the identical ResNet-targeted corpus described in the CIFAR-10 setup. DINOv2 remains an observer, rather than the attacked classifier. The following existing representation results are preserved beside the newly completed explanation results above. ResNet pooled features have 2,048 dimensions and DINOv2 CLS features have 384, versus 196/392 for explanation maps; these are complete-pipeline comparisons rather than dimension-matched ablations.

| Features | PGD | APGD | CW | PATCH | FGSM | Clean FPR |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet pooled | 1.000 / 99.7% | 1.000 / 99.7% | 0.645 / 11.3% | 0.872 / 50.5% | 0.864 / 38.2% | 2.6% |
| DINOv2 | 0.992 / 97.2% | 0.993 / 97.2% | 0.668 / 20.6% | 0.514 / 6.1% | 0.998 / 99.6% | 7.1% |

| Features | gauss2 FPR | gauss4 FPR | gauss8 FPR | unif8 FPR |
| --- | --- | --- | --- | --- |
| ResNet pooled | 6.0% | 32.0% | 42.0% | 37.0% |
| DINOv2 | 39.0% | 89.0% | 100.0% | 92.0% |

| Features | Holdout | Arm | N per class | AUC mean ± SD | TPR | Clean FPR | Mean noise FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet pooled | APGD | mixture | 221 | 0.985 ± 0.005 | 93.4% | 4.0% | 18.0% |
| ResNet pooled | APGD | pgd_only | 221 | 0.999 ± 0.000 | 99.4% | 2.6% | 21.8% |
| ResNet pooled | CW | mixture | 289 | 0.624 ± 0.003 | 10.1% | 2.9% | 8.2% |
| ResNet pooled | CW | pgd_only | 289 | 0.644 ± 0.000 | 12.5% | 2.9% | 27.3% |
| ResNet pooled | PATCH | mixture | 235 | 0.878 ± 0.025 | 56.0% | 5.0% | 22.6% |
| ResNet pooled | PATCH | pgd_only | 235 | 0.898 ± 0.000 | 55.6% | 2.0% | 18.8% |
| DINOv2 | APGD | mixture | 221 | 0.971 ± 0.002 | 85.4% | 5.8% | 59.2% |
| DINOv2 | APGD | pgd_only | 221 | 0.992 ± 0.000 | 96.2% | 6.6% | 75.5% |
| DINOv2 | CW | mixture | 289 | 0.628 ± 0.003 | 15.9% | 6.8% | 67.0% |
| DINOv2 | CW | pgd_only | 289 | 0.648 ± 0.000 | 17.7% | 5.7% | 75.8% |
| DINOv2 | PATCH | mixture | 235 | 0.492 ± 0.021 | 3.7% | 5.6% | 71.2% |
| DINOv2 | PATCH | pgd_only | 235 | 0.506 ± 0.000 | 5.4% | 6.3% | 76.5% |

Existing representation comparisons use the primary seed 42 and compare mixture minus refitted PGD-only within each pipeline, with test-source and calibration bootstrap resampling. Displayed estimates are bootstrap means, not three-seed means.

| Features | Holdout | Bootstrap mean AUC gain [95% interval] | Bootstrap mean TPR gain [95% interval] |
| --- | --- | --- | --- |
| ResNet pooled | CW | -0.016 [-0.039, 0.006] | 0.1% [-8.3%, 7.8%] |
| ResNet pooled | PATCH | -0.044 [-0.067, -0.022] | -7.4% [-24.0%, 7.5%] |
| ResNet pooled | APGD | -0.008 [-0.013, -0.004] | -5.4% [-12.3%, -2.2%] |
| DINOv2 | CW | -0.016 [-0.037, 0.003] | -5.2% [-15.4%, 4.0%] |
| DINOv2 | PATCH | -0.038 [-0.047, -0.028] | -1.8% [-5.4%, 1.7%] |
| DINOv2 | APGD | -0.023 [-0.033, -0.014] | -10.6% [-17.4%, -5.2%] |

Sources: [ResNet Stage 2 bootstrap](results/resnet_representation_cifar10_native32/stage2_bootstrap.csv), [DINOv2 Stage 2 bootstrap](results/dinov2_observer_cifar10_native32_stage2/stage2_bootstrap.csv). These comparisons do not test representation features against explanation features.

| Features | PGD-trained CW | CW-trained CW | CW-trained clean FPR | gauss2 FPR | gauss4 FPR | gauss8 FPR | unif8 FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet pooled | 0.645 / 11.3% | 0.698 / 14.5% | 4.9% | 12.0% | 34.0% | 47.0% | 34.0% |
| DINOv2 | 0.668 / 20.6% | 0.674 / 20.2% | 4.9% | 32.0% | 71.0% | 92.0% | 79.0% |

Cells show AUC / TPR at the 5% calibration target. Stage 2 uses three-seed means and sample AUC SD; the available paired bootstrap concerns mixture versus matched PGD-only within each representation pipeline at seed 42. No paired cross-pipeline significance was computed. Sources: [ResNet native-32](results/resnet_representation_cifar10_native32/), [DINOv2 native-32](results/dinov2_observer_cifar10_native32/), [DINOv2 Stage 2](results/dinov2_observer_cifar10_native32_stage2/), [ResNet CW training](results/resnet_representation_cifar10_native32_cw_train/), [DINOv2 CW training](results/dinov2_observer_cifar10_native32_cw_train/).

**Native-32 CIFAR-10 findings.** The highest observed Stage 1 AUC among the five explanation-map feature sets is PGD: Grad-CAM 0.917 versus ResNet pooled 1.000; APGD: SLICE signed 0.961 versus ResNet pooled 1.000; CW: SLICE signed 0.528 versus ResNet pooled 0.645; PATCH: SLICE signed 0.583 versus ResNet pooled 0.872; FGSM: Grad-CAM 0.586 versus ResNet pooled 0.864. These are descriptive comparisons selected after evaluation; the detailed tables retain all methods, TPRs and actual clean/noise FPRs. Mixture training raises the three-seed mean held-out AUC in 5/15 explanation-map × holdout comparisons. The [primary-seed bootstrap intervals](results/slice_belief_cifar10_native32_fixed1000/tables/stage2_bootstrap.csv) quantify individual within-pipeline arm comparisons; they are not cross-pipeline tests or a multiple-comparison-adjusted claim. The CIFAR-10 classifier has ten outputs, so this result does not inherit Imagenette's out-of-class membership confound.
<!-- cifar10-native32-representation:end -->


### 3.8 What the comparison supports

**ResNet offers stronger attack discrimination, with a noise cost.** On native-32 CIFAR-10 it detects 99.7% of successful PGD/APGD attacks and has higher Stage-1 AUC than every explanation-map feature set for all five attack families. Yet CW detection remains only 11.3%. Its Gaussian-8/255 false-alarm rate is 42%, compared with 1% for Grad-CAM and 5% for SLICE signed. Actual clean FPRs are 2.6%, 4.9% and 4.6%, respectively. The representation advantage therefore depends on which attacks and benign perturbations matter; these are not comparisons at identical realized test FPR.

**DINOv2's FGSM result is qualified by its noise sensitivity.** It detects 99.6% of successful FGSM attacks, but also flags 100% of Gaussian-8/255 inputs and 39% even at Gaussian-2/255, with 7.1% clean FPR. This is consistent with sensitivity to perturbations that are not specific to adversarial intent; it does not prove the underlying mechanism. DINOv2 observes attacks aimed at ResNet, so the experiment does not measure detection of attacks targeted at DINOv2 itself.

**Mixture training can reduce noise alarms without improving held-out AUC.** For example, the ResNet CW holdout's mean noise FPR falls from 27.3% to 8.2%, while CW AUC falls from 0.644 to 0.624 and TPR from 12.5% to 10.1%. Calling this simply an improvement or a failure would omit the tradeoff. Which operating point is preferable depends on the cost of missed attacks versus false alarms.

**The supported conclusion is limited to these complete pipelines and saved corpora.** Explanation maps do not outperform ResNet representations in PGD-trained Stage-1 AUC, but some yield fewer noise alarms. Empty-map behavior prevents interpreting successful detection as evidence of better explanations. CIFAR-10 removes the Imagenette out-of-class membership confound, but feature dimensions and extraction precision still differ, and no cross-pipeline significance test was performed. The attacks are non-adaptive and do not optimize against the detector.

The most informative follow-up comparisons would isolate the empty-map signal, check explanation precision sensitivity, and test whether training with ordinary noisy negatives reduces representation false alarms while retaining attack detection. These are proposed experiments, not demonstrated improvements.
