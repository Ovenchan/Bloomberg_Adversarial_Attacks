# CIFAR-10 native-32: completed explanation experiment

## Shared setup and validation

The primary CIFAR-10 experiment uses 1,000 class-balanced official test sources, partitioned 350/100/200/350 into train/development/clean calibration/test. Attacks and noise modify the native 32×32 pixels before bilinear resizing to the stored 224×224 tensors. The classifier is a frozen ImageNet-pretrained ResNet-50 with a linear head trained on the official CIFAR-10 training split. DINOv2 is an independent observer of the same ResNet-targeted attacks. The older 224-resolution attack experiment remains a separate diagnostic.

Native-space PGD/APGD/FGSM use an L∞ budget of 8/255. PGD and APGD each use 20 steps; the patch is 7×7 native pixels. CW uses c=1.0, kappa=0.0 and 100 steps with its own distortion objective; it is not constrained by the shared L∞ budget.

All 21 tensor blocks and three corpus metadata files match the saved ResNet provenance. The local classifier audit has **zero prediction mismatches**. All **4,282** required explanation records, Grad-CAM features, stability repetitions and fidelity records were read and validated. The saved development-selected sigma **0.3** and completed segmentation are reused.

SLICE and BELIEF share segmentation and the first 500-sample perturbation draw. SLICE uses 1000 bootstrap fits, at most 10 elimination steps and tolerance 3; BELIEF uses slack 1e-06. Training-only scaling precedes logistic regression; C is selected from [0.01, 0.1, 1.0, 10.0] using development AUC. Independent clean calibration freezes thresholds at targets [0.01, 0.05]. Settings are recorded in the execution environment and [native-32 configuration](../../configs/experiments/slice_belief_cifar10_native32_fixed1000.yaml).

| Test attack | Eligible sources | Successful attacks | ASR |
| --- | --- | --- | --- |
| PGD | 319 | 319 | 100.0% |
| APGD | 319 | 319 | 100.0% |
| CW | 319 | 248 | 77.7% |
| PATCH | 319 | 297 | 93.1% |
| FGSM | 319 | 238 | 74.6% |

Detection TPR divides by successful attacks; ASR and undetected-success rate divide by eligible clean-correct sources. Every attack is compared with all 350 clean test sources. Each noise control has 100 test sources. Failed attacks do not enter detector positives. FGSM is test-only.

Independent refitting of 16 PGD/CW-trained Stage 1 detectors verified 288 attack/noise operating points. AUC is recomputed with sklearn, while thresholds, TPR, clean/noise FPR and undetected rates are checked with direct score comparisons. The validation retains [all recomputed scores](validation_scores.csv) and [calibration thresholds](validation_thresholds.json).

| Stage 2 holdout | Permitted mixture | Matched N clean + N attacked | Matched dev sources |
| --- | --- | --- | --- |
| CW | PGD+APGD+PATCH | 289 + 289 | 83 |
| PATCH | PGD+APGD+CW | 235 + 235 | 64 |
| APGD | PGD+CW+PATCH | 221 + 221 | 60 |

Stage 2 refits PGD-only and mixture controls on identical successful source pools. The source allocation varies over seeds 42, 123 and 2026; neither the held-out family nor FGSM enters fitting or tuning. Both arms retain the original development-negative and clean-calibration protocol.

Matched development counts refer to successful positive-source pools; clean development negatives remain all 100 sources in each arm.

Sources: [validation](validation.json), [prediction audit](prediction_audit.json), [ASR counts](tables/asr.csv), [matched source pools](tables/stage2_matching.json), [per-seed assignments](tables/stage2_assignments.csv).

## Explanation detection, quality and cost

#### PGD-trained Stage 1 and false alarms

Cells show **AUC / TPR at the 5% clean-calibration target**. Actual clean test FPR is a separate column; the target does not guarantee a 5% test FPR. Positive explanation maps have 196 features and signed variants have 392. Confidence features retain the original corpus's saved maximum probabilities and entropies.

| Features | PGD | APGD | CW | PATCH | FGSM | Actual clean FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Grad-CAM | 0.917 / 59.6% | 0.928 / 63.9% | 0.500 / 8.1% | 0.381 / 3.0% | 0.586 / 14.3% | 4.9% |
| SLICE | 0.889 / 85.3% | 0.948 / 94.4% | 0.491 / 3.6% | 0.573 / 14.5% | 0.518 / 1.7% | 4.3% |
| BELIEF | 0.696 / 0.0% | 0.697 / 0.0% | 0.208 / 0.0% | 0.553 / 0.0% | 0.294 / 0.0% | 0.0% |
| SLICE signed | 0.910 / 85.3% | 0.961 / 94.7% | 0.528 / 3.2% | 0.583 / 15.2% | 0.548 / 2.1% | 4.6% |
| BELIEF signed | 0.715 / 0.6% | 0.716 / 0.6% | 0.235 / 0.0% | 0.561 / 0.3% | 0.311 / 0.4% | 0.3% |
| Confidence | 0.983 / 92.8% | 0.992 / 96.6% | 0.113 / 0.0% | 0.590 / 15.5% | 0.217 / 0.0% | 5.1% |
| Superpixel count | 0.717 / 29.8% | 0.711 / 32.0% | 0.557 / 14.9% | 0.456 / 6.4% | 0.772 / 41.2% | 10.3% |
| ResNet pooled | 1.000 / 99.7% | 1.000 / 99.7% | 0.645 / 11.3% | 0.872 / 50.5% | 0.864 / 38.2% | 2.6% |
| DINOv2 | 0.992 / 97.2% | 0.993 / 97.2% | 0.668 / 20.6% | 0.514 / 6.1% | 0.998 / 99.6% | 7.1% |

| PGD-trained features | gauss2 FPR | gauss4 FPR | gauss8 FPR | unif8 FPR |
| --- | --- | --- | --- | --- |
| Grad-CAM | 4.0% | 6.0% | 1.0% | 4.0% |
| SLICE | 1.0% | 5.0% | 3.0% | 5.0% |
| BELIEF | 0.0% | 0.0% | 0.0% | 0.0% |
| SLICE signed | 4.0% | 3.0% | 5.0% | 6.0% |
| BELIEF signed | 0.0% | 0.0% | 0.0% | 2.0% |
| Confidence | 3.0% | 2.0% | 2.0% | 2.0% |
| Superpixel count | 19.0% | 31.0% | 54.0% | 27.0% |

Sources: [Stage 1](tables/stage1_detection.csv), [noise controls](tables/stage1_noise.csv), [ResNet](../resnet_representation_cifar10_native32/stage1_detection.csv), [DINOv2](../dinov2_observer_cifar10_native32/stage1_detection.csv). The constant ten-class membership indicator is retained in raw CSVs but omitted here because every CIFAR-10 prediction belongs to its label set.

Positive-only BELIEF has zero TPR on every attack as well as zero clean/noise FPR at the 5% target. Its independent calibration check records zero calibration FPR under the strict decision rule. The zero false-alarm rate therefore does not indicate useful detection at this operating point.

#### Stage 2: matched mixtures and uncertainty

| Features | Holdout | Arm | N per class | AUC mean ± SD | TPR mean ± SD | Clean FPR | Mean noise FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BELIEF | APGD | mixture | 221 | 0.423 ± 0.040 | 0.0% ± 0.0% | 6.7% | 7.3% |
| BELIEF signed | APGD | mixture | 221 | 0.448 ± 0.029 | 0.0% ± 0.0% | 5.9% | 7.3% |
| Confidence | APGD | mixture | 221 | 0.008 ± 0.000 | 0.0% ± 0.0% | 6.3% | 10.5% |
| Grad-CAM | APGD | mixture | 221 | 0.675 ± 0.008 | 19.5% ± 3.6% | 5.9% | 5.6% |
| Superpixel count | APGD | mixture | 221 | 0.711 ± 0.000 | 32.0% ± 0.0% | 10.3% | 32.8% |
| SLICE | APGD | mixture | 221 | 0.840 ± 0.050 | 0.1% ± 0.2% | 6.6% | 5.0% |
| SLICE signed | APGD | mixture | 221 | 0.919 ± 0.024 | 31.6% ± 54.7% | 5.9% | 5.8% |
| BELIEF | APGD | pgd_only | 221 | 0.680 ± 0.000 | 0.6% ± 0.0% | 2.0% | 1.5% |
| BELIEF signed | APGD | pgd_only | 221 | 0.716 ± 0.000 | 0.0% ± 0.0% | 0.0% | 0.5% |
| Confidence | APGD | pgd_only | 221 | 0.992 ± 0.000 | 96.6% ± 0.0% | 5.1% | 2.3% |
| Grad-CAM | APGD | pgd_only | 221 | 0.921 ± 0.000 | 68.3% ± 0.0% | 6.9% | 5.5% |
| Superpixel count | APGD | pgd_only | 221 | 0.711 ± 0.000 | 32.0% ± 0.0% | 10.3% | 32.8% |
| SLICE | APGD | pgd_only | 221 | 0.947 ± 0.000 | 94.4% ± 0.0% | 4.0% | 2.8% |
| SLICE signed | APGD | pgd_only | 221 | 0.963 ± 0.000 | 94.7% ± 0.0% | 3.7% | 5.5% |
| BELIEF | CW | mixture | 289 | 0.247 ± 0.032 | 6.5% ± 4.3% | 4.4% | 4.0% |
| BELIEF signed | CW | mixture | 289 | 0.313 ± 0.011 | 10.5% ± 0.8% | 5.9% | 5.8% |
| Confidence | CW | mixture | 289 | 0.112 ± 0.000 | 0.0% ± 0.0% | 5.1% | 2.3% |
| Grad-CAM | CW | mixture | 289 | 0.705 ± 0.001 | 21.2% ± 0.9% | 5.9% | 4.6% |
| Superpixel count | CW | mixture | 289 | 0.557 ± 0.000 | 14.9% ± 0.0% | 10.3% | 32.8% |
| SLICE | CW | mixture | 289 | 0.499 ± 0.013 | 4.0% ± 3.0% | 2.4% | 2.3% |
| SLICE signed | CW | mixture | 289 | 0.518 ± 0.012 | 1.6% ± 1.2% | 4.3% | 4.9% |
| BELIEF | CW | pgd_only | 289 | 0.214 ± 0.000 | 0.8% ± 0.0% | 1.7% | 1.0% |
| BELIEF signed | CW | pgd_only | 289 | 0.219 ± 0.000 | 0.0% ± 0.0% | 0.0% | 0.5% |
| Confidence | CW | pgd_only | 289 | 0.117 ± 0.000 | 0.0% ± 0.0% | 5.1% | 2.3% |
| Grad-CAM | CW | pgd_only | 289 | 0.531 ± 0.000 | 9.3% ± 0.0% | 4.3% | 3.5% |
| Superpixel count | CW | pgd_only | 289 | 0.557 ± 0.000 | 14.9% ± 0.0% | 10.3% | 32.8% |
| SLICE | CW | pgd_only | 289 | 0.494 ± 0.000 | 4.4% ± 0.0% | 4.9% | 4.5% |
| SLICE signed | CW | pgd_only | 289 | 0.528 ± 0.000 | 2.4% ± 0.0% | 4.3% | 4.0% |
| BELIEF | PATCH | mixture | 235 | 0.476 ± 0.002 | 3.8% ± 1.9% | 7.1% | 8.4% |
| BELIEF signed | PATCH | mixture | 235 | 0.498 ± 0.009 | 5.7% ± 1.0% | 7.3% | 9.4% |
| Confidence | PATCH | mixture | 235 | 0.471 ± 0.002 | 4.7% ± 0.0% | 7.4% | 10.5% |
| Grad-CAM | PATCH | mixture | 235 | 0.663 ± 0.008 | 10.7% ± 1.0% | 5.1% | 3.7% |
| Superpixel count | PATCH | mixture | 235 | 0.456 ± 0.000 | 6.4% ± 0.0% | 10.3% | 32.8% |
| SLICE | PATCH | mixture | 235 | 0.571 ± 0.007 | 1.9% ± 1.2% | 1.9% | 1.8% |
| SLICE signed | PATCH | mixture | 235 | 0.590 ± 0.007 | 16.3% ± 0.7% | 5.0% | 5.0% |
| BELIEF | PATCH | pgd_only | 235 | 0.549 ± 0.000 | 0.7% ± 0.0% | 1.1% | 1.0% |
| BELIEF signed | PATCH | pgd_only | 235 | 0.557 ± 0.000 | 0.3% ± 0.0% | 0.0% | 0.5% |
| Confidence | PATCH | pgd_only | 235 | 0.589 ± 0.000 | 15.5% ± 0.0% | 5.1% | 2.3% |
| Grad-CAM | PATCH | pgd_only | 235 | 0.478 ± 0.000 | 6.4% ± 0.0% | 6.3% | 4.8% |
| Superpixel count | PATCH | pgd_only | 235 | 0.456 ± 0.000 | 6.4% ± 0.0% | 10.3% | 32.8% |
| SLICE | PATCH | pgd_only | 235 | 0.580 ± 0.000 | 0.3% ± 0.0% | 0.9% | 0.8% |
| SLICE signed | PATCH | pgd_only | 235 | 0.601 ± 0.000 | 16.2% ± 0.0% | 4.0% | 6.2% |

The following paired source-bootstrap intervals compare mixture minus its refitted PGD-only control at seed 42 (2,000 test-source resamples, with calibration resampled). They describe the primary seed, not the uncertainty of the three-seed mean. The displayed gain estimates are bootstrap means.

| Features | Holdout | Bootstrap mean AUC gain [95% interval] | Bootstrap mean TPR gain [95% interval] |
| --- | --- | --- | --- |
| Grad-CAM | CW | 0.175 [0.143, 0.209] | 9.1% [0.7%, 18.1%] |
| SLICE | CW | 0.015 [-0.017, 0.048] | 3.4% [-1.6%, 8.4%] |
| BELIEF | CW | 0.025 [-0.003, 0.054] | 4.0% [1.2%, 7.0%] |
| SLICE signed | CW | 0.003 [-0.035, 0.042] | 0.1% [-3.9%, 6.4%] |
| BELIEF signed | CW | 0.086 [0.043, 0.130] | 11.7% [7.0%, 17.9%] |
| Confidence | CW | -0.005 [-0.011, 0.000] | 0.0% [0.0%, 0.0%] |
| Superpixel count | CW | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| Grad-CAM | PATCH | 0.192 [0.160, 0.225] | 5.2% [-0.3%, 11.7%] |
| SLICE | PATCH | -0.013 [-0.041, 0.014] | -5.0% [-18.1%, 1.4%] |
| BELIEF | PATCH | -0.076 [-0.134, -0.020] | 1.7% [-0.7%, 6.0%] |
| SLICE signed | PATCH | -0.002 [-0.030, 0.025] | -0.1% [-8.3%, 4.6%] |
| BELIEF signed | PATCH | -0.055 [-0.110, -0.001] | 4.7% [1.3%, 8.4%] |
| Confidence | PATCH | -0.115 [-0.198, -0.029] | -10.6% [-17.5%, -3.6%] |
| Superpixel count | PATCH | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| Grad-CAM | APGD | -0.243 [-0.279, -0.209] | -46.7% [-58.6%, -30.9%] |
| SLICE | APGD | -0.148 [-0.187, -0.110] | -78.3% [-96.6%, 0.0%] |
| BELIEF | APGD | -0.236 [-0.286, -0.191] | -0.6% [-1.6%, 0.0%] |
| SLICE signed | APGD | -0.068 [-0.095, -0.044] | -83.2% [-96.9%, 0.3%] |
| BELIEF signed | APGD | -0.246 [-0.291, -0.203] | 0.0% [0.0%, 0.0%] |
| Confidence | APGD | -0.984 [-0.993, -0.972] | -96.2% [-98.7%, -93.0%] |
| Superpixel count | APGD | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |

Sources: [three-seed summaries](tables/stage2_summary.csv), [per-seed metrics](tables/stage2_per_seed.csv), [per-seed noise](tables/stage2_noise.csv), [training-arm bootstrap](tables/stage2_bootstrap.csv), [Stage 1 comparisons against Grad-CAM](tables/stage1_bootstrap.csv). Bootstrap comparisons are within the explanation pipeline; no cross-pipeline significance test was computed.

#### CW-trained Stage 1

| Features | PGD | APGD | CW | PATCH | FGSM | Actual clean FPR |
| --- | --- | --- | --- | --- | --- | --- |
| Grad-CAM | 0.445 / 3.4% | 0.387 / 2.2% | 0.774 / 28.6% | 0.784 / 23.6% | 0.719 / 21.8% | 4.3% |
| SLICE | 0.397 / 1.3% | 0.375 / 0.0% | 0.515 / 5.2% | 0.480 / 4.0% | 0.498 / 2.5% | 5.7% |
| BELIEF | 0.318 / 0.0% | 0.317 / 0.0% | 0.767 / 15.3% | 0.453 / 3.4% | 0.698 / 8.8% | 4.6% |
| SLICE signed | 0.345 / 0.6% | 0.320 / 0.0% | 0.485 / 4.4% | 0.482 / 6.1% | 0.512 / 3.8% | 4.6% |
| BELIEF signed | 0.297 / 0.0% | 0.296 / 0.0% | 0.778 / 14.9% | 0.448 / 3.0% | 0.702 / 9.7% | 6.3% |
| Confidence | 0.017 / 0.0% | 0.008 / 0.0% | 0.887 / 55.6% | 0.410 / 0.3% | 0.783 / 23.5% | 4.6% |
| Superpixel count | 0.717 / 29.8% | 0.711 / 32.0% | 0.557 / 14.9% | 0.456 / 6.4% | 0.772 / 41.2% | 10.3% |

| CW-trained features | gauss2 FPR | gauss4 FPR | gauss8 FPR | unif8 FPR |
| --- | --- | --- | --- | --- |
| Grad-CAM | 3.0% | 5.0% | 8.0% | 3.0% |
| SLICE | 7.0% | 3.0% | 6.0% | 7.0% |
| BELIEF | 6.0% | 6.0% | 10.0% | 5.0% |
| SLICE signed | 7.0% | 7.0% | 4.0% | 7.0% |
| BELIEF signed | 6.0% | 6.0% | 8.0% | 5.0% |
| Confidence | 6.0% | 5.0% | 11.0% | 11.0% |
| Superpixel count | 19.0% | 31.0% | 54.0% | 27.0% |

CW training uses successful CW training positives and CW development tuning. Sources: [CW-trained detection](cw_train/stage1_detection.csv), [CW-trained noise](cw_train/stage1_noise.csv), [CW bootstrap](cw_train/stage1_bootstrap.csv).

#### Explanation quality, stability and cost

| Method | Records | Zero positive maps | LIME fallback | All eliminated | Nonfinite status |
| --- | --- | --- | --- | --- | --- |
| SLICE | 4282 | 1442/4282 (33.7%) | 1543 | 0 | 0 |
| BELIEF | 4282 | 2789/4282 (65.1%) | 0 | 2698 | 0 |

Grad-CAM has 0/4282 (0.0%) zero positive feature maps on the required explanation inputs and zero prediction mismatches across all stored rows.

Every stored coefficient array is readable, dimension-checked and finite. Zero maps and all-eliminated or fallback outputs are retained and reported; they are not silently dropped.

| Test variant | Method | N | Zero positive maps | LIME fallback | All eliminated | Mean positive superpixels |
| --- | --- | --- | --- | --- | --- | --- |
| clean | SLICE | 350 | 3.1% | 7.4% | 0.0% | 4.731 |
| clean | BELIEF | 350 | 60.6% | 0.0% | 56.9% | 1.291 |
| pgd | SLICE | 319 | 85.3% | 92.5% | 0.0% | 1.508 |
| pgd | BELIEF | 319 | 99.7% | 0.0% | 99.7% | 0.009 |
| apgd | SLICE | 319 | 94.4% | 96.9% | 0.0% | 0.589 |
| apgd | BELIEF | 319 | 100.0% | 0.0% | 100.0% | 0.000 |
| cw | SLICE | 248 | 1.2% | 0.0% | 0.0% | 5.016 |
| cw | BELIEF | 248 | 7.7% | 0.0% | 6.5% | 3.786 |
| patch | SLICE | 297 | 13.8% | 16.2% | 0.0% | 4.101 |
| patch | BELIEF | 297 | 71.4% | 0.0% | 67.3% | 0.933 |
| fgsm | SLICE | 238 | 1.3% | 0.0% | 0.0% | 5.197 |
| fgsm | BELIEF | 238 | 19.3% | 0.0% | 16.8% | 3.277 |
| gauss2 | SLICE | 100 | 1.0% | 4.0% | 0.0% | 5.150 |
| gauss2 | BELIEF | 100 | 52.0% | 0.0% | 48.0% | 1.640 |
| gauss4 | SLICE | 100 | 3.0% | 5.0% | 0.0% | 4.960 |
| gauss4 | BELIEF | 100 | 57.0% | 0.0% | 51.0% | 1.770 |
| gauss8 | SLICE | 100 | 2.0% | 3.0% | 0.0% | 5.360 |
| gauss8 | BELIEF | 100 | 45.0% | 0.0% | 40.0% | 2.230 |
| unif8 | SLICE | 100 | 4.0% | 3.0% | 0.0% | 5.230 |
| unif8 | BELIEF | 100 | 53.0% | 0.0% | 48.0% | 1.760 |

High zero-map, all-eliminated or fallback rates limit interpretation as consistently informative local attributions. The detection results describe this fixed sigma and CUDA precision setting; they do not establish how a full-precision rerun would compare.

| Panel | Method | Inputs | Repeats | Zero positive maps | Sign agreement | Top-3 Jaccard | Spearman | Valid Spearman pairs | Detector-score SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | SLICE | 20 | 5 | 0.0% | 0.889 | 0.708 | 0.897 | 200 | 0.048 |
| clean | BELIEF | 20 | 5 | 55.0% | 0.971 | 0.924 | 0.912 | 90 | 0.005 |
| clean | Grad-CAM | 20 | deterministic | — | 1.000 | 1.000 | 1.000 | — | 0.000 |
| pgd | SLICE | 18 | 5 | 80.0% | 0.930 | 0.871 | 0.578 | 33 | 0.036 |
| pgd | BELIEF | 18 | 5 | 100.0% | 1.000 | 1.000 | undefined | 0 | 0.000 |
| pgd | Grad-CAM | 18 | deterministic | — | 1.000 | 1.000 | 1.000 | — | 0.000 |

The Grad-CAM stability row is the pipeline's deterministic-control assumption, not a repeated CUDA measurement. SLICE and BELIEF are the methods actually recomputed five times per selected input. Two empty positive maps receive Jaccard 1; Spearman excludes constant coefficient arrays. Interpret stability together with panel zero-map rates and valid Spearman-pair counts.

| Panel | Method | N | AOPC deletion mean ± SE |
| --- | --- | --- | --- |
| clean | Grad-CAM | 350 | -4.779e-05 ± 4.208e-05 |
| clean | SLICE | 350 | 0.000679 ± 8.492e-05 |
| clean | BELIEF | 350 | 0.0006248 ± 8.174e-05 |
| clean | random | 350 | -4.977e-05 ± 3.321e-05 |
| pgd | Grad-CAM | 319 | 8.386e-07 ± 9.06e-07 |
| pgd | SLICE | 319 | 2.406e-06 ± 1.965e-06 |
| pgd | BELIEF | 319 | 1.406e-06 ± 1.492e-06 |
| pgd | random | 319 | 1.383e-06 ± 1.395e-06 |

| Method / timing origin | Median seconds/input | Mean seconds/input | P90 seconds/input | Median queries | P90 queries |
| --- | --- | --- | --- | --- | --- |
| slice (local CUDA) | 2.627 | 2.4 | 4.129 | 2995.0 | 4991.0 |
| belief (local CUDA) | 0.2875 | 0.2974 | 0.3892 | 500.0 | 500.0 |
| gradcam (local CUDA) | 0.001244 | 0.001216 | — | 1 | — |
| quickshift (shared by SLICE/BELIEF) (transferred CPU segmentation) | 0.6436 | 0.6608 | — | — | — |

SLICE and BELIEF standalone costs each include their shared initial sample, so their columns should not be summed as the joint run time. These timers exclude shared context creation and segmentation. Grad-CAM's displayed mean/median are unweighted aggregates of 21 shard-level per-image batch times, rather than individual-image latency percentiles. Quickshift measurements come from the transferred segmentation and are kept separate from the CUDA extraction benchmark. Fidelity uses the saved blur-fill policy and five top-positive-superpixel deletion ranks (plus a random-ranking control); it does not separately measure deletion of negative evidence and is not a detector robustness metric.

Sources: [explanation status](tables/explanation_status.csv), [stability](tables/stability.csv), [fidelity summary](tables/fidelity_summary.csv), [per-input fidelity](fidelity.csv), [runtime/query cost](tables/cost.csv).

**Execution and cache decision:** Python 3.12.10, NVIDIA GeForce RTX 4070 Ti, CUDA runtime 12.4; torch 2.4.1+cu124, torchvision 0.19.1+cu124, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2, scikit-image 0.23.2, torchattacks 3.5.1. All primary Grad-CAM, SLICE/BELIEF and repeat caches were regenerated locally. Perturbation-model queries use CUDA float16 autocast/channels-last, Grad-CAM and the audit use float32, and regression uses CPU float64. TF32 is disabled: the default cuDNN TF32 path changed one clean prediction, while full CUDA FP32 agrees with CPU FP32/FP64 and the saved baseline. The strict zero-mismatch audit was then rerun over every block. The transferred 220 MPS full-precision explanations and partial repeat directory are preserved under `data/processed/slice_belief_cifar10_native32_fixed1000/archives/mps_transfer_20261123/`; none enters the primary explanation results or CUDA timing. The handoff stated 200 records; the unchanged checkpoint and original MPS log show 220. Segmentation and the MPS development-selected sigma 0.3 are reused. Seeds, splits, attack tensors, explanation budgets, and detector/calibration protocol are unchanged. The requested tests passed with 53 passed and 1 skipped (MPS unavailable); the added CUDA mixed-precision query test passed. The pinned upstream reference cases also ran locally; [reference provenance](reference_validation.json) records source commits, hashes, optional test dependencies and the separate parity run. Sources: [environment and archive hashes](execution_environment.json), [numerical diagnosis](audit_attempts/numerical_diagnosis.json), [package lock](requirements-windows-cuda.lock.txt), [test transcript](pytest.txt).

## Representation comparison

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

Sources: [ResNet Stage 2 bootstrap](../resnet_representation_cifar10_native32/stage2_bootstrap.csv), [DINOv2 Stage 2 bootstrap](../dinov2_observer_cifar10_native32_stage2/stage2_bootstrap.csv). These comparisons do not test representation features against explanation features.

| Features | PGD-trained CW | CW-trained CW | CW-trained clean FPR | gauss2 FPR | gauss4 FPR | gauss8 FPR | unif8 FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet pooled | 0.645 / 11.3% | 0.698 / 14.5% | 4.9% | 12.0% | 34.0% | 47.0% | 34.0% |
| DINOv2 | 0.668 / 20.6% | 0.674 / 20.2% | 4.9% | 32.0% | 71.0% | 92.0% | 79.0% |

Cells show AUC / TPR at the 5% calibration target. Stage 2 uses three-seed means and sample AUC SD; the available paired bootstrap concerns mixture versus matched PGD-only within each representation pipeline at seed 42. No paired cross-pipeline significance was computed. Sources: [ResNet native-32](../resnet_representation_cifar10_native32/), [DINOv2 native-32](../dinov2_observer_cifar10_native32/), [DINOv2 Stage 2](../dinov2_observer_cifar10_native32_stage2/), [ResNet CW training](../resnet_representation_cifar10_native32_cw_train/), [DINOv2 CW training](../dinov2_observer_cifar10_native32_cw_train/).

**Native-32 CIFAR-10 findings.** The highest observed Stage 1 AUC among the five explanation-map feature sets is PGD: Grad-CAM 0.917 versus ResNet pooled 1.000; APGD: SLICE signed 0.961 versus ResNet pooled 1.000; CW: SLICE signed 0.528 versus ResNet pooled 0.645; PATCH: SLICE signed 0.583 versus ResNet pooled 0.872; FGSM: Grad-CAM 0.586 versus ResNet pooled 0.864. These are descriptive comparisons selected after evaluation; the detailed tables retain all methods, TPRs and actual clean/noise FPRs. Mixture training raises the three-seed mean held-out AUC in 5/15 explanation-map × holdout comparisons. The [primary-seed bootstrap intervals](tables/stage2_bootstrap.csv) quantify individual within-pipeline arm comparisons; they are not cross-pipeline tests or a multiple-comparison-adjusted claim. The CIFAR-10 classifier has ten outputs, so this result does not inherit Imagenette's out-of-class membership confound.
