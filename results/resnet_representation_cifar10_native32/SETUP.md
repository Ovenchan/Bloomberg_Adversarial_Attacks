# Experiment setup

Frozen ResNet-50 (`IMAGENET1K_V2`) supplies 2,048-dimensional pooled features.
Both stages reuse the explanation experiment's exact saved images and original
ResNet eligibility/success masks; local prediction differences are recorded in
the feature provenance. No attacks or SLICE/BELIEF explanations are regenerated.

- Source partitions: 350 train, 100 development,
  200 clean calibration, 350 test; all variants stay with their source.
- Detector: training-only StandardScaler + logistic regression; C selected on development AUC.
- Stage 1: all clean training samples and successful PGD examples.
- Stage 2: fresh matched PGD-only and mixture detectors, each with N clean + N attacked
  samples from identical sources. The held-out family never enters fitting or tuning.
- Stage 2 seeds: [42, 123, 2026]. Thresholds are independently calibrated
  on clean calibration data at 1% and 5%, then frozen for all test conditions.
- FGSM is used only as an additional unseen-attack test in both stages. It is never
  included in detector training or hyperparameter selection, and is not a separate
  Stage 2 holdout setting. Thus, `attack=fgsm` with `is_holdout=False` does not
  mean FGSM was seen during training.
- Test detection includes only originally successful attacks. Clean FPR includes all clean
  test images; noise FPR is measured on the shared noise panel.

## Reading combined_results.csv

Each row is a stage/training strategy/held-out setting/test attack combination.
Stage 1 has one run; its `_mean` columns are the single-run results and `_sd` is blank.
Stage 2 averages over seeds; `_sd` is sample standard deviation, not a confidence interval.
Filter `stage=stage2` and `is_holdout=True` for the main generalization comparisons.
Stage 1 has no Stage 2 holdout designation. `num_train_clean` and
`num_train_adversarial` explicitly distinguish the full and matched training pools.

`tpr@0.05_mean` uses the clean-calibrated threshold; `test_roc_tpr@0.05_mean`
is a different, descriptive test-ROC statistic. Always compare like columns and
report actual test FPR. `undetected` divides missed successful attacks by eligible
attempts, whereas TPR divides detected attacks by successful attacks.
Noise means average the noise conditions within each run, then average runs.

The original detailed CSVs, confidence intervals, scores, thresholds and fitted
models are retained in this folder. Stage 2 paired-bootstrap gains are in
`stage2_bootstrap.csv` and apply to primary seed 42, not the three-seed mean.
