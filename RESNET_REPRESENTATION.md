# ResNet representation baseline

This addition extracts 2,048-dimensional ResNet-50 global-average-pooled features
from the **saved explanation corpus**, then trains a PGD-only logistic detector.
The same file implements Stage 1 and Stage 2 for ResNet. An optional DINOv2
observer uses the same corpus and evaluation code for Stage 1.

## Run both ResNet stages in one command

```bash
source .venv-corpus/bin/activate
python -m experiments.resnet_representation \
  --encoder resnet --stage both --device mps \
  --prediction-policy record \
  --output results/resnet_representation_fixed1000_complete
```

Both stages are implemented in `experiments/resnet_representation.py`. Features
are extracted exactly once. Stage 1 files go in the output root, cached vectors
in `features/`, and Stage 2 files in the same output root.
`combined_results.csv` summarizes both stages, `SETUP.md` explains the protocol,
and `MAIN_RESULTS.md` presents the main results. Detailed per-stage audit files
remain available alongside the combined table. The saved original corpus is read
without regenerating attacks or explanations. Use a new output directory.

If a ResNet feature cache already exists, skip extraction:

```bash
python -m experiments.resnet_representation \
  --encoder resnet --stage both \
  --features /absolute/path/to/existing/features \
  --output results/resnet_representation_fixed1000_complete_from_cache
```

`--stage stage1` remains the default for compatibility; `--stage stage2` runs
only mixture evaluation. DINOv2 currently supports `--stage stage1` only.
`run.json` records which stages ran and the feature-cache location. The older
`resnet_representation_stage2` module remains a compatibility wrapper, with no
separate implementation of the Stage 2 experiment.

## DINOv2 on the same images

```bash
python -m experiments.resnet_representation --encoder dinov2 --check
python -m experiments.resnet_representation --encoder dinov2 --device cuda --batch-size 20
```

On this Mac, replace `python` with `/opt/anaconda3/bin/python` and use `--device
mps` or `--device cpu`. DINOv2 outputs default to
`results/dinov2_observer_fixed1000/`, separate from the ResNet baseline. Existing
output directories are never overwritten.

This mode loads frozen `dinov2_vits14` via the existing Torch Hub loader. First
use needs the upstream code/checkpoint download unless already cached. No DINO
classification head is trained. Input pixels are unchanged; the observer applies
ImageNet normalization once and extracts 384-dimensional CLS embeddings.
The loaded backbone's weight SHA-256 is saved in the feature provenance. The
existing Hub source is unpinned, so keep the same cached code/version for repeated
runs; a weight fingerprint alone does not guarantee implementation equivalence.

**The attack target remains ResNet-50.** The script verifies saved ResNet
predictions even when extracting DINO features. Success filtering, source splits,
clean calibration, noise panels, regularization grid and detector-training code
are identical. Each feature method independently selects C using the same
development rule and calibrates its own thresholds on the same clean images.
This is a comparison of feature encoders observing a common ResNet attack corpus,
not an evaluation of attacks against a DINO classifier. Do not combine it with
the historical DINOv2-native CIFAR-10 results as if the threat models matched.

Compare the two runs' `stage1_detection.csv` / `stage1_noise.csv`; confirm equal
corpus metadata hashes and per-block tensor hashes in `features/provenance.json`.
Both runs retain source IDs in `scores.csv` for subsequent paired analysis.
The DINO extraction timing excludes the separate ResNet prediction check.

## Required input

Use the original explanation-run configuration, defaulting to
`configs/experiments/slice_belief_imagenette_fixed1000.yaml`. Its `paths.work_dir`
must contain:

- `manifest.csv`: original source identities and partitions;
- `variants.csv`: image indices, predictions, eligibility and success masks;
- `corpus.json`: original configuration, weights and tensor hashes;
- `images/*.pt`: the original clean, attacked and noisy image tensors.

These files are ignored by Git and are not included in the clone. Obtain them
from the machine that ran the explanation experiment. The committed
`results/slice_belief_fixed1000/source_files.csv` identifies the sampled original
images but cannot reconstruct the attacked tensors. Original JPEGs, segmentation
artifacts and saved explanations are not needed once this corpus is available.
The script never generates attacks or modifies the corpus. Do not regenerate a
separate attack corpus and compare it as if it were the original paired run.

## Run

A separate local environment `.venv-corpus` now matches all seven versions
recorded in `corpus.json`. Activate it from the repository root:

```bash
source .venv-corpus/bin/activate
python -m experiments.resnet_representation --check
```

`requirements-corpus.txt` pins those recorded versions; the complete installed
Mac environment is captured in `requirements-corpus-macos.lock.txt`. The original
Python version and other dependency versions were not recorded, so this is not
an exact reconstruction of the original environment. It uses Python 3.12.11 on
arm64 macOS. CUDA versus MPS/CPU numerical differences may persist. Use a new
output directory for experiments under this environment so older results remain
available. The environment is excluded from Git.

From the repository root, using an environment with the repository dependencies:

```bash
python -m experiments.resnet_representation --check
python -m experiments.resnet_representation --device cuda --batch-size 20
```

Use `--device cpu` (default) or `--device mps` as appropriate. The locally verified
Python environment is `/opt/anaconda3/bin/python` (PyTorch 2.5.1 / torchvision
0.20.1). No new environment or package installation was performed.

To use another existing explanation experiment:

```bash
python -m experiments.resnet_representation \
  --config configs/experiments/slice_belief_imagenette.yaml \
  --output results/resnet_representation_900 --device cpu
```

The output directory must be new and outside the explanation corpus/results.
There is no overwrite or automatic resume option. If a run fails, fix the cause
and choose a new output directory. Preflight checks metadata and file presence;
extraction additionally validates tensor hashes and classifier predictions.
A prediction mismatch aborts the run; investigate weights, preprocessing and
numerical differences before treating the experiment as a paired comparison.

### Prediction differences across machines

Strict mode (default) now writes `features/prediction_mismatches.csv` before
aborting, including source IDs, original/local classes, correctness and logit
margins. GPU kernels and library versions can change the winning class for
near-tied adversarial outputs. Identical image tensors do not guarantee identical
predictions across machines. A mismatch is not automatically a harmless tie.

After reviewing the differences, an explicit audited run is available:

```bash
python -m experiments.resnet_representation --encoder resnet --device mps \
  --prediction-policy record --output results/resnet_representation_fixed1000_audited
```

The same option works for `--encoder dinov2` with a separate output directory.
This mode keeps all tensor-hash, configuration, identity and feature checks.
It records every local prediction difference, but continues using the original
ResNet eligibility/success masks so the comparison evaluates the original cohort.
It does not change the images or regenerate SLICE/BELIEF. Review the complete
report, particularly any changes in correctness, before interpreting results.
Report these as features evaluated on the original corpus, not exact numerical
reproduction of its classifier. To require exact prediction reproduction, retain
strict mode and use the original software/hardware environment.

An initial local CPU diagnostic of all 350 test APGD images found one difference:
tensor index 206, original predicted class 107 versus local class 78, true label
569; both predictions are incorrect. The local top-two logit gap was approximately
0.00124. This is consistent with numerical sensitivity but does not prove the
cause or establish that other corpus blocks have no differences. Original
PyTorch/torchvision versions were 2.4.1/0.19.1; local versions were 2.5.1/0.20.1.

## Evaluation and outputs

The extractor converts stored uint8 clean tensors to [0,1] and leaves stored
float attack/noise pixels at their original scale. ImageNet normalization is
applied once. There is no resizing, augmentation or clean/adversarial subtraction.
`ResNetEncoder.pooled_features()` returns the pre-classifier features; existing
`forward()` and `encode()` retain their original behavior.

All corpus variants are extracted, including unsuccessful attacks, with source
IDs. Detector evaluation selects the exact `explain` cohort: all clean/noise
images and successful attacks. Source identities must remain within one partition.

The detector uses the existing StandardScaler + logistic regression implementation.
Only training features fit the scaler and detector. Regularization is selected
by clean-versus-PGD development AUC. Thresholds are calibrated on independent
clean calibration images and frozen across test attacks and noise conditions.
Reported TPR is therefore at a **calibration target** FPR, not a threshold fitted
to the test ROC. Always report the observed test FPR alongside it.

New outputs under `results/resnet_representation_fixed1000/`:

- `features/*.npz`: features and source IDs, in original tensor-index order;
- `features/provenance.json`: corpus hashes, tensor hashes, weights, device,
  PyTorch version and feature-extraction timing (excludes I/O and prediction checks);
- `stage1_detection.csv`: AUC, calibrated TPR and intervals, actual clean FPR,
  attack-success counts/rates, and undetected-success rate;
- `stage1_noise.csv`: noise false-alarm rates and intervals;
- `scores.csv`: test and calibration scores with source IDs for later paired analysis;
- `thresholds.json`, `evaluation.json`, `detector.joblib`: frozen operating points,
  configuration and fitted detector including scaler.

Compare `stage1_detection.csv` and `stage1_noise.csv` with the corresponding
explanation tables only after verifying that both runs reference the same corpus.
The output does not claim statistical superiority; paired bootstrap comparisons
with explanation scores can be added after those scores are available.

## Stage 1 metric comparability audit

The representation evaluator now includes every detection/noise metric column
from `SliceBeliefAnalysis.evaluate`, including `test_roc_tpr@...`, noise
`prediction_changed`, and noise false alarms split by original clean correctness.
It retains the additional ASR and lower noise-confidence-bound columns. Development
selection uses the same `fast_auc` implementation as the explanation study.

Do not mix `tpr@...` (threshold frozen on clean calibration) with
`test_roc_tpr@...` (descriptive operating point selected on the test ROC).
Compare the same column across methods and report actual clean test FPR.
The confidence intervals in the detection CSV use the shared exact binomial
implementation; they are not the paired-bootstrap intervals in the teammate's
separate `stage1_bootstrap.csv`.

The completed matched ResNet run was re-evaluated from its cached features into
`results/resnet_representation_fixed1000_matched_metrics_aligned/`, preserving
the original run. Its audit JSON records 129 passing same-score comparisons
against the teammate's evaluator, unchanged existing metrics, and matching
test sample counts across all methods in the teammate's saved detection table.
No images, attacks, explanations, or backbone features were regenerated.

## Offline verification

```bash
python -m pytest tests/test_resnet_representation.py -q
```

Tests use random weights or a tiny injected encoder, with no image or checkpoint
download. They cover pooled-feature/classifier equivalence, raw-pixel conversion,
failed-attack filtering, unchanged input artifacts, source leakage, changed
attack tensors and classifier prediction mismatches. They do not establish
Imagenette detection performance.

## Stage 2 ResNet mixture training and held-out testing

Run the new entry point in the matched environment:

```bash
source .venv-corpus/bin/activate
python -m experiments.resnet_representation_stage2 \
  --features results/resnet_representation_fixed1000_matched_audited/features \
  --output results/resnet_representation_fixed1000_stage2
```

The feature path must contain all cached ResNet `.npz` blocks and
`provenance.json`. The original feature cache was no longer present when Stage 2
was implemented. Metric tables alone cannot train new detectors. Restore the
cache, or regenerate only ResNet features (plus its inexpensive Stage 1 analysis):

```bash
python -m experiments.resnet_representation \
  --encoder resnet --device mps --prediction-policy record \
  --output results/resnet_representation_fixed1000_reextracted
python -m experiments.resnet_representation_stage2 \
  --features results/resnet_representation_fixed1000_reextracted/features \
  --output results/resnet_representation_fixed1000_stage2
```

This does not regenerate images, attacks, segmentation, SLICE or BELIEF. Stage 2
itself uses cached vectors on CPU, with no backbone inference or GPU requirement.
Existing output directories are rejected rather than overwritten.

The runner inherits the teammate's matched-pool, allocation, detector fitting,
threshold calibration, and evaluation methods. Its Stage 2 loop follows the same
reference implementation with extra audit artifacts. For each held-out family,
train/dev pools contain sources successful under PGD and all permitted training
families; withheld-family success never selects the fitting pool. The PGD-only
and mixture arms have N clean and N successful adversarial samples from the same
source IDs, with one attack per source in the mixture. These are newly fitted,
sample-matched baselines, not the full-sample Stage 1 detector.

Verified counts from the actual corpus, matching the teammate's Stage 2 tables:

| Held out | Permitted mixture | N training sources per arm | Matched positive dev sources |
| --- | --- | ---: | ---: |
| CW | PGD + APGD + patch | 195 | 62 |
| Patch | PGD + APGD + CW | 320 | 94 |
| APGD | PGD + CW + patch | 195 | 62 |

Each arm uses all 100 clean development images, the same 200 independent clean
calibration images, and the same test/noise cohorts. Seeds are 42, 123, 2026.
Only permitted families select regularization; the held-out family is never used
for detector tuning. Each fitted detector evaluates every test attack, with
`is_holdout=True` marking the primary generalization result.

Outputs in the new directory:

- `stage2_per_seed.csv`: both arms, each held-out family/seed/test attack;
- `stage2_summary.csv`: mean and sample standard deviation over the three seeds;
- `stage2_bootstrap.csv`: mixture-minus-matched-PGD gains with paired source and
  calibration bootstrap (2,000 replicates, primary seed 42 only, as in the reference);
- `stage2_assignments.csv`: exact source pools and assigned training attacks;
- `stage2_scores.csv`, `stage2_noise.csv`, `stage2_thresholds.json`: auditable scores,
  noise false alarms and independently calibrated operating points per detector;
- `models/*.joblib`: all 18 fitted detectors including their scalers;
- `stage2_provenance.json`: feature source/hashes, corpus metadata hashes, config,
  original feature audit and reference code fingerprint.

Original success/eligibility masks remain authoritative, including the previously
observed CW training-image correctness disagreement. That preserves the original
comparison cohort but is not exact local reproduction of all classifier outputs.
A sensitivity analysis excluding such sources across all methods would be a
separate study; this runner does not silently change their eligibility.

Offline integration tests compare all three primary Stage 2 CSV outputs
byte-for-byte with the teammate's evaluator using the same synthetic features,
including a multi-family mixture, held-out-family exclusion, and bootstrap.
Actual Stage 2 performance still requires an available ResNet feature cache.
