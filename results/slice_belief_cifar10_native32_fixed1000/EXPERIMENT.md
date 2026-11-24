# Native-32 CIFAR-10 explanation experiment

This run completes Grad-CAM, SLICE and BELIEF on the exact saved corpus used
by the existing ResNet and DINOv2 representation experiments. It does not
regenerate attacks or fine-tune the classifier. The 21 tensor blocks and all
three metadata files were checked against the representation provenance.

From the repository root:

```powershell
# Fresh Windows Python 3.12 environment (the transferred macOS venv is unused).
# This machine's project-local Python 3.12 interpreter; use a Python 3.12
# installation path on a different machine rather than its default Python.
tmp/python312/python.exe -m venv .venv-corpus-win
.venv-corpus-win/Scripts/python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
.venv-corpus-win/Scripts/python.exe -m pip install -r requirements-corpus.txt
$env:PYTHONUTF8 = '1'
$env:TORCH_HOME = "$PWD/tmp/torch"
.venv-corpus-win/Scripts/python.exe -m experiments.cifar10_explanations
```

Verify that both imported package versions have the `+cu124` suffix and
`torch.cuda.is_available()` is true. The actual fresh environment used
Python 3.12.10, Torch 2.4.1+cu124 and Torchvision 0.19.1+cu124 on an NVIDIA
GeForce RTX 4070 Ti. The exact package inventory is saved in
[requirements-windows-cuda.lock.txt](requirements-windows-cuda.lock.txt).
The original macOS environment remains unused. On Apple MPS only, set
`PYTORCH_ENABLE_MPS_FALLBACK=1` before invoking the runner.

## Device migration and numerical audit

Before extraction, transferred MPS explanation/repeat caches were moved to
`data/processed/slice_belief_cifar10_native32_fixed1000/archives/mps_transfer_20261123/`.
That archive also preserves the transferred source snapshots and result/status
files. It contains 220 partial `train_clean.pt` explanation records (indices
0–219), rather than the 200 stated in the handoff. Its original MPS log ends
at `220/350`, and the unchanged checkpoint SHA-256 is
`e0049619acdfc4f992aca5e728676197a45d534a0b72db0a55ea541764ed90b1`.
All primary Grad-CAM, SLICE, BELIEF and repeat caches are regenerated on CUDA;
MPS records and timings are not pooled with CUDA results. Completed segmentation
and the saved development-selected sigma 0.3 are reused. Segmentation timings
are transferred CPU measurements and must be labeled separately from CUDA cost.

The first CUDA audit identified one changed test-clean prediction with cuDNN
TF32 enabled. The tensor hashes and model weights were correct. Disabling TF32
restored the baseline class and agreed with CPU FP32 and CPU FP64 inference.
See [the saved diagnosis](audit_attempts/numerical_diagnosis.json) and
[initial failed audit](audit_attempts/cuda_default_prediction_audit.json).
The runner disables TF32 for its full FP32 audit and Grad-CAM path, retaining
configured CUDA float16 autocast for perturbation queries and CPU float64
regression. The subsequent strict audit requires zero mismatches over all 21
blocks; no masks or attacks are regenerated. The full official backbone weight
SHA-256 is `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

The runner holds an OS process lock and, on Windows, a system-awake request.
Monitor the process ID together with explanation checkpoint timestamps and
the progress log; `status: running` alone is insufficient.

The runner first checks all tensor hashes and requires zero differences in
classifier predictions. It then runs segmentation, development-only sigma
selection, explanation extraction, stability repetitions, fidelity evaluation,
PGD-trained Stage 1 and matched-mixture Stage 2, CW-trained Stage 1, and a
comparison report. The existing native-32 configuration controls all sample
counts, query budgets, seeds and calibration settings.

Only run one instance at a time. Repeating the command rechecks provenance,
reuses existing segmentation and sigma files, and resumes explanation records
at their saved checkpoints. The checkpoint interval is 20 explained images.
Analysis and reports are regenerated from completed saved artifacts.

## Reference-parity validation

The initial required suite passed 39 tests and skipped 13: 12 missing pinned
reference cases and one MPS-only case. The two upstream sources were then
downloaded at the exact commits named in `explainers/perturbation.py` into
`tmp/research_references/{SLICE,BELIEF}/slice/slice_explainer.py`.
Optional test-only dependencies `pandas==2.2.3` and `matplotlib==3.9.2` were
installed with `-c requirements-corpus.txt`; the extraction package pins remain
unchanged. A focused run passed 23 CPU tests, including all 12 pinned-reference
cases, with three unrelated cases deselected. Matplotlib/Pyparsing emitted
deprecation warnings during reference imports; no numerical-parity test failed.

[Reference provenance](reference_validation.json) records the commit URLs,
source hashes, optional dependency versions and scope. TensorFlow, Keras and
OpenCV imports use the repository's test stubs, and the documented per-channel
blur shim is retained. This checks CPU float64 numerical parity, rather than
equivalence between MPS full-precision and CUDA mixed-precision explanations.
The final complete suite is recorded in [pytest.txt](pytest.txt) and
[pytest.xml](pytest.xml).

## Outputs

- `experiment_status.json`: current stage, process ID, completed stages, and
  any traceback. `status: complete` is written only when requested stages finish.
- `prediction_audit.json`: hashes and prediction agreement on all corpus rows.
- `tables/`: PGD-trained detection, matched Stage 2, noise, bootstrap,
  explanation status, stability, fidelity and runtime tables.
- `cw_train/`: CW-trained detection and noise tables.
- `MAIN_RESULTS.md`: generated comparison with the existing representations;
  this file is produced only after analysis completes.
- `execution_environment.json`: device/precision, versions, source hashes,
  segmentation/sigma reuse and transferred archive checksums.
- `validation.json`: independent post-run checks of every main/repeat record,
  splits, attack denominators, matched sources and three-seed summaries.
- `validation_scores.csv`, `validation_thresholds.json`: scores and frozen
  thresholds from independently refitted PGD/CW-trained Stage 1 detectors,
  checked against published AUC/TPR/clean/noise-FPR operating points.
- `report_input_inventory.json`, `report_validation.json`: report dependency
  hashes, regeneration equality, link checks and preservation of original tables.
- `tables/stage2_assignments.csv`, `stage2_matching.json`, `stage2_noise.csv`:
  additional provenance for matched sources and per-seed false alarms.
- `logs/slice_belief.log`: detailed progress and checkpoint estimates.
- `runner-windows-cuda.log`: stdout/stderr from the homogeneous CUDA run.
  The shared log retains historical MPS/stop and initial audit entries; cost
  tables use the new CUDA caches rather than pooling historical log timings.

Large segmentation and explanation checkpoints remain under the ignored
`data/processed/slice_belief_cifar10_native32_fixed1000/` folder. An active
process or partial checkpoints do not mean the experiment has completed.

After the full runner reports completion, validate and regenerate all reporting
layers from the saved artifacts:

```powershell
.venv-corpus-win/Scripts/python.exe -m scripts.validate_cifar10_explanations
.venv-corpus-win/Scripts/python.exe -m scripts.check_cifar10_report_inputs
.venv-corpus-win/Scripts/python.exe -m scripts.cifar10_explanation_reporting
.venv-corpus-win/Scripts/python.exe scripts/make_new_experiments_report.py
.venv-corpus-win/Scripts/python.exe -m scripts.validate_cifar10_reports
```

The report generator also reads the preserved Imagenette and 224-diagnostic
corpora's `variants.csv`, their representation/CW/Stage-2 result tables, and
the Imagenette explanation Stage-1 reproduction tables. These transferred
dependencies must remain present. Generation stops if native-32 validation
has not passed or if the historical Stage-1 reproduction comparison changes.

The classifier is a frozen ImageNet-pretrained ResNet-50 with a CIFAR-10-trained
linear head. The primary detector setting trains on successful PGD attacks;
FGSM stays test-only. This is a non-adaptive detection study. Actual clean and
noise false-positive rates must accompany the calibrated detection rates.
