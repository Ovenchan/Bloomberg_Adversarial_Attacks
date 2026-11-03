# Research Plan: Controlled Adversarial Detection with Explanations and Representations

Prepared November 2, 2026, for Yunxuan Chen, Ying Jin, and Cara Zhu.

This plan operationalizes [Research Proposal.md](Research%20Proposal.md), informed by the [previous team's report](Adversarial_Attacks_Report_Final.pdf), the supplied code and saved results, and the authors' SLICE and BELIEF implementations. It is a plan for implementation and experiments; the new experiments have not been run.

The central question is whether explanation and representation features detect successful adversarial manipulation beyond the signatures of the attacks used for detector training. The study will first establish a trustworthy evaluation protocol, then implement SLICE and BELIEF in PyTorch, and finally compare single-attack and multiple-attack detector training.

## 1. Scope and research questions

### Required study

- **Dataset:** Imagenette first, shared across methods.
- **Explanation features:** Grad-CAM, SLICE, and BELIEF, explaining the same frozen ResNet-50 classifier.
- **Representation features:** pooled ResNet-50 features and frozen DINOv2 ViT-S/14 embeddings.
- **Detector:** a separate standardized, regularized logistic regression for each feature method, using a common fitting and evaluation procedure.
- **Attacks:** PGD, APGD, CW-L2, and a per-image optimized patch. FGSM remains a small diagnostic because it was a weak-transfer case in the report.
- **Training regimes:** PGD-only training and sample-matched multiple-attack training, with entire attack families withheld from detector development.
- **Controls:** clean inputs, noisy non-adversarial inputs, source-image separation, independent threshold calibration, and explanation repeatability/fidelity checks.

### Conditional extensions

DINOv3 ViT-S/16 and CIFAR-10 follow completion of the Imagenette core. DINOv3 is an additional frozen encoder, not a backbone-training project. Its official implementation provides `dinov3_vits16`; checkpoint availability and access must be established before scheduling it. [DINOv3 reference implementation](https://github.com/facebookresearch/dinov3)

Adaptive detector-evasion attacks, detector ensembles, broad architecture/layer searches, and full ImageNet evaluation are outside this study. Consequently, results will concern **non-adaptive attacks against classifiers**, without claiming robustness to attackers who optimize against the detector.

The new proposal supersedes historical dataset priorities, personnel assignments, and the September deadline in `CLAUDE.md`. Retain its useful engineering conventions: shared attack/metric code, configuration-driven experiments, documented interfaces, and reusable components.

| Question | Experiment | Evidence needed |
| --- | --- | --- |
| RQ1: Does the explanation method change detection and generalization? | Grad-CAM vs SLICE vs BELIEF on identical ResNet inputs and attacks | Paired differences in detection, noise false alarms, fidelity, stability, and cost |
| RQ2: How do explanation and representation signals compare? | Add pooled ResNet-50 features; evaluate DINOv2 under explicitly identified attack targets | Comparisons with dataset, source identities, detector family, and threat model held fixed |
| RQ3: Does attack diversity help at fixed training size? | PGD-only vs mixtures, evaluated on a withheld family | Improvements on unseen attacks without unacceptable clean/noise false alarms |
| RQ4: What explains unusually strong or weak historical results? | Protocol audit, threshold correction, attack-strength checks, and focused ablations | A documented distinction between implementation problems, calibration effects, and persistent attack-dependent behavior |

Hypotheses are directional expectations, not required outcomes: stabilization may improve explanation repeatability without improving detection; diverse training may help CW/patch transfer without solving it; apparent performance advantages may shrink under stricter controls. Negative results remain valid contributions.

## 2. What must be checked in the previous work

### Reported observations

| Observation in the report | Location | Research implication |
| --- | --- | --- |
| Grad-CAM heatmap CNN: PGD AUC 0.9781; APGD about 0.981; CW about 0.494 | pp. 26-27, Table 1 | Strong within-attack detection and weak CW transfer coexist; distinguish explainer effects from detector capacity |
| Grad-CAM logistic detector: PGD AUC 0.9524; FGSM 0.4524; CW 0.5229 | p. 27, Table 2 | A lightweight baseline already captures the broad pattern and is suitable for controlled comparisons |
| DINOv2 logistic detector: PGD AUC 1.0000; shuffled-label AUC 0.5133 | pp. 28, 32-33 | Near-perfect separation is an audit target, not evidence by itself of an error |
| DINOv2 on CIFAR-10: CW ASR 100%, detection AUC 0.7642, recall 5% | p. 28, Tables 3-4 | A successful attack can evade the chosen detector threshold even with some ranking signal |
| ImageNet-5 CW: AUC 0.9646 but recall 30% | pp. 30-31 | Low thresholded recall must be separated from poor ranking |

The report explicitly describes correcting early classifier/test overlap, with perfect PGD separation persisting afterward. It also acknowledges classifier-development use of test performance (p. 35). Recheck provenance rather than assuming either that all results are invalid or that the earlier audit resolves every issue.

### Findings from the supplied files

These are static inspection findings, not newly reproduced performance measurements.

1. **Threshold selection:** `analysis/metrics.py::compute_tpr_at_fpr` chooses the best attainable point from the supplied evaluation ROC curve. This is a descriptive ROC statistic, but it does not implement the proposal's independent clean-validation threshold. Add an explicit calibration/evaluation interface; preserve the old statistic under a clearly descriptive name.
2. **Attack-success denominator:** `DetectionEvaluator.evaluate` reports retained successful attacks divided by all supplied attempts. If initially misclassified images are included, this differs from ASR conditional on clean correctness. The newer Imagenette files report clean accuracy 0.881 and PGD success fraction 0.881, illustrating why numerator and eligibility counts must be stored separately.
3. **Success filtering differs across scripts:** `experiments/08_train_pgd_detector.py` and `10c_evaluate_pgd_detector_on_cw.py` label all loaded adversarial rows positive without applying a success mask in those scripts. The grouped Grad-CAM implementation does apply a success mask. Verify upstream inclusion rules and recalculate from predictions; do not treat these paths as equivalent.
4. **ResNet features:** `encoders/cnn.py::ResNetEncoder.encode` currently returns classifier logits. Implement a separate pooled-feature interface for the proposed representation detector; do not relabel logits as penultimate features.
5. **Historical and current results differ:** `results_original/tables/gradcam_cross_attack_results.csv` agrees with the report's logistic table. The current grouped Imagenette summary reports PGD AUC about 0.9504 and APGD about 0.9316, with a different cohort/protocol. Preserve both versions and explain the differences before interpreting a trend.
6. **Configuration drift:** the report uses a 48x48 DINO patch; the current Grad-CAM config uses 50x50. The current ResNet config uses resize 232 and crop 224. Record the actual weights, preprocessing, attack parameters, and class mapping for every result.

Create a provenance table connecting each historical claim to its script, configuration, source IDs, checkpoint, saved predictions, and metric definition. Mark claims without sufficient artifacts as unreproduced. Recompute from saved arrays where possible; avoid expensive historical retraining merely to force a matching number.

## 3. Comparison design and threat models

### A. Primary comparison: a shared ResNet-50 attack corpus

Generate each attack once against a frozen ImageNet-pretrained ResNet-50. Feed those exact image tensors to all feature extractors:

| Method | Feature source | Detector input |
| --- | --- | --- |
| Grad-CAM | ResNet-50 final convolutional stage, current predicted class | Common heatmap vector |
| SLICE | Same ResNet-50, same current predicted class | Common heatmap vector |
| BELIEF | Same ResNet-50, same current predicted class | Common heatmap vector |
| ResNet representation | Same ResNet-50, before the final classification layer | 2,048-dimensional global-average-pooled feature |
| DINOv2 representation | Frozen DINOv2 observing the same input | 384-dimensional image embedding |

The DINOv2 row here is an **observer of attacks against ResNet-50**. Its success mask is the ResNet-50 attack-success mask. It does not measure attacks against a DINOv2 classifier. This inexpensive shared-corpus comparison prevents changes in attack target from being mistaken for changes in feature quality.

Keep ResNet's original 1,000-class prediction space in this primary experiment, using the explicit Imagenette-to-ImageNet label mapping already present in `utils/gradcam_data.py`. Attacks may cause predictions outside the ten Imagenette classes. Every ResNet explainer must explain the same resulting class, including such predictions.

### B. Companion experiment: attacks against DINOv2 itself

Train a ten-class linear head on frozen DINOv2 using only classifier-training data. Attack the differentiable encoder-plus-head and detect from its embeddings. Use the same source partitions, budgets, detector protocol, and training regimes as in A, but store a separate corpus and report a separate table.

This preserves the proposal's evaluation against each relevant classifier and tests whether the historical PGD/CW pattern persists on Imagenette. Native DINOv2 results cannot establish a pure feature advantage over ResNet results: the attacked classifier and class space differ. Cross-model success intersections may be reported as a sensitivity analysis, with their reduced counts and selection effect disclosed.

All core detectors receive features from the **current input alone**. Clean-adversarial differences, original clean labels/predictions, perturbation norms, attack names, source IDs, and attack-success flags are evaluation metadata, never detector inputs.

## 4. Data partitions and reproducibility

### Imagenette partition plan

The supplied `data/raw/imagenette2` contains 9,469 training images and 3,925 validation images. Verify these counts and content hashes when implementation begins.

| Partition | Source and initial size | Permitted use |
| --- | --- | --- |
| Classifier train / classifier validation | Class-stratified 80/20 division of official training images | Fit/select the DINO head; ResNet remains pretrained and frozen |
| Detector training | 1,000 official-validation identities | Fit detector scalers and detectors; generate training attacks |
| Detector development | 300 different official-validation identities | Select detector regularization and development settings using permitted attack families |
| Clean calibration | 1,000 different official-validation identities | Set detector thresholds after all fitting/selection is complete |
| Final test | 1,000 different official-validation identities | Locked clean, adversarial, and noise evaluation |
| Reserve | Remaining 625 official-validation identities | Unused initially; any use is documented before final testing |

These are initial targets, subject to the timing pilot and eligibility counts. Draw class-stratified random samples, never the first N filesystem entries. Calibration/test should represent the sampled clean population, including initially misclassified inputs; report false alarms separately by clean correctness as well.

Use a globally unique `source_id` containing dataset, official split, and file identity/hash. All attacks, corruptions, crops, repeated explanations, and cached features from one source inherit its partition. Fit PCA, scalers, feature selection, and classifiers only on their permitted training partitions.

Inspect legacy source IDs before locking the test cohort. Prefer identities not used in earlier development where enough remain, and record historical exposure honestly. Task-level split isolation does not prove absence from pretrained models' original training data.

Use a fixed primary split seed (42) and three prespecified experimental seed values (42, 123, 2026) for attack initialization, explainer sampling, and training-mixture allocation. Keep final test identities fixed for paired comparisons. Derive independent RNG streams from `(run_seed, source_id, purpose)` so labels and batch order cannot determine randomness. A separate split-sensitivity study is optional; do not select the best seed.

Cache raw attack tensors and extracted features with configuration hashes. Store image-space preprocessing once, keep normalization inside the model adapter, and invalidate caches when weights, class mapping, target, explainer, or preprocessing changes.

## 5. Attack generation and benign controls

### Shared input contract

Use RGB float tensors `[N, 3, 224, 224]` in `[0, 1]`. Resize/crop before attack optimization, then keep the attacked tensor unchanged across feature methods. Pixel-space budgets apply to this 224x224 tensor. A future native-32x32 CIFAR experiment must be labeled separately.

For primary ResNet V2 weights, use resize 232, center crop 224, and the documented ImageNet normalization. The official V1 and V2 preprocessing differ, so pin the weight name rather than using a floating default. [Torchvision ResNet-50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)

Freeze model parameters and use evaluation mode. Preserve gradients with respect to inputs during attacks and Grad-CAM; do not wrap those paths in `no_grad` or detach the encoder. Test that inference and attack adapters give identical clean logits.

### Attack configuration

| Attack | Historical replication setting | Main-study starting setting / check |
| --- | --- | --- |
| PGD-Linf | epsilon 8/255, alpha 2/255, 20 steps, random start | 50 steps and 5 restarts; retain a successful candidate before preferring higher loss; audit convergence on development data |
| APGD-Linf | epsilon 8/255, 20 steps, 1 restart | 100 steps and 5 restarts, CE objective; optional DLR sensitivity after checking the implementation |
| CW-L2 | c=1, kappa=0, 100 steps, learning rate 0.01 | Validate convergence with a prespecified c grid {0.01, 0.1, 1, 10} and up to 1,000 steps; retain minimum-L2 successful candidates |
| Per-image patch | Report: 48x48 at 224x224, 100 Adam steps, learning rate 0.05 | Fix 48x48 (4.59% area), identical source-specific locations across compared methods; assess optimization length on development images |
| FGSM diagnostic | epsilon 8/255 | Fixed one-step implementation; never substitute it for iterative attack validation |

The stronger starting settings are proposed checks, not a claim that they are optimal. Finalize bounded compute settings using attack efficacy and convergence on development data, then freeze them before test evaluation. Any global tuning for a withheld-family fold must exclude that family; its configuration comes from the prespecified recipe, never its detector results.

PGD and APGD have matched Linf budgets. CW and patch have different constraints: report L2, Linf, RMS, and patch area as appropriate rather than describing all four as equally strong. PSNR is descriptive and does not establish perceptual invisibility.

For every attack save clean/adversarial predictions, true labels, target model/checkpoint, eligibility, success, seed, steps/restarts, loss, perturbation norms, and tensor hashes. Confirm bounds, finite/nonzero gradients, unchanged pixels outside a patch, and recomputed predictions from saved tensors. Include an epsilon-zero identity check. Select candidates only by the attack objective/success, never by detector score.

### Noise controls

Evaluate Gaussian noise at fixed standard deviations `{1/255, 2/255, 4/255, 8/255}`, clipped to `[0,1]`, using independent source-specific seeds. Also include uniform noise bounded by the PGD epsilon; if affordable, add a non-optimized random patch with the same area as the optimized patch.

Use the same corrupted tensors across methods. Noise remains a **negative/non-adversarial** input even when it changes the classifier's prediction. Report both full-cohort and correctness-stratified noise false alarms. A noise-matched-to-attack-RMS diagnostic may use clean pairs during offline construction, but those pairs never enter a detector.

Primary training uses clean negatives. Adding noisy negatives is a separately labeled ablation with unchanged total training count, so improved false-alarm performance cannot be attributed silently to additional data.

## 6. PyTorch replication of SLICE and BELIEF

### Reference versions inspected

| Reference | Pinned revision | Main implementation |
| --- | --- | --- |
| [SLICE](https://github.com/rebathip/SLICE-Stabilized-LIME-for-Consistent-Explanations-for-Image-Classification) | `61d35c09b9f8bd9f770f5d48fae97a181a28f974` | `slice/slice_explainer.py`, `slice_test.py` |
| [BELIEF](https://github.com/rebathip/BELIEF) | `70f373c0aab5aa6b7e679ed4ecb7571e7a79ba5b` | `slice/slice_explainer.py`, including `SEMBayesianRidge`; `belief_20runs.py` |

The supplied implementations combine TensorFlow/Keras prediction with NumPy/SciPy/scikit-learn explanation calculations. Port the prediction interface and numerical algorithm deliberately; replacing a model import alone is insufficient.

### Components to preserve

- **Shared machinery:** binary superpixel masks, the all-present sample, probability of the fixed explained class, cosine-distance locality weights, and the reference perturbation/blur selection behavior. SLICE's driver uses Quickshift (`kernel_size=5`, `max_dist=200`, `ratio=0.2`, segmentation seed 42); do not assume it uses SLIC because of its name. [SLICE driver](https://github.com/rebathip/SLICE-Stabilized-LIME-for-Consistent-Explanations-for-Image-Classification/blob/61d35c09b9f8bd9f770f5d48fae97a181a28f974/slice_test.py)
- **SLICE:** preserve locality-weighted bootstrap sampling, weighted ridge fitting, coefficient sign-entropy estimation, iterative removal of unstable features, re-sampling with eliminated features held present, and final coefficient mapping. Reference defaults include 500 perturbations per step, 1,000 bootstrap fits, up to 10 steps, and tolerance 3; the driver disables rank stabilization. [SLICE algorithm](https://github.com/rebathip/SLICE-Stabilized-LIME-for-Consistent-Explanations-for-Image-Classification/blob/61d35c09b9f8bd9f770f5d48fae97a181a28f974/slice/slice_explainer.py)
- **BELIEF:** preserve weighted Bayesian ridge estimation, posterior covariance, sign probabilities/entropy, feature elimination, precision updates, and restoration of the original coefficient indexing. The wrapper passes `zeta=1e-6` as entropy slack, while the underlying class has a different default; use the public wrapper setting. The driver reuses stored segmentation/blur settings. [BELIEF algorithm](https://github.com/rebathip/BELIEF/blob/70f373c0aab5aa6b7e679ed4ecb7571e7a79ba5b/slice/slice_explainer.py)

Neither method should be replaced by generic repeated-LIME averaging. BELIEF's claimed explanation efficiency and fidelity motivate measurement, but do not predict superior adversarial detection. [BELIEF paper](https://proceedings.mlr.press/v286/bora25a.html)

### Implementation contract

1. Build a PyTorch classifier adapter exposing differentiable logits and batched probabilities. Convert layout/range explicitly, apply normalization once, and apply softmax once for surrogate targets. Grad-CAM continues to explain class logits.
2. Accept in-memory adversarial tensors; avoid JPEG round-trips and file-based image reloads. A shared perturbation component produces masks and model queries in bounded batches.
3. Implement weighted ridge, Bayesian posterior/precision calculations, and elimination with tested PyTorch linear algebra. Use float64 initially for numerical fixtures. NumPy/SciPy and scikit-image may remain explicit CPU utilities for reference-compatible KDE or segmentation; the production pipeline must not require TensorFlow/Keras. Keep sklearn logistic regression for the common detector.
4. Record numerical corner cases: zero-variance probabilities, singular systems, no retained superpixels, constant/zero maps, entropy boundaries, and non-finite outputs. Preserve valid reference fallback behavior with an explicit status; never silently replace failed explanations with successful ones. Report extraction coverage and failures by method/condition. Do not drop failed inputs to improve headline metrics: a condition remains incomplete until extraction works or a prespecified fallback can score every intended input.
5. Separate a **reference-compatible profile** from the **controlled study profile**. The reference code masks/blurs preprocessed arrays; raw black masking is not equivalent to setting a normalized channel to zero. Reconstruct the intended baseline in raw pixels or document a deliberate change. Verify blur channel handling, interpolation, segmentation labels, and RNG API behavior against pinned versions.
6. Use one segmentation computed from each current input for both SLICE and BELIEF, and the same explained class `argmax f(x_current)`. Share initial masks where algorithmically possible. Freeze that class during its perturbed queries; do not explain each perturbation's changing class. Do not borrow segmentation or the class from a clean counterpart at deployment.
7. For the controlled profile, use a common, documented blur-selection policy for SLICE/BELIEF. Resolve the reference cache/refresh semantics before adoption, and include selection queries in timing. Keep any fixes separate from literal reference behavior in a deviation log.

### Parity gates before full experiments

**Numerical fixture parity:** export masks, targets, weights, bootstrap indices, and expected intermediate/output arrays from the pinned reference implementation. Feed the exact arrays to the PyTorch implementation. Compare ridge coefficients, Bayesian posterior quantities, entropy, retained features, and rankings. Start with `atol=1e-6, rtol=1e-5` on well-conditioned float64 fixtures; handle threshold/tie-sensitive cases explicitly instead of masking them with relaxed tolerances.

**Prediction/perturbation parity:** use a tiny deterministic model with identical parameters across frameworks, or archived query probabilities, to isolate layout, normalization, masking, and blur differences. Verify chunked and unchunked querying agree. Matching random seed numbers across TensorFlow, Python, NumPy, and PyTorch is insufficient; use fixed sampled arrays for exact tests.

**End-to-end validation:** run both explainers on a small fixed image set, retaining traces and visual maps. Keras ResNet-50 and torchvision ResNet-50 differ in weights/implementation; identical heatmaps are not an appropriate acceptance criterion unless an equivalent model has been established. Label the torchvision study as algorithm replication on a controlled target model.

Pin the reference runtime separately if its private sklearn APIs or old scikit-image arguments require it. Reference code is an oracle for comparison, not an unreviewed production dependency. Finish a short paper-to-code specification and record any paper/code disagreement before freezing the study profile.

## 7. Features and detector fitting

Use a common spatial conversion for explanation methods. Rasterize superpixel coefficients to pixels, preserving their signs and original IDs. The primary capacity-matched comparison uses the positive-evidence map for all three explainers, scales nonzero maps by their positive maximum, resizes to 14x14, and flattens to 196 features. Record zero maps and do not add plot colors, boundaries, or overlayed images to detector inputs.

Because Grad-CAM is nonnegative while SLICE/BELIEF are signed, also prespecify a signed-evidence ablation: concatenate separate positive and negative 14x14 maps for SLICE/BELIEF. Label its 392-dimensional input explicitly; it is not the capacity-matched primary result. Evaluate explanation fidelity using the full signed explanation, not only the detector's positive projection.

Extract ResNet global-average-pooled activations before `fc`, without replacing the classifier needed by attacks/explainers. Verify the 2,048-dimensional output and its relationship to the original logits. Use the pinned DINOv2 global embedding and verify its dimension and differentiable attack path.

For each feature method, fit `StandardScaler -> LogisticRegression`, selecting C from `{0.01, 0.1, 1, 10}` using only detector-development data and training-permitted attack families. Use the same search budget and validation objective (mean validation ROC-AUC across permitted families). Calibration data never select C. Different native representation dimensions remain a disclosed limitation; a training-only dimension-matched projection is optional.

Train on successful adversarial examples for the primary detection task. Keep failed attempts and initially incorrect inputs in the manifest for explicit secondary analyses, rather than deleting their records. Explanations must not encode success status. Use one explanation realization per input in each experimental seed; repeated explanations are correlated measurements, not additional independent training images.

## 8. Single-attack and multiple-attack experiments

### Stage 1

Fit each detector on clean inputs and successful PGD examples. Select its parameters using clean/PGD development data. Calibrate thresholds on clean calibration data, freeze the full pipeline, and evaluate on PGD, APGD, CW, patch, and the diagnostic FGSM/noise conditions.

### Stage 2

| Withheld test family | Permitted detector-training mixture | Matched single-attack comparator | Priority |
| --- | --- | --- | --- |
| CW | PGD + APGD + patch | PGD only | Core |
| Patch | PGD + APGD + CW | PGD only | Core |
| APGD | PGD + CW + patch | PGD only | Extension after core |
| PGD | APGD + CW + patch | APGD only | Optional rotation; PGD-only is not an unseen-PGD comparator |

Withheld-family data cannot influence detector hyperparameters, feature choices, early stopping, mixture weights, or threshold calibration. Freeze the protocol for every planned fold before opening final-test results. A family used for training in one fold is still wholly excluded from detector development in another fold.

For a matched comparison, choose N eligible training source identities that have successful attacks for the baseline and every permitted mixture family, using **training data only**. Refit the single-attack comparator on those same identities. Each arm receives N clean rows and N adversarial rows, with exactly one adversarial variant per source. Allocate mixture families as evenly as possible, stratified by source class. Thus both total rows and unique image identities are matched; the mixture does not get three times as many examples.

Use a prespecified N up to 1,000, lowered to the available common-success training pool if necessary, and report all exclusions/class counts. This restriction conditions training on attackability; describe that limitation. The final test cohort is never selected by multi-family success. Apply the same comparability rule to development data and keep the calibration/test identities fixed between arms. Training-size learning curves, if needed, use additional prespecified smaller N values.

PGD and APGD are closely related Linf attacks. Their separation measures transfer between implementations/objectives, not entirely novel perturbation geometry. CW and patch are therefore the main held-out tests. A later grouped holdout of both PGD/APGD can test a stronger notion of attack-family novelty.

## 9. Metrics, calibration, and statistical analysis

### Correct attack-success definitions

For an untargeted attack on classifier f, define eligibility `E_i = 1[f(x_i) = y_i]` and success `S_i = E_i * 1[f(x_adv_i) != y_i]`.

`ASR = sum(S_i) / sum(E_i)`.

For a targeted attack with `t_i != y_i`, success instead requires `f(x_adv_i) == t_i`. The proposal's inequality against an intended target is incorrect and must not be implemented. All planned core attacks are untargeted.

Store total sources, eligible sources, attempted attacks, successful attacks, and evaluated positives. Report classifier clean accuracy and attacked accuracy separately. If no eligible attacks or successful test attacks exist, return an explicit undefined metric with counts, rather than an invented zero or perfect result.

### Frozen operational thresholds

For each fitted detector and seed, select separate thresholds for target clean FPR alpha in `{0.01, 0.05}` using **only clean calibration scores**. Use a documented conservative empirical upper-quantile rule, strict `score > threshold` decisions, and explicit handling of tied scores. Save the threshold, achieved calibration FPR, calibration IDs, and rule/version.

Apply these thresholds unchanged to every test attack and noise condition for that detector. Report:

- TPR on successful test attacks at the calibration-derived 1% and 5% operating points.
- Actual clean test FPR and each noise-control FPR at those same thresholds.
- ROC-AUC on the fixed clean test population versus successful adversarial inputs.
- Secondary detection of all attack attempts, clearly separated from successful-attack detection.
- Undetected successful-attack rate among eligible sources, `sum(S_i * 1[score_i <= threshold]) / sum(E_i)`, to connect attack effectiveness with detection.

The realized test FPR need not equal its calibration target. Do not choose a new threshold from test labels, report accuracy as the primary metric, or flip a score's sign after seeing a below-chance test AUC. Keep historical default-0.5 recall and test-ROC TPR in separately labeled replication tables.

Treat 5% FPR as the primary operational endpoint and 1% as a more data-demanding secondary endpoint. With 1,000 clean test images, 1% corresponds to only ten false alarms; report counts and binomial confidence intervals. If feasibility forces smaller cohorts, retain the caveat and do not claim precise low-FPR behavior.

### Explanation quality and cost

On a class-stratified, prespecified 100-image audit panel, repeat each stochastic explanation 20 times with fixed segmentation/class to measure coefficient sign agreement, top-k Jaccard overlap, rank correlation, and resulting detector-score variability. Select the panel by source IDs before observing outputs and retain every attack outcome. Use separate development and final-test panels; never tune on the final panel. Grad-CAM repeats provide a deterministic reference.

Measure fidelity using held-out perturbation predictions (weighted error/R2 for local surrogates) and common insertion/deletion or AOPC curves for all explainers. Fix the explained class, masking baseline, step schedule, and pixel-area budget. Use common evaluation regions so differences in superpixel count do not give a method an easier metric; include a random-ranking control. Report negative evidence separately and acknowledge that masking probes can themselves be out of distribution.

Report median and tail runtime per image, total classifier queries, feature-extraction time, detector fitting/inference time, and peak GPU memory/host memory when measurable. Count segmentation, blur selection, retries, and fallback work. Synchronize GPU timing, warm up models, use the same hardware, and distinguish cached from end-to-end timing. Compare reference defaults first, then a prespecified matched-query-budget sensitivity; any reduced SLICE bootstrap/step budget is an approximation that must be labeled.

### Uncertainty and comparisons

Save per-source scores and use paired bootstrap resampling of source identities (2,000 replicates) for AUC/TPR differences. Resample all variants of an identity together and recompute success-filtered metrics; never treat 20 explanations as 20 independent images. For headline operational uncertainty, resample the independent calibration cohort too and recompute its threshold within each bootstrap replicate. Show seed variability separately; a bootstrap of fixed trained models does not include retraining uncertainty.

Report mean/dispersion across the three prespecified seeds, effect sizes, 95% intervals, and exact denominators. Primary comparisons are SLICE/BELIEF versus Grad-CAM, explanation versus pooled ResNet features, and mixed versus matched PGD-only training on CW and patch. Use a multiple-comparison correction if making formal significance claims. Do not select favorable attacks or seeds for the headline conclusion.

## 10. Required controls and focused ablations

| Priority | Check | What it can reveal |
| --- | --- | --- |
| Required | Exact/near-duplicate and source-group split checks | Source leakage or misleading row alignment |
| Required | Train-label permutation over repeated seeds | Label/artifact leakage; compare with a null distribution rather than demanding exactly 0.5 AUC |
| Required | Same clean images through every extraction/cache path; epsilon-zero identity | Normalization, resizing, serialization, or cache artifacts masquerading as attack signals |
| Required | Frozen calibration thresholds and successful-only vs all-attempt scoring | Threshold and denominator explanations for strong/weak results |
| Required | Noise controls at multiple severities | Whether high attack detection is accompanied by general corruption sensitivity |
| Required | Shared SLICE/BELIEF segmentation; fixed-grid sensitivity on a subset | How much signal is due to segmentation changes rather than attribution |
| Focused | PGD/APGD budgets 2/255, 4/255, 8/255 | Whether high detection depends on large perturbations; report ASR at every budget |
| Focused | Confidence-only, predicted-class, and segmentation-statistic baselines | Whether expensive features add information beyond output-class changes, prediction uncertainty, or segmentation artifacts |
| Focused | Signed vs positive-only maps; normalized vs unnormalized features | Sensitivity to the feature representation used by the detector |
| Focused | Shared-target DINO observer vs native-target DINO | Association between attacked representation and ease of detection |

A fixed geometric grid can be used on every current input without a clean reference. Reusing a clean input's segmentation or class for its adversarial counterpart is permitted only in an explicitly labeled offline diagnostic. Pairwise embedding displacement and cosine/alignment analysis are similarly explanatory analyses, not deployable detector inputs or causal evidence.

For the 1,000-class ResNet target, specifically report how often predictions leave the ten Imagenette classes. A detector may exploit that class-support change; compare against the simple current-prediction-in-Imagenette indicator and inspect class/confidence-stratified results before attributing the signal to explanation quality.

## 11. Implementation work packages and acceptance gates

Extend existing modules rather than building parallel copies of attacks and metrics. New filenames below describe planned work, not already implemented commands.

| Package | Main files/modules | Deliverable and acceptance gate |
| --- | --- | --- |
| P0: provenance and protocol | `utils/`, `configs/`, historical `experiments/08b_pipeline_audit.py` | Frozen source manifest, artifact/claim inventory, environment versions and hashes; no source overlaps |
| P1: common evaluation | `analysis/metrics.py`, `analysis/grouped_cv.py`, proposed `analysis/calibration.py` and `analysis/bootstrap.py` | Correct ASR counts, independent calibration, group-safe selection, paired intervals; hand-calculated metric fixtures pass |
| P2: classifiers and attacks | `encoders/cnn.py`, `encoders/dinov2.py`, `models/`, `attacks/` | Explicit logits/features interfaces; same clean predictions through all paths; constraints and success recomputation pass |
| P3: PyTorch explainers | proposed `explainers/base.py`, `perturbation.py`, `slice.py`, `belief.py`; extend `vectorizer.py` | Pinned-reference traces, numerical parity, documented deviations, deterministic replay and failure handling |
| P4: shared experiments | `pipelines/`, `detectors/logistic.py`, proposed `experiments/controlled_detection.py` | Stage-1 and Stage-2 runs from config; equal counts/source IDs; held-out families absent from fitting/selection |
| P5: quality and reporting | proposed `analysis/explanation_quality.py`; existing plotting/results helpers | Noise/fidelity/stability/cost tables, paired plots, full per-source results and reproducible report generation |
| P6: optional extensions | proposed `encoders/dinov3.py`; CIFAR configs | Same interface/protocol, isolated output labels, and an explicit resource go/no-go decision |

Add focused tests for split isolation across every variant; feature/metadata alignment; correct targeted/untargeted success denominators; frozen thresholds and ties; zero-success cases; raw-pixel bounds and patch locality; gradient flow through frozen encoders; pooled ResNet features; explanation fixture parity; no hidden clean-reference dependency; and cache invalidation. Do not launch the full benchmark until these checks pass.

Use versioned directories such as `results/controlled_v1/<run_id>/` and `data/processed/controlled_v1/`. Preserve `results_original/` and existing result files. Keep TensorFlow reference fixtures/runtime isolated from the PyTorch production environment. Record all package versions rather than assuming the current unverified requirements list reproduces either upstream repository.

## 12. Execution order, compute gates, and completion criteria

| Milestone | Work | Exit condition |
| --- | --- | --- |
| M0: establish trustworthy baselines | P0-P2; reconstruct available historical metrics and complete the new split/calibration design | Each discrepancy is explained or explicitly unresolved; smoke tests pass |
| M1: validate explainer ports | P3; tiny numerical fixtures, image-panel checks, paper/code specification | SLICE/BELIEF parity established at the claimed level; deviations recorded |
| M2: measure feasibility | Run 100 development images through all core methods/attacks; profile queries, CPU/GPU time, memory and storage | Freeze sample counts, query budgets, seed count, and final configurations before accessing test outputs |
| M3: Stage 1 development | Generate/cache development corpora; fit/select PGD detectors and calibrate on the separate clean cohort | Frozen baseline models/thresholds and reproducible development results |
| M4: Stage 2 development | CW and patch holdouts with matched training size; additional rotations only if affordable | Frozen mixture models and refitted, sample-matched single-attack baselines |
| M5: final evaluation and delivery | Run the locked test manifest for both stages together; stability/fidelity/cost, prescribed ablations, intervals, figures/report | Claims trace to artifacts; limitations and negative findings are explicit |
| M6: optional expansion | DINOv3 and then CIFAR-10, or another order justified by measured cost | Core completed first; extensions use the locked protocol and separate result labels |

No hardware budget or deadline has been supplied, so calendar dates and GPU-hour promises would be speculative. At M2 estimate cost from measured per-image latency/query counts, actual successful-attack counts, number of seeds, and the explicit experiment manifest. Include clean calibration extraction and the noise/fidelity panels, not just adversarial images. Reuse each cached explanation/embedding across detector-training regimes; do not regenerate features for each detector fit.

If compute is constrained, first defer DINOv3, CIFAR-10, extra holdout rotations, and optional ablations. Then reduce detector-training size identically across methods/arms using a prespecified rule. Keep the three required explainers, ResNet/DINOv2 representations, CW/patch tests, independent calibration, and benign-noise evaluation. If sample sizes or explainer approximations must change, freeze and disclose that change before final testing; do not silently drop the slowest method.

### Final artifacts

- A protocol/provenance manifest and a historical-result audit table.
- Tested PyTorch SLICE/BELIEF implementations, reference fixtures, and a paper/code deviation log.
- Configuration files for every classifier, attack, explainer, feature transform, training regime, and seed.
- Per-source CSV records: split, source hash, target model, attack/control, eligibility/success, predictions, detector score, thresholds, timing, feature provenance, and failure status. Store large tensors separately with hashes.
- Main tables for classifier accuracy/ASR/distortion, Stage-1 detection, Stage-2 held-out-family detection, clean/noise false alarms, and explanation quality/cost, with sample counts and intervals.
- ROC/score-distribution plots with frozen thresholds, attack-transfer matrices, noise-severity curves, paired method differences, and representative explanation maps selected by a prespecified sampling rule.
- A reproducible final report distinguishing literal historical replication, corrected evaluation, and new experiments.

The study is complete when all required methods have comparable results under the locked protocol, numerical replication claims are supported, and every headline detection result is accompanied by attack effectiveness, clean/noise false alarms, uncertainty, and cost. Completion does not require exceeding the previous team's AUC or showing that BELIEF/SLICE outperform Grad-CAM.

## Source notes

- Local primary sources: [proposal](Research%20Proposal.md); [previous report](Adversarial_Attacks_Report_Final.pdf), especially pp. 19-40; `results_original/metrics/`, `results_original/tables/`, and current source/configuration files cited above.
- SLICE: Bora et al., CVPR 2024, pp. 10988-10996. [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Bora_SLICE_Stabilized_LIME_for_Consistent_Explanations_for_Image_Classification_CVPR_2024_paper.pdf), [code](https://github.com/rebathip/SLICE-Stabilized-LIME-for-Consistent-Explanations-for-Image-Classification).
- BELIEF: Bora et al., UAI 2025, PMLR 286:332-354. [Paper record](https://proceedings.mlr.press/v286/bora25a.html), [code](https://github.com/rebathip/BELIEF).
- Dataset/model references: [Imagenette](https://github.com/fastai/imagenette), [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [DINOv3](https://github.com/facebookresearch/dinov3).
- Source inspection date: November 2, 2026. Upstream SLICE/BELIEF revisions are pinned above; paper/code parity remains an implementation gate, not a completed result.
