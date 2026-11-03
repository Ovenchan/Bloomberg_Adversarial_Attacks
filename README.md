# Adversarial Explainability Benchmark

This repository contains the codebase for our Summer AI in Finance project. The goal is to evaluate modern explainability methods under contemporary adversarial attacks and compare different feature extraction pipelines.

Our starting point is previous work using LIME for adversarial detection. Rather than reproducing those results, we aim to evaluate newer explainability techniques and visual encoders against modern attack methods and benchmark their strengths and weaknesses under a common evaluation framework.

---

## Project Goals

- Compare CNN-based and visual encoder-based pipelines.
- Evaluate modern explainability methods (Grad-CAM, Integrated Gradients, etc.).
- Test against recent adversarial attacks rather than older benchmark attacks.
- Analyze robustness, explainability, and generalization across models and attacks.
- Keep experiments reproducible and easy to extend.

---

## Repository Layout

```
configs/        Experiment configuration files (datasets, attacks, models)

data/           Datasets (not committed to Git)
    raw/
    processed/

attacks/        Adversarial attack implementations

encoders/       Vision backbone models
                (ResNet, DINOv2, CLIP, etc.)

explainers/     Explainability methods
                (Grad-CAM, Integrated Gradients, LayerCAM, ...)

detectors/      Detection / classification models built on extracted features

analysis/       Evaluation metrics, visualization, plots, dimensionality reduction

experiments/    Experiment entry points

results/
    figures/
    tables/
    logs/

notebooks/      Exploratory notebooks and quick experiments

utils/          Shared helper functions

tests/          Unit tests
```

---

## Team Workflow

Each experiment should ideally follow the same high-level pipeline:

```
Dataset
    ↓
Model / Encoder
    ↓
Adversarial Attack
    ↓
Explainability Method
    ↓
Feature Extraction
    ↓
Evaluation
```

Where possible, experiments should reuse shared components instead of duplicating code.

For example:

- attack implementations belong in `attacks/`
- DINOv2 loading belongs in `encoders/`
- Grad-CAM belongs in `explainers/`
- evaluation metrics belong in `analysis/`

This makes it easier to compare different approaches under the same experimental setup.

---

## Configs

Experiment settings should live in `configs/` rather than being hardcoded.

Examples include:

- dataset
- batch size
- attack parameters
- model selection
- evaluation settings

This should allow experiments to be reproduced by changing configuration files instead of modifying source code.

---

## Branching

Please develop new features on separate branches whenever possible.

Suggested naming:

```
feature/dinov2
feature/gradcam
feature/attacks
feature/metrics
```

Open a PR into `main` once experiments are working.

---

## Notes

This repository is expected to evolve as we narrow the research direction. The initial goal is to build a flexible experimental framework rather than optimize for a single paper implementation.# IEORE4737
