"""Source-grouped detector evaluation using the unmodified shared metrics."""

import numpy as np
from sklearn.model_selection import KFold

from analysis.metrics import (
    DetectionEvaluator,
    filter_successful_attacks,
    summarize,
)
from detectors.logistic import GradCAMLogisticDetector

DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
REPORT_METRICS = (
    "auc",
    "attack_success_rate",
    "num_adversarial_evaluated",
    "num_clean",
)


class GroupedDetectionCV:
    """Input: fold count, seed and detector config. Output: grouped evaluator."""

    def __init__(
        self,
        n_splits=DEFAULT_FOLDS,
        seed=DEFAULT_SEED,
        detector_config=None,
    ):
        """Input: int n_splits/seed, dict|None detector_config. Output: None."""
        self.n_splits = n_splits
        self.seed = seed
        self.detector_config = detector_config or {}
        self.evaluator = DetectionEvaluator()
        self.detectors = []

    def split(self, groups):
        """Input: array[N] source IDs. Output: iterator of (train,test) index arrays."""
        groups = np.asarray(groups)
        unique = np.unique(groups)
        splitter = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )
        for train, test in splitter.split(unique):
            train_rows = np.flatnonzero(np.isin(groups, unique[train]))
            test_rows = np.flatnonzero(np.isin(groups, unique[test]))
            yield train_rows, test_rows

    def evaluate(self, clean, attacks, groups, labels, clean_predictions):
        """Evaluate aligned feature matrices and source IDs.

        Input: clean array[N,D], attack dicts with features/predictions/source_ids,
            and groups, labels, clean_predictions arrays[N].
        Output: dict of fold metrics, group assignments and summaries.
        """
        clean, groups, labels, clean_predictions = map(
            np.asarray, (clean, groups, labels, clean_predictions)
        )
        if clean.ndim != 2 or not np.isfinite(clean).all():
            raise ValueError("Clean features must be a finite matrix")
        if any(
            a.shape != (len(clean),)
            for a in (groups, labels, clean_predictions)
        ):
            raise ValueError("Metadata must align exactly with clean features")
        if "pgd" not in attacks:
            raise ValueError("PGD is required for detector training")
        masks = {}
        for name, attack in attacks.items():
            if "source_ids" not in attack or not np.array_equal(
                attack["source_ids"], groups
            ):
                raise ValueError(
                    f"{name}: source IDs must align exactly with clean rows"
                )
            attack_features = np.asarray(attack["features"])
            if (
                attack_features.shape != clean.shape
                or not np.isfinite(attack_features).all()
            ):
                raise ValueError(
                    f"{name}: features must align exactly; no truncation allowed"
                )
            masks[name] = filter_successful_attacks(
                clean_predictions, attack["predictions"], labels
            )
        folds = []
        self.detectors = []
        for fold, (train, test) in enumerate(self.split(groups)):
            successful_train = train[masks["pgd"][train]]
            if not len(successful_train):
                raise ValueError(
                    f"Fold {fold}: no successful training PGD attacks"
                )
            detector = GradCAMLogisticDetector(
                seed=self.seed, **self.detector_config
            )
            pgd_train = np.asarray(attacks["pgd"]["features"])[
                successful_train
            ]
            training_features = np.concatenate([clean[train], pgd_train])
            training_labels = np.concatenate(
                [np.zeros(len(train)), np.ones(len(successful_train))]
            )
            detector.fit(training_features, training_labels)
            self.detectors.append(detector)
            clean_scores = detector.score_samples(clean[test])
            metrics = {}
            for name, attack in attacks.items():
                metrics[name] = self.evaluator.evaluate(
                    clean_scores,
                    detector.score_samples(
                        np.asarray(attack["features"])[test]
                    ),
                    masks[name][test],
                )
            folds.append(
                {
                    "fold": fold,
                    "train_groups": np.unique(groups[train]).tolist(),
                    "test_groups": np.unique(groups[test]).tolist(),
                    "metrics": metrics,
                }
            )
        summary = {}
        for name in attacks:
            values = {
                metric: [f["metrics"][name][metric] for f in folds]
                for metric in REPORT_METRICS
            }
            for target in self.evaluator.fpr_targets:
                values[f"tpr_at_{target}"] = [
                    f["metrics"][name]["tpr_at_fpr"][str(target)]
                    for f in folds
                ]

            summary[name] = {
                key: {
                    **summarize(value),
                    "valid_folds": int(np.isfinite(value).sum()),
                }
                for key, value in values.items()
            }
        return {"folds": folds, "summary": summary}
