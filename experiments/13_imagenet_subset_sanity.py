"""
Experiment 13: ImageNet Subset Sanity Check

Objective
---------
Construct and validate a small, balanced ImageNet subset for external
validation of the DINOv2 adversarial-representation pipeline.

Inputs
------
- Downloaded ImageNet training images:
    data/raw/imagenet/train/<WNID>/*.JPEG
- Pre-organized official ImageNet validation images:
    data/raw/imagenet/val/<WNID>/*.JPEG

Method
------
Use five fixed ImageNet classes for which sufficient training data are
available locally.

For each class:
    - 75 official TRAIN images -> classifier training
    - 20 official TRAIN images -> internal model-selection validation
    - 50 official VAL images   -> untouched final evaluation

The TRAIN split is deterministic. Images are sorted by ImageNet image ID
before selecting the first 95 examples. The first 75 are assigned to
classifier training and the next 20 to internal validation.

The official ImageNet validation images are never used for classifier
training or model selection.

Outputs
-------
- data/raw/imagenet/meta/imagenet5_train_75_per_class.txt
- data/raw/imagenet/meta/imagenet5_internal_val_20_per_class.txt
- data/raw/imagenet/meta/imagenet5_final_val_50_per_class.txt
- results/metrics/13_imagenet_subset_sanity.json

Research Goal
-------------
Test whether adversarial representation-space behavior observed on
CIFAR-10 also appears on genuine ImageNet-source photographs.

Important Limitation
--------------------
This is a five-class ImageNet subset rather than a full ImageNet-1K
evaluation. The reduced scope was chosen to preserve a balanced,
strictly separated experimental protocol using the ImageNet data
available locally.

Next Experiment
---------------
Extract frozen DINOv2 embeddings for the classifier-training,
internal-validation, and untouched official-validation partitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_ROOT = Path("data/raw/imagenet/train")
VAL_ROOT = Path("data/raw/imagenet/val")
META_ROOT = Path("data/raw/imagenet/meta")
RESULTS_ROOT = Path("results/metrics")

TRAIN_PER_CLASS = 75
INTERNAL_VAL_PER_CLASS = 20
FINAL_VAL_PER_CLASS = 50

# Fixed five-class subset.
#
# These classes were selected from the previously fixed ImageNet subset
# because enough official TRAIN images had already been downloaded for
# every class.
CLASSES: Dict[str, str] = {
    "n01440764": "tench",
    "n02106662": "German shepherd",
    "n02391049": "zebra",
    "n02504458": "African elephant",
    "n02814533": "beach wagon",
}

TRAIN_MANIFEST = META_ROOT / "imagenet5_train_75_per_class.txt"
INTERNAL_VAL_MANIFEST = (
    META_ROOT / "imagenet5_internal_val_20_per_class.txt"
)
FINAL_VAL_MANIFEST = META_ROOT / "imagenet5_final_val_50_per_class.txt"

RESULT_PATH = RESULTS_ROOT / "13_imagenet_subset_sanity.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_id(path: Path) -> str:
    """Return an ImageNet image ID without the .JPEG suffix."""
    return path.stem


def get_images(directory: Path) -> List[Path]:
    """Return JPEG images in deterministic image-ID order."""
    if not directory.exists():
        return []

    return sorted(
        directory.glob("*.JPEG"),
        key=lambda p: p.stem,
    )


def write_manifest(
    path: Path,
    entries: List[tuple[str, Path]],
) -> None:
    """
    Write a manifest as:

        WNID,ImageId

    Paths are intentionally not stored so that experiments can choose
    the appropriate dataset root explicitly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for wnid, image_path in entries:
            f.write(f"{wnid},{image_id(image_path)}\n")


def verify_no_overlap(
    train_entries: List[tuple[str, Path]],
    internal_val_entries: List[tuple[str, Path]],
    final_val_entries: List[tuple[str, Path]],
) -> None:
    """Verify that no ImageNet image ID appears across partitions."""

    train_ids = {image_id(path) for _, path in train_entries}
    internal_val_ids = {
        image_id(path) for _, path in internal_val_entries
    }
    final_val_ids = {
        image_id(path) for _, path in final_val_entries
    }

    assert train_ids.isdisjoint(internal_val_ids), (
        "TRAIN and internal-validation image IDs overlap."
    )

    assert train_ids.isdisjoint(final_val_ids), (
        "TRAIN and final-validation image IDs overlap."
    )

    assert internal_val_ids.isdisjoint(final_val_ids), (
        "Internal-validation and final-validation image IDs overlap."
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    META_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    train_entries: List[tuple[str, Path]] = []
    internal_val_entries: List[tuple[str, Path]] = []
    final_val_entries: List[tuple[str, Path]] = []

    per_class_results = {}

    print("=" * 72)
    print("Experiment 13: ImageNet 5-Class Subset Sanity Check")
    print("=" * 72)

    print()
    print("Protocol:")
    print(f"  Classifier TRAIN:       {TRAIN_PER_CLASS} / class")
    print(
        f"  Internal validation:    "
        f"{INTERNAL_VAL_PER_CLASS} / class"
    )
    print(f"  Official final VAL:     {FINAL_VAL_PER_CLASS} / class")
    print(f"  Number of classes:      {len(CLASSES)}")

    required_train_images = (
        TRAIN_PER_CLASS + INTERNAL_VAL_PER_CLASS
    )

    for wnid, class_name in CLASSES.items():
        train_dir = TRAIN_ROOT / wnid
        val_dir = VAL_ROOT / wnid

        available_train = get_images(train_dir)
        available_val = get_images(val_dir)

        print()
        print(f"{wnid} ({class_name})")
        print(f"  Available TRAIN: {len(available_train)}")
        print(f"  Available VAL:   {len(available_val)}")

        if len(available_train) < required_train_images:
            raise RuntimeError(
                f"{wnid} has only {len(available_train)} TRAIN images; "
                f"{required_train_images} are required."
            )

        if len(available_val) < FINAL_VAL_PER_CLASS:
            raise RuntimeError(
                f"{wnid} has only {len(available_val)} official VAL "
                f"images; {FINAL_VAL_PER_CLASS} are required."
            )

        # Deterministic split from official TRAIN.
        selected_train = available_train[:TRAIN_PER_CLASS]

        selected_internal_val = available_train[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + INTERNAL_VAL_PER_CLASS
        ]

        # The organized VAL folders contain exactly the fixed class-specific
        # official validation images. We still cap at the protocol size.
        selected_final_val = available_val[:FINAL_VAL_PER_CLASS]

        train_entries.extend(
            (wnid, path) for path in selected_train
        )

        internal_val_entries.extend(
            (wnid, path) for path in selected_internal_val
        )

        final_val_entries.extend(
            (wnid, path) for path in selected_final_val
        )

        per_class_results[wnid] = {
            "class_name": class_name,
            "available_train": len(available_train),
            "available_official_val": len(available_val),
            "classifier_train": len(selected_train),
            "internal_val": len(selected_internal_val),
            "final_official_val": len(selected_final_val),
        }

        print(f"  Selected TRAIN:       {len(selected_train)}")
        print(
            f"  Selected internal VAL:"
            f" {len(selected_internal_val)}"
        )
        print(
            f"  Selected official VAL:"
            f" {len(selected_final_val)}"
        )

    # -----------------------------------------------------------------------
    # Global validation
    # -----------------------------------------------------------------------

    expected_train = TRAIN_PER_CLASS * len(CLASSES)
    expected_internal_val = INTERNAL_VAL_PER_CLASS * len(CLASSES)
    expected_final_val = FINAL_VAL_PER_CLASS * len(CLASSES)

    assert len(train_entries) == expected_train
    assert len(internal_val_entries) == expected_internal_val
    assert len(final_val_entries) == expected_final_val

    verify_no_overlap(
        train_entries,
        internal_val_entries,
        final_val_entries,
    )

    # Verify every selected file physically exists.
    all_entries = (
        train_entries
        + internal_val_entries
        + final_val_entries
    )

    missing = [
        str(path)
        for _, path in all_entries
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} selected images are missing."
        )

    # -----------------------------------------------------------------------
    # Save manifests
    # -----------------------------------------------------------------------

    write_manifest(TRAIN_MANIFEST, train_entries)
    write_manifest(
        INTERNAL_VAL_MANIFEST,
        internal_val_entries,
    )
    write_manifest(FINAL_VAL_MANIFEST, final_val_entries)

    # -----------------------------------------------------------------------
    # Save experiment metadata
    # -----------------------------------------------------------------------

    results = {
        "experiment": "13_imagenet_subset_sanity",
        "dataset": "ImageNet Object Localization Challenge subset",
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "protocol": {
            "classifier_train_per_class": TRAIN_PER_CLASS,
            "internal_val_per_class": INTERNAL_VAL_PER_CLASS,
            "final_official_val_per_class": FINAL_VAL_PER_CLASS,
            "classifier_train_total": expected_train,
            "internal_val_total": expected_internal_val,
            "final_official_val_total": expected_final_val,
        },
        "separation": {
            "classifier_train_source": "official ImageNet TRAIN",
            "internal_val_source": "official ImageNet TRAIN",
            "final_eval_source": "official ImageNet VAL",
            "train_internal_val_overlap": 0,
            "train_final_val_overlap": 0,
            "internal_val_final_val_overlap": 0,
        },
        "per_class": per_class_results,
        "manifests": {
            "classifier_train": str(TRAIN_MANIFEST),
            "internal_val": str(INTERNAL_VAL_MANIFEST),
            "final_official_val": str(FINAL_VAL_MANIFEST),
        },
        "status": "passed",
    }

    with RESULT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SANITY CHECK PASSED")
    print("=" * 72)
    print()
    print(f"Classifier TRAIN:      {len(train_entries)}")
    print(f"Internal validation:   {len(internal_val_entries)}")
    print(f"Official final VAL:    {len(final_val_entries)}")
    print(
        f"Total selected images: "
        f"{len(train_entries) + len(internal_val_entries) + len(final_val_entries)}"
    )
    print()
    print("Partition overlap:")
    print("  TRAIN vs internal VAL: 0")
    print("  TRAIN vs official VAL:  0")
    print("  Internal vs official:   0")
    print()
    print("Manifests:")
    print(f"  {TRAIN_MANIFEST}")
    print(f"  {INTERNAL_VAL_MANIFEST}")
    print(f"  {FINAL_VAL_MANIFEST}")
    print()
    print(f"Metrics: {RESULT_PATH}")
    print()
    print(
        "Official ImageNet VAL remains reserved for final "
        "evaluation and adversarial generation."
    )


if __name__ == "__main__":
    main()
