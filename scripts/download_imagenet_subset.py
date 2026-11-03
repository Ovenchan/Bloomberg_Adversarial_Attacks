#!/usr/bin/env python3

"""
Download the fixed 10-class ImageNet subset for Experiment 13.

Design
------
Official TRAIN:
    400 classifier-train images per class
    50 internal-validation images per class

Official VAL:
    50 final-evaluation images per class

Outputs
-------
data/raw/imagenet/train/<WNID>/*.JPEG
data/raw/imagenet/internal_val/<WNID>/*.JPEG
data/raw/imagenet/val/<WNID>/*.JPEG

Important
---------
- Image IDs are deduplicated before manifests are created.
- Downloads are resumable.
- HTTP 429 errors are retried with exponential backoff.
- A small delay is added between Kaggle requests.
"""

import csv
import random
import subprocess
import time
from pathlib import Path


COMPETITION = "imagenet-object-localization-challenge"

WNIDS = [
    "n01440764",
    "n02106662",
    "n02391049",
    "n02504458",
    "n02814533",
    "n03417042",
    "n03770679",
    "n04254680",
    "n04285008",
    "n04591157",
]

TRAIN_PER_CLASS = 400
INTERNAL_VAL_PER_CLASS = 50
FINAL_VAL_PER_CLASS = 50

SEED = 42

# Delay between successful Kaggle requests.
REQUEST_DELAY_SECONDS = 1.0

# Retry schedule for rate limits / temporary failures.
MAX_RETRIES = 8
INITIAL_BACKOFF_SECONDS = 30


ROOT = Path("data/raw/imagenet")

TRAIN_ROOT = ROOT / "train"
INTERNAL_VAL_ROOT = ROOT / "internal_val"
FINAL_VAL_ROOT = ROOT / "val"
META_ROOT = ROOT / "meta"

TRAIN_CSV = META_ROOT / "LOC_train_solution.csv"

TRAIN_MANIFEST = META_ROOT / "train_400_per_class.txt"
INTERNAL_VAL_MANIFEST = META_ROOT / "internal_val_50_per_class.txt"


def load_train_ids():
    """
    Read ImageNet TRAIN metadata and group UNIQUE image IDs by WNID.
    """

    grouped = {wnid: set() for wnid in WNIDS}

    with TRAIN_CSV.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            prediction = row["PredictionString"].split()

            if not prediction:
                continue

            wnid = prediction[0]

            if wnid in grouped:
                grouped[wnid].add(row["ImageId"])

    # Convert sets to sorted lists for deterministic behavior.
    return {
        wnid: sorted(image_ids)
        for wnid, image_ids in grouped.items()
    }


def build_manifests(force=False):
    """
    Deterministically select:
        400 TRAIN images/class
        50 internal-validation images/class

    IDs are unique within each class.
    """

    if (
        not force
        and TRAIN_MANIFEST.exists()
        and INTERNAL_VAL_MANIFEST.exists()
    ):
        print("Using existing manifests.")
        return

    grouped = load_train_ids()

    rng = random.Random(SEED)

    train_rows = []
    internal_val_rows = []

    print("=== Available UNIQUE TRAIN images ===")

    for wnid in WNIDS:
        image_ids = list(grouped[wnid])

        print(f"{wnid}: {len(image_ids)}")

        required = TRAIN_PER_CLASS + INTERNAL_VAL_PER_CLASS

        if len(image_ids) < required:
            raise RuntimeError(
                f"{wnid}: need {required} unique images, "
                f"but only found {len(image_ids)}"
            )

        rng.shuffle(image_ids)

        selected = image_ids[:required]

        train_ids = selected[:TRAIN_PER_CLASS]
        internal_val_ids = selected[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + INTERNAL_VAL_PER_CLASS
        ]

        for image_id in train_ids:
            train_rows.append((wnid, image_id))

        for image_id in internal_val_ids:
            internal_val_rows.append((wnid, image_id))

    META_ROOT.mkdir(parents=True, exist_ok=True)

    with TRAIN_MANIFEST.open("w") as f:
        for wnid, image_id in train_rows:
            f.write(f"{wnid},{image_id}\n")

    with INTERNAL_VAL_MANIFEST.open("w") as f:
        for wnid, image_id in internal_val_rows:
            f.write(f"{wnid},{image_id}\n")

    print()
    print(f"Saved: {TRAIN_MANIFEST}")
    print(f"Saved: {INTERNAL_VAL_MANIFEST}")


def download_file(wnid, image_id, output_root):
    """
    Download one ImageNet TRAIN image with retries.

    Returns:
        True if downloaded now.
        False if it already existed.
    """

    output_dir = output_root / wnid
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{image_id}.JPEG"
    local_path = output_dir / filename

    if local_path.exists():
        return False

    remote_path = (
        f"ILSVRC/Data/CLS-LOC/train/"
        f"{wnid}/{filename}"
    )

    delay = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):

        result = subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                COMPETITION,
                "-f",
                remote_path,
                "-p",
                str(output_dir),
                "-q",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            time.sleep(REQUEST_DELAY_SECONDS)
            return True

        combined_output = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )

        if "429" in combined_output or "Too Many Requests" in combined_output:
            print(
                f"\n  Kaggle rate limit hit. "
                f"Sleeping {delay}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            time.sleep(delay)

            delay *= 2
            continue

        print("\nDownload failed:")
        print(combined_output)

        raise RuntimeError(
            f"Failed downloading {remote_path}"
        )

    raise RuntimeError(
        f"Exceeded retry limit for {remote_path}"
    )


def load_manifest(path):
    rows = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            wnid, image_id = line.split(",", 1)
            rows.append((wnid, image_id))

    return rows


def download_manifest(manifest_path, output_root, label):
    rows = load_manifest(manifest_path)

    grouped = {wnid: [] for wnid in WNIDS}

    for wnid, image_id in rows:
        grouped[wnid].append(image_id)

    print(f"\n=== {label} download ===")

    for wnid in WNIDS:
        image_ids = grouped[wnid]

        output_dir = output_root / wnid
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{wnid}: target {len(image_ids)}")

        for i, image_id in enumerate(image_ids, start=1):

            download_file(
                wnid=wnid,
                image_id=image_id,
                output_root=output_root,
            )

            if i % 25 == 0 or i == len(image_ids):
                count = len(
                    list(output_dir.glob("*.JPEG"))
                )

                print(
                    f"  processed {i}/{len(image_ids)} "
                    f"(local: {count})"
                )


def verify_split(root, expected_per_class, label):
    print(f"\n=== {label} verification ===")

    for wnid in WNIDS:
        folder = root / wnid

        count = len(
            list(folder.glob("*.JPEG"))
        )

        print(f"{wnid}: {count}")

        if count != expected_per_class:
            raise RuntimeError(
                f"{label} {wnid}: "
                f"expected {expected_per_class}, "
                f"found {count}"
            )


def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_CSV}"
        )

    build_manifests()

    download_manifest(
        TRAIN_MANIFEST,
        TRAIN_ROOT,
        "TRAIN",
    )

    download_manifest(
        INTERNAL_VAL_MANIFEST,
        INTERNAL_VAL_ROOT,
        "INTERNAL VAL",
    )

    verify_split(
        TRAIN_ROOT,
        TRAIN_PER_CLASS,
        "TRAIN",
    )

    verify_split(
        INTERNAL_VAL_ROOT,
        INTERNAL_VAL_PER_CLASS,
        "INTERNAL VAL",
    )

    verify_split(
        FINAL_VAL_ROOT,
        FINAL_VAL_PER_CLASS,
        "FINAL VAL",
    )

    print("\nImageNet subset complete.")
    print("TRAIN:        4,000 images")
    print("INTERNAL VAL:   500 images")
    print("FINAL VAL:      500 images")
    print("TOTAL:        5,000 images")


if __name__ == "__main__":
    main()
