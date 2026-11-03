"""
ImageNet Subset Downloader

Downloads only missing images listed in fixed ImageNet manifests.

Protocol:
    - 200 training images per class
    - 50 internal-validation images per class
    - 50 official validation images per class already downloaded

The downloader is resumable and uses limited concurrency to reduce
the large overhead of downloading thousands of files sequentially.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import time


COMPETITION = "imagenet-object-localization-challenge"

TRAIN_MANIFEST = Path(
    "data/raw/imagenet/meta/train_200_per_class.txt"
)

INTERNAL_VAL_MANIFEST = Path(
    "data/raw/imagenet/meta/internal_val_50_per_class.txt"
)

TRAIN_ROOT = Path("data/raw/imagenet/train")
INTERNAL_VAL_ROOT = Path("data/raw/imagenet/internal_val")

MAX_WORKERS = 1
REQUEST_DELAY = 3.0
MAX_RETRIES = 8
INITIAL_BACKOFF = 60


def load_manifest(path):
    entries = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        wnid, image_id = line.split(",", 1)
        entries.append((wnid, image_id))

    return entries


def download_one(wnid, image_id, local_root):
    out_dir = local_root / wnid
    out_dir.mkdir(parents=True, exist_ok=True)

    output = out_dir / f"{image_id}.JPEG"

    if output.exists():
        return "existing", wnid, image_id

    kaggle_path = (
        f"ILSVRC/Data/CLS-LOC/train/"
        f"{wnid}/{image_id}.JPEG"
    )

    delay = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        cmd = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            kaggle_path,
            "-p",
            str(out_dir),
            "-q",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            zip_path = out_dir / f"{image_id}.JPEG.zip"

            if zip_path.exists():
                subprocess.run(
                    ["unzip", "-oq", str(zip_path), "-d", str(out_dir)],
                    check=True,
                )
                zip_path.unlink()

            if output.exists():
                time.sleep(REQUEST_DELAY)
                return "downloaded", wnid, image_id

        combined = (
            (result.stdout or "") +
            (result.stderr or "")
        )

        if "429" in combined:
            print(
                f"[429] {wnid}/{image_id}; "
                f"waiting {delay}s"
            )
            time.sleep(delay)
            delay *= 2
            continue

        if attempt < MAX_RETRIES:
            time.sleep(5)

    return "failed", wnid, image_id


def run_split(name, manifest, root):
    entries = load_manifest(manifest)

    missing = [
        (wnid, image_id)
        for wnid, image_id in entries
        if not (root / wnid / f"{image_id}.JPEG").exists()
    ]

    print()
    print(f"{name}")
    print(f"Expected: {len(entries)}")
    print(f"Already present: {len(entries) - len(missing)}")
    print(f"Missing: {len(missing)}")
    print()

    if not missing:
        print(f"{name} already complete.")
        return

    downloaded = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                download_one,
                wnid,
                image_id,
                root,
            ): (wnid, image_id)
            for wnid, image_id in missing
        }

        for i, future in enumerate(as_completed(futures), 1):
            status, wnid, image_id = future.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "failed":
                failed.append((wnid, image_id))

            if i % 25 == 0 or i == len(missing):
                print(
                    f"[{name}] processed {i}/{len(missing)} "
                    f"| downloaded {downloaded} "
                    f"| failed {len(failed)}"
                )

    print()
    print(f"{name} complete.")
    print(f"Downloaded this run: {downloaded}")
    print(f"Failures: {len(failed)}")

    if failed:
        print("Failed entries:")
        for wnid, image_id in failed:
            print(f"  {wnid},{image_id}")


def main():
    run_split(
        "TRAIN",
        TRAIN_MANIFEST,
        TRAIN_ROOT,
    )

    run_split(
        "INTERNAL VAL",
        INTERNAL_VAL_MANIFEST,
        INTERNAL_VAL_ROOT,
    )


if __name__ == "__main__":
    main()
