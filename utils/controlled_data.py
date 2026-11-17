"""Source-identity manifest and class-stratified partitions for controlled runs.

Every Imagenette validation file (or CIFAR-10 official test image) receives a
globally unique `source_id` (dataset, split, file path or index, content hash). Partitions are drawn per
class from a seeded permutation (never the first N filesystem entries);
all variants of a source (attacks, noise, repeated explanations) inherit
its partition.
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from utils.gradcam_data import IMAGENET_LABELS

PARTITION_ORDER = ("train", "dev", "cal", "test")
RESERVE_PARTITION = "reserve"
HASH_PREFIX_LENGTH = 12
SEED_MODULUS = 2 ** 63 - 1
MANIFEST_FIELDS = (
    "source_id", "path", "wnid", "class_index", "imagenet_label",
    "partition", "partition_row", "noise_panel", "stability_panel",
)


def derive_seed(*parts):
    """Input: any printable parts (run seed, source id, purpose...).

    Output: int in [0, 2^63-1) from SHA-256, independent of batch order.
    """
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % SEED_MODULUS


def file_hash(path):
    """Input: str/Path. Output: str SHA-1 hex digest prefix of the file bytes."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:HASH_PREFIX_LENGTH]


def partition_counts(pool_size, per_class):
    """Per-class partition sizes from absolute counts or pool fractions.

    Input: int pool_size (images available for this class), dict partition ->
        int count, or float fraction of the pool (fractions summing to 1 use
        the whole pool; rounding remainders go to the last partition).
    Output: dict partition -> int count (sum <= pool_size).
    """
    if all(isinstance(per_class[name], int) for name in PARTITION_ORDER):
        counts = {name: per_class[name] for name in PARTITION_ORDER}
    else:
        counts = {name: int(np.floor(per_class[name] * pool_size)) for name in PARTITION_ORDER}
        if np.isclose(sum(per_class[name] for name in PARTITION_ORDER), 1.0):
            counts[PARTITION_ORDER[-1]] += pool_size - sum(counts.values())
    if sum(counts.values()) > pool_size:
        raise ValueError(f"Partitions need {sum(counts.values())} images per class, only {pool_size} available")
    return counts


CIFAR10_BATCH = "cifar-10-batches-py/test_batch"


def cifar10_test_set(root):
    """Input: data root. Output: torchvision CIFAR10 official TEST split (PIL images)."""
    return datasets.CIFAR10(root=root, train=False, download=False)


def _imagenette_sources(root, size):
    """Output: (dict label -> sorted list[Path], callable(item, label) -> identity fields)."""
    dataset = datasets.Imagenette(root=root, split="val", size=size, download=False)
    by_class = {}
    for path, label in dataset._samples:
        by_class.setdefault(label, []).append(Path(path))

    def describe(path, label):
        wnid = dataset.wnids[label]
        relative = f"{wnid}/{path.name}"
        return {"source_id": f"imagenette2:val:{relative}:{file_hash(path)}", "path": str(path),
                "wnid": wnid, "class_index": label, "imagenet_label": IMAGENET_LABELS[wnid]}
    return {label: sorted(paths) for label, paths in by_class.items()}, describe


def _cifar10_sources(root):
    """CIFAR-10 test images keyed by class; `imagenet_label` holds the 10-class head label."""
    dataset = cifar10_test_set(root)
    by_class = {}
    for index, label in enumerate(dataset.targets):
        by_class.setdefault(int(label), []).append(index)

    def describe(index, label):
        digest = hashlib.sha1(dataset.data[index].tobytes()).hexdigest()[:HASH_PREFIX_LENGTH]
        return {"source_id": f"cifar10:test:{index:05d}:{digest}",
                "path": f"{Path(root) / CIFAR10_BATCH}#{index}",
                "wnid": dataset.classes[label], "class_index": label, "imagenet_label": label}
    return by_class, describe


def build_manifest(root, per_class, noise_per_class, stability_per_class, seed, size="full",
                   subset_per_class=None, dataset_name="imagenette"):
    """Class-stratified partition manifest of the official Imagenette val split.

    Input: str root, dict partition -> images per class (int counts or float
        fractions, keys in PARTITION_ORDER), int noise/stability panel sizes
        per class (drawn from the test partition), int seed, str Imagenette
        size, optional int subset_per_class: if set, each class is first
        reduced to a fixed seeded subset of that many files (e.g. 100 ->
        a fixed 1,000-file, 10-class subset) and partitions are drawn inside it;
        if None the whole val split is the pool.
        dataset_name: "imagenette" (official val split) or "cifar10" (official
        test split, which the CIFAR-10 classifier head never trained on).
    Output: list[dict] rows with MANIFEST_FIELDS (unused images: reserve).
    """
    if dataset_name == "cifar10":
        by_class, describe = _cifar10_sources(root)
    elif dataset_name == "imagenette":
        by_class, describe = _imagenette_sources(root, size)
    else:
        raise ValueError(f"Unsupported dataset {dataset_name!r}")
    rows = []
    counters = {name: 0 for name in (*PARTITION_ORDER, RESERVE_PARTITION)}
    rng = np.random.default_rng(seed)
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        order = rng.permutation(len(paths))
        pool = len(paths) if subset_per_class is None else subset_per_class
        if pool > len(paths):
            raise ValueError(f"Class {label} has {len(paths)} files, subset needs {pool}")
        counts = partition_counts(pool, per_class)
        cursor = 0
        assigned = []
        for name in PARTITION_ORDER:
            assigned += [(paths[i], name) for i in order[cursor:cursor + counts[name]]]
            cursor += counts[name]
        assigned += [(paths[i], RESERVE_PARTITION) for i in order[cursor:]]
        test_seen = 0
        for path, name in assigned:
            rows.append(
                {
                    **describe(path, label),
                    "partition": name,
                    "partition_row": -1,
                    "noise_panel": name == "test" and test_seen < noise_per_class,
                    "stability_panel": name == "test" and test_seen < stability_per_class,
                }
            )
            test_seen += name == "test"
    # Interleave classes inside each partition so row order is class-balanced.
    for name in counters:
        members = [r for r in rows if r["partition"] == name]
        per_label = {}
        for r in members:
            per_label.setdefault(r["class_index"], []).append(r)
        position = 0
        depth = max((len(v) for v in per_label.values()), default=0)
        for k in range(depth):
            for label in sorted(per_label):
                if k < len(per_label[label]):
                    per_label[label][k]["partition_row"] = position
                    position += 1
    return rows


def write_manifest(rows, path):
    """Input: list[dict], str/Path. Output: None; writes CSV with MANIFEST_FIELDS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    """Input: str/Path. Output: list[dict] with typed ints/bools."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("class_index", "imagenet_label", "partition_row"):
            row[key] = int(row[key])
        for key in ("noise_panel", "stability_panel"):
            row[key] = row[key] == "True"
    return rows


def partition_rows(rows, partition):
    """Input: manifest rows, str partition. Output: rows sorted by partition_row."""
    return sorted((r for r in rows if r["partition"] == partition), key=lambda r: r["partition_row"])


def image_transform(resize_size, image_size):
    """Input: ints. Output: torchvision transform PIL -> Tensor[3,S,S] in [0,1]."""
    return transforms.Compose(
        [transforms.Resize(resize_size), transforms.CenterCrop(image_size), transforms.ToTensor()]
    )


def load_native_images(rows):
    """CIFAR-10 rows only. Output: Tensor[N,3,32,32] uint8, the original stored pixels."""
    if not rows or not all(r["source_id"].startswith("cifar10:") for r in rows):
        raise ValueError("Native-resolution loading is defined for CIFAR-10 rows only")
    dataset = cifar10_test_set(Path(rows[0]["path"].split("#")[0]).parent.parent)
    pixels = np.stack([dataset.data[int(r["path"].split("#")[1])] for r in rows])   # [N,32,32,3]
    return torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous()


def load_images(rows, resize_size, image_size):
    """Input: manifest rows, ints. Output: Tensor[N,3,S,S] uint8 (exact k/255 pixels).

    CIFAR-10 rows are resized 32 -> (S, S) exactly as the classifier head was
    trained (utils.datasets.get_cifar10_loader); resize_size is unused for them.
    """
    if rows and rows[0]["source_id"].startswith("cifar10:"):
        root = Path(rows[0]["path"].split("#")[0]).parent.parent
        dataset = cifar10_test_set(root)
        transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        images = [transform(dataset[int(r["path"].split("#")[1])][0]) for r in rows]
    else:
        transform = image_transform(resize_size, image_size)
        images = [transform(Image.open(r["path"]).convert("RGB")) for r in rows]
    return (torch.stack(images) * 255).round().to(torch.uint8)
