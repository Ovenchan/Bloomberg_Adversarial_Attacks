"""
Experiment 14: ImageNet DINOv2 Embedding Extraction

Objective
---------
Extract frozen DINOv2 ViT-S/14 representations for the balanced
five-class ImageNet subset created in Experiment 13.

Inputs
------
- data/raw/imagenet/meta/imagenet5_train_75_per_class.txt
- data/raw/imagenet/meta/imagenet5_internal_val_20_per_class.txt
- data/raw/imagenet/meta/imagenet5_final_val_50_per_class.txt
- data/raw/imagenet/train/<WNID>/*.JPEG
- data/raw/imagenet/val/<WNID>/*.JPEG

Method
------
Each source image is transformed using the standard ImageNet preprocessing
used for DINOv2 experiments:
    - resize shorter side to 256
    - center crop to 224 x 224
    - convert to tensor
    - normalize using ImageNet mean/std

The pretrained DINOv2 ViT-S/14 encoder is frozen and used only for
gradient-free feature extraction. The resulting representation is
384-dimensional.

Outputs
-------
- data/processed/embeddings/imagenet5/train_dinov2_vits14.pt
- data/processed/embeddings/imagenet5/internal_val_dinov2_vits14.pt
- data/processed/embeddings/imagenet5/final_val_dinov2_vits14.pt
- results/metrics/14_imagenet_dinov2_embeddings.json

Research Goal
-------------
Establish clean DINOv2 representations for genuine ImageNet-source
photographs before classifier training and adversarial evaluation.

Important Limitation
--------------------
This experiment uses a balanced five-class ImageNet subset rather than
the complete ImageNet-1K benchmark. Images are evaluated after the
standard 224 x 224 DINOv2 preprocessing crop.

Next Experiment
---------------
Train a linear 384 -> 5 classifier using only the classifier-training
embeddings, select the best epoch using the internal-validation split,
and evaluate once on the untouched official ImageNet validation set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from encoders.dinov2 import DinoV2Encoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "dinov2_vits14"
EMBEDDING_DIM = 384

TRAIN_ROOT = Path("data/raw/imagenet/train")
FINAL_VAL_ROOT = Path("data/raw/imagenet/val")

META_ROOT = Path("data/raw/imagenet/meta")

TRAIN_MANIFEST = META_ROOT / "imagenet5_train_75_per_class.txt"
INTERNAL_VAL_MANIFEST = (
    META_ROOT / "imagenet5_internal_val_20_per_class.txt"
)
FINAL_VAL_MANIFEST = META_ROOT / "imagenet5_final_val_50_per_class.txt"

OUTPUT_ROOT = Path("data/processed/embeddings/imagenet5")
RESULTS_ROOT = Path("results/metrics")

TRAIN_OUTPUT = OUTPUT_ROOT / "train_dinov2_vits14.pt"
INTERNAL_VAL_OUTPUT = (
    OUTPUT_ROOT / "internal_val_dinov2_vits14.pt"
)
FINAL_VAL_OUTPUT = OUTPUT_ROOT / "final_val_dinov2_vits14.pt"

RESULT_PATH = RESULTS_ROOT / "14_imagenet_dinov2_embeddings.json"

BATCH_SIZE = 64
NUM_WORKERS = 4

CLASSES: Dict[str, int] = {
    "n01440764": 0,  # tench
    "n02106662": 1,  # German shepherd
    "n02391049": 2,  # zebra
    "n02504458": 3,  # African elephant
    "n02814533": 4,  # beach wagon
}

CLASS_NAMES: Dict[int, str] = {
    0: "tench",
    1: "German shepherd",
    2: "zebra",
    3: "African elephant",
    4: "beach wagon",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ManifestImageNetDataset(Dataset):
    """
    ImageNet subset backed by a fixed manifest.

    Each manifest row has the form:
        WNID,ImageId
    """

    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        transform,
    ):
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.transform = transform

        self.samples: List[Tuple[Path, int, str]] = []

        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue

            wnid, image_id = line.split(",", 1)

            if wnid not in CLASSES:
                raise ValueError(
                    f"Unexpected WNID {wnid} in {manifest_path}"
                )

            image_path = image_root / wnid / f"{image_id}.JPEG"

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing image: {image_path}"
                )

            label = CLASSES[wnid]

            self.samples.append(
                (image_path, label, image_id)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, image_id = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label, image_id


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    encoder: DinoV2Encoder,
    loader: DataLoader,
):
    all_embeddings = []
    all_labels = []
    all_image_ids = []

    for images, labels, image_ids in tqdm(
        loader,
        desc="Extracting embeddings",
    ):
        embeddings = encoder.encode(images)

        all_embeddings.append(embeddings)
        all_labels.append(labels.cpu())
        all_image_ids.extend(list(image_ids))

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return embeddings, labels, all_image_ids


def save_split(
    encoder: DinoV2Encoder,
    dataset: ManifestImageNetDataset,
    output_path: Path,
    split_name: str,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(f"Extracting {split_name}")
    print(f"Images: {len(dataset)}")

    embeddings, labels, image_ids = extract_embeddings(
        encoder,
        loader,
    )

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"{split_name}: expected 2-D embeddings, "
            f"got shape {tuple(embeddings.shape)}"
        )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"{split_name}: expected embedding dimension "
            f"{EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )

    if embeddings.shape[0] != len(dataset):
        raise RuntimeError(
            f"{split_name}: embedding count mismatch."
        )

    if labels.shape[0] != len(dataset):
        raise RuntimeError(
            f"{split_name}: label count mismatch."
        )

    if len(image_ids) != len(dataset):
        raise RuntimeError(
            f"{split_name}: image ID count mismatch."
        )

    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError(
            f"{split_name}: duplicate image IDs detected."
        )

    if not torch.isfinite(embeddings).all():
        raise RuntimeError(
            f"{split_name}: non-finite embedding values detected."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
            "image_ids": image_ids,
            "model_name": MODEL_NAME,
            "class_names": CLASS_NAMES,
        },
        output_path,
    )

    print(
        f"Saved {split_name}: "
        f"{tuple(embeddings.shape)} -> {output_path}"
    )

    counts = torch.bincount(
        labels,
        minlength=len(CLASSES),
    ).tolist()

    return {
        "num_images": len(dataset),
        "embedding_shape": list(embeddings.shape),
        "label_counts": {
            CLASS_NAMES[i]: int(counts[i])
            for i in range(len(CLASSES))
        },
        "unique_image_ids": len(set(image_ids)),
        "finite_embeddings": True,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    train_dataset = ManifestImageNetDataset(
        manifest_path=TRAIN_MANIFEST,
        image_root=TRAIN_ROOT,
        transform=transform,
    )

    internal_val_dataset = ManifestImageNetDataset(
        manifest_path=INTERNAL_VAL_MANIFEST,
        image_root=TRAIN_ROOT,
        transform=transform,
    )

    final_val_dataset = ManifestImageNetDataset(
        manifest_path=FINAL_VAL_MANIFEST,
        image_root=FINAL_VAL_ROOT,
        transform=transform,
    )

    print("=" * 72)
    print("Experiment 14: ImageNet DINOv2 Embedding Extraction")
    print("=" * 72)
    print()
    print(f"Classifier TRAIN:    {len(train_dataset)}")
    print(f"Internal validation: {len(internal_val_dataset)}")
    print(f"Official final VAL:  {len(final_val_dataset)}")

    # -----------------------------------------------------------------------
    # Confirm split identity separation before model execution
    # -----------------------------------------------------------------------

    train_ids = {
        image_id
        for _, _, image_id in train_dataset.samples
    }

    internal_ids = {
        image_id
        for _, _, image_id in internal_val_dataset.samples
    }

    final_ids = {
        image_id
        for _, _, image_id in final_val_dataset.samples
    }

    if not train_ids.isdisjoint(internal_ids):
        raise RuntimeError(
            "TRAIN and internal-validation image IDs overlap."
        )

    if not train_ids.isdisjoint(final_ids):
        raise RuntimeError(
            "TRAIN and final-validation image IDs overlap."
        )

    if not internal_ids.isdisjoint(final_ids):
        raise RuntimeError(
            "Internal-validation and final-validation image IDs overlap."
        )

    encoder = DinoV2Encoder(
        model_name=MODEL_NAME,
    )

    train_metrics = save_split(
        encoder,
        train_dataset,
        TRAIN_OUTPUT,
        "classifier TRAIN",
    )

    internal_metrics = save_split(
        encoder,
        internal_val_dataset,
        INTERNAL_VAL_OUTPUT,
        "internal validation",
    )

    final_metrics = save_split(
        encoder,
        final_val_dataset,
        FINAL_VAL_OUTPUT,
        "official final VAL",
    )

    results = {
        "experiment": "14_imagenet_dinov2_embeddings",
        "model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "preprocessing": {
            "resize_shorter_side": 256,
            "center_crop": 224,
            "normalization_mean": [
                0.485,
                0.456,
                0.406,
            ],
            "normalization_std": [
                0.229,
                0.224,
                0.225,
            ],
        },
        "splits": {
            "classifier_train": train_metrics,
            "internal_validation": internal_metrics,
            "official_final_validation": final_metrics,
        },
        "identity_overlap": {
            "train_internal": 0,
            "train_final": 0,
            "internal_final": 0,
        },
        "status": "passed",
    }

    with RESULT_PATH.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("EMBEDDING EXTRACTION PASSED")
    print("=" * 72)
    print()
    print(
        f"TRAIN embeddings:        "
        f"{train_metrics['embedding_shape']}"
    )
    print(
        f"Internal VAL embeddings: "
        f"{internal_metrics['embedding_shape']}"
    )
    print(
        f"Final VAL embeddings:    "
        f"{final_metrics['embedding_shape']}"
    )
    print()
    print(f"Metrics: {RESULT_PATH}")


if __name__ == "__main__":
    main()
