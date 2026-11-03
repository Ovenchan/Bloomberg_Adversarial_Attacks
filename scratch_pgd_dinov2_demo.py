"""
Throwaway standalone script — PGD attack on ResNet18, DINOv2 embedding shift.

Not part of the shared pipeline. No imports from this repo. Run directly:
    python scratch_pgd_dinov2_demo.py

Pipeline: CIFAR-10 -> ResNet18 (ImageNet weights, attack target) -> PGD
-> DINOv2 embeddings of clean vs adversarial -> cosine similarity, compared
against a clean-vs-clean baseline (two different clean images) to check
whether the attacked similarity is actually low, or just sitting at the
background noise level for "two different images of anything".

The ResNet18 is ImageNet-pretrained, so its class space has nothing to do
with CIFAR-10 labels. "Attack success" therefore means "the predicted
ImageNet class changed", using the model's own clean prediction as the
attack's target label (there's no meaningful ground truth to compare to
here) — this differs from the repo's real eval protocol, which requires a
correct clean prediction against true labels. Fine for a first number;
not fine for final results.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import torchvision
from torchvision import transforms

NUM_IMAGES = 100
BATCH_SIZE = 20  # chunk size for attack/embedding passes, keeps memory bounded
IMAGE_SIZE = 224  # DINOv2 patch size is 14; 224 / 14 = 16 patches per side
PGD_EPS = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10
SEED = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    """Pick the fastest available backend. Input: none. Output: torch.device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NormalizedResNet18(nn.Module):
    """ImageNet-pretrained ResNet18 with normalization baked in.

    Input: image batch tensor in [0, 1], shape (B, 3, H, W).
    Output: logits, shape (B, 1000).

    Normalization lives inside the model (not in the data transform) so
    that attacks operate directly on [0, 1] pixel values — required for
    epsilon and PSNR to mean what they say.
    """

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def load_cifar10_images(num_images: int) -> torch.Tensor:
    """Load the first `num_images` CIFAR-10 test images, resized for ResNet18/DINOv2.

    Input: num_images (int). Output: tensor (num_images, 3, 224, 224) in [0, 1].
    """
    transform = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    images = torch.stack([dataset[i][0] for i in range(num_images)])
    return images


def psnr_batch(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR between two [0, 1]-range image batches.

    Input: clean, adv — tensors of identical shape (B, C, H, W), values in [0, 1].
    Output: tensor of shape (B,), PSNR in dB per image.
    """
    mse = F.mse_loss(clean, adv, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-12)  # avoid log(0) for identical images
    return 10 * torch.log10(1.0 / mse)


def normalize_for_dinov2(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Apply ImageNet normalization for DINOv2 input.

    Input: x in [0, 1], shape (B, 3, H, W). Output: normalized tensor, same shape.
    """
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def dinov2_embed(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """DINOv2 CLS-token embeddings, computed in chunks of BATCH_SIZE.

    Input: images in [0, 1], shape (N, 3, H, W). Output: embeddings, shape (N, D).
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, images.shape[0], BATCH_SIZE):
            chunk = images[start : start + BATCH_SIZE]
            chunks.append(model(normalize_for_dinov2(chunk, device)))
    return torch.cat(chunks, dim=0)


def stats(x: torch.Tensor) -> str:
    """Format mean/std/min/max of a 1D tensor. Input: tensor. Output: str."""
    if x.numel() == 0:
        return "n=0"
    return (
        f"mean={x.mean().item():.4f} std={x.std().item():.4f} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} (n={x.numel()})"
    )


def main():
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading {NUM_IMAGES} CIFAR-10 images...")
    images = load_cifar10_images(NUM_IMAGES).to(device)

    print("Loading ResNet18 (ImageNet weights, attack target)...")
    resnet = NormalizedResNet18().to(device).eval()

    print("Loading DINOv2 (dinov2_vits14, encoder of interest)...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

    atk = torchattacks.PGD(resnet, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)

    # Clean predictions (used as PGD's pseudo-label — see module docstring).
    clean_preds = []
    with torch.no_grad():
        for start in range(0, NUM_IMAGES, BATCH_SIZE):
            chunk = images[start : start + BATCH_SIZE]
            clean_preds.append(resnet(chunk).argmax(dim=1))
    clean_preds = torch.cat(clean_preds)

    # PGD attack, batched.
    print("Running PGD...")
    adv_chunks = []
    for start in range(0, NUM_IMAGES, BATCH_SIZE):
        chunk = images[start : start + BATCH_SIZE]
        pred_chunk = clean_preds[start : start + BATCH_SIZE]
        adv_chunks.append(atk(chunk, pred_chunk))
    adv_images = torch.cat(adv_chunks, dim=0)

    adv_preds = []
    with torch.no_grad():
        for start in range(0, NUM_IMAGES, BATCH_SIZE):
            chunk = adv_images[start : start + BATCH_SIZE]
            adv_preds.append(resnet(chunk).argmax(dim=1))
    adv_preds = torch.cat(adv_preds)

    success = adv_preds != clean_preds
    image_psnr = psnr_batch(images, adv_images)

    # DINOv2 embeddings for all clean images (reused for both the attack
    # comparison and the clean-vs-clean baseline) and all adversarial images.
    print("Computing DINOv2 embeddings...")
    clean_emb = dinov2_embed(dinov2, images, device)
    adv_emb = dinov2_embed(dinov2, adv_images, device)

    cosine_attack_all = F.cosine_similarity(clean_emb, adv_emb, dim=1)
    cosine_attack_success = cosine_attack_all[success]

    # Clean-vs-clean baseline: pair each image with a different clean image
    # (fixed offset of half the batch, so no image is paired with itself).
    offset = NUM_IMAGES // 2
    baseline_idx = (torch.arange(NUM_IMAGES, device=device) + offset) % NUM_IMAGES
    cosine_baseline = F.cosine_similarity(clean_emb, clean_emb[baseline_idx], dim=1)

    num_success = int(success.sum().item())
    success_rate = num_success / NUM_IMAGES
    mean_psnr_success = (
        image_psnr[success].mean().item() if num_success > 0 else float("nan")
    )

    print()
    print(f"Attack success rate: {num_success}/{NUM_IMAGES} ({success_rate:.1%})")
    print(f"Mean PSNR (successful attacks only): {mean_psnr_success:.2f} dB")
    print()
    print("Cosine similarity, clean vs adversarial (successful attacks only):")
    print(f"  {stats(cosine_attack_success)}")
    print("Cosine similarity, clean vs a DIFFERENT clean image (background baseline):")
    print(f"  {stats(cosine_baseline)}")
    print()

    print(f"{'idx':>3} | {'success':>7} | {'cosine_sim(clean, adv)':>23}")
    print("-" * 40)
    for i in range(NUM_IMAGES):
        cos_str = f"{cosine_attack_all[i].item():.4f}" if success[i] else "-"
        print(f"{i:>3} | {str(bool(success[i])):>7} | {cos_str:>23}")


if __name__ == "__main__":
    main()
