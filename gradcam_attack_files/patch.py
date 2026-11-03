"""Per-image, untargeted optimized patch, raw [0, 1] images.

Unlike a universal patch, each image receives an independently learned patch.
Locations must be supplied by the experiment for cross-run reproducibility.
"""
import torch
import torch.nn.functional as F


def apply_patches(images, patch_values, top_positions, left_positions):
    """Place one patch in each image, preserving gradient through patch_values."""
    size = patch_values.shape[-1]
    pieces = []
    for i in range(images.shape[0]):
        top, left = int(top_positions[i]), int(left_positions[i])
        h, w = images.shape[-2:]
        if top < 0 or left < 0 or top + size > h or left + size > w:
            raise ValueError("Patch location outside image")
        mask = torch.zeros_like(images[i])
        mask[:, top:top + size, left:left + size] = 1
        canvas = F.pad(patch_values[i], (left, w - left - size, top, h - top - size))
        pieces.append(images[i] * (1 - mask) + canvas)
    return torch.stack(pieces, dim=0)


def generate_patch_attack(
    model, images, labels, top_positions, left_positions,
    patch_size=50, steps=100, learning_rate=0.05,
):
    if patch_size <= 0 or patch_size > min(images.shape[-2:]):
        raise ValueError("Invalid patch size")
    if steps < 0 or learning_rate <= 0:
        raise ValueError("Invalid optimization parameters")
    source = images.detach()
    patches = torch.rand(
        source.shape[0], source.shape[1], patch_size, patch_size,
        device=source.device, dtype=source.dtype, requires_grad=True,
    )
    optimizer = torch.optim.Adam([patches], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        adversarial = apply_patches(source, patches, top_positions, left_positions)
        loss = F.cross_entropy(model(adversarial), labels)
        (-loss).backward()
        optimizer.step()
        with torch.no_grad():
            patches.clamp_(0, 1)
    return apply_patches(source, patches.detach(), top_positions, left_positions).detach()
