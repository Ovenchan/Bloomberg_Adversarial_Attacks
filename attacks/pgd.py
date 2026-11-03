"""
Projected Gradient Descent (PGD) adversarial attack.

Generates L-infinity bounded adversarial images by repeatedly updating
the input in the direction that increases classification loss, then
projecting the result back into the allowed epsilon neighborhood.

The attack operates on image tensors in raw [0, 1] pixel space.
"""

import torch
import torch.nn.functional as F


def pgd_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=10,
    random_start=True,
):
    original_images = images.detach().clone()

    if random_start:
        noise = torch.empty_like(
            original_images
        ).uniform_(
            -epsilon,
            epsilon,
        )

        adversarial_images = (
            original_images + noise
        ).clamp(0.0, 1.0)

    else:
        adversarial_images = (
            original_images.clone()
        )

    for _ in range(steps):
        adversarial_images.requires_grad_(True)

        logits = model(adversarial_images)

        loss = F.cross_entropy(
            logits,
            labels,
        )

        gradient = torch.autograd.grad(
            loss,
            adversarial_images,
            retain_graph=False,
            create_graph=False,
        )[0]

        adversarial_images = (
            adversarial_images.detach()
            + alpha * gradient.sign()
        )

        perturbation = torch.clamp(
            adversarial_images
            - original_images,
            min=-epsilon,
            max=epsilon,
        )

        adversarial_images = torch.clamp(
            original_images + perturbation,
            min=0.0,
            max=1.0,
        ).detach()

    return adversarial_images