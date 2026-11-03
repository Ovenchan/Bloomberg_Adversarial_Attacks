"""Untargeted FGSM for raw [0, 1] images."""
import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, eps=8 / 255):
    source = images.detach()
    x = source.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), labels)
    gradient = torch.autograd.grad(loss, x)[0]
    return (source + eps * gradient.sign()).clamp(0, 1).detach()
