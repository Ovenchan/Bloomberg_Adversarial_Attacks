"""Attack factory compatible with experiments/gradcam.py.

The classifier must accept raw [0,1] images and return logits.
"""
from functools import partial
from attacks.pgd import pgd_attack
from attacks.fgsm import fgsm_attack
from attacks.patch import generate_patch_attack


def build_attack(name, model, params):
    params = dict(params)
    name = name.lower()
    if name == 'pgd':
        return partial(pgd_attack, model, **params)
    if name == 'fgsm':
        return partial(fgsm_attack, model, **params)
    if name == 'apgd':
        from attacks.apgd import make_apgd_attack
        return make_apgd_attack(model, **params)
    if name == 'cw':
        from attacks.cw import make_cw_attack
        return make_cw_attack(model, **params)
    if name == 'patch':
        return partial(generate_patch_attack, model, **params)
    raise ValueError(f'Unsupported attack: {name}')
