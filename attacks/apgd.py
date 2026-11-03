"""Auto-PGD via torchattacks; raw [0, 1] images."""
import torchattacks


def make_apgd_attack(model, **params):
    return torchattacks.APGD(model=model, **params)
