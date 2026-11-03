"""Untargeted Carlini-Wagner L2 via torchattacks."""
import torchattacks


def make_cw_attack(model, **params):
    return torchattacks.CW(model=model, **params)
