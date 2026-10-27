"""Explicit variant dispatch for a separately registered054 continuation."""


def select_readout(variant, *, zero, probe):
    if variant == "raw":
        return None
    if variant == "zero":
        return zero
    if variant == "probe052":
        return probe
    raise ValueError("Unregistered synchronous-state variant: " + str(variant))
