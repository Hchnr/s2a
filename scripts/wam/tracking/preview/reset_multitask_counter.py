"""Construct the real denoise boundary for a bounded multi-request pilot."""

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.readout.run_wam_readout_features import BoundaryCounter


def make_counter(model, registration):
    """Permit five scopes of 35 actual denoises, with a global limit of 175."""
    prefix.require(
        type(registration) is dict
        and type(registration.get("max_requests")) is int
        and registration["max_requests"] == 5
        and type(registration.get("denoise_steps")) is int
        and registration["denoise_steps"] == 35,
        "The registered five-request, 35-denoise budget is required",
    )
    return BoundaryCounter(model, registration["max_requests"] * registration["denoise_steps"])
