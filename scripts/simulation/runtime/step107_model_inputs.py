"""Load the verified step-107 model inputs without binding legacy globals."""

import hashlib
import json
from pathlib import Path

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_model as historical
from scripts.wam.tracking.preview import ordered_feedback_wam_inference_20261111 as engine


def load_inputs():
    """Recheck the original capture and model manifests without loading a model."""
    prefix.environment()
    config = prefix.read(prefix.reference(historical.CONFIG))
    prefix.require(
        type(config) is dict
        and set(config) == set(historical.FIXED)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in historical.FIXED.items()),
        "The exact historical image, model, device and budgets are required",
    )
    initial, capture = historical.validate_capture(prefix.read(historical.CAPTURE))
    prefix.checked(capture["canvas_png"])
    prepared, bridge, _completed, template, _response = engine.previous_inputs()
    model_config = prepared["model_config"]
    cosmos = Path(model_config["cosmos_source_root"]).resolve()
    prefix.require(
        cosmos == Path(bridge["config"]["cosmos_source_root"]).resolve()
        and cosmos.is_relative_to(prefix.REPO / "outputs")
        and Path(model_config["model_python"]).is_file()
        and model_config["checkpoint"]["name"] == "8400_masked"
        and model_config["checkpoint"]["iteration"] == "iter_000008400",
        "The registered model Python, source and checkpoint are required",
    )
    sources = {path: digest for path, digest in bridge["source_sha256"].items() if Path(path).is_relative_to(cosmos)}
    weights = {path: detail["sha256"] for path, detail in prepared["weight_files"].items()}
    fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prefix.require(
        len(sources) >= 100 and len(weights) == 163 and fingerprint == config["model_sha256"],
        "The historical model source and weight manifests differ",
    )
    return dict(
        config=config,
        capture=capture,
        initial=initial,
        prepared=prepared,
        bridge=bridge,
        template=template,
        model_source_hashes=sources,
        weight_hashes=weights,
    )
