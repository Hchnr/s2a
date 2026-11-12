"""Validate an immutable model bundle before loading a model in a fresh process."""

import hashlib
import json
from pathlib import Path

from scripts.simulation.runtime import adaptive_live_bowl_source_manifest as manifest
from scripts.simulation.runtime import adaptive_live_bowl_supervisor as historical
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview import run_causal_stream_live_wam as model_owner


def prior_adaptive_source_refs():
    """Reproduce the original adaptive owner's complete, bound source list."""
    recorded_layout = historical.source_refs()
    execution = historical.previous.source_refs()
    inherited_model = historical.previous_model.source_refs()
    prefix_length = len(execution) + len(inherited_model)
    prefix.require(
        recorded_layout[:prefix_length] == execution + inherited_model,
        "The historical adaptive source list has an unexpected structure",
    )
    return execution + manifest.causal_model_source_refs() + recorded_layout[prefix_length:]


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_registered_model_bundle(*, output, qa_path, dry_path, gpu_uuid, source_refs):
    """Recheck one preflight and bundle without binding or importing a model.

    The caller owns source-list construction, GPU authorization and model
    registration validation; this function verifies that those sources and the
    exact preflight, bundle, Python and checkpoint metadata still agree.
    """
    output = Path(output).resolve()
    prefix.require(
        output.is_relative_to(prefix.REPO / "outputs")
        and callable(source_refs)
        and type(gpu_uuid) is str
        and gpu_uuid.startswith("GPU-"),
        "A local output, source provider and registered GPU UUID are required",
    )
    registration = prefix.read(prefix.reference(output / "producer/registration.json"))
    bundle_ref = registration["model_bundle_ref"]
    prefix.require(
        Path(bundle_ref["path"]).resolve() == output / "producer/model/registered_inputs.json",
        "The model bundle must belong to the registered producer",
    )
    bundle = prefix.read(bundle_ref)
    preflight = prefix.read(prefix.reference(output / "preflight.json"))
    config = bundle["prepared"]["model_config"]
    weights = {path: detail["sha256"] for path, detail in bundle["prepared"]["weight_files"].items()}
    source_root = Path(config["cosmos_source_root"]).resolve()
    model_sources = {
        path: digest
        for path, digest in bundle["bridge"]["source_sha256"].items()
        if Path(path).is_relative_to(source_root)
    }
    prefix.require(
        source_root.is_relative_to(prefix.REPO / "outputs")
        and registration["phase"] == "producer"
        and registration["inputs"] == bundle["input_receipt"] == preflight["inputs"]
        and registration["source_refs"] == preflight["source_refs"] == source_refs()
        and preflight["complete"] is preflight["passed"] is True
        and preflight["qa_ref"] == prefix.reference(qa_path)
        and preflight["dry_ref"] == prefix.reference(dry_path)
        and preflight["model_bundle_sha256"] == _digest(bundle)
        and preflight["gpu_uuid"] == gpu_uuid
        and preflight["source_files_verified"] == len(model_sources) >= 100
        and preflight["weight_files_verified"] == len(weights) == 163
        and bundle["weight_hashes"] == weights
        and preflight["weight_stats"] == model_owner.file_stats(weights)
        and preflight["model_python_sha256"] == repro.sha256(config["model_python"]),
        "The model worker requires the exact preflight, registered sources and bundle",
    )
    return bundle, preflight
