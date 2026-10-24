"""Admission and online-only request contracts for the fixed038 workers.

Metadata preparation cannot launch a model. A separate, fully audited execution
registration must freeze the final entrypoints and their evidence first. This
module opens JSON, current PNG bytes and source files; it never decodes labels.
"""

import hashlib
import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_execution_proof import check_execution_proof
from s2a.clean_training_seed_plan import apply_seed_plan
from s2a.wam_clean_readout import CleanResidualReadout
from s2a.wam_readout import LIBERO_DOMAIN

GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
VISION_SHAPE = (1, 48, 11, 60, 40)
PREP_SCHEMA = "s2a.clean-training-preparation.v1"
REG_SCHEMA = "s2a.clean-training-execution-registration.v1"
REQUEST_KEYS = frozenset(
    (
        "action_chunk_size",
        "domain_name",
        "extra",
        "fps",
        "guidance",
        "image_size",
        "model_mode",
        "name",
        "num_frames",
        "num_steps",
        "prompt",
        "seed",
        "shift",
        "vision_path",
    )
)
ACTION_KEYS = frozenset(
    ("action_dim_mask", "action_domain", "action_fps", "camera_layout", "return_video", "selected_roles", "task_title")
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "requires exact path/SHA reference")
    path = Path(ref["path"])
    require(path.is_absolute() and path.is_file() and sha(path) == ref["sha256"], "artifact path/SHA differs")
    return path


def read(ref):
    path = checked(ref)
    require(path.suffix == ".json", "JSON admission refuses binary labels")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == ref["sha256"], "JSON changed during admission")
    return json.loads(data)


def publish(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO / "outputs"), "output must stay inside S2A outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return reference(path)


def check_map(mapping):
    require(isinstance(mapping, dict) and bool(mapping), "source/input closure is empty")
    for path, digest in mapping.items():
        checked(dict(path=path, sha256=digest))


def admit_execution(prepared_ref, registration_ref, *, required_sources):
    """Check concrete evidence and frozen sources before any model import.

    This verifies a registration; it does not manufacture execution eligibility
    from entrypoint presence or from the earlier metadata-only preparation.
    """
    prepared, registration = read(prepared_ref), read(registration_ref)
    require(prepared["schema_version"] == PREP_SCHEMA, "038 preparation schema differs")
    require(
        registration["schema_version"] == REG_SCHEMA
        and registration["prepared_ref"] == prepared_ref
        and registration.get("complete") is True
        and registration.get("passed") is True
        and registration.get("readiness", {}).get("ready_for_execution") is True
        and registration.get("automatic_retry") is False,
        "a complete independent execution registration is required",
    )
    check_execution_proof(registration, prepared_ref=prepared_ref, read_json=read, check_file=checked)
    metadata = read(registration["metadata_verification_ref"])
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("metadata_qualified") is True
        and metadata.get("execution_ready") is False
        and metadata.get("execution_authorized") is False
        and metadata.get("consumed_before_after_exact") is True
        and metadata.get("complete") is True
        and metadata.get("passed") is True
        and metadata.get("prepared_ref") == prepared_ref,
        "independent metadata verification does not bind this preparation",
    )
    verifier = metadata["verifier_ref"]
    checked(verifier)
    require(
        registration["source_sha256"].get(verifier["path"]) == verifier["sha256"],
        "independent metadata verifier is not in the final execution source closure",
    )
    require(registration.get("fresh_performance_authorized") is False, "038 cannot authorize fresh performance")
    root = Path(prepared["root"]).resolve()
    require(
        root == REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023",
        "038 output root differs",
    )
    sources = {}
    for mapping in (
        prepared["qualified_model_source_sha256"],
        prepared["source_sha256"],
        registration["source_sha256"],
    ):
        check_map(mapping)
        for path, digest in mapping.items():
            require(path not in sources or sources[path] == digest, "source closures conflict")
            sources[path] = digest
    check_map(prepared["snapshot_sha256"])
    check_map(registration["snapshot_sha256"])
    check_map(prepared["input_sha256"])
    qualified = read(prepared["prerequisite_refs"]["bridge_prepared"])
    check_map(qualified["snapshot_sha256"])
    check_map(qualified["input_sha256"])
    require(
        len(qualified["checkpoint_small_metadata_sha256"]) == 3,
        "qualified checkpoint must retain its three small metadata SHA references",
    )
    check_map(qualified["checkpoint_small_metadata_sha256"])
    for path in required_sources:
        absolute = str(Path(path).resolve())
        require(registration["source_sha256"].get(absolute) == sha(absolute), "worker source is not execution-frozen")
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "weight stat changed")
    require(prepared["config"]["gpu_uuid"] == GPU, "038 GPU UUID differs")
    seed_ref = registration["training_seed_plan_ref"]
    prepared = apply_seed_plan(prepared, prepared_ref, read(seed_ref), seed_ref)
    return prepared, registration, sources


def validate_current_request(request, *, name, seed, image_ref, task, template):
    """Exact qualified template equality plus an explicit current-only schema."""
    image = checked(image_ref)
    expected = dict(template, name=name, seed=seed, vision_path=str(image), prompt=task)
    expected["extra"] = json.loads(json.dumps(template["extra"]))
    expected["extra"]["action128"]["task_title"] = task
    require(request == expected, "request differs from its frozen task/current-image template")
    require(set(request) <= REQUEST_KEYS, "unregistered online or future request field")
    require(set(request["extra"]) == {"action128"}, "unregistered conditioning namespace")
    action = request["extra"]["action128"]
    require(set(action) == ACTION_KEYS, "future/privileged action conditioning is forbidden")
    require(
        request["model_mode"] == "wam"
        and request["action_chunk_size"] == 40
        and request["num_frames"] == 41
        and request["num_steps"] == 35
        and request["guidance"] == 1.0
        and request["shift"] == 5.0
        and request["fps"] == action["action_fps"] == 20
        and action["action_domain"] == LIBERO_DOMAIN
        and action["return_video"] is False
        and action["action_dim_mask"] == [True] * 10 + [False] * 118,
        "fixed H40/20Hz masked request contract differs",
    )


def trained_head(fit_ref, verification_ref, *, stage, prepared_ref, registration_ref):
    fit, verification = read(fit_ref), read(verification_ref)
    require(
        fit.get("schema_version") == "s2a.clean-training-fit.v1"
        and fit.get("complete") is True
        and fit.get("passed") is True
        and fit.get("stage") == stage
        and fit.get("prepared_ref") == prepared_ref
        and fit.get("execution_registration_ref") == registration_ref
        and fit.get("producer_fits") == 1,
        "wrong, incomplete, or unrelated fixed training fit",
    )
    require(
        verification.get("schema_version") == "s2a.clean-training-fit-verification.v1"
        and verification.get("complete") is True
        and verification.get("passed") is True
        and verification.get("fit_ref") == fit_ref
        and verification.get("prepared_ref") == prepared_ref
        and verification.get("execution_registration_ref") == registration_ref
        and verification.get("stage") == stage
        and verification.get("params_ref") == fit.get("params_ref"),
        "independent full-fit verification does not bind this head",
    )
    head = CleanResidualReadout.from_parameters(read(fit["params_ref"]))
    require(len(head.center) == 1290 and head.ridge == 0.1, "fixed1290/.1 head contract differs")
    return head, fit["params_ref"]
