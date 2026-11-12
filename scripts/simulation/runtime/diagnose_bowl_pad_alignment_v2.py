"""Recover saved pad geometry with owned per-query snapshots."""

import copy
import hashlib
from pathlib import Path

from scripts.simulation.runtime import diagnose_bowl_pad_alignment as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_pad_alignment_recovery_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_pad_alignment_recovery_20261112.md"
OUTPUT = ROOT / "outputs/bowl_pad_alignment_recovery_20261112"
QA = ROOT / "outputs/qa/bowl_pad_alignment_recovery_20261112/result.json"
DRY = ROOT / "outputs/bowl_pad_alignment_recovery_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.diagnose_bowl_pad_alignment_v2"
FAILURE = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_snapshot_failure_20261112/result.json"),
    "sha256": "43d178f76fb90eed3f003952b7b6ec6078bd8a4fa9fdc9bafe2d6ec5ba198a10",
}
FIXED = copy.deepcopy(previous.FIXED)
FIXED.update(
    schema="s2a.bowl-pad-alignment-recovery.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    failed_alignment_sha256="19d2537aed166cda156c5c3152e00d679593425bf66e5f1d53cd62ce98f251be",
    failed_audit_sha256=FAILURE["sha256"],
    failed_alignment_models=1,
    failed_alignment_forwards=97,
    before_recovery_static_models=4,
    before_recovery_static_forwards=301,
    snapshot_ownership="deep-copy-before-next-forward",
)
_OLD_CONFIG, _OLD_PLAN = previous.CONFIG, previous.PLAN
_OLD_LOAD, _OLD_SOURCES, _OLD_RUN = previous.load_inputs, previous.source_refs, previous.run_geometry
_PINNED = (
    (
        "scripts/simulation/runtime/diagnose_bowl_pad_alignment.py",
        "1174c415334841c7529c6a60770d72690808e67640eb677caa86f94f78f057fb",
    ),
    (
        "src/s2a/research/native_runtime/pad_alignment.py",
        "0c3c508eb8ae6ea7bad2c727f9a11033f7fe791424337a470d5caee01f51fb7e",
    ),
)


def load_inputs(*, verify_assets=True):
    inputs = _OLD_LOAD(verify_assets=verify_assets)
    failure = prefix.read(FAILURE)
    prefix.require(
        failure["complete"] is failure["passed"] is True
        and failure["source_diagnostic_qualified"] is False
        and failure["source_ref"]["sha256"] == inputs["config"]["failed_alignment_sha256"]
        and failure["source_models"] == inputs["config"]["failed_alignment_models"]
        and failure["source_forwards"] == inputs["config"]["failed_alignment_forwards"]
        and failure["cumulative_static_models"] == inputs["config"]["before_recovery_static_models"]
        and failure["cumulative_static_forwards"] == inputs["config"]["before_recovery_static_forwards"]
        and failure["audit_models"] == failure["audit_forwards"] == failure["audit_physics_steps"] == 0
        and failure["error_type"] == "AssertionError"
        and failure["first_failed_query"] == "reset"
        and failure["mismatch_summary"]["left_pad_world_m"]["mismatch_count"] > 0
        and failure["mismatch_summary"]["right_pad_world_m"]["mismatch_count"] > 0,
        "The failed snapshot audit and all previously spent work must be preserved",
    )
    for key, ref in (
        ("snapshot_failure_audit", FAILURE),
        ("snapshot_failure_source", failure["source_ref"]),
        ("snapshot_failure_worker", failure["worker_ref"]),
    ):
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    return inputs


def source_refs():
    files = (
        _OLD_CONFIG,
        _OLD_PLAN,
        Path(__file__).resolve(),
        ROOT / "tests/research/native_runtime/test_diagnose_bowl_pad_alignment_v2.py",
        ROOT / "scripts/simulation/runtime/audit_bowl_pad_alignment.py",
        ROOT / "scripts/simulation/runtime/audit_bowl_pad_alignment_snapshot_failure.py",
        ROOT / "reports/tracking/gripper/bowl_pad_alignment_snapshot_failure_20261112.md",
    )
    return _OLD_SOURCES() + [prefix.reference(path) for path in files]


def reconstruct_endpoint(model, data, state, *, site_id, body_id, site_from_tcp, forward):
    """Detach every result array before the caller can forward the shared data again."""
    return copy.deepcopy(
        previous.reconstruct_endpoint(
            model, data, state, site_id=site_id, body_id=body_id, site_from_tcp=site_from_tcp, forward=forward
        )
    )


def bind():
    prefix.require(
        all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha for path, sha in _PINNED),
        "The failed alignment sources must remain unchanged",
    )
    engine = previous.bind()
    engine.CONFIG, engine.PLAN, engine.OUTPUT, engine.QA, engine.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    engine.MODULE, engine.FIXED = MODULE, FIXED
    engine.reconstruct_endpoint = reconstruct_endpoint
    engine.load_inputs, engine.source_refs, engine.run_geometry = load_inputs, source_refs, _OLD_RUN
    return engine


def main():
    bind().main()


if __name__ == "__main__":
    main()
