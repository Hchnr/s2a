"""Recover saved-state geometry with explicitly audited quaternion normalization."""

import copy
import hashlib
from pathlib import Path

from s2a.research.native_runtime.saved_geometry_v2 import reconstruct_endpoint
from scripts.simulation.runtime import diagnose_reset_bowl_saved_geometry as previous
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/reset_bowl_saved_geometry_recovery_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/reset_bowl_saved_geometry_recovery_20261112.md"
OUTPUT = ROOT / "outputs/reset_bowl_saved_geometry_recovery_20261112"
QA = ROOT / "outputs/qa/reset_bowl_saved_geometry_recovery_20261112/result.json"
DRY = ROOT / "outputs/reset_bowl_saved_geometry_recovery_dry_20261112/result.json"
MODULE = "scripts.simulation.runtime.diagnose_reset_bowl_saved_geometry_v2"
FAILURE = {
    "path": str(ROOT / "outputs/reset_bowl_saved_geometry_20261112/result.json"),
    "sha256": "18ea6ee0293f0dcb627eacfefb3673699be21fc9b0623b0ce09367836faaec57",
}
_OLD_SCRIPT_SHA = "9e03f92f84f2ce2ab37676a94d2ec2595b4254544fea32a24fb5fdfe49f08b04"
_OLD_GEOMETRY_SHA = "edeefa7172bc16011a682e02e4d988ac76761ad4dfde8afac4069159f75c33e4"
_OLD_CONFIG, _OLD_PLAN = previous.CONFIG, previous.PLAN
_OLD_LOAD, _OLD_SOURCES, _OLD_RUN = previous.load_inputs, previous.source_refs, previous.run_geometry
FIXED = copy.deepcopy(previous.FIXED)
FIXED.update(
    schema="s2a.reset-bowl-saved-geometry-recovery.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    previous_failure_sha256=FAILURE["sha256"],
    previous_failed_models=1,
    previous_failed_forwards=10,
    max_quaternion_component_change=1e-12,
    max_quaternion_norm_error=1e-12,
    normalized_component_tolerance=2e-15,
)


def load_inputs(*, verify_assets=True):
    inputs = _OLD_LOAD(verify_assets=verify_assets)
    failure = prefix.read(FAILURE)
    prefix.require(
        failure["complete"] is failure["passed"] is False
        and failure["automatic_retry"] is False
        and failure["error"]["type"] == "ValueError",
        "The prior failed diagnostic must remain preserved",
    )
    worker = prefix.read(failure["available_worker_ref"])
    lifecycle = prefix.read(failure["workers_ref"])
    counts = worker["counters"]
    prefix.require(
        worker["complete"] is worker["passed"] is False
        and worker["error"]["message"] == "A non-stepping geometry query must not change the saved state"
        and counts["models"] == counts["data"] == 1
        and counts["forwards_attempted"] == counts["forwards_completed"] == 10
        and counts["physics_steps"] == counts["contexts"] == counts["model_requests"] == 0
        and len(lifecycle["workers"]) == 1
        and lifecycle["workers"][0]["actual_wait"] is lifecycle["workers"][0]["reaped"] is True
        and lifecycle["workers"][0]["signal_sent"] is False,
        "The previous one-model, ten-forward failure and child lifecycle must match",
    )
    for key, ref in (
        ("prior_failure", FAILURE),
        ("prior_failure_worker", failure["available_worker_ref"]),
        ("prior_failure_workers", failure["workers_ref"]),
    ):
        prefix.checked(ref, inputs["source_sha256"])
        inputs["refs"][key] = ref
    return inputs


def source_refs():
    extra = (
        _OLD_CONFIG,
        _OLD_PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/native_runtime/saved_geometry_v2.py",
        ROOT / "tests/research/native_runtime/test_saved_geometry_v2.py",
        ROOT / "tests/research/native_runtime/test_diagnose_reset_bowl_saved_geometry_v2.py",
        ROOT / "reports/tracking/references/reset_bowl_saved_geometry_failure_20261112.md",
    )
    return _OLD_SOURCES() + [prefix.reference(path) for path in extra]


def run_geometry(inputs, counters, *, mujoco):
    geometry = _OLD_RUN(inputs, counters, mujoco=mujoco)
    entries = [("reset", 0, geometry["reset"])] + [
        (branch, row["endpoint_step"], row) for branch, rows in geometry["points"].items() for row in rows
    ]
    changed = [
        (branch, step, row["state_change"]) for branch, step, row in entries if row["state_change"]["changed_indices"]
    ]
    geometry["normalization_summary"] = dict(
        queries=len(entries),
        queries_with_change=len(changed),
        max_absolute_change=max((row["state_change"]["max_absolute_change"] for _, _, row in entries), default=0),
        changed_state_indices=sorted({index for _, _, change in changed for index in change["changed_indices"]}),
        first_change=None if not changed else dict(branch=changed[0][0], endpoint_step=changed[0][1], **changed[0][2]),
    )
    return geometry


def bind_recovery():
    files = (
        (Path(previous.__file__), _OLD_SCRIPT_SHA),
        (ROOT / "src/s2a/research/native_runtime/saved_geometry.py", _OLD_GEOMETRY_SHA),
    )
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in files):
        raise ValueError("The failed diagnostic's original sources must remain unchanged")
    previous.CONFIG, previous.PLAN, previous.OUTPUT, previous.QA, previous.DRY = CONFIG, PLAN, OUTPUT, QA, DRY
    previous.MODULE, previous.FIXED = MODULE, FIXED
    previous.reconstruct_endpoint = reconstruct_endpoint
    previous.load_inputs, previous.source_refs, previous.run_geometry = load_inputs, source_refs, run_geometry
    return previous


def main():
    bind_recovery().main()


if __name__ == "__main__":
    main()
