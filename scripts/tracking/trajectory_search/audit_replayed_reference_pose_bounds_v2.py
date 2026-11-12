"""Bind the explicitly verified software library for the replayed-pose audit."""

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_early_quality_feasibility as static
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds as original

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/replayed_reference_pose_bounds_recovery_20261112.md"
FIRST_ATTEMPT = {
    "path": str(ROOT / "outputs/replayed_reference_pose_bounds_20261112/result.json"),
    "sha256": "7c363cbf72992fe2b34bc2a81f1b06779402e36362f77c5261feeda94d2c8e97",
}
FIRST_WORKER = {
    "path": str(ROOT / "outputs/replayed_reference_pose_bounds_20261112/worker/result.json"),
    "sha256": "8aae8ef05f4634d0b5c02f8783f9be4edcc9a5760883aae1531b4379ffddadaf",
}
MODULE = "scripts.tracking.trajectory_search.audit_replayed_reference_pose_bounds_v2"
_LOAD_INPUTS = original.load_inputs


def bind_software_gl(inputs):
    """Add the already pinned library to the current run's consumed sources."""
    library = static.OSMESA
    prefix.require(inputs["case"]["config"]["osmesa_library"] == library["path"], "Configured OSMesa path changed")
    prefix.checked(library, inputs["source"]["source_sha256"])
    inputs["refs"]["software_gl"] = dict(library)


def load_inputs():
    inputs = _LOAD_INPUTS()
    failed, worker = prefix.read(FIRST_ATTEMPT), prefix.read(FIRST_WORKER)
    stats = worker["stats"]
    prefix.require(
        failed["complete"] is failed["passed"] is worker["complete"] is worker["passed"] is False
        and worker["error"]["message"] == "Bound software GL identity"
        and all(value == 0 for key, value in stats.items() if key != "max_qpos_restore_difference")
        and failed["physics_steps"] == failed["world_model_calls"] == failed["fresh_cases"] == 0,
        "The first failed, zero-model diagnostic must remain accounted for",
    )
    bind_software_gl(inputs)
    inputs["refs"].update(recovery_plan=prefix.reference(PLAN), first_attempt=FIRST_ATTEMPT, first_worker=FIRST_WORKER)
    return inputs


def main():
    original.QA = ROOT / "outputs/qa/replayed_reference_pose_bounds_v3_20261112/result.json"
    original.DRY = ROOT / "outputs/replayed_reference_pose_bounds_dry_v2_20261112/result.json"
    original.OUTPUT = ROOT / "outputs/replayed_reference_pose_bounds_v2_20261112"
    original.MODULE = MODULE
    original.load_inputs = load_inputs
    original.main()


if __name__ == "__main__":
    main()
