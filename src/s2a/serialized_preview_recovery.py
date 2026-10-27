"""Bind serialization repair to the stopped run and its actual exit evidence."""

from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, read, reference, require


def recovery_evidence():
    root = REPO / "outputs/current_coupled_wam_development_20261026"
    refs = {}

    def record(name, path):
        ref = reference(path)
        refs[name] = ref
        return read(ref)

    prepared = record("failed_prepared", root / "prepared.json")
    require(
        refs["failed_prepared"]["sha256"] == "beb26bb9cb6e7822039ab56314a33bb14487bd7ae29541748e5d333cded106e5",
        "Unexpected failed preparation",
    )
    for path, digest in prepared["source_sha256"].items():
        checked(dict(path=path, sha256=digest))
    owner = record("failed_owner", root / "development/orchestration/result.json")
    require(owner["complete"] is owner["passed"] is False, "Original failure must remain a failure")
    for name, directory in (
        ("supervisor", root / "supervisor"),
        ("service", root / "development/orchestration/service_process"),
        ("worker", root / "development/orchestration/task0_init0_s42_raw"),
    ):
        launch = record(name + "_launch", directory / "launch.json")
        receipt = record(name + "_exit", directory / "exit.json")
        release = record(
            name + "_release", directory / ("release.json" if name == "supervisor" else "ownership/release.json")
        )
        require(receipt["actual_wait"] is True and receipt["returncode"] == 1, "Actual failed exit required")
        require(release["reaped"] is True and release["returncode"] == 1, "Failed child must be reaped")
        child = launch["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            require(int(fields[19]) != child["start_ticks"] or fields[0] in ("Z", "X"), "Original process still live")
    gpu = record("released_gpu", root / "development/orchestration/gpu_after.json")
    require(gpu["released"] is True and gpu["memory_mib"] == 0 and not gpu["compute_apps"], "Original GPU not released")
    audit = record("serialization_audit", REPO / "outputs/preview_result_publication_20261026/verified.json")
    require(
        audit["complete"] is audit["passed"] is audit["pose_values_equal"] is True, "Actual-value repair proof required"
    )
    for ref in [audit["failed_worker_ref"], audit["arrays_ref"], *audit["source_refs"]]:
        checked(ref)
        refs[ref["path"]] = ref
    refs["derivation"] = reference(REPO / "tests/fixtures/serialized_preview_derivation.json")
    return dict(
        refs=refs,
        original_execution_failed=True,
        recovery_mode="explicit new complete12 run; no reuse of failed-worker qualification",
        prior_failed_run_budget=dict(model_initializations=1, requests=75, denoise_calls=2625, physics_steps=600),
        automatic_retry=False,
        fresh_authorized=False,
    )
