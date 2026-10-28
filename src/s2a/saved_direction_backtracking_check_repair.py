"""Bind one metadata-only checker repair to the preserved failed execution."""

from s2a.saved_direction_backtracking_evidence import (
    NATIVE_PYTHON,
    REPO,
    SOURCES,
    finish,
    read,
    reference,
    remember,
    require,
)

REPAIR_ROOT = REPO / "outputs/tracking_v5_saved_direction_backtracking_check_v2_20261028"
EXECUTION_ROOT = REPO / "outputs/tracking_v5_saved_direction_backtracking_check_v2_execution_20261028"
PLAN = REPO / "plans/v0.0.1/saved_direction_backtracking_check_repair_20261028.md"
NEW_TEST = "tests/test_saved_direction_backtracking_check_v2.py"
REPAIR_SOURCES = (
    "src/s2a/saved_direction_backtracking_check_repair.py",
    "scripts/check_tracking_v5_saved_direction_backtracking_v2.py",
    "scripts/execute_tracking_v5_saved_direction_backtracking_check_v2.py",
    "plans/v0.0.1/saved_direction_backtracking_check_repair_20261028.md",
    NEW_TEST,
)
QA_SOURCES = (*SOURCES, *REPAIR_SOURCES)
FAILED_RESULT_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_direction_backtracking_check_20261028/verified.json"),
    sha256="78c5df22d2dc767f145ea144138c0296b916c6ce66e4c21189fab78eaf9cacd3",
)
FAILED_OWNER_REF = dict(
    path=str(REPO / "outputs/tracking_v5_saved_direction_backtracking_check_execution_20261028/result.json"),
    sha256="7e2c78e5da7fbfa985d5d8e53b7d6b4d413779b31afb6c3a7925dd1b556cbd42",
)


def repair_prerequisites():
    """Read only: prove failure termination and charge its executed numerical prefix."""
    refs = {}
    failed = read(remember(FAILED_RESULT_REF, refs))
    owner = read(remember(FAILED_OWNER_REF, refs))
    registration = read(remember(owner["registration_ref"], refs))
    check_reg = read(remember(failed["registration_ref"], refs))
    require(
        failed["complete"] is failed["passed"] is owner["complete"] is owner["passed"] is False
        and registration["stage"] == "check"
        and "ufunc 'isfinite'" in failed["error"]
        and "failed with1" in owner["error"]
        and list(failed["cases"]) == ["prefix"]
        and failed["cases"]["prefix"]["numerical_inner_steps"] == 300
        and failed["cases"]["prefix"]["mj_forward_queries"] == 313
        and failed["cases"]["prefix"]["entries"] == []
        and all(failed[key] == 0 for key in ("response_evaluations", "numerical_inner_steps", "query_count"))
        and failed["boundary_classification_disagreements"] == []
        and check_reg["producer_ref"]["sha256"] == "0957f81930cd65d08761220c76000374e1b7c9826872145319e311a42e26028e",
        "Only the preserved first-endpoint metadata failure, including its300 inner steps and313 queries",
    )
    capture = REPO / "outputs/tracking_v5_saved_direction_backtracking_check_execution_20261028/capture"
    lifecycle = {}
    for relative, digest in {
        "launch.json": "73760b838648a4eea2782fdfe3acdc9d4f93fdcf397ea064ad5348cef09d4b34",
        "exit.json": "06bc82811a76aeacd289ab37671b1de45bf96c9669da6f43c4d9525bb1997b6b",
        "ownership/release.json": "e2104d295fe397fa7e7f0e6eff79b29ae95029705c2e2d9fc23956c1aecd1e34",
    }.items():
        lifecycle[relative] = read(remember(dict(path=str(capture / relative), sha256=digest), refs))
    launch, exited, release = (lifecycle[key] for key in ("launch.json", "exit.json", "ownership/release.json"))
    remember(reference(capture / "command.log"), refs)
    require(
        launch["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_saved_direction_backtracking.py")]
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is exited["automatic_retry"] is False
        and exited["actual_wait"] is True
        and exited["returncode"] == 1
        and release["reaped"] is True
        and release["returncode"] == 1
        and release["signal_sent"] is False
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"]
        and release["original_error"] == owner["error"],
        "Failed independent child actually waited/reaped with no signal or automatic retry",
    )
    for path, digest in registration["source_sha256"].items():
        remember(dict(path=path, sha256=digest), refs)
    for relative in REPAIR_SOURCES:
        remember(reference(REPO / relative), refs)
    finish(refs)
    return refs
