"""Run the separately registered capture-path recovery without erasing the incident."""

from pathlib import Path

from scripts.simulation.runtime import diagnose_wine_capture_identity as diagnostic
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/references/wine_capture_identity_recovery_20261116.json"
PLAN = ROOT / "plans/v0.0.1/tracking/references/wine_capture_identity_recovery_20261116.md"
OUTPUT = ROOT / "outputs/wine_capture_identity_recovery_20261116"
QA = ROOT / "outputs/qa/wine_capture_identity_recovery_final_20261116/result.json"
MODULE = "scripts.simulation.runtime.recover_wine_capture_identity"
INCIDENT = ROOT / "reports/tracking/references/wine_capture_identity_incident_20261116.md"
_validate_original_config = diagnostic.validate_config
_prepare_original = diagnostic.prepare
_original_source_refs = diagnostic.source_refs
_FAILURE = dict(
    path="outputs/wine_capture_identity_diagnostic_20261116/result.json",
    sha256="09f8cb77d3379fdfb0f94758f6dfa3e3b72c24ae8a055b80a635e021b057c915",
)
_WORKER_SHAS = dict(
    producer="6668e1e6f1581cbf52e1a37c52904a2a0cbc3f8ffa3b281ca3503cfc117f8d88",
    checker="c18801f1de6e424b898d77aecdc54a73a4480c90bb231d558496164e80c20b66",
)


def validate_config(config):
    if (
        not isinstance(config, dict)
        or config.get("schema") != "s2a.wine-capture-identity-recovery.v1"
        or config.get("output") != "outputs/wine_capture_identity_recovery_20261116"
        or type(config.get("failed_diagnostic")) is not dict
        or config["failed_diagnostic"] != _FAILURE
    ):
        raise ValueError("The separately registered recovery and preserved failed diagnostic are required")
    original = dict(config)
    original.pop("failed_diagnostic")
    original.update(
        schema="s2a.wine-capture-identity-diagnostic.v1",
        output="outputs/wine_capture_identity_diagnostic_20261116",
    )
    _validate_original_config(original)


def prepare():
    data = _prepare_original()
    ref = dict(path=str(ROOT / _FAILURE["path"]), sha256=_FAILURE["sha256"])
    failure = prefix.read(ref)
    prefix.require(
        failure["complete"] is failure["owner_closed"] is failure["sources_unchanged"] is True
        and failure["passed"] is False
        and failure["automatic_retry"] is False,
        "The original diagnostic owner and failed result must remain intact",
    )
    worker_refs = []
    for phase, sha in _WORKER_SHAS.items():
        worker_ref = failure["phases"][phase]["available_branch_ref"]
        expected_path = ROOT / "outputs/wine_capture_identity_diagnostic_20261116" / phase / "worker/result.json"
        prefix.require(
            failure["phases"][phase]["status"] == "failed" and worker_ref == dict(path=str(expected_path), sha256=sha),
            "Both prior failed worker identities must remain fixed",
        )
        worker = prefix.read(worker_ref)
        actual = worker["runtime"]
        prefix.require(
            worker["complete"] is worker["passed"] is False
            and worker["error"]["type"] == "PrefixReplayFailure"
            and worker["error"]["message"].startswith("path outside approved roots:")
            and actual["control_steps_attempted"] == actual["control_steps_completed"] == 104
            and actual["counters"]["mj_step"]["completed"] == 2600
            and actual["counters"]["model_construct"]["completed"] == actual["counters"]["context"]["completed"] == 3
            and actual["cleanup_complete"] is True,
            "The two prior 104-period path failures and their costs are required",
        )
        worker_refs.append(worker_ref)
    lifecycle = prefix.read(failure["lifecycle_ref"])
    prefix.require(
        len(lifecycle["workers"]) == 2
        and {item["key"] for item in lifecycle["workers"]} == set(_WORKER_SHAS)
        and all(item["actual_wait"] is item["reaped"] is True for item in lifecycle["workers"]),
        "The original two worker processes must actually have been reaped",
    )
    data["identity"]["failed_diagnostic_ref"] = ref
    data["identity"]["failed_diagnostic_worker_refs"] = worker_refs
    return data


def source_refs():
    return _original_source_refs() + [
        prefix.reference(path)
        for path in (
            Path(__file__).resolve(),
            INCIDENT,
            ROOT / "tests/research/native_runtime/test_wine_capture_identity_recovery.py",
        )
    ]


def configure():
    """Select a new explicit profile in this process, including worker invocations."""
    diagnostic.CONFIG = CONFIG
    diagnostic.PLAN = PLAN
    diagnostic.OUTPUT = OUTPUT
    diagnostic.DRY = OUTPUT / "dry.json"
    diagnostic.QA = QA
    diagnostic.MODULE = MODULE
    diagnostic.validate_config = validate_config
    diagnostic.prepare = prepare
    diagnostic.source_refs = source_refs


def main():
    configure()
    diagnostic.main()


if __name__ == "__main__":
    main()
