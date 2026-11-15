"""Recovery must bind the failed diagnostic and its actual recorded cost."""

import copy
import json

import pytest

from scripts.simulation.runtime import recover_wine_capture_identity as recovery


def test_recovery_config_keeps_prior_incident_and_new_budget():
    config = json.loads(recovery.CONFIG.read_text())
    recovery.validate_config(config)
    altered = copy.deepcopy(config)
    altered["failed_diagnostic"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        recovery.validate_config(altered)
    with pytest.raises(ValueError):
        recovery.validate_config(dict(config, max_control_steps_per_phase=105))
    with pytest.raises(ValueError):
        recovery.validate_config(dict(config, automatic_retry=True))


def test_recovery_rejects_missing_or_unaccounted_prior_physics(monkeypatch):
    workers = {}
    phases = {}
    for phase, sha in recovery._WORKER_SHAS.items():
        path = recovery.ROOT / "outputs/wine_capture_identity_diagnostic_20261116" / phase / "worker/result.json"
        ref = dict(path=str(path), sha256=sha)
        phases[phase] = dict(status="failed", available_branch_ref=ref)
        workers[str(path)] = dict(
            complete=False,
            passed=False,
            error=dict(type="PrefixReplayFailure", message="path outside approved roots: worker"),
            runtime=dict(
                control_steps_attempted=104,
                control_steps_completed=104,
                counters=dict(
                    mj_step=dict(completed=2600), model_construct=dict(completed=3), context=dict(completed=3)
                ),
                cleanup_complete=True,
            ),
        )
    lifecycle_ref = dict(path="lifecycle", sha256="lifecycle")
    failure = dict(
        complete=True,
        owner_closed=True,
        sources_unchanged=True,
        passed=False,
        automatic_retry=False,
        phases=phases,
        lifecycle_ref=lifecycle_ref,
    )
    lifecycle = dict(workers=[dict(key=phase, actual_wait=True, reaped=True) for phase in phases])

    def read(ref):
        if ref == lifecycle_ref:
            return lifecycle
        if ref["path"] in workers:
            return workers[ref["path"]]
        return failure

    monkeypatch.setattr(recovery, "_prepare_original", lambda: dict(identity={}))
    monkeypatch.setattr(recovery.prefix, "read", read)
    data = recovery.prepare()
    assert len(data["identity"]["failed_diagnostic_worker_refs"]) == 2
    workers[phases["checker"]["available_branch_ref"]["path"]]["runtime"]["counters"]["mj_step"]["completed"] = 0
    with pytest.raises(ValueError, match="prior 104-period path failures"):
        recovery.prepare()
