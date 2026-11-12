"""Audit recovery session identity and conservative physical cost ranges."""

import pytest

from scripts.simulation.runtime import adaptive_live_bowl_recovery_admission as admission
from scripts.simulation.runtime import adaptive_live_bowl_recovery_audit as audit
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from tests.research.native_runtime.test_adaptive_live_bowl_audit import rewrapped


def recorded():
    _old_config, producer, checker = rewrapped()
    config = prefix.read({"path": str(admission.CONFIG), "sha256": admission.CONFIG_SHA})
    producer["runtime"]["adaptive_stop_frame"]["frame"]["session"] = config["session"]
    return config, producer, checker


def test_saved_physics_can_be_audited_under_the_distinct_recovery_session():
    config, producer, checker = recorded()
    left = audit.summarize_branch(producer, "producer", config)
    right = audit.summarize_branch(checker, "checker", config)
    assert left["control_steps"] == right["control_steps"] == 113
    assert left["adaptive_steps"] == right["adaptive_steps"] == 6
    assert left["adaptive_task_success_periods"] == right["adaptive_task_success_periods"] == 5
    assert all(value == 0 for value in audit.compare_branches(producer, checker).values())
    assert right["candidate_budget"] is None


@pytest.mark.parametrize("change", ["old_session", "wrong_endpoint", "uncommitted", "buffer_loss", "checker_select"])
def test_recovery_audit_rejects_stale_or_unsafe_evidence(change):
    config, producer, checker = recorded()
    if change == "old_session":
        producer["runtime"]["adaptive_stop_frame"]["frame"]["session"] = admission.historical.SESSION
    elif change == "wrong_endpoint":
        producer["runtime"]["adaptive_stop_frame"]["frame"]["next_step"] += 1
    elif change == "uncommitted":
        producer["runtime"]["continuation"]["adaptive"]["segments"][0]["inflight"] = {"inner_completed": 3}
    elif change == "buffer_loss":
        audit.historical.periods(producer, "producer")[0][3]["inner_below_buffer_samples"] = 1
    else:
        checker["runtime"]["newly_selected_commands"] = 1
    with pytest.raises((ValueError, AssertionError)):
        audit.summarize_branch(
            checker if change == "checker_select" else producer,
            "checker" if change == "checker_select" else "producer",
            config,
        )


def test_cumulative_ranges_keep_the_old_unknown_cost_after_both_replays():
    config, producer, checker = recorded()
    previous = {
        "control_steps": {"lower": 1531, "upper": 1646},
        "native_steps": {"lower": 38275, "upper": 41150},
        "models": {"lower": 78, "upper": 81},
        "contexts": {"lower": 78, "upper": 81},
    }
    for phase, branch in (("producer", producer), ("checker", checker)):
        previous = audit.add_cost_bounds(
            previous, audit.exact_cost_bounds(audit.summarize_branch(branch, phase, config))
        )
    assert previous == dict(
        control_steps=dict(lower=1757, upper=1872),
        native_steps=dict(lower=43925, upper=46800),
        models=dict(lower=84, upper=87),
        contexts=dict(lower=84, upper=87),
    )


@pytest.mark.parametrize("change", ["missing", "negative", "inverted", "boolean"])
def test_invalid_physical_ranges_cannot_be_added(change):
    left = {key: {"lower": 1, "upper": 2} for key in ("control_steps", "native_steps", "models", "contexts")}
    right = {key: {"lower": 0, "upper": 3} for key in left}
    if change == "missing":
        right.pop("contexts")
    elif change == "negative":
        right["models"]["lower"] = -1
    elif change == "inverted":
        right["models"]["lower"] = 4
    else:
        right["models"]["lower"] = True
    with pytest.raises(ValueError):
        audit.add_cost_bounds(left, right)
