"""The wine comparison activates and audits only its fixed candidate boundary."""

import copy
import json

import pytest

from s2a.research.joint_control.preview import cached_nominal_fallback
from scripts.simulation.runtime import run_wine_cached_nominal_candidate as runner


def test_config_keeps_one_boundary_and_original_budgets():
    config = json.loads(runner.CONFIG.read_text())
    runner.validate_config(config)
    for key, value in (("activation_step", 142), ("max_extra_periods", 2), ("automatic_retry", True)):
        with pytest.raises(ValueError):
            runner.validate_config(dict(config, **{key: value}))
    altered = copy.deepcopy(config)
    altered["baseline_checker"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        runner.validate_config(altered)


def test_producer_injection_activates_once_at_registered_boundary(monkeypatch):
    events = []

    class Adapter:
        def __init__(self, receipt):
            events.append(("factory", receipt))

        def activate(self, policy, owner, receipt, *, step):
            events.append(("activate", policy, owner, receipt, step))

    def probe(*_args, **kwargs):
        factory, activate = kwargs["adapter_factory"], kwargs["activate_probe"]
        factory("segment")
        with pytest.raises(ValueError, match="registered completed boundary"):
            activate("policy", "owner", "receipt", 142)
        activate("policy", "owner", "receipt", 143)
        with pytest.raises(ValueError, match="one fresh candidate"):
            factory("second")
        return "done"

    monkeypatch.setattr(cached_nominal_fallback, "CachedNominalFallbackAudit", Adapter)
    monkeypatch.setattr(runner, "_run_original_probe", probe)
    assert runner.run_probe(phase="producer") == "done"
    assert events == [("factory", "segment"), ("activate", "policy", "owner", "receipt", 143)]


def test_checker_injection_never_supplies_candidate_callbacks(monkeypatch):
    events = []
    monkeypatch.setattr(runner, "checker_isolation", lambda: events.append("isolation"))
    monkeypatch.setattr(runner, "_run_original_probe", lambda **kwargs: kwargs)
    kwargs = runner.run_probe(phase="checker")
    assert kwargs == {"phase": "checker"} and events == ["isolation"]


def test_extra_action_requires_cached_scope_and_independent_two_point_audit(monkeypatch):
    trace = dict(enabled=True, extra_model_responses=0, selected=True)
    verification = dict(
        passed=True,
        prefix_pose_caps_satisfied=True,
        selection_scope="cached_nominal_two_step",
        model_margin_enforced=False,
    )
    branch = dict(
        runtime=dict(
            cached_nominal_activation=dict(step=143, completed_periods=143, future_observations=0),
            segment_receipts=[dict(cached_nominal_traces=[trace])],
            continuation=dict(probe=dict(periods=[dict(decision=dict(quality=dict(verification=verification)))])),
        )
    )
    baseline = dict(periods=[{}] * 143)
    monkeypatch.setattr(runner, "_validate_original_branch", lambda *_args: {"probe_steps": 1})
    monkeypatch.setattr(
        runner.prefix,
        "read",
        lambda ref: baseline if ref == "baseline-branch" else {"case": {"branch_ref": "baseline-branch"}},
    )
    monkeypatch.setattr(
        runner.original, "periods", lambda value: value["periods"] if "periods" in value else [{}] * 144
    )
    monkeypatch.setattr(runner.comparison, "compare_periods", lambda left, right: dict(equal=len(left) == len(right)))
    counts = runner.validate_branch(branch, "producer", runner.CASE)
    assert counts["baseline_prefix_max_difference"]["equal"] is True
    assert counts["cached_nominal_trace"]["selected"] is True
    verification["selection_scope"] = "guarded_two_step"
    with pytest.raises(ValueError, match="independently audited"):
        runner.validate_branch(branch, "producer", runner.CASE)
