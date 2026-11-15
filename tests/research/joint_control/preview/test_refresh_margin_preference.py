"""An experimental preference cannot affect the recorded prefix."""

from types import SimpleNamespace

import pytest

from s2a.research.joint_control.preview import refresh_margin_preference as candidate


def test_preference_activates_only_on_idle_completed_step_16(monkeypatch):
    choice = candidate.RefreshMarginPreference()
    adapter = choice.adapter_factory({})
    calls = []
    monkeypatch.setattr(
        candidate.margin_guarded_fallback,
        "solve_margin_guarded_fallback",
        lambda *args, **kwargs: calls.append(kwargs["prefer_refresh"]),
    )
    choice.solve()
    assert calls == [False]
    owner = object()
    inner = SimpleNamespace(
        _owner=owner,
        snapshot=lambda: {"completed_periods": 8, "last_completed_step": 15, "future_guarantee": False},
    )
    refiner = SimpleNamespace(history=inner, solver=adapter.solve)
    tracking = SimpleNamespace(
        _refiner=refiner, phase="ready", history=SimpleNamespace(completed_periods=16, pending=None, pending_plan=None)
    )
    arm = SimpleNamespace(pending=None, coupled=SimpleNamespace(pending=None, completed_periods=16), tracking=tracking)
    policy = SimpleNamespace(
        arm_policy=arm, inner_history=inner, start_step=8, max_steps=8, pending=None, last_inner_loss=False
    )
    receipt = {"control_steps_completed": 15}
    with pytest.raises(ValueError, match="step-16"):
        choice.activate(policy, owner, receipt)
    assert choice.enabled is False
    receipt["control_steps_completed"] = 16
    choice.activate(policy, owner, receipt)
    choice.solve()
    assert calls == [False, True]
    assert receipt["refresh_preference_activation"]["future_observations"] == 0
    with pytest.raises(ValueError, match="step-16"):
        choice.activate(policy, owner, receipt)
    with pytest.raises(ValueError, match="isolated adapter"):
        choice.adapter_factory({})
