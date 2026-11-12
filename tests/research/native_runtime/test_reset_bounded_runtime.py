"""Longer reset execution preserves causal reset and explicit physical bounds."""

from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime import development_prefix_runtime_bounded as bounded
from s2a.research.native_runtime.reset_bounded_runtime import ResetBoundedRuntime
from tests.research.native_runtime.test_reset_endpoint import sample


def make_runtime():
    expected, endpoint, chunk, controller = sample()
    return ResetBoundedRuntime(
        expected=expected, expected_receipt=endpoint, chunk=chunk, controller=controller, tool=np.eye(4)
    )


@pytest.mark.parametrize("budget", [1, 8, 48, 56, 600])
def test_forwards_an_explicit_budget_without_saved_actions(budget):
    runtime = make_runtime()

    def continuation(*_args, **_kwargs):
        return None

    def fake_run(*args, **kwargs):
        assert len(args[3]) == 0 and args[3][-1] is None
        assert kwargs["continuation_budget"] == kwargs["continuation_limit"] == budget
        assert kwargs["registered_seed"] == 42 and kwargs["continuation"] is continuation
        return {"behavioural_replay_verified": True}

    runtime._run = fake_run
    result = runtime.run_feedback(
        {}, {}, "", Path("."), {}, registered_seed=42, continuation=continuation, max_control_steps=budget
    )
    assert result["reset_observation_verified"] is True and result["behavioural_replay_verified"] is False


@pytest.mark.parametrize("budget", [0, -1, 601, True, 56.0, None])
def test_rejects_implicit_invalid_or_overlong_budgets(budget):
    runtime = make_runtime()
    with pytest.raises(ValueError, match="explicit 1–600 period budget"):
        runtime.run_feedback(
            {}, {}, "", Path("."), {}, registered_seed=42, continuation=lambda: None, max_control_steps=budget
        )


def test_rejects_a_missing_continuation():
    runtime = make_runtime()
    with pytest.raises(ValueError, match="continuation"):
        runtime.run_feedback({}, {}, "", Path("."), {}, registered_seed=42, continuation=None, max_control_steps=56)


@pytest.mark.parametrize(
    ("saved_count", "budget", "limit"), [(0, 601, 601), (550, 51, 600), (0, 56, 48), (0, -1, 56), (0, True, 56)]
)
def test_low_level_rejects_combined_prefix_and_continuation_over_budget(saved_count, budget, limit):
    with pytest.raises(ValueError, match="bounded continuation budget"):
        bounded.run_prefix_runtime(
            {},
            {},
            "",
            [None] * saved_count,
            {},
            [],
            Path("."),
            {},
            {},
            continuation=lambda *_args, **_kwargs: None,
            continuation_budget=budget,
            continuation_limit=limit,
        )


def test_legacy_runtime_is_unchanged_apart_from_bounded_admission():
    directory = Path(bounded.__file__).parent
    current = (directory / "development_prefix_runtime_bounded.py").read_text()
    legacy = (directory / "development_prefix_runtime_48.py").read_text()
    current = current.replace(
        '"""Compatibility copy admitting an explicitly bounded 600-period continuation."""',
        '"""Frozen compatibility copy admitting one bounded 48-period continuation."""',
    )
    current = current.replace("or not 0 < continuation_limit <= 600", "or continuation_limit not in (8, 16, 48)")
    current = current.replace("        or count + continuation_budget > 600\n", "")
    assert current == legacy
