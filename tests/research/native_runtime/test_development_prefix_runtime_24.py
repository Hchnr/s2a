"""Check the bounded extension without changing the frozen historical runtime."""

from pathlib import Path

import pytest

from s2a.research.native_runtime import development_prefix_runtime as historical
from s2a.research.native_runtime import development_prefix_runtime_24 as extended


def test_compatibility_copy_changes_only_explicit_admission():
    original = Path(historical.__file__).read_text()
    expected = original.replace(
        '"""One bounded OSMesa saved-prefix replay with explicit lifecycle accounting."""',
        '"""Frozen compatibility copy admitting one bounded 24-period continuation."""',
    ).replace("continuation_limit not in (8, 16)", "continuation_limit not in (8, 16, 24)")
    assert Path(extended.__file__).read_text() == expected


def test_extended_budget_is_counted_before_any_environment_initialization(monkeypatch):
    def stop_before_initialization():
        raise RuntimeError("no model initialized")

    monkeypatch.setattr(extended.bootstrap, "_cpu_environment", stop_before_initialization)
    with pytest.raises(extended.PrefixReplayFailure, match="no model initialized") as error:
        extended.run_prefix_runtime(
            {},
            {},
            "",
            [None] * 360,
            {},
            [],
            "unused",
            [],
            {},
            continuation=lambda *_args, **_kwargs: None,
            continuation_budget=24,
            continuation_limit=24,
        )
    assert error.value.receipt["failure_stage"] == "admission"
    assert error.value.receipt["limits"]["mj_step"] == 9600
    assert error.value.receipt["counters"]["model_construct"]["attempted"] == 0


@pytest.mark.parametrize("limit,budget", [(24, 25), (24, 0), (25, 24), (True, 1)])
def test_extended_runtime_rejects_unregistered_or_excess_budget(limit, budget):
    with pytest.raises(ValueError, match="bounded continuation budget"):
        extended.run_prefix_runtime(
            {},
            {},
            "",
            [],
            {},
            [],
            "unused",
            [],
            {},
            continuation=lambda *_args, **_kwargs: None,
            continuation_budget=budget,
            continuation_limit=limit,
        )


def test_historical_runtime_still_rejects_24_periods():
    with pytest.raises(ValueError, match="bounded continuation budget"):
        historical.run_prefix_runtime(
            {},
            {},
            "",
            [],
            {},
            [],
            "unused",
            [],
            {},
            continuation=lambda *_args, **_kwargs: None,
            continuation_budget=24,
            continuation_limit=24,
        )
