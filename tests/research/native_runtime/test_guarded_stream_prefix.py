"""Check the bounded stream extension without initializing physics."""

from pathlib import Path

import pytest

from s2a.research.native_runtime import development_prefix_runtime as historical
from s2a.research.native_runtime import development_prefix_runtime_48 as extended
from s2a.research.native_runtime.guarded_stream_prefix import GuardedStreamPrefixRuntime


def test_extended_copy_differs_only_in_docstring_and_explicit_admission():
    expected = (
        Path(historical.__file__)
        .read_text()
        .replace(
            '"""One bounded OSMesa saved-prefix replay with explicit lifecycle accounting."""',
            '"""Frozen compatibility copy admitting one bounded 48-period continuation."""',
        )
        .replace("continuation_limit not in (8, 16)", "continuation_limit not in (8, 16, 48)")
    )
    assert Path(extended.__file__).read_text() == expected


def test_stream_budget_is_registered_before_environment_initialization(monkeypatch):
    def stop_before_initialization():
        raise RuntimeError("no model initialized")

    monkeypatch.setattr(extended.bootstrap, "_cpu_environment", stop_before_initialization)
    runtime = GuardedStreamPrefixRuntime()
    with pytest.raises(extended.PrefixReplayFailure, match="no model initialized") as error:
        runtime.run(
            {},
            {},
            "",
            [None] * 80,
            {},
            [],
            "unused",
            [],
            {},
            continuation=lambda *_args, **_kwargs: None,
            continuation_budget=48,
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )
    assert error.value.receipt["limits"]["mj_step"] == 3200
    assert error.value.receipt["counters"]["model_construct"]["attempted"] == 0
    assert error.value.receipt["failure_stage"] == "admission"


@pytest.mark.parametrize(
    "changes",
    [
        {"registered_seed": 43},
        {"capture_final_prefix_inner": False},
        {"continuation_budget": 49},
        {"continuation_limit": 16},
    ],
)
def test_stream_guard_rejects_unregistered_settings_before_initialization(changes):
    kwargs = dict(registered_seed=42, capture_final_prefix_inner=True, continuation_budget=48, continuation_limit=48)
    kwargs.update(changes)
    with pytest.raises(ValueError, match="48-period budget"):
        GuardedStreamPrefixRuntime().run(**kwargs)


def test_historical_runtime_still_rejects_extended_budget():
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
            continuation_budget=48,
            continuation_limit=48,
        )


def test_stream_wrapper_does_not_mutate_extended_runtime():
    functions = (extended.run_prefix_runtime, extended.replay_saved_prefix, extended.FinalPrefixInner)
    runtime = GuardedStreamPrefixRuntime()
    assert runtime._run.__code__ is functions[0].__code__
    assert (extended.run_prefix_runtime, extended.replay_saved_prefix, extended.FinalPrefixInner) == functions
