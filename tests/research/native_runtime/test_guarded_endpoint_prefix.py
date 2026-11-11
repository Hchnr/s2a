"""The diagnostic runtime cannot exceed its separate 40-period budget."""

from pathlib import Path

import pytest

from s2a.research.native_runtime import development_prefix_runtime_48 as extended
from s2a.research.native_runtime import guarded_endpoint_prefix as endpoint
from s2a.research.native_runtime import guarded_stream_prefix as stream


def test_diagnostic_guard_has_only_the_registered_changes():
    expected = Path(stream.__file__).read_text()
    for old, new in (
        (
            '"""Guard the registered 80-period prefix before a 48-period stream."""',
            '"""Guard the registered prefix before one 40-period diagnostic replay."""',
        ),
        ("class GuardedStreamPrefixRuntime:", "class GuardedEndpointPrefixRuntime:"),
        ('kwargs.get("continuation_budget") != 48', 'kwargs.get("continuation_budget") != 40'),
        (
            'raise ValueError("Unchanged runtime, seed 42, full-prefix recording and a 48-period budget are required")',
            'raise ValueError(\n                "Unchanged runtime, seed 42, full-prefix recording and a 40-period '
            'diagnostic budget are required"\n            )',
        ),
    ):
        expected = expected.replace(old, new)
    assert Path(endpoint.__file__).read_text() == expected


def test_diagnostic_registers_3000_steps_before_environment_initialization(monkeypatch):
    def stop_before_initialization():
        raise RuntimeError("no model initialized")

    monkeypatch.setattr(extended.bootstrap, "_cpu_environment", stop_before_initialization)
    with pytest.raises(extended.PrefixReplayFailure, match="no model initialized") as error:
        endpoint.GuardedEndpointPrefixRuntime().run(
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
            continuation_budget=40,
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )
    assert error.value.receipt["limits"]["mj_step"] == 3000
    assert error.value.receipt["counters"]["model_construct"]["attempted"] == 0


@pytest.mark.parametrize("budget", [0, 39, 41, 48, True])
def test_diagnostic_refuses_another_budget(budget):
    with pytest.raises(ValueError, match="40-period diagnostic budget"):
        endpoint.GuardedEndpointPrefixRuntime().run(
            continuation_budget=budget, continuation_limit=48, registered_seed=42, capture_final_prefix_inner=True
        )
