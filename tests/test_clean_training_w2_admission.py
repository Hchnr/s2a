"""047 accepts only explicit050 D1 qualification and explicit receipt formats."""

import copy

import pytest

from s2a import clean_training_d1_lifecycle_addendum as addendum
from s2a import clean_training_w2_admission as gate


class Evidence:
    def __init__(self, data):
        self.data = data

    def json(self, key):
        return copy.deepcopy(self.data[key])


def test_d1_handoff_delegates_to_explicit_addendum_without_flag_conversion(monkeypatch):
    calls = []
    marker = dict(complete=True, original_supervisor_false_preserved=True)
    monkeypatch.setattr(addendum, "read_qualified", lambda spec, e: (calls.append((spec, e)) or marker))
    spec = dict(d1_lifecycle_addendum_ref=dict(path="050", sha256="a" * 64))
    e = object()
    assert gate.qualified_d1(spec, e) is marker and calls == [(spec, e)]


def test_d1_without_addendum_is_rejected():
    with pytest.raises(ValueError, match="addendum is required"):
        gate.qualified_d1({})


def test_explicit_legacy_releases_do_not_require_invented_wait_field():
    values = dict(
        launch=dict(child=dict(pid=1, start_ticks=2)),
        exited=dict(actual_wait=True, returncode=0, automatic_retry=False),
        released=dict(
            pid=1,
            start_ticks=2,
            reaped=True,
            returncode=0,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
    )
    e = Evidence(values)
    assert gate.normal_receipts(e, "launch", "exited", "released", legacy=True) == values["launch"]
    with pytest.raises(ValueError):
        gate.normal_receipts(e, "launch", "exited", "released")
    assert "actual_wait" not in values["released"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("budget", dict(gate.BUDGET, new_producer_attempts=2)),
        ("stage_scope", "W1_W2"),
        ("execution_authorized", True),
    ],
)
def test_changed_w2_budget_or_scope_rejected_before_data(monkeypatch, key, value):
    p = dict(
        schema_version=gate.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        protocol="047",
        stage_scope="W2_only",
        root=str(gate.ROOT),
        budget=gate.BUDGET,
        automatic_retry=False,
    )
    p[key] = value
    monkeypatch.setattr(gate.Evidence, "json", lambda e, r: p)
    monkeypatch.setattr(gate, "check_cpu_environment", lambda env: None)
    monkeypatch.setattr(gate, "qualified_d1", lambda *args: pytest.fail("invalid scope reached D1"))
    with pytest.raises(ValueError):
        gate.validate({})


def test_runtime_rejects_other_versions_before_binary_matching():
    with pytest.raises(ValueError, match="actual worker runtime"):
        gate.check_runtime(
            dict(runtime=dict(python="0.0", numpy="0.0", scipy="0.0", executable="other"), runtime_file_refs=[])
        )


def test_runtime_rejects_wrong_loaded_binary_fingerprints():
    import platform
    import sys

    import numpy as np
    import scipy

    actual = dict(
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, executable=sys.executable
    )
    with pytest.raises(ValueError, match="loaded numerical binary"):
        gate.check_runtime(dict(runtime=actual, runtime_file_refs=[dict(path="other", sha256="a" * 64)] * 3))
