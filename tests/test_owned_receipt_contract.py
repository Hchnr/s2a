"""Actual legacy/durable schema distinctions and the exact046 post-wait error."""

import copy

import pytest

from s2a.owned_receipt_contract import d1_postcheck_failure, normal_receipts


class Evidence:
    def __init__(self, data):
        self.data = data

    def json(self, ref):
        return copy.deepcopy(self.data[ref])


def data(format):
    release = dict(
        pid=123,
        start_ticks=456,
        reaped=True,
        returncode=0,
        signal_sent=False,
        original_error=None,
        only_owned_child_signalled=True,
    )
    exited = dict(returncode=0, actual_wait=True, automatic_retry=False)
    if format == "durable":
        release.update(actual_wait=True, kill_sent=False)
        exited["observed_utc"] = "2026-10-24T00:00:00+00:00"
    return dict(
        launch=dict(child=dict(pid=123, ppid=122, proc_state="R", start_ticks=456)), exited=exited, released=release
    )


@pytest.mark.parametrize("format", ["legacy", "durable"])
def test_normal_formats_preserve_actual_receipts_without_adding_flags(format):
    values = data(format)
    before = copy.deepcopy(values)
    assert normal_receipts(Evidence(values), "launch", "exited", "released", format=format) == values["launch"]
    assert values == before
    assert ("actual_wait" in values["released"]) is (format == "durable")


@pytest.mark.parametrize("actual,declared", [("legacy", "durable"), ("durable", "legacy")])
def test_schema_misclassification_rejected(actual, declared):
    with pytest.raises(ValueError):
        normal_receipts(Evidence(data(actual)), "launch", "exited", "released", format=declared)


@pytest.mark.parametrize("format", ["legacy", "durable"])
@pytest.mark.parametrize(
    "record,key,value",
    [
        ("exited", "actual_wait", False),
        ("exited", "returncode", 1),
        ("released", "reaped", False),
        ("released", "signal_sent", True),
        ("released", "start_ticks", 457),
    ],
)
def test_non_normal_or_unrelated_child_is_never_accepted(format, record, key, value):
    values = data(format)
    values[record][key] = value
    with pytest.raises(ValueError):
        normal_receipts(Evidence(values), "launch", "exited", "released", format=format)


def postcheck():
    values = data("durable")
    values["released"]["original_error"] = "KeyError('actual_wait')"
    owned = dict(
        complete=False,
        passed=False,
        error=dict(type="KeyError", message="'actual_wait'"),
        timed_out=False,
        automatic_retry=False,
        launch_ref="launch",
        exit_ref="exited",
        release_ref="released",
    )
    return values, owned


def test_exact_postcheck_exception_preserves_false_but_proves_actual_exit0_reap():
    values, owned = postcheck()
    before = copy.deepcopy((values, owned))
    assert d1_postcheck_failure(Evidence(values), owned) == values["launch"]
    assert (values, owned) == before and owned["passed"] is False
    with pytest.raises(ValueError):
        normal_receipts(Evidence(values), "launch", "exited", "released", format="durable")


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("exited", "returncode", 1),
        ("released", "original_error", "other error"),
        ("released", "signal_sent", True),
        ("released", "reaped", False),
        ("released", "actual_wait", False),
    ],
)
def test_postcheck_exception_does_not_waive_other_failures(record, key, value):
    values, owned = postcheck()
    values[record][key] = value
    with pytest.raises(ValueError):
        d1_postcheck_failure(Evidence(values), owned)
