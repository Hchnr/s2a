"""The registered physical worker admits only its own immutable case."""

import copy

import pytest

from scripts.simulation.runtime import run_broader_recovery_pilot as run


@pytest.fixture
def registration():
    return dict(
        phase="producer",
        eligible_cases=list(range(24)),
        source_refs=[dict(path="fixed", sha256="digest")],
        inputs=dict(cases=[dict(index=index) for index in range(24)]),
        producer_branch_refs={},
        automatic_retry=False,
    )


def test_worker_admission_preserves_exact_case_and_source(registration):
    assert run.validate_registration("producer", 7, registration, registration["source_refs"]) == dict(index=7)
    for mutate in (
        lambda data: data["eligible_cases"].remove(7),
        lambda data: data["inputs"]["cases"][7].update(index=8),
        lambda data: data.update(automatic_retry=True),
        lambda data: data["source_refs"][0].update(sha256="changed"),
        lambda data: data.update(producer_branch_refs={"7": {}}),
    ):
        altered = copy.deepcopy(registration)
        mutate(altered)
        with pytest.raises(ValueError):
            run.validate_registration("producer", 7, altered, registration["source_refs"])
    for index in (-1, 24, True):
        with pytest.raises(ValueError, match="One registered case"):
            run.validate_registration("producer", index, registration, registration["source_refs"])


def test_probe_is_reset_only_case_and_checker_requires_producer(registration):
    probe = copy.deepcopy(registration)
    probe.update(phase="probe", eligible_cases=[0])
    assert run.validate_registration("probe", 0, probe, registration["source_refs"]) == dict(index=0)
    with pytest.raises(ValueError, match="One registered case"):
        run.validate_registration("probe", 1, probe, registration["source_refs"])
    checker = copy.deepcopy(registration)
    checker.update(phase="checker", producer_branch_refs={"7": {"sha256": "producer"}})
    assert run.validate_registration("checker", 7, checker, registration["source_refs"]) == dict(index=7)
    with pytest.raises(ValueError, match="replayed producer"):
        run.validate_registration("checker", 8, checker, registration["source_refs"])
