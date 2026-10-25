"""Joint048 metadata checks; delegated full STATE checks have their own suite."""

import copy
import importlib

import pytest


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    m = importlib.import_module("check_wam_clean_training_state_terminal")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "check_cpu_environment", lambda env: None)
    registration = dict(path="registration", sha256="registration-hash")
    a = dict(w2_owner_ref="w2-owner", w2_params_ref="w2-params", source_sha256={}, snapshot_sha256={})
    data = {}

    def ref(path):
        return str(path)

    monkeypatch.setattr(m, "reference", ref)
    monkeypatch.setattr(m, "admit", lambda r, role: a)
    worker_ref = str(tmp_path / "worker/result.json")
    monkeypatch.setattr(m, "check_worker", lambda a, r: worker_ref)
    birth = dict(pid=123, start_ticks=456, boot_id="boot")
    data["launch"] = dict(child=birth, command=m.worker_argv(registration))
    data["exit"] = dict(actual_wait=True, returncode=0, automatic_retry=False)
    data["release"] = dict(
        **birth,
        actual_wait=True,
        reaped=True,
        returncode=0,
        signal_sent=False,
        kill_sent=False,
        original_error=None,
        only_owned_child_signalled=True,
    )
    owned_ref = str(tmp_path / "supervisor/STATE/result.json")
    data[owned_ref] = dict(
        complete=True,
        passed=True,
        timed_out=False,
        automatic_retry=False,
        launch_ref="launch",
        exit_ref="exit",
        release_ref="release",
    )
    data[worker_ref] = dict(
        identity=copy.deepcopy(birth),
        delegate_result_ref="state",
        verification_ref="audit",
        state_gate_passed=False,
    )
    owner_ref = str(tmp_path / "supervisor/result.json")
    data[owner_ref] = dict(
        schema_version=m.PREFIX + "owner.v1",
        complete=True,
        passed=True,
        registration_ref=registration,
        stage="STATE",
        **a,
        automatic_retry=False,
        development_authorized=False,
        fresh_performance_authorized=False,
        numerical_contract_passed=True,
        eligible_for_development=True,
        owned_wrapper_ref=owned_ref,
        worker_ref=worker_ref,
        state_owner_ref="state",
        verification_ref="audit",
        state_gate_passed=False,
    )

    class Evidence:
        def json(self, r):
            return copy.deepcopy(data[r])

        def mapping(self, values):
            assert values == {}

        def finish(self):
            pass

    monkeypatch.setattr(m, "Evidence", Evidence)
    return m, registration, data, owner_ref, worker_ref


def test_diagnostic_false_allowed_only_with_complete_joint_terminal(terminal):
    m, registration, data, _, _ = terminal
    before = copy.deepcopy(data)
    assert m.qualify(registration)["state_gate_passed"] is False
    assert data == before


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("owner", "passed", False),
        ("owner", "numerical_contract_passed", False),
        ("owner", "worker_ref", "another-worker"),
        ("owner", "state_gate_passed", True),
        ("exit", "returncode", 1),
        ("release", "reaped", False),
        ("release", "actual_wait", False),
        ("release", "signal_sent", True),
        ("release", "start_ticks", 999),
    ],
)
def test_partial_or_mismatched_terminal_rejected(terminal, target, key, value):
    m, registration, data, owner_ref, _ = terminal
    data[owner_ref if target == "owner" else target][key] = value
    with pytest.raises(ValueError):
        m.qualify(registration)


def test_worker_identity_must_match_actual_owned_child(terminal):
    m, registration, data, _, worker_ref = terminal
    data[worker_ref]["identity"]["boot_id"] = "other-boot"
    with pytest.raises(ValueError):
        m.qualify(registration)
