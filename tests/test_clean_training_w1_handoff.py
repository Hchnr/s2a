"""W1 metadata rejection, lookup isolation and exception-safe restoration."""

import copy
from types import SimpleNamespace

import pytest

import s2a.clean_training_w1_handoff as handoff


def ref(name):
    return dict(path=name, sha256="a" * 64)


@pytest.fixture
def valid(monkeypatch):
    spec = dict(w1_admitted_ref=handoff.ADMITTED_REF, w1_terminal_ref=handoff.TERMINAL_REF)
    spec.update(
        {
            k: ref(k)
            for k in (
                "w1_owner_ref",
                "w1_fit_ref",
                "w1_verification_ref",
                "w1_params_ref",
                "prepared_ref",
                "execution_registration_ref",
                "d0_owner_ref",
                "d0_collection_ref",
                "d0_capture_audit_ref",
            )
        }
    )

    def result(schema, **kwargs):
        return dict(schema_version=schema, complete=True, passed=True, **kwargs)

    admitted = result(
        "s2a.clean-training-runtime-fit-registration.v1",
        **{
            k: spec[k]
            for k in (
                "prepared_ref",
                "execution_registration_ref",
                "d0_owner_ref",
                "d0_collection_ref",
                "d0_capture_audit_ref",
            )
        },
        source_sha256={"source": "1"},
        snapshot_sha256={"snapshot": "1"},
    )
    terminal = dict(
        complete=True,
        passed=True,
        admitted_ref=handoff.ADMITTED_REF,
        source_before_after_exact=True,
        source_files=508,
        snapshot_files=508,
        cumulative_w1_attempts=2,
        new_successful_fits=1,
        independent_reconstructions=1,
        supervisor_ref=spec["w1_owner_ref"],
        fit_ref=spec["w1_fit_ref"],
        independent_ref=spec["w1_verification_ref"],
        params_ref=spec["w1_params_ref"],
    )
    fit = result(
        "s2a.clean-training-fit.v1",
        stage="W1",
        prepared_ref=spec["prepared_ref"],
        execution_registration_ref=spec["execution_registration_ref"],
        actual_rows=420000,
        fit_attempts=1,
        producer_fits=1,
        rounds=["D0"],
        collection_refs=[spec["d0_collection_ref"]],
        capture_audit_refs=[spec["d0_capture_audit_ref"]],
        params_ref=spec["w1_params_ref"],
    )
    verified = result(
        "s2a.clean-training-fit-verification.v1",
        stage="W1",
        prepared_ref=spec["prepared_ref"],
        execution_registration_ref=spec["execution_registration_ref"],
        actual_rows=420000,
        independent_reconstructions=1,
        fit_ref=spec["w1_fit_ref"],
        params_ref=spec["w1_params_ref"],
        full_training_row_reconstruction=True,
        rtol=1e-9,
        atol=1e-11,
    )
    commands = [dict(role=role, owner_ref=ref(role + "_owner")) for role in ("producer", "independent")]
    owner = result(
        "s2a.clean-training-runtime-fit-owner.v1",
        repair_ref=handoff.ADMITTED_REF,
        producer_ref=spec["w1_fit_ref"],
        verification_ref=spec["w1_verification_ref"],
        d1_authorized=False,
        fresh_performance_authorized=False,
        automatic_retry=False,
        owned_commands=commands,
    )
    d0 = result(
        "s2a.clean-training-owner.v1",
        stage="D0",
        producer_ref=spec["d0_collection_ref"],
        verification_ref=spec["d0_capture_audit_ref"],
    )
    data = {
        handoff.ADMITTED_REF["path"]: admitted,
        handoff.TERMINAL_REF["path"]: terminal,
        spec["w1_owner_ref"]["path"]: owner,
        spec["w1_fit_ref"]["path"]: fit,
        spec["w1_verification_ref"]["path"]: verified,
        spec["w1_params_ref"]["path"]: {"coefficients": "metadata_only"},
        spec["d0_owner_ref"]["path"]: d0,
        spec["prepared_ref"]["path"]: {"root": str(handoff.PREPARED_ROOT)},
    }
    for role in ("producer", "independent"):
        role_owner = result(
            "s2a.durable-stage-owner.v1",
            timed_out=False,
            automatic_retry=False,
            launch_ref=ref(role + "_launch"),
            exit_ref=ref(role + "_exit"),
            release_ref=ref(role + "_release"),
            stage_result_ref=ref(role + "_worker"),
        )
        worker = result(
            "s2a.clean-training-runtime-fit-worker.v1",
            repair_ref=handoff.ADMITTED_REF,
            role=role,
            delegate_result_ref=spec["w1_fit_ref"] if role == "producer" else spec["w1_verification_ref"],
            delegate_calls_attempted=1,
            producer_fits=int(role == "producer"),
            independent_reconstructions=int(role == "independent"),
            source_before_after_exact=True,
            source_proof=dict(callback_corrected_count=1, source_guard_completed=True),
            new_model_initializations=0,
            new_denoise_calls=0,
            new_physics_steps=0,
            new_fresh_cases=0,
        )
        data.update(
            {
                role + "_owner": role_owner,
                role + "_launch": dict(child=dict(pid=1, start_ticks=2)),
                role + "_exit": dict(returncode=0, actual_wait=True),
                role + "_release": dict(
                    returncode=0,
                    actual_wait=True,
                    reaped=True,
                    signal_sent=False,
                    kill_sent=False,
                    original_error=None,
                    pid=1,
                    start_ticks=2,
                ),
                role + "_worker": worker,
            }
        )
    checks = []
    monkeypatch.setattr(handoff, "read", lambda r: copy.deepcopy(data[r["path"]]))
    monkeypatch.setattr(handoff, "check_sources", lambda m: checks.append(copy.deepcopy(m)))
    return spec, data, checks


def test_valid_original_receipts_are_read_without_rewriting(valid):
    spec, data, checks = valid
    before = copy.deepcopy(data)
    assert handoff.check_handoff(spec) == data["w1_owner_ref"]
    assert data == before and len(checks) == 4


@pytest.mark.parametrize(
    "path,key,value",
    [
        ("w1_fit_ref", "passed", False),
        ("w1_verification_ref", "full_training_row_reconstruction", False),
        ("w1_verification_ref", "actual_rows", 419999),
        ("w1_verification_ref", "rtol", 1e-5),
        ("producer_exit", "returncode", 1),
        ("independent_release", "reaped", False),
        ("independent_release", "signal_sent", True),
        ("producer_release", "start_ticks", 3),
        ("producer_worker", "delegate_calls_attempted", 2),
        ("independent_worker", "new_model_initializations", 1),
        ("w1_owner_ref", "automatic_retry", True),
        ("w1_owner_ref", "d1_authorized", True),
    ],
)
def test_incomplete_or_changed_evidence_rejected(valid, path, key, value):
    spec, data, _ = valid
    data[path][key] = value
    with pytest.raises(ValueError):
        handoff.check_handoff(spec)


def test_another_parameter_and_old_failed_owner_cannot_be_substituted(valid):
    spec, _, _ = valid
    for key in ("w1_params_ref", "w1_owner_ref", "w1_admitted_ref", "w1_terminal_ref", "d0_collection_ref"):
        changed = copy.deepcopy(spec)
        changed[key] = ref("old_or_other")
        with pytest.raises(ValueError):
            handoff.check_handoff(changed)


@pytest.fixture
def scoped(monkeypatch):
    calls = []

    def original(*args):
        calls.append(args)
        return dict(original_stage=args[-1])

    owner = SimpleNamespace(prior=original)
    spec = dict(prepared_ref=ref("prepared"), execution_registration_ref=ref("registered"))
    qualified = dict(producer_ref=ref("new_fit"), verification_ref=ref("new_verified"))
    monkeypatch.setattr(handoff, "_check_owner", lambda module, spec: original)
    monkeypatch.setattr(handoff, "check_handoff", lambda spec: copy.deepcopy(qualified))
    args = ({"root": str(handoff.PREPARED_ROOT)}, spec["prepared_ref"], spec["execution_registration_ref"])
    return owner, spec, qualified, args, calls, original


def test_only_w1_redirected_and_return_is_independent_copy(scoped):
    owner, spec, qualified, args, calls, original = scoped
    with handoff.qualified_w1_prior(owner, spec) as receipt:
        value = owner.prior(*args, "W1")
        value["producer_ref"] = ref("changed_return")
        assert owner.prior(*args, "W1") == qualified
        assert owner.prior(*args, "D0") == dict(original_stage="D0")
        assert len(calls) == 1 and calls[0][-1] == "D0"
    assert owner.prior is original
    assert receipt == dict(w1_lookups=2, original_lookups=1, restored=True, source_guard_completed=True)


def test_body_failure_restores_and_propagates(scoped):
    owner, spec, _, _, _, original = scoped
    with pytest.raises(RuntimeError, match="delegate failed"):
        with handoff.qualified_w1_prior(owner, spec) as receipt:
            raise RuntimeError("delegate failed")
    assert owner.prior is original and receipt["restored"] is True
    with handoff.qualified_w1_prior(owner, spec):
        pass


def test_nested_scope_and_foreign_preparation_rejected(scoped):
    owner, spec, _, args, _, original = scoped
    with handoff.qualified_w1_prior(owner, spec):
        with pytest.raises(ValueError, match="already active"):
            with handoff.qualified_w1_prior(owner, spec):
                pass
        with pytest.raises(ValueError, match="another preparation"):
            owner.prior(args[0], ref("foreign"), args[2], "W1")
    assert owner.prior is original


def test_callback_tamper_restores_then_rejects(scoped):
    owner, spec, _, _, _, original = scoped
    with pytest.raises(ValueError, match="replaced"):
        with handoff.qualified_w1_prior(owner, spec) as receipt:
            owner.prior = lambda *args: None
    assert owner.prior is original and receipt["restored"] is True and receipt["source_guard_completed"] is False
    with handoff.qualified_w1_prior(owner, spec):
        pass


def test_post_scope_source_failure_restores_and_unlocks(scoped, monkeypatch):
    owner, spec, _, _, _, original = scoped
    with pytest.raises(ValueError, match="source drift"):
        with handoff.qualified_w1_prior(owner, spec) as receipt:
            monkeypatch.setattr(
                handoff, "check_handoff", lambda spec: (_ for _ in ()).throw(ValueError("source drift"))
            )
    assert owner.prior is original and receipt["source_guard_completed"] is False
    monkeypatch.setattr(handoff, "check_handoff", lambda spec: {})
    with handoff.qualified_w1_prior(owner, spec):
        pass
