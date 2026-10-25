"""Synthetic metadata boundaries; full upstream048 qualification is mocked explicitly."""

import copy
import importlib

import pytest

from s2a import clean_training_development_handoff as dev


@pytest.fixture
def proposal(monkeypatch):
    def ref(path):
        return dict(path=str(path), sha256="a" * 64)

    monkeypatch.setattr(dev, "reference", ref)
    registered = dict(
        prepared_ref=ref("prepared"),
        execution_registration_ref=ref("original"),
        source_sha256={"baseline": "a" * 64},
        snapshot_sha256={"snapshot": "a" * 64},
    )
    terminal = dict(
        state_owner_ref=ref("state-owner"),
        verification_ref=ref("state-audit"),
        eligible_for_development=True,
        numerical_contract_passed=True,
        state_gate_passed=False,
    )
    monkeypatch.setattr(dev, "qualified_state", lambda r: copy.deepcopy(terminal))
    p = dict(
        schema_version=dev.PREFIX + "proposal.v1",
        complete=True,
        passed=True,
        execution_authorized=False,
        protocol="049",
        stage_scope="DEVELOPMENT_only",
        root=str(dev.ROOT),
        gpu_uuid=dev.GPU,
        budget=copy.deepcopy(dev.BUDGET),
        automatic_retry=False,
        state_terminal=terminal,
        state_registration_ref=ref("state-registration"),
        state_owner_ref=terminal["state_owner_ref"],
        state_verification_ref=terminal["verification_ref"],
        owner_ref=ref(dev.OWNER_PATH),
        private_port=31339,
        prepared_ref=registered["prepared_ref"],
        execution_registration_ref=registered["execution_registration_ref"],
        case_order=[dict(case_index=i, expected_requests=75) for i in range(12)],
        owner_timeout_s=172980,
        source_sha256=dict(registered["source_sha256"], **{str(dev.REPO / r): "a" * 64 for r in dev.SOURCES}),
        snapshot_sha256=registered["snapshot_sha256"],
        qa_ref=ref("qa"),
    )
    p["command"] = dev.command(p)
    for key, relative in (
        ("config_ref", "configs/wam_clean_training_development_continuation_20261024.json"),
        ("policy_ref", "configs/wam_clean_training_development_continuation_policy_20261024.json"),
        ("plan_ref", "plans/v0.0.1/049_clean_training_development_continuation_20261024.md"),
    ):
        p[key] = ref(dev.REPO / relative)
    pr = ref("proposal")
    data = {
        pr["path"]: p,
        "state-registration": registered,
        "prepared": dict(resources=dict(development_timeout_s=172800)),
        "original": dict(development=dict(case_order=copy.deepcopy(p["case_order"]))),
        p["policy_ref"]["path"]: dict(execution_authorized=False, budget=copy.deepcopy(dev.BUDGET)),
        p["config_ref"]["path"]: dict(protocol="049", stage_scope="DEVELOPMENT_only"),
        "qa": dict(
            complete=True,
            passed=True,
            ruff_passed=True,
            format_passed=True,
            pytest_returncode=0,
            actual_model_calls=0,
            actual_physics_steps=0,
            pytest_log_ref=ref("qa-log"),
            tested_source_sha256={str(dev.REPO / r): "a" * 64 for r in dev.SOURCES},
        ),
    }

    class Evidence:
        def json(self, r):
            return copy.deepcopy(data[r["path"]])

        def file(self, r):
            return r["path"]

        def mapping(self, values):
            pass  # Actual SHA readers are covered separately; these are synthetic documents.

        def finish(self):
            pass

    monkeypatch.setattr(dev, "Evidence", Evidence)
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(key, "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "8")
    return p, pr, data


def test_development_full_metadata_envelope_is_read_without_mutation(proposal):
    _, ref, data = proposal
    before = copy.deepcopy(data)
    assert dev.validate(ref)["stage_scope"] == "DEVELOPMENT_only"
    assert data == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("requests", 901),
        ("cases", 11),
        ("physics_steps", 7201),
        ("fresh_cases", 1),
        ("fit_calls", 1),
        ("qr_calls", 1),
        ("model_initializations", 2),
    ],
)
def test_changed_execution_budget_rejected(proposal, key, value):
    p, ref, _ = proposal
    p["budget"][key] = value
    with pytest.raises(ValueError, match="fixed development-only"):
        dev.validate(ref)


@pytest.mark.parametrize(
    "key,value",
    [
        ("private_port", 31338),
        ("state_owner_ref", {}),
        ("owner_ref", {}),
        ("gpu_uuid", "forbidden"),
        ("stage_scope", "FRESH"),
        ("owner_timeout_s", 172981),
    ],
)
def test_wrong_scope_or_upstream_binding_rejected(proposal, key, value):
    p, ref, _ = proposal
    p[key] = value
    with pytest.raises(ValueError):
        dev.validate(ref)


@pytest.mark.parametrize("fault", ("case_order", "command", "source_baseline", "qa_coverage", "qa_actual_physics"))
def test_order_source_and_qa_contracts_rejected(proposal, fault):
    p, ref, data = proposal
    if fault == "case_order":
        p["case_order"].reverse()
    elif fault == "command":
        p["command"][-1] = "43"
    elif fault == "source_baseline":
        del p["source_sha256"]["baseline"]
    elif fault == "qa_coverage":
        del data["qa"]["tested_source_sha256"][str(dev.REPO / dev.SOURCES[0])]
    else:
        data["qa"]["actual_physics_steps"] = 1
    with pytest.raises(ValueError):
        dev.validate(ref)


@pytest.fixture
def development_result(monkeypatch):
    owner = importlib.import_module("run_wam_clean_training_development_handoff")
    monkeypatch.setattr(owner, "reference", lambda p: str(p))
    a = dict(
        prepared_ref="prepared",
        execution_registration_ref="registration",
        state_owner_ref="state-owner",
        state_verification_ref="state-audit",
        gpu_uuid=dev.GPU,
    )
    command = dict(launch="launch", exit="exit", release="release")
    data = {
        "launch": dict(child=dict(pid=1, start_ticks=2)),
        "exit": dict(actual_wait=True, returncode=0, automatic_retry=False),
        "release": dict(
            only_owned_child_signalled=True,
            original_error=None,
            pid=1,
            reaped=True,
            returncode=0,
            signal_sent=False,
            start_ticks=2,
        ),
    }
    audit_path = str(dev.PREPARED_ROOT / "review/DEVELOPMENT_verified.json")
    owner_path = str(dev.PREPARED_ROOT / "orchestration/DEVELOPMENT/result.json")
    data[owner_path] = dict(
        schema_version="s2a.clean-training-development-audit-owner.v1",
        complete=True,
        passed=True,
        automatic_retry=False,
        fresh_performance_authorized=False,
        **a,
        verification_ref=audit_path,
        development_ref="run",
        score=dict(promote=False),
        owned_commands=[command],
        service_owned_command=command,
        gpu_after_ref="gpu",
    )
    data["run"] = dict(
        schema_version="s2a.clean-training-development-owner.v1",
        complete=True,
        passed=True,
        **a,
        owned_commands=[command] * 12,
    )
    data[audit_path] = dict(
        schema_version="s2a.clean-training-development-verification.v1",
        complete=True,
        passed=True,
        prepared_ref=a["prepared_ref"],
        execution_registration_ref=a["execution_registration_ref"],
        development_ref="run",
        score=dict(promote=False),
        cases=[{}] * 12,
        actual_requests=900,
        actual_denoise_calls=31500,
        actual_head_calls=31500,
        physics_steps=7200,
        model_initializations=1,
        paired_first_current_conditions=6,
        independently_verified_compact_points=31500,
        future_label_archive_decodes=0,
        fresh_cases=0,
        consumed_before_after_exact=True,
    )
    data["gpu"] = dict(released=True, uuid=dev.GPU, memory_mib=0, compute_apps=[])

    class Evidence:
        def json(self, r):
            return copy.deepcopy(data[r])

        def finish(self):
            pass

    monkeypatch.setattr(owner, "Evidence", Evidence)
    return owner, a, data, owner_path, audit_path


def test_complete_mechanics_with_failed_promotion_remains_reportable(development_result):
    owner, a, data, path, _ = development_result
    before = copy.deepcopy(data)
    assert owner.check_result(a) == path
    assert data == before


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("audit", "actual_requests", 899),
        ("audit", "physics_steps", 7199),
        ("audit", "actual_head_calls", 31499),
        ("audit", "fresh_cases", 1),
        ("audit", "paired_first_current_conditions", 5),
        ("audit", "cases", [{}] * 11),
        ("owner", "state_owner_ref", "other"),
        ("release", "reaped", False),
        ("release", "signal_sent", True),
        ("gpu", "memory_mib", 1),
        ("gpu", "uuid", "forbidden"),
    ],
)
def test_incomplete_or_mismatched_actual_results_rejected(development_result, target, key, value):
    owner, a, data, path, audit = development_result
    data[audit if target == "audit" else path if target == "owner" else target][key] = value
    with pytest.raises(ValueError):
        owner.check_result(a)
