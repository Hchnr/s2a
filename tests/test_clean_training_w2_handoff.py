"""Read-only048 W2 handoff gates; all complete-result fixtures are synthetic."""

import copy

import pytest

from s2a import clean_training_w2_handoff as gate


def ref(path):
    return dict(path=str(path), sha256="a" * 64)


class Evidence:
    def __init__(self, data):
        self.data = data
        self.maps = 0
        self.finished = False

    def json(self, r):
        return copy.deepcopy(self.data[r["path"]])

    def mapping(self, m):
        self.maps += 1

    def finish(self):
        self.finished = True


def test_pending_w2_refused_without_any_metadata_or_array_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path / "pending")
    with pytest.raises(ValueError, match="unavailable"):
        gate.qualified_w2({})


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    root = tmp_path / "w2"
    monkeypatch.setattr(gate, "ROOT", root)
    names = dict(
        w2_registration_ref=root / "execution/admitted.json",
        w2_owner_ref=root / "supervisor/result.json",
        w2_fit_ref=root / "fit/W2/result.json",
        w2_verification_ref=root / "review/W2_verified.json",
        w2_params_ref=root / "fit/W2/parameters.json",
    )
    for path in names.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setattr(gate, "reference", lambda p: ref(p))
    spec = {k: ref(path) for k, path in names.items()}
    spec.update(prepared_ref=ref("prepared"), execution_registration_ref=ref("original_registration"))

    def complete(schema, **kwargs):
        return dict(schema_version=schema, complete=True, passed=True, **kwargs)

    verifier = ref(gate.REPO / "scripts/verify_wam_clean_training_w2_fit.py")
    proposal = complete(
        gate.PREFIX + "proposal.v1",
        execution_authorized=False,
        stage_scope="W2_only",
        budget=gate.BUDGET,
        automatic_retry=False,
        source_sha256={verifier["path"]: verifier["sha256"]},
        snapshot_sha256={"snapshot": "a" * 64},
        prepared_ref=spec["prepared_ref"],
        execution_registration_ref=spec["execution_registration_ref"],
        runtime=dict(executable="sim_python"),
        d0_collection_ref=ref("D0"),
        d1_collection_ref=ref("D1"),
        d0_capture_audit_ref=ref("D0_audit"),
        d1_capture_audit_ref=ref("D1_audit"),
        d1_owner_ref=ref("D1_owner"),
        d1_handoff_owner_ref=ref("046_owner"),
    )
    registered = copy.deepcopy(proposal)
    registered.update(
        schema_version=gate.PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=ref("proposal"),
        independent_verification_ref=ref("metadata"),
    )
    metadata = complete(
        gate.PREFIX + "verification.v1",
        proposal_ref=ref("proposal"),
        eligible_for_execution=True,
        verifier_ref=verifier,
    )
    commands = [
        dict(role=role, owner_ref=ref(root / "supervisor" / role / "result.json"))
        for role in ("producer", "independent")
    ]
    owner = complete(
        gate.PREFIX + "owner.v1",
        stage="W2",
        repair_ref=spec["w2_registration_ref"],
        prepared_ref=spec["prepared_ref"],
        execution_registration_ref=spec["execution_registration_ref"],
        producer_ref=spec["w2_fit_ref"],
        verification_ref=spec["w2_verification_ref"],
        automatic_retry=False,
        fresh_performance_authorized=False,
        d1_authorized=False,
        owned_commands=commands,
    )
    features = [*(f"hidden_{i:04d}" for i in range(1280)), *(f"c0_{i:02d}" for i in range(10))]
    common = dict(
        stage="W2",
        prepared_ref=spec["prepared_ref"],
        execution_registration_ref=spec["execution_registration_ref"],
        actual_rows=840000,
        params_ref=spec["w2_params_ref"],
        label_reconstruction_receipts=[{}] * 150,
    )
    fit = complete(
        "s2a.clean-training-fit.v1",
        **common,
        fit_attempts=1,
        producer_fits=1,
        rounds=["D0", "D1"],
        collection_refs=[registered["d0_collection_ref"], registered["d1_collection_ref"]],
        capture_audit_refs=[registered["d0_capture_audit_ref"], registered["d1_capture_audit_ref"]],
        label_anchor_count=150,
        fit_metadata=dict(
            rows=840000,
            feature_columns=features,
            ridge=0.1,
            replay_verified=True,
            replay_sha256=dict(features="f", target_clean_residual="t", weights="w"),
        ),
    )
    audit = complete(
        "s2a.clean-training-fit-verification.v1",
        **common,
        independent_reconstructions=1,
        fit_ref=spec["w2_fit_ref"],
        full_training_row_reconstruction=True,
        rtol=1e-9,
        atol=1e-11,
        qr_metadata=dict(
            rows=840000,
            feature_columns=features,
            ridge=0.1,
            replay_verified=True,
            replay_sha256=dict(features="f", targets="t", weights="w"),
        ),
    )
    params = dict(
        schema_version="s2a.clean-residual-readout.v1",
        coefficients=[[0.0] * 10 for _ in range(1291)],
        center=[0.0] * 1290,
        scale=1.0,
        ridge=0.1,
    )
    data = {
        spec["w2_registration_ref"]["path"]: registered,
        "proposal": proposal,
        "metadata": metadata,
        spec["w2_owner_ref"]["path"]: owner,
        spec["w2_fit_ref"]["path"]: fit,
        spec["w2_verification_ref"]["path"]: audit,
        spec["w2_params_ref"]["path"]: params,
        "original_registration": dict(execution_verification_ref=ref("original_check")),
        "original_check": dict(verifier_ref=ref("original_verifier")),
    }
    for role in ("producer", "independent"):
        worker_ref = ref(root / "workers" / role / "result.json")
        owned = complete(
            "s2a.durable-stage-owner.v1",
            timed_out=False,
            automatic_retry=False,
            launch_ref=ref(role + "_launch"),
            exit_ref=ref(role + "_exit"),
            release_ref=ref(role + "_release"),
            stage_result_ref=worker_ref,
        )
        data[str(root / "supervisor" / role / "result.json")] = owned
        data[role + "_launch"] = dict(
            child=dict(pid=1, start_ticks=2), command=["sim_python", "scripts/execute_wam_clean_training_w2_fit.py"]
        )
        data[role + "_exit"] = dict(actual_wait=True, returncode=0, automatic_retry=False)
        data[role + "_release"] = dict(
            only_owned_child_signalled=True,
            actual_wait=True,
            reaped=True,
            returncode=0,
            signal_sent=False,
            kill_sent=False,
            original_error=None,
            pid=1,
            start_ticks=2,
        )
        data[worker_ref["path"]] = complete(
            gate.PREFIX + "worker.v1",
            repair_ref=spec["w2_registration_ref"],
            role=role,
            delegate_result_ref=spec["w2_fit_ref"] if role == "producer" else spec["w2_verification_ref"],
            delegate_calls_attempted=1,
            producer_fits=int(role == "producer"),
            independent_reconstructions=int(role == "independent"),
            source_before_after_exact=True,
            runtime=registered["runtime"],
            d1_owner_ref=registered["d1_owner_ref"],
            d1_handoff_owner_ref=registered["d1_handoff_owner_ref"],
            source_proof=dict(
                registration_ref=spec["execution_registration_ref"],
                source_verifier_ref=ref("original_verifier"),
                callback_corrected_count=1,
                source_guard_completed=True,
            ),
            new_model_initializations=0,
            new_denoise_calls=0,
            new_physics_steps=0,
            new_fresh_cases=0,
        )
    return spec, data


def test_complete_real_schema_chain_is_read_without_mutation_or_conversion(qualified):
    spec, data = qualified
    before = copy.deepcopy(data)
    e = Evidence(data)
    owner = gate.qualified_w2(spec, e)
    assert owner == data[spec["w2_owner_ref"]["path"]] and data == before and e.finished and e.maps == 2


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("w2_owner_ref", "passed", False),
        ("w2_fit_ref", "actual_rows", 839999),
        ("w2_fit_ref", "rounds", ["D1"]),
        ("w2_fit_ref", "label_anchor_count", 149),
        ("w2_verification_ref", "rtol", 1e-5),
        ("w2_verification_ref", "full_training_row_reconstruction", False),
        ("producer_exit", "returncode", 1),
        ("independent_release", "reaped", False),
        ("producer_release", "start_ticks", 3),
        ("independent_release", "signal_sent", True),
        ("w2_params_ref", "ridge", 1.0),
        ("w2_params_ref", "scale", float("inf")),
        ("metadata", "eligible_for_execution", False),
    ],
)
def test_partial_failed_or_different_fit_is_rejected(qualified, target, key, value):
    spec, data = qualified
    name = spec[target]["path"] if target in spec else target
    data[name][key] = value
    with pytest.raises(ValueError):
        gate.qualified_w2(spec, Evidence(data))


def test_row_hash_change_and_source_proof_failure_rejected(qualified):
    spec, data = qualified
    data[spec["w2_verification_ref"]["path"]]["qr_metadata"]["replay_sha256"]["features"] = "changed"
    with pytest.raises(ValueError, match="full-row"):
        gate.qualified_w2(spec, Evidence(data))
    data[spec["w2_verification_ref"]["path"]]["qr_metadata"]["replay_sha256"]["features"] = "f"
    producer = next(v for k, v in data.items() if k.endswith("/workers/producer/result.json"))
    producer["source_proof"]["source_guard_completed"] = False
    with pytest.raises(ValueError, match="source-proof"):
        gate.qualified_w2(spec, Evidence(data))


def test_semantic_feature_order_and_parameter_shape_rejected(qualified):
    spec, data = qualified
    # Both math receipts agreeing is insufficient if their semantic columns differ.
    for key, meta in (("w2_fit_ref", "fit_metadata"), ("w2_verification_ref", "qr_metadata")):
        data[spec[key]["path"]][meta]["feature_columns"] = ["other"] * 1290
    with pytest.raises(ValueError, match="full-row"):
        gate.qualified_w2(spec, Evidence(data))
    value = data[spec["w2_params_ref"]["path"]]
    value["coefficients"].pop()
    with pytest.raises(ValueError, match="geometry"):
        gate.parameters(value)
