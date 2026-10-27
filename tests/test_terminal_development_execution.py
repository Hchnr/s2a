import pytest

import s2a.terminal_development_execution as execution


def evidence(tmp_path, monkeypatch):
    def ref(name):
        return dict(path=str(tmp_path / name), sha256="a" * 64)

    pref, rref, mref = ref("prepared.json"), ref("registration.json"), ref("metadata.json")
    sources = {str(execution.REPO / "scripts" / name): "a" * 64 for name in execution.ROLES.values()}
    verifier = dict(path=str(execution.REPO / "scripts/verify_wam_terminal_development_metadata.py"), sha256="a" * 64)
    sources[verifier["path"]] = verifier["sha256"]
    weights = {}
    for i in range(163):
        p = tmp_path / f"weight{i}"
        p.write_bytes(b"fixture")
        stat = p.stat()
        weights[str(p)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    params = ref("params.json")
    admission = dict(
        candidate_params_ref=params,
        development_preconditions_passed=True,
        execution_authorized=False,
        fresh_authorized=False,
    )
    prepared = dict(
        schema_version="s2a.terminal-development-preparation.v1",
        root=str(execution.ROOT),
        budget=execution.BUDGET,
        source_sha256=sources,
        snapshot_sha256={str(tmp_path / "snapshot"): "a" * 64},
        input_sha256={str(tmp_path / "input"): "a" * 64},
        weight_files=weights,
        state_owner_ref=ref("owner.json"),
        candidate_params_ref=params,
    )
    registration = dict(
        schema_version="s2a.terminal-development-registration.v1",
        passed=True,
        complete=True,
        execution_authorized=True,
        automatic_retry=False,
        fresh_authorized=False,
        prepared_ref=pref,
        budget=execution.BUDGET,
        development=dict(port=execution.PORT, master_port=execution.MASTER_PORT, case_order=execution.case_order()),
        metadata_verification_ref=mref,
        roles={k: dict(path=str(execution.REPO / "scripts" / v), sha256="a" * 64) for k, v in execution.ROLES.items()},
        state_admission=admission,
    )
    metadata = dict(
        schema_version="s2a.terminal-development-metadata.v1",
        passed=True,
        complete=True,
        prepared_ref=pref,
        cases_verified=12,
        execution_authorized=False,
        verifier_ref=verifier,
    )
    docs = {pref["path"]: prepared, rref["path"]: registration, mref["path"]: metadata}
    monkeypatch.setattr(execution, "read", lambda r: docs[r["path"]])
    monkeypatch.setattr(execution, "checked", lambda r: None)
    monkeypatch.setattr(execution, "admit_state_gate", lambda *a: (None, admission))
    return pref, rref, prepared, registration, metadata


def test_explicit_execution_requires_all_roles_closure_and_weights(tmp_path, monkeypatch):
    pref, rref, p, r, m = evidence(tmp_path, monkeypatch)
    actual, _, sources = execution.admit_execution(
        pref, rref, required_sources=[execution.REPO / "scripts" / execution.ROLES["service"]]
    )
    assert actual == p and len(sources) == 5


@pytest.mark.parametrize(
    "fault",
    [
        "metadata-only",
        "partial-cases",
        "metadata-source",
        "missing-role",
        "role-source",
        "snapshot",
        "weight-roster",
        "weight-changed",
        "candidate",
        "fresh",
    ],
)
def test_incomplete_registration_cannot_initialize_service(tmp_path, monkeypatch, fault):
    pref, rref, p, r, m = evidence(tmp_path, monkeypatch)
    if fault == "metadata-only":
        r["execution_authorized"] = False
    elif fault == "partial-cases":
        m["cases_verified"] = 11
    elif fault == "metadata-source":
        m["verifier_ref"]["sha256"] = "b" * 64
    elif fault == "missing-role":
        r["roles"].pop("client")
    elif fault == "role-source":
        r["roles"]["service"]["sha256"] = "b" * 64
    elif fault == "snapshot":
        p["snapshot_sha256"] = {}
    elif fault == "weight-roster":
        p["weight_files"].pop(next(iter(p["weight_files"])))
    elif fault == "weight-changed":
        (tmp_path / "weight0").write_bytes(b"changed weight bytes")
    elif fault == "candidate":
        p["candidate_params_ref"] = dict(path="/other", sha256="b" * 64)
    else:
        r["fresh_authorized"] = True
    with pytest.raises(ValueError):
        execution.admit_execution(
            pref, rref, required_sources=[execution.REPO / "scripts" / execution.ROLES["service"]]
        )
