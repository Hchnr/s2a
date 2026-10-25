"""047 metadata gate: only full normally audited D1 can precede fixed W2 math."""

import copy
import os
import platform
import sys
from pathlib import Path

from s2a.clean_training_d1_admission import Evidence, canonical
from s2a.clean_training_w1_handoff import ADMITTED_REF, PREPARED_ROOT, check_handoff
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, reference, require

ROOT = PREPARED_ROOT / "runtime_w2_fit_20261024"
D1_ROOT = REPO / "outputs/clean_training_d1_continuation_20261024"
D1_REGISTRATION_REF = dict(
    path=str(D1_ROOT / "execution/admitted.json"),
    sha256="76d8751d0213e585b9660b704556dff4c4ae7849aa80c7c067dbdcb694749d49",
)
PREFIX = "s2a.clean-training-w2-runtime-fit-"
BUDGET = dict(
    new_producer_attempts=1,
    producer_fits=1,
    independent_reconstructions=1,
    cumulative_clean_producer_attempts=3,
    new_model_initializations=0,
    new_denoise_calls=0,
    new_physics_steps=0,
    new_fresh_cases=0,
    automatic_retry=False,
)
SOURCES = (
    "src/s2a/clean_training_w2_admission.py",
    "scripts/execute_wam_clean_training_w2_fit.py",
    "scripts/run_wam_clean_training_w2_fit.py",
    "scripts/verify_wam_clean_training_w2_fit.py",
    "tests/test_clean_training_w2_fit.py",
    "tests/test_clean_training_w2_admission.py",
    "src/s2a/owned_receipt_contract.py",
    "src/s2a/clean_training_d1_lifecycle_addendum.py",
    "scripts/verify_wam_clean_training_d1_lifecycle.py",
    "tests/test_owned_receipt_contract.py",
)


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "047 evidence is incomplete or wrong schema",
    )


def normal_receipts(e, launch_ref, exit_ref, release_ref, *, legacy=False):
    from s2a.owned_receipt_contract import normal_receipts as typed_receipts

    return typed_receipts(e, launch_ref, exit_ref, release_ref, format="legacy" if legacy else "durable")


def qualified_d1(spec, e=None):
    """Consume050's exact post-check-only addendum, never rewrite046 False."""
    from s2a.clean_training_d1_lifecycle_addendum import read_qualified

    return read_qualified(spec, e)


def validate(proposal_ref, *, initial=False):
    require(Path.cwd().resolve() == REPO, "047 must run from S2A")
    check_cpu_environment(os.environ)
    e = Evidence()
    p = e.json(proposal_ref)
    complete(p, PREFIX + "proposal.v1")
    require(
        p["execution_authorized"] is False
        and p["protocol"] == "047"
        and p["stage_scope"] == "W2_only"
        and p["root"] == str(ROOT)
        and p["budget"] == BUDGET
        and p["automatic_retry"] is False,
        "047 finite W2-only proposal required",
    )
    require(
        p["fit_output_dir"] == str(ROOT / "fit/W2")
        and p["verification_output"] == str(ROOT / "review/W2_verified.json"),
        "047 fresh W2 paths differ",
    )
    require(
        p["config_ref"]["path"] == str(REPO / "configs/wam_clean_training_w2_runtime_fit_20261024.json")
        and p["policy_ref"]["path"] == str(REPO / "configs/wam_clean_training_w2_policy_20261024.json")
        and p["plan_ref"]["path"] == str(REPO / "plans/v0.0.1/047_clean_training_w2_runtime_fit_20261024.md"),
        "047 actual config/policy/plan required",
    )
    config = e.json(p["config_ref"])
    policy = e.json(p["policy_ref"])
    e.file(p["plan_ref"])
    require(all(p.get(k) == v for k, v in config.items()), "047 proposal/config differs")
    require(
        policy["execution_authorized"] is False
        and policy["budget"] == BUDGET
        and policy["expected_rows"] == 840000
        and policy["feature_count"] == 1290
        and policy["ridge"] == 0.1
        and policy["rounds"] == ["D0", "D1"],
        "047 frozen policy changed",
    )
    d1 = qualified_d1(p, e)
    check_handoff(p)
    w1 = e.json(ADMITTED_REF)
    require(
        p["runtime"] == w1["runtime"] == policy["runtime"]
        and p["runtime_file_refs"] == w1["runtime_file_refs"] == policy["runtime_file_refs"],
        "047 qualified actual runtime differs",
    )
    for ref in p["runtime_file_refs"]:
        e.file(ref)
    require(len(d1["source_sha256"]) == 519, "046 source baseline incomplete")
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in d1[key].items()),
            "047 changed or omitted519 baseline sources/snapshots",
        )
        e.mapping(p[key])
    supplement = e.json(p["d1_lifecycle_addendum_ref"])
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in supplement[key].items()),
            "047 omitted050 frozen qualification sources",
        )
    require(set(p["source_sha256"].values()) <= set(p["snapshot_sha256"].values()), "047 snapshots incomplete")
    for relative in SOURCES:
        actual = REPO / relative
        require(
            p["source_sha256"].get(str(actual)) == reference(actual)["sha256"],
            "actual047 source not frozen: " + relative,
        )
    qa = e.json(p["qa_ref"])
    require(
        qa["complete"] is qa["passed"] is qa["ruff_passed"] is qa["format_passed"] is True
        and qa["pytest_returncode"] == 0
        and qa["actual_fit_calls"] == qa["actual_model_calls"] == 0,
        "047 pre-fit QA incomplete",
    )
    e.file(qa["pytest_log_ref"])
    require(
        all(p["source_sha256"].get(path) == digest for path, digest in qa["tested_source_sha256"].items())
        and set(str(REPO / r) for r in SOURCES) <= set(qa["tested_source_sha256"]),
        "047 QA did not bind actual code",
    )
    if initial:
        require(
            not (ROOT / "fit/W2").exists()
            and not (ROOT / "review/W2_verified.json").exists()
            and not (ROOT / "supervisor").exists(),
            "W2 outputs exist; no retry",
        )
    original = e.json(p["execution_registration_ref"])
    verified = e.json(original["execution_verification_ref"])
    e.file(verified["verifier_ref"])
    e.finish()
    return p, e, verified["verifier_ref"]


def verify_proposal(proposal_ref):
    p, e, _ = validate(proposal_ref, initial=True)
    verifier = reference(REPO / "scripts/verify_wam_clean_training_w2_fit.py")
    require(p["source_sha256"][verifier["path"]] == verifier["sha256"], "047 metadata verifier not frozen")
    return dict(
        schema_version=PREFIX + "verification.v1",
        complete=True,
        passed=True,
        proposal_ref=proposal_ref,
        eligible_for_execution=True,
        verifier_ref=verifier,
        metadata_only=True,
        consumed_sha256=e.consumed,
        actual_fit_calls=0,
        actual_array_decodes=0,
        actual_model_calls=0,
    )


def check_runtime(configuration):
    """Check actual045 numerical runtime only; this cannot authorize any fit."""
    import numpy
    import numpy.core._multiarray_umath as multiarray
    import scipy
    import scipy.linalg._fblas as fblas

    actual = dict(
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        executable=sys.executable,
    )
    require(actual == configuration["runtime"], "047 actual worker runtime differs")
    require(
        [
            reference(Path(sys.executable).resolve()),
            reference(Path(multiarray.__file__).resolve()),
            reference(Path(fblas.__file__).resolve()),
        ]
        == configuration["runtime_file_refs"],
        "047 loaded numerical binary differs",
    )


def worker_admit(registration_ref, *, runtime):
    e = Evidence()
    a = e.json(registration_ref)
    complete(a, PREFIX + "registration.v1")
    require(
        a["execution_authorized"] is True and registration_ref["path"] == str(ROOT / "execution/admitted.json"),
        "actual047 admission required",
    )
    p, guard, old_verifier = validate(a["proposal_ref"])
    e.consumed.update(guard.consumed)
    verified = e.json(a["independent_verification_ref"])
    complete(verified, PREFIX + "verification.v1")
    require(
        verified["eligible_for_execution"] is True
        and verified["proposal_ref"] == a["proposal_ref"]
        and verified["verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_training_w2_fit.py"),
        "047 independent verification differs",
    )
    expected = copy.deepcopy(p)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=a["proposal_ref"],
        independent_verification_ref=a["independent_verification_ref"],
    )
    require(canonical(expected) == canonical(a), "047 exact admitted envelope differs")
    if runtime:
        check_runtime(a)
    e.finish()
    return a, e, old_verifier


def admit(registration_ref):
    return worker_admit(registration_ref, runtime=False)[0]
