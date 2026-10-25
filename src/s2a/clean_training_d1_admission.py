"""046 metadata admission for one original041 D1 collection using qualified045 W1."""

import copy
import json
import os
from pathlib import Path

from s2a.clean_training_w1_handoff import ADMITTED_REF, OWNER_PATH, PREPARED_ROOT, check_handoff
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, check_sources, checked, read, reference, require

ROOT = REPO / "outputs/clean_training_d1_continuation_20261024"
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
PREFIX = "s2a.clean-training-d1-handoff-"
BUDGET = dict(
    model_initialization_attempts=1,
    model_initializations=1,
    requests=300,
    denoise_calls=10500,
    head_calls=10500,
    fit_calls=0,
    independent_fit_reconstructions=0,
    physics_steps=0,
    fresh_cases=0,
    automatic_retry=False,
)
SOURCES = (
    "src/s2a/clean_training_w1_handoff.py",
    "src/s2a/clean_training_d1_admission.py",
    "scripts/execute_wam_clean_training_d1_handoff.py",
    "scripts/run_wam_clean_training_d1_handoff.py",
    "scripts/verify_wam_clean_training_d1_handoff.py",
    "tests/test_clean_training_w1_handoff.py",
    "tests/test_clean_training_d1_handoff.py",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Evidence:
    def __init__(self):
        self.consumed = {}

    def file(self, ref):
        path = checked(ref)
        require(
            path.is_absolute() and path.resolve() == path and path.is_relative_to(REPO),
            "046 metadata/source escaped canonical S2A path",
        )
        old = self.consumed.setdefault(str(path), ref["sha256"])
        require(old == ref["sha256"], "conflicting046 evidence SHA")
        return path

    def json(self, ref):
        path = self.file(ref)
        require(path.suffix == ".json", "046 metadata reader refuses array archives")
        value = read(ref)
        require(isinstance(value, dict), "046 JSON object required")
        return value

    def mapping(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), "046 source closure missing")
        for path, digest in mapping.items():
            self.file(dict(path=path, sha256=digest))

    def finish(self):
        check_sources(self.consumed)


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "046 evidence incomplete or wrong schema",
    )


def absent_outputs():
    for path in (
        PREPARED_ROOT / "capture/D1",
        PREPARED_ROOT / "orchestration/D1",
        PREPARED_ROOT / "review/D1_verified.json",
        ROOT / "worker",
        ROOT / "supervisor",
    ):
        require(not path.exists(), "D1 output already exists; no retry: " + str(path))


def validate_proposal(proposal_ref, *, initial=False):
    require(Path.cwd().resolve() == REPO, "046 must run from S2A")
    check_cpu_environment(os.environ)
    e = Evidence()
    p = e.json(proposal_ref)
    complete(p, PREFIX + "proposal.v1")
    require(
        p.get("execution_authorized") is False and p.get("protocol") == "046" and p.get("stage_scope") == "D1_only",
        "046 unexecuted D1-only proposal required",
    )
    require(
        p["root"] == str(ROOT) and p["budget"] == BUDGET and p["automatic_retry"] is False,
        "046 root or finite budget differs",
    )
    require(
        p["gpu_uuid"] == GPU and p["private_port"] == 31338 and p["owner_timeout_s"] == 88400,
        "046 resource routing differs",
    )
    require(
        p["config_ref"]["path"] == str(REPO / "configs/wam_clean_training_d1_handoff_20261024.json"),
        "actual046 config required",
    )
    config = e.json(p["config_ref"])
    require(all(p.get(k) == v for k, v in config.items()), "046 proposal/config differs")
    require(
        p["plan_ref"]["path"] == str(REPO / "plans/v0.0.1/046_clean_training_d1_continuation_20261024.md"),
        "046 plan differs",
    )
    e.file(p["plan_ref"])
    require(p["original_owner_ref"] == reference(OWNER_PATH), "original041 owner differs")
    e.file(p["original_owner_ref"])
    for key, basename in (
        ("collector_ref", "collect_wam_clean_training.py"),
        ("capture_verifier_ref", "verify_wam_clean_training_capture.py"),
    ):
        require(p[key] == reference(OWNER_PATH.parent / basename), "original041 collection/audit delegate differs")
        e.file(p[key])
    check_handoff(p)
    old = e.json(ADMITTED_REF)
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in old[key].items()),
            "046 changed or omitted original508 sources/snapshots",
        )
        e.mapping(p[key])
    require(
        set(p["source_sha256"].values()) <= set(p["snapshot_sha256"].values()),
        "046 snapshots do not cover source contents",
    )
    for relative in SOURCES:
        actual = REPO / relative
        require(
            p["source_sha256"].get(str(actual)) == reference(actual)["sha256"],
            "actual046 source is not frozen: " + relative,
        )
    for key in (
        "w1_admitted_ref",
        "w1_terminal_ref",
        "w1_owner_ref",
        "w1_fit_ref",
        "w1_verification_ref",
        "w1_params_ref",
        "prepared_ref",
        "execution_registration_ref",
        "d0_owner_ref",
        "d0_collection_ref",
        "d0_capture_audit_ref",
    ):
        e.json(p[key])
    require(
        p["old_failed_w1_owner_ref"]["path"] == str(PREPARED_ROOT / "orchestration/W1/result.json"),
        "old failure path differs",
    )
    require(e.json(p["old_failed_w1_owner_ref"])["passed"] is False, "old failure was replaced")
    qa = e.json(p["qa_ref"])
    require(
        qa["complete"] is qa["passed"] is qa["ruff_passed"] is qa["format_passed"] is True
        and qa["actual_model_calls"] == 0,
        "046 CPU QA incomplete",
    )
    require(qa["pytest_returncode"] == 0 and qa["tests_passed"] >= 19, "046 targeted tests failed")
    e.file(qa["pytest_log_ref"])
    require(
        all(p["source_sha256"].get(path) == digest for path, digest in qa["tested_source_sha256"].items()),
        "046 QA source drift",
    )
    require(
        set(str(REPO / r) for r in SOURCES) <= set(qa["tested_source_sha256"]),
        "046 QA did not bind every actual entry/test",
    )
    if initial:
        absent_outputs()
    e.finish()
    return p, e


def verify_proposal(proposal_ref):
    p, e = validate_proposal(proposal_ref, initial=True)
    verifier = reference(REPO / "scripts/verify_wam_clean_training_d1_handoff.py")
    require(p["source_sha256"][verifier["path"]] == verifier["sha256"], "046 independent metadata verifier not frozen")
    return dict(
        schema_version=PREFIX + "verification.v1",
        complete=True,
        passed=True,
        proposal_ref=proposal_ref,
        eligible_for_execution=True,
        verifier_ref=verifier,
        metadata_only=True,
        source_before_after_exact=True,
        consumed_sha256=e.consumed,
        actual_model_calls=0,
        actual_array_decodes=0,
    )


def admit(registration_ref):
    e = Evidence()
    a = e.json(registration_ref)
    complete(a, PREFIX + "registration.v1")
    require(
        a.get("execution_authorized") is True and registration_ref["path"] == str(ROOT / "execution/admitted.json"),
        "actual046 admitted registration required",
    )
    p, guard = validate_proposal(a["proposal_ref"])
    e.consumed.update(guard.consumed)
    verified = e.json(a["independent_verification_ref"])
    complete(verified, PREFIX + "verification.v1")
    require(
        verified["proposal_ref"] == a["proposal_ref"]
        and verified["eligible_for_execution"] is True
        and verified["verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_training_d1_handoff.py"),
        "046 independent verification differs",
    )
    expected = copy.deepcopy(p)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=a["proposal_ref"],
        independent_verification_ref=a["independent_verification_ref"],
    )
    require(canonical(a) == canonical(expected), "046 admitted envelope differs from exact proposal")
    e.finish()
    return a
