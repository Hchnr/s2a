"""049 metadata admission for the original fixed twelve-case development run."""

import copy
import os
from pathlib import Path

from s2a.clean_training_d1_admission import Evidence, canonical
from s2a.clean_training_state_handoff import BUNDLE, GPU
from s2a.clean_training_state_handoff import ROOT as STATE_ROOT
from s2a.clean_training_w1_handoff import PREPARED_ROOT
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, reference, require

ROOT = REPO / "outputs/clean_training_development_continuation_20261024"
PREFIX = "s2a.clean-training-development-handoff-"
BUDGET = dict(
    model_initialization_attempts=1,
    model_initializations=1,
    cases=12,
    requests=900,
    denoise_calls=31500,
    physics_steps=7200,
    fit_calls=0,
    qr_calls=0,
    fresh_cases=0,
    automatic_retry=False,
)
OWNER_PATH = BUNDLE / "run_wam_clean_training_development.py"
SOURCES = (
    "src/s2a/clean_training_development_handoff.py",
    "scripts/prepare_wam_clean_training_development_handoff.py",
    "scripts/verify_wam_clean_training_development_handoff.py",
    "scripts/run_wam_clean_training_development_handoff.py",
    "scripts/check_wam_clean_training_state_terminal.py",
    "tests/test_clean_training_development_handoff.py",
    "tests/test_clean_training_development_contract.py",
    "tests/test_clean_training_state_terminal.py",
)


def command(config):
    args = [str(REPO / ".venv/bin/python"), str(OWNER_PATH)]
    for name, key in (
        ("prepared", "prepared_ref"),
        ("execution-registration", "execution_registration_ref"),
        ("state-owner", "state_owner_ref"),
        ("state-verification", "state_verification_ref"),
    ):
        ref = config[key]
        args.extend(("--" + name, ref["path"], "--" + name + "-sha256", ref["sha256"]))
    return [*args, "--seed", "42"]


def environment():
    from s2a.durable_stage_supervisor import cpu_environment

    result = cpu_environment()
    for name in ("S2A_STATE_HANDOFF_PATH", "S2A_STATE_HANDOFF_SHA256", "S2A_STATE_HANDOFF_DRY_RUN", "PYTHONSAFEPATH"):
        result.pop(name, None)
    result["PYTHONPATH"] = os.pathsep.join(map(str, (BUNDLE, REPO / "src", REPO / "scripts")))
    return result


def absent_outputs():
    require(
        all(
            not p.exists()
            for p in (
                ROOT / "supervisor",
                PREPARED_ROOT / "development",
                PREPARED_ROOT / "orchestration/DEVELOPMENT",
                PREPARED_ROOT / "review/DEVELOPMENT_verified.json",
            )
        ),
        "development outputs exist; no automatic retry",
    )


def qualified_state(ref):
    # Existing script checks the actual048 owner, wrapper, original STATE and both lookups.
    from check_wam_clean_training_state_terminal import qualify

    require(ref == reference(STATE_ROOT / "execution/admitted.json"), "049 requires the exact048 registration")
    return qualify(ref)


def validate(proposal_ref, *, initial=False):
    require(Path.cwd().resolve() == REPO, "049 must run from S2A")
    check_cpu_environment(os.environ)
    e = Evidence()
    p = e.json(proposal_ref)
    require(
        p["schema_version"] == PREFIX + "proposal.v1"
        and p["complete"] is p["passed"] is True
        and p["execution_authorized"] is False
        and p["protocol"] == "049"
        and p["stage_scope"] == "DEVELOPMENT_only"
        and p["root"] == str(ROOT)
        and p["gpu_uuid"] == GPU
        and p["budget"] == BUDGET
        and p["automatic_retry"] is False,
        "049 fixed development-only envelope required",
    )
    terminal = qualified_state(p["state_registration_ref"])
    require(p["state_terminal"] == terminal, "049 joint048 terminal differs")
    registered = e.json(p["state_registration_ref"])
    for key in ("prepared_ref", "execution_registration_ref"):
        require(p[key] == registered[key], "049 original041 identity differs")
    require(
        p["state_owner_ref"] == terminal["state_owner_ref"]
        and p["state_verification_ref"] == terminal["verification_ref"]
        and p["owner_ref"] == reference(OWNER_PATH)
        and p["private_port"] == 31339,
        "049 STATE or original owner binding differs",
    )
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in registered[key].items()),
            "049 changed upstream551 source closure",
        )
        e.mapping(p[key])
    require(set(p["source_sha256"].values()) <= set(p["snapshot_sha256"].values()), "049 snapshots incomplete")
    require(p["command"] == command(p), "049 actual original owner command differs")
    original = e.json(p["execution_registration_ref"])
    require(
        p["case_order"] == original["development"]["case_order"]
        and len(p["case_order"]) == 12
        and sum(case["expected_requests"] for case in p["case_order"]) == 900,
        "049 original fixed case order differs",
    )
    prepared = e.json(p["prepared_ref"])
    require(
        p["owner_timeout_s"] == prepared["resources"]["development_timeout_s"] + 180
        and prepared["resources"]["development_timeout_s"] == 172800,
        "049 fixed owner time budget differs",
    )
    for key, relative in (
        ("config_ref", "configs/wam_clean_training_development_continuation_20261024.json"),
        ("policy_ref", "configs/wam_clean_training_development_continuation_policy_20261024.json"),
        ("plan_ref", "plans/v0.0.1/049_clean_training_development_continuation_20261024.md"),
    ):
        require(p[key]["path"] == str(REPO / relative), "049 final config/policy/plan path differs")
        e.file(p[key])
    config = e.json(p["config_ref"])
    require(all(p.get(key) == value for key, value in config.items()), "049 config differs")
    policy = e.json(p["policy_ref"])
    require(policy["execution_authorized"] is False and policy["budget"] == BUDGET, "049 policy budget differs")
    for relative in SOURCES:
        path = REPO / relative
        require(p["source_sha256"].get(str(path)) == reference(path)["sha256"], "049 actual source not frozen")
    qa = e.json(p["qa_ref"])
    require(
        qa["complete"] is qa["passed"] is qa["ruff_passed"] is qa["format_passed"] is True
        and qa["pytest_returncode"] == 0
        and qa["actual_model_calls"] == qa["actual_physics_steps"] == 0,
        "049 preparation QA missing",
    )
    e.file(qa["pytest_log_ref"])
    require(set(str(REPO / r) for r in SOURCES) <= set(qa["tested_source_sha256"]), "049 QA source coverage missing")
    for path, digest in qa["tested_source_sha256"].items():
        require(p["source_sha256"].get(path) == digest, "049 tested source differs")
    if initial:
        absent_outputs()
    e.finish()
    return p


def admit(ref):
    e = Evidence()
    a = e.json(ref)
    require(
        ref["path"] == str(ROOT / "execution/admitted.json")
        and a["schema_version"] == PREFIX + "registration.v1"
        and a["execution_authorized"] is True,
        "actual049 admission required",
    )
    p = validate(a["proposal_ref"])
    v = e.json(a["independent_verification_ref"])
    require(
        v["schema_version"] == PREFIX + "verification.v1"
        and v["complete"] is v["passed"] is v["eligible_for_execution"] is True
        and v["proposal_ref"] == a["proposal_ref"]
        and v["verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_training_development_handoff.py"),
        "049 independent admission differs",
    )
    expected = copy.deepcopy(p)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=a["proposal_ref"],
        independent_verification_ref=a["independent_verification_ref"],
    )
    require(canonical(expected) == canonical(a), "049 exact admission envelope differs")
    e.finish()
    return a
