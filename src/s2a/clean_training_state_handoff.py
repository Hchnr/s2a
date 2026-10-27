"""048 STATE admission and explicit metadata bridge; no numerical modifications."""

import copy
import inspect
import os
import sys
from pathlib import Path

from s2a.clean_training_d1_admission import Evidence, canonical
from s2a.clean_training_import_bridge import ENV_PATH, ENV_SHA, child_environment
from s2a.clean_training_w1_handoff import PREPARED_ROOT
from s2a.clean_training_w2_handoff import qualified_w2
from s2a.durable_stage_supervisor import REPO, check_cpu_environment, identity, publish, reference, require

ROOT = REPO / "outputs/clean_training_state_continuation_20261024"
BUNDLE = PREPARED_ROOT / "source/entrypoints"
BRIDGE_DIRECTORY = ROOT / "import_bridge"
BRIDGE_PATH = BRIDGE_DIRECTORY / "run_wam_clean_training.py"
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
PREFIX = "s2a.clean-training-state-handoff-"
DRY_ENV = "S2A_STATE_HANDOFF_DRY_RUN"
BUDGET = dict(
    model_initialization_attempts=1,
    model_initializations=1,
    requests=99,
    denoise_calls=3465,
    head_calls=3465,
    fit_calls=0,
    qr_calls=0,
    physics_steps=0,
    fresh_cases=0,
    automatic_retry=False,
)
SOURCES = (
    "src/s2a/clean_training_state_handoff.py",
    "src/s2a/clean_training_w2_handoff.py",
    "src/s2a/clean_training_import_bridge.py",
    "scripts/execute_wam_clean_training_state_handoff.py",
    "scripts/run_wam_clean_training_state_handoff.py",
    "scripts/verify_wam_clean_training_state_handoff.py",
    "scripts/prepare_wam_clean_training_state_handoff.py",
    "tests/test_clean_training_state_handoff.py",
    "tests/test_clean_training_w2_handoff.py",
    "tests/test_clean_training_import_bridge.py",
)
_DRY_RECEIPTS = []


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "048 metadata is incomplete or wrong schema",
    )


def absent_outputs():
    for path in (
        PREPARED_ROOT / "state_sampling",
        PREPARED_ROOT / "orchestration/STATE",
        PREPARED_ROOT / "review/STATE_verified.json",
        ROOT / "worker",
        ROOT / "role_receipts",
        ROOT / "supervisor",
    ):
        require(not path.exists(), "048 refuses existing outputs or retry: " + str(path))


def validate(proposal_ref, *, initial=False):
    """Pure metadata validation, usable by a GPU process before model imports."""
    e = Evidence()
    p = e.json(proposal_ref)
    complete(p, PREFIX + "proposal.v1")
    require(
        p["protocol"] == "048"
        and p["execution_authorized"] is False
        and p["stage_scope"] == "STATE_only"
        and p["root"] == str(ROOT)
        and p["budget"] == BUDGET
        and p["gpu_uuid"] == GPU
        and p["automatic_retry"] is False,
        "048 fixed STATE-only scope differs",
    )
    require(p["owner_timeout_s"] == 45500 and p["private_port"] == 31338, "048 resources differ")
    require(
        p["config_ref"]["path"] == str(REPO / "configs/wam_clean_training_state_handoff_20261024.json")
        and p["policy_ref"]["path"] == str(REPO / "configs/wam_clean_training_state_handoff_policy_20261024.json")
        and p["plan_ref"]["path"] == str(REPO / "plans/v0.0.1/048_clean_training_state_handoff_20261024.md"),
        "048 actual config/policy/plan differs",
    )
    config = e.json(p["config_ref"])
    policy = e.json(p["policy_ref"])
    e.file(p["plan_ref"])
    require(
        all(p.get(k) == v for k, v in config.items())
        and policy["execution_authorized"] is False
        and policy["budget"] == BUDGET
        and policy["python_safepath_required"] is True,
        "048 config or original policy differs",
    )
    qualified_w2(p, e)
    w2 = e.json(p["w2_registration_ref"])
    for key in ("source_sha256", "snapshot_sha256"):
        require(
            all(p[key].get(path) == digest for path, digest in w2[key].items()),
            "048 changed or omitted upstream source closure",
        )
        e.mapping(p[key])
    require(set(p["source_sha256"].values()) <= set(p["snapshot_sha256"].values()), "048 snapshots incomplete")
    for relative in SOURCES:
        path = REPO / relative
        require(
            p["source_sha256"].get(str(path)) == reference(path)["sha256"],
            "actual048 source is not frozen: " + relative,
        )
    for key, name in (
        ("original_owner_ref", "run_wam_clean_training_state.py"),
        ("original_sampler_ref", "sample_wam_clean_training_state.py"),
        ("original_verifier_ref", "verify_wam_clean_training_state.py"),
        ("original_training_owner_ref", "run_wam_clean_training.py"),
    ):
        require(p[key] == reference(BUNDLE / name), "048 original delegate differs: " + key)
        e.file(p[key])
    require(
        p["bridge_ref"] == reference(BRIDGE_PATH)
        and p["source_sha256"].get(str(BRIDGE_PATH)) == p["bridge_ref"]["sha256"],
        "048 production bridge not frozen",
    )
    expected_modules = {
        "run_wam_clean_training": p["bridge_ref"],
        "continuation_runtime": reference(BUNDLE / "continuation_runtime.py"),
        "collect_wam_clean_training": reference(BUNDLE / "collect_wam_clean_training.py"),
    }
    require(p["module_refs"] == expected_modules, "048 actual module origin contract differs")
    qa = e.json(p["qa_ref"])
    require(
        qa["complete"] is qa["passed"] is qa["ruff_passed"] is qa["format_passed"] is True
        and qa["pytest_returncode"] == 0
        and qa["actual_model_calls"] == 0,
        "048 QA incomplete",
    )
    e.file(qa["pytest_log_ref"])
    require(
        all(p["source_sha256"].get(path) == digest for path, digest in qa["tested_source_sha256"].items())
        and set(str(REPO / r) for r in SOURCES) <= set(qa["tested_source_sha256"]),
        "048 QA source drift",
    )
    if initial:
        absent_outputs()
    e.finish()
    return p, e


def verify_proposal(proposal_ref):
    require(Path.cwd().resolve() == REPO, "048 metadata review must run from S2A")
    check_cpu_environment(os.environ)
    p, e = validate(proposal_ref, initial=True)
    verifier = reference(REPO / "scripts/verify_wam_clean_training_state_handoff.py")
    require(p["source_sha256"].get(verifier["path"]) == verifier["sha256"], "048 verifier is not frozen")
    return dict(
        schema_version=PREFIX + "verification.v1",
        complete=True,
        passed=True,
        proposal_ref=proposal_ref,
        verifier_ref=verifier,
        eligible_for_execution=True,
        metadata_only=True,
        consumed_sha256=e.consumed,
        actual_model_calls=0,
    )


def admit(registration_ref, *, role="owner"):
    require(Path.cwd().resolve() == REPO and role in ("owner", "model"), "048 role/cwd differs")
    require(
        os.environ.get("LD_LIBRARY_PATH") == ""
        and all(os.environ.get(k) == "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "048 explicit empty LD and CPU limits required",
    )
    if role == "owner":
        check_cpu_environment(os.environ)
    else:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
            and sys.flags.safe_path is True
            and Path(sys.argv[0]).resolve() == BUNDLE / "sample_wam_clean_training_state.py",
            "048 model bootstrap identity differs",
        )
        require(os.environ.get(DRY_ENV) != "1", "048 model cannot use metadata dry mode")
    e = Evidence()
    a = e.json(registration_ref)
    complete(a, PREFIX + "registration.v1")
    require(
        registration_ref["path"] == str(ROOT / "execution/admitted.json") and a["execution_authorized"] is True,
        "actual048 admitted envelope required",
    )
    p, guard = validate(a["proposal_ref"])
    e.consumed.update(guard.consumed)
    v = e.json(a["independent_verification_ref"])
    complete(v, PREFIX + "verification.v1")
    require(
        v["proposal_ref"] == a["proposal_ref"]
        and v["eligible_for_execution"] is True
        and v["verifier_ref"] == reference(REPO / "scripts/verify_wam_clean_training_state_handoff.py"),
        "048 independent verification differs",
    )
    expected = copy.deepcopy(p)
    expected.update(
        schema_version=PREFIX + "registration.v1",
        execution_authorized=True,
        proposal_ref=a["proposal_ref"],
        independent_verification_ref=a["independent_verification_ref"],
    )
    require(canonical(expected) == canonical(a), "048 admitted proposal envelope differs")
    e.finish()
    return a


def environment_reference():
    require(ENV_PATH in os.environ and ENV_SHA in os.environ, "048 bridge registration declaration missing")
    return dict(path=os.environ[ENV_PATH], sha256=os.environ[ENV_SHA])


def model_environment(base):
    ref = environment_reference()
    admit(ref, role="owner")
    require(
        os.environ.get(DRY_ENV) != "1"
        and base.get("CUDA_VISIBLE_DEVICES") == GPU
        and base.get("LD_LIBRARY_PATH") == "",
        "048 model environment is not the admitted GPU role",
    )
    return child_environment(base, bridge_directory=BRIDGE_DIRECTORY, bundle_directory=BUNDLE, registration_ref=ref)


def audit_environment(base):
    """The independent old checker keeps its original script-directory imports."""
    result = dict(base)
    for key in (ENV_PATH, ENV_SHA, DRY_ENV, "PYTHONSAFEPATH"):
        result.pop(key, None)
    require(
        str(BRIDGE_DIRECTORY) not in result.get("PYTHONPATH", "").split(os.pathsep),
        "independent audit inherited bridge imports",
    )
    return result


def reset_dry_receipts():
    _DRY_RECEIPTS.clear()


def dry_receipts():
    return copy.deepcopy(_DRY_RECEIPTS)


def metadata_prior(prepared, prepared_ref, registration_ref):
    """Called only from the explicit bridge; never replace a mathematical function."""
    role = "model" if os.environ.get("CUDA_VISIBLE_DEVICES") == GPU else "owner"
    ref = environment_reference()
    a = admit(ref, role=role)
    require(
        prepared_ref == a["prepared_ref"]
        and registration_ref == a["execution_registration_ref"]
        and prepared["root"] == str(PREPARED_ROOT),
        "048 lookup belongs to another preparation",
    )
    dry = os.environ.get(DRY_ENV) == "1"
    frame = inspect.currentframe()
    shim = frame.f_back
    caller = shim.f_back
    try:
        require(
            Path(shim.f_code.co_filename).resolve() == BRIDGE_PATH,
            "048 lookup did not originate from production bridge",
        )
        caller_path = Path(caller.f_code.co_filename).resolve()
        caller_name = caller.f_code.co_name
        expected = (
            {(BUNDLE / "run_wam_clean_training_state.py", "planned_worker")}
            if role == "owner" and not dry
            else {
                (BUNDLE / "run_wam_clean_training_state.py", "planned_worker"),
                (BUNDLE / "sample_wam_clean_training_state.py", "admit_sampling"),
            }
        )
        if role == "model":
            expected = {(BUNDLE / "sample_wam_clean_training_state.py", "admit_sampling")}
        require((caller_path, caller_name) in expected, "048 original metadata caller differs")
    finally:
        del frame, shim, caller
    bridge = sys.modules.get("run_wam_clean_training")
    require(
        bridge is not None
        and reference(Path(bridge.__file__).resolve()) == a["bridge_ref"]
        and reference(Path(bridge._original.__file__).resolve()) == a["original_training_owner_ref"],
        "048 bridge/private delegate actual origin differs",
    )
    require(sys.path[:2] == [str(BRIDGE_DIRECTORY), str(BUNDLE)], "048 explicit import precedence differs")
    owner = qualified_w2(a)
    receipt = dict(
        schema_version=PREFIX + "lookup.v1",
        complete=True,
        passed=True,
        role=role,
        dry_run=dry,
        registration_ref=ref,
        prepared_ref=prepared_ref,
        execution_registration_ref=registration_ref,
        w2_owner_ref=a["w2_owner_ref"],
        w2_fit_ref=a["w2_fit_ref"],
        w2_verification_ref=a["w2_verification_ref"],
        w2_params_ref=a["w2_params_ref"],
        bridge_ref=a["bridge_ref"],
        original_training_owner_ref=a["original_training_owner_ref"],
        caller_source_ref=reference(caller_path),
        caller_function=caller_name,
        identity=identity(os.getpid()),
        safe_path=bool(sys.flags.safe_path),
        sys_path_prefix=sys.path[:2],
        source_guard_completed=True,
    )
    if dry:
        _DRY_RECEIPTS.append(receipt)
    else:
        publish(ROOT / "role_receipts" / f"{role}.json", receipt)
    return owner


def check_lookup_receipts(a, registration_ref, stage):
    refs = {role: reference(ROOT / "role_receipts" / f"{role}.json") for role in ("owner", "model")}
    model_launch = Evidence().json(stage["owned_commands"][0]["launch"])
    for role, ref in refs.items():
        value = Evidence().json(ref)
        complete(value, PREFIX + "lookup.v1")
        require(
            value["role"] == role
            and value["dry_run"] is False
            and value["registration_ref"] == registration_ref
            and value["w2_owner_ref"] == a["w2_owner_ref"]
            and value["prepared_ref"] == a["prepared_ref"]
            and value["execution_registration_ref"] == a["execution_registration_ref"]
            and value["w2_fit_ref"] == a["w2_fit_ref"]
            and value["w2_verification_ref"] == a["w2_verification_ref"]
            and value["sys_path_prefix"] == [str(BRIDGE_DIRECTORY), str(BUNDLE)]
            and value["w2_params_ref"] == a["w2_params_ref"]
            and value["bridge_ref"] == a["bridge_ref"]
            and value["original_training_owner_ref"] == a["original_training_owner_ref"]
            and value["source_guard_completed"] is True,
            "048 lookup receipt differs",
        )
        if role == "model":
            require(
                value["safe_path"] is True
                and all(value["identity"][k] == model_launch["child"][k] for k in ("pid", "start_ticks"))
                and value["identity"]["ppid"] == model_launch["child"]["ppid"]
                and value["identity"]["environment"]["CUDA_VISIBLE_DEVICES"] == GPU
                and value["caller_source_ref"] == a["original_sampler_ref"]
                and value["caller_function"] == "admit_sampling",
                "048 model lookup did not occur in actual sampler child",
            )
        else:
            require(
                value["caller_source_ref"] == a["original_owner_ref"] and value["caller_function"] == "planned_worker",
                "048 owner lookup origin differs",
            )
    owner_lookup = Evidence().json(refs["owner"])
    require(
        owner_lookup["identity"]["pid"] == model_launch["child"]["ppid"],
        "048 lookups came from different owner/model chain",
    )
    return refs
