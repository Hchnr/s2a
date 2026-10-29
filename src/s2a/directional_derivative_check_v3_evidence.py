"""Bind a check-only correction to the closed producer and preserved failed replay."""

from s2a.directional_derivative_v2_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.directional_derivative_v2_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.directional_derivative_v2_evidence import PLAN as PRODUCER_PLAN
from s2a.directional_derivative_v2_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.directional_derivative_v2_evidence import QA_SOURCES as PRODUCER_QA_SOURCES
from s2a.directional_derivative_v2_evidence import QA_TESTS as PRODUCER_QA_TESTS
from s2a.directional_derivative_v2_evidence import REPO, finish, read, reference, remember, require
from s2a.directional_derivative_v2_evidence import SQP_CLOSURE_REF as SQP_CLOSURE_REF
from s2a.directional_derivative_v2_evidence import SQP_PRODUCER_REF as SQP_PRODUCER_REF
from s2a.directional_derivative_v2_evidence import V1_ADMISSION_REF as V1_ADMISSION_REF
from s2a.directional_derivative_v2_evidence import V1_QA_REF as V1_QA_REF
from s2a.directional_derivative_v2_evidence import case_root as producer_root
from s2a.directional_derivative_v2_evidence import checked as checked
from s2a.directional_derivative_v2_evidence import closed_owner as closed_owner
from s2a.directional_derivative_v2_evidence import load_prerequisites as load_v2_prerequisites
from s2a.directional_derivative_v2_evidence import model_registration as model_registration

PLAN = REPO / "plans/v0.0.1/directional_derivative_check_v3_20261029.md"
SOURCES = (
    "scripts/directional_derivative_stencil_audit_v2.py",
    "scripts/check_tracking_v5_directional_derivative_v3.py",
    "scripts/execute_tracking_v5_directional_derivative_check_v3.py",
    "src/s2a/directional_derivative_check_v3_evidence.py",
    "plans/v0.0.1/directional_derivative_check_v3_20261029.md",
    "tests/test_directional_derivative_stencil_audit_v2.py",
    "tests/test_directional_derivative_check_v3.py",
)
QA_SOURCES = (*PRODUCER_QA_SOURCES, *SOURCES)
QA_TESTS = (*PRODUCER_QA_TESTS, *SOURCES[-2:])
PRODUCER_REF = dict(
    path=str(producer_root() / "verified.json"),
    sha256="a626d66c66b4fba7dd0509092c1067c615b69cfe8545bba42e6f2b4d8d13951c",
)
PRODUCER_OWNER_REF = dict(
    path=str(producer_root(owner=True) / "result.json"),
    sha256="ebf9d63e7612e939370ce40ebfe3e698a2996912c167db0e5f08686f0564dbe6",
)
PRODUCER_QA_REF = dict(
    path=str(REPO / "outputs/directional_derivative_v2_native_qa_20261029_yuinc_wl/result.json"),
    sha256="8499d64a179c44c0b6364e588cd9d64c87b3d981aec3d07654cf898d36a5673a",
)
FAILED_CHECK_REF = dict(
    path=str(producer_root(check=True) / "verified.json"),
    sha256="f764512e07864a8dc89774059c43ed71ac76dca5914ad4531fa8a132b2ee5aab",
)
FAILED_OWNER_REF = dict(
    path=str(producer_root(check=True, owner=True) / "result.json"),
    sha256="feebe6faa1ff11491e94e8ed2c32b15573f5cf2c9dfe18517f392a71069bf852",
)
FAILED_CAPTURE_REFS = tuple(
    dict(path=str(producer_root(check=True, owner=True) / "capture" / path), sha256=digest)
    for path, digest in (
        ("launch.json", "d895d4ff97145affe72298c9cffa924df32a0bed29f22f33a658f9846e5d928d"),
        ("exit.json", "06bc82811a76aeacd289ab37671b1de45bf96c9669da6f43c4d9525bb1997b6b"),
        ("ownership/release.json", "de7542c07dd5f53392f9be135a15b8fdd17512ba377e0c13c7d474bf8e941d3e"),
    )
)


def case_root(*, check=False, owner=False):
    if not check:
        return producer_root(owner=owner)
    suffix = "_execution" if owner else ""
    return REPO / f"outputs/tracking_v5_directional_derivative_v3_check{suffix}_20261029"


def verify_preserved_failure():
    """Verify failed lifecycle and original source bytes, without granting qualification."""
    refs = {}
    failed = read(remember(FAILED_CHECK_REF, refs))
    owner = read(remember(FAILED_OWNER_REF, refs))
    reg = read(remember(failed["registration_ref"], refs))
    owner_reg = read(remember(owner["registration_ref"], refs))
    launch, exited, release = [read(remember(ref, refs)) for ref in FAILED_CAPTURE_REFS]
    require(
        failed["complete"] is failed["passed"] is owner["complete"] is owner["passed"] is False
        and failed["error"] == "ValueError('Anchor inside old stage bounds')"
        and owner["error"] == "ChildProcessError('025 command failed with1; no automatic retry')"
        and failed["response_evaluations"] == failed["verified_responses"] == 77
        and failed["unique_command_count"] == 76
        and failed["numerical_inner_steps"] == 23100
        and failed["query_count"] == failed["mj_forward_queries"] == 24025
        and failed["ledger_check"]["passed"] is True
        and failed["boundary_classification_disagreements"] == []
        and failed["control_qualified"] is False
        and all(
            failed[key] == 0
            for key in ("physics_steps", "world_model_calls", "fresh_cases", "actual_consumed_reference_count")
        )
        and not any(
            key in failed for key in ("saved_stencil_check", "historical_anchor_check", "source_refs_unchanged")
        ),
        "Preserve the exact incomplete v2 stencil audit failure; no completed qualification",
    )
    require(
        launch["command"] == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_directional_derivative_v2.py")]
        and launch["selected_environment"] == {"CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""}
        and launch["child"]["pid"] == release["pid"] == 205307
        and launch["child"]["start_ticks"] == release["start_ticks"] == 616309119
        and launch["child"]["ppid"] == 201498
        and launch["automatic_retry"] is exited["automatic_retry"] is release["signal_sent"] is False
        and exited["actual_wait"] is release["reaped"] is True
        and exited["returncode"] == release["returncode"] == 1
        and release["original_error"] == owner["error"],
        "The original failed child was actually waited and reaped without signal or retry",
    )
    require(
        reg["producer_ref"] == owner_reg["producer_ref"] == PRODUCER_REF
        and reg["producer_owner_ref"] == owner_reg["producer_owner_ref"] == PRODUCER_OWNER_REF
        and reg["plan_ref"] == owner_reg["plan_ref"] == reference(PRODUCER_PLAN)
        and owner_reg["native_component_qa_ref"] == PRODUCER_QA_REF
        and owner_reg["stage"] == owner["stage"] == "check"
        and owner_reg["timeout_seconds"] == 1800,
        "Original failed checker registrations retain the v2 plan, producer and QA",
    )
    qa = read(remember(PRODUCER_QA_REF, refs))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is True
        and qa["returncode"] == 0
        and set(qa["source_sha256"]) == set(PRODUCER_QA_SOURCES),
        "The closed original68-source QA is immutable",
    )
    remember(owner_reg["native_component_qa_log_ref"], refs)
    for path, digest in {**owner_reg["source_sha256"], **qa["source_sha256"]}.items():
        remember(dict(path=str(REPO / path), sha256=digest), refs)
    for ref in reg["source_refs"] + reg["input_refs"]:
        remember(ref, refs)
    finish(refs)
    return refs


def load_prerequisites():
    original, registration, sqp, closure, refs = load_v2_prerequisites()
    for ref in verify_preserved_failure().values():
        remember(ref, refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return original, registration, sqp, closure, refs
