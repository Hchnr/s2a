"""Bind a check-only repair to the unchanged producer and terminated failed replay."""

from s2a import tail_translation_reserve_evidence as prior
from s2a.tail_translation_reserve_evidence import CPU_ENVIRONMENT as CPU_ENVIRONMENT
from s2a.tail_translation_reserve_evidence import NATIVE_PYTHON as NATIVE_PYTHON
from s2a.tail_translation_reserve_evidence import PLAN as PLAN
from s2a.tail_translation_reserve_evidence import PREFIX_CHECK_REF as PREFIX_CHECK_REF
from s2a.tail_translation_reserve_evidence import PRIOR_ANALYSIS_REF as PRIOR_ANALYSIS_REF
from s2a.tail_translation_reserve_evidence import PRIOR_CLOSURE_REF as PRIOR_CLOSURE_REF
from s2a.tail_translation_reserve_evidence import PRIOR_PRODUCER_REF as PRIOR_PRODUCER_REF
from s2a.tail_translation_reserve_evidence import (
    REPO,
    closed_owner,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.tail_translation_reserve_evidence import model_registration as model_registration

REPAIR_PLAN = REPO / "plans/v0.0.1/tail_translation_check_v2_20261029.md"
SOURCES = (
    "src/s2a/tail_translation_check_v2_evidence.py",
    "scripts/check_tracking_v5_tail_translation_reserve_v2.py",
    "scripts/execute_tracking_v5_tail_translation_check_v2.py",
    "plans/v0.0.1/tail_translation_check_v2_20261029.md",
    "tests/test_tail_translation_check_v2.py",
)
QA_SOURCES = (*prior.QA_SOURCES, *SOURCES)
QA_TESTS = (*prior.QA_TESTS, SOURCES[-1])
PRIOR_QA_REF = dict(
    path=str(REPO / "outputs/tail_translation_reserve_native_qa_20261029_lkpp55kq/result.json"),
    sha256="e421b3860faa93407194fad3cc57eabd56d16d6eea58bb46998a90dc62dde663",
)
PRODUCER_REF = dict(
    path=str(prior.case_root() / "verified.json"),
    sha256="0305dc39480abf4512205f8628729fd789595d40aeca189343bc19779f029d70",
)
PRODUCER_OWNER_REF = dict(
    path=str(prior.case_root(owner=True) / "result.json"),
    sha256="f604b9e31b1a4aa00a576b099065334fd04aae8722ac5710165b080c2c76b6cb",
)
FAILED_CHECK_REF = dict(
    path=str(prior.case_root(check=True) / "verified.json"),
    sha256="f7582560bc9dce92f56632b8c7688c9b4403b50ce23be4dc1d68b818d3600751",
)
FAILED_OWNER_REF = dict(
    path=str(prior.case_root(check=True, owner=True) / "result.json"),
    sha256="22a0be80635c2023c2cc95b2a088eb0c9b51aa5e2c362c7b4a52c97eae88da8f",
)
FAILED_CAPTURE = {
    "launch": ("launch.json", "04e7969e9c056b40ae538674b1f5c24e7c747ce2367f34eaa4d523e432ad1896"),
    "exit": ("exit.json", "06bc82811a76aeacd289ab37671b1de45bf96c9669da6f43c4d9525bb1997b6b"),
    "release": ("ownership/release.json", "867d73a4f1b75a855c0486738016b8a023db101755abf3435ca4d3d7bb82307b"),
    "log": ("command.log", "876e35928e2245bf60f093cf22cd2e1abd24af0e23506bafb1ad234b6d3df7a9"),
}


def case_root(*, check=False, owner=False):
    if not check:
        return prior.case_root(owner=owner)
    suffix = "_execution" if owner else ""
    return REPO / f"outputs/tracking_v5_tail_translation_reserve_check_v2{suffix}_20261029"


def verify_failed_replay(refs):
    failed = read(remember(FAILED_CHECK_REF, refs))
    owner = read(remember(FAILED_OWNER_REF, refs))
    registration = read(remember(owner["registration_ref"], refs))
    child_registration = read(remember(failed["registration_ref"], refs))
    captures = {
        key: remember(dict(path=str(prior.case_root(check=True, owner=True) / "capture" / name), sha256=sha), refs)
        for key, (name, sha) in FAILED_CAPTURE.items()
    }
    launch, exited, release = (read(captures[key]) for key in ("launch", "exit", "release"))
    require(
        failed["complete"] is failed["passed"] is owner["complete"] is owner["passed"] is False
        and failed["response_evaluations"] == failed["verified_responses"] == 14
        and failed["numerical_inner_steps"] == 4200
        and failed["query_count"] == 4369
        and failed["boundary_classification_disagreements"] == []
        and failed["error"].startswith("TypeError('numpy boolean subtract")
        and "ledger_check" not in failed
        and "independent_restoration_check" not in failed,
        "Preserve complete numerical replay followed by the exact unfinished metadata failure",
    )
    require(
        exited["actual_wait"] is release["reaped"] is True
        and exited["returncode"] == release["returncode"] == 1
        and release["signal_sent"] is False
        and exited["automatic_retry"] is launch["automatic_retry"] is False
        and release["pid"] == launch["child"]["pid"] == 569716
        and release["start_ticks"] == launch["child"]["start_ticks"] == 616909863
        and launch["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_tail_translation_reserve.py")]
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        "The original failed checker was actually waited and reaped without signals or retries",
    )
    require(
        registration["native_component_qa_ref"] == PRIOR_QA_REF
        and registration["producer_ref"] == child_registration["producer_ref"] == PRODUCER_REF
        and registration["producer_owner_ref"] == child_registration["producer_owner_ref"] == PRODUCER_OWNER_REF
        and child_registration["plan_ref"] == reference(PLAN),
        "Failed replay binds the same closed producer",
    )
    for path, sha in registration["source_sha256"].items():
        remember(dict(path=path, sha256=sha), refs)
    for ref in child_registration["input_refs"] + child_registration["source_refs"]:
        remember(ref, refs)
    return dict(
        failed_checker_ref=FAILED_CHECK_REF,
        failed_owner_ref=FAILED_OWNER_REF,
        capture_refs=captures,
        response_evaluations=14,
        numerical_inner_steps=4200,
        query_count=4369,
        actual_wait=True,
        reaped=True,
        returncode=1,
        signal_sent=False,
        automatic_retry=False,
    )


def load_prerequisites():
    original, registration, secant, closure, refs = prior.load_prerequisites()
    owner, producer = closed_owner(remember(PRODUCER_OWNER_REF, refs), refs)
    remember(PRODUCER_REF, refs)
    require(
        owner["result_ref"] == PRODUCER_REF and producer["complete"] is producer["passed"] is True,
        "Original successful producer remains the only response source",
    )
    verify_failed_replay(refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return original, registration, secant, closure, refs
