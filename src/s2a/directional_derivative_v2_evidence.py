"""Preserve the failed warm-clipping admission; admit exact diagnostic bytes."""

import numpy as np

from s2a.directional_derivative_evidence import (
    CPU_ENVIRONMENT as CPU_ENVIRONMENT,
)
from s2a.directional_derivative_evidence import (
    NATIVE_PYTHON as NATIVE_PYTHON,
)
from s2a.directional_derivative_evidence import (
    PREFIX_CHECK_REF as PREFIX_CHECK_REF,
)
from s2a.directional_derivative_evidence import (
    QA_SOURCES as V1_QA_SOURCES,
)
from s2a.directional_derivative_evidence import (
    QA_TESTS as V1_QA_TESTS,
)
from s2a.directional_derivative_evidence import (
    REPO,
    finish,
    read,
    reference,
    remember,
    require,
)
from s2a.directional_derivative_evidence import (
    SQP_CHECK_OWNER_REF as SQP_CHECK_OWNER_REF,
)
from s2a.directional_derivative_evidence import (
    SQP_CHECK_REF as SQP_CHECK_REF,
)
from s2a.directional_derivative_evidence import (
    SQP_CLOSURE_REF as SQP_CLOSURE_REF,
)
from s2a.directional_derivative_evidence import (
    SQP_OWNER_REF as SQP_OWNER_REF,
)
from s2a.directional_derivative_evidence import (
    SQP_PRODUCER_REF as SQP_PRODUCER_REF,
)
from s2a.directional_derivative_evidence import (
    checked as checked,
)
from s2a.directional_derivative_evidence import (
    closed_owner as closed_owner,
)
from s2a.directional_derivative_evidence import (
    compare_saved_stencil as compare_saved_stencil,
)
from s2a.directional_derivative_evidence import (
    load_prerequisites as load_v1_prerequisites,
)
from s2a.directional_derivative_evidence import (
    model_registration as model_registration,
)
from s2a.directional_derivative_evidence import (
    saved_stencil as saved_stencil,
)

PLAN = REPO / "plans/v0.0.1/directional_derivative_v2_admission_20261029.md"
SOURCES = (
    "src/s2a/directional_derivative_v2_evidence.py",
    "scripts/probe_tracking_v5_directional_derivative_v2.py",
    "scripts/check_tracking_v5_directional_derivative_v2.py",
    "scripts/execute_tracking_v5_directional_derivative_v2.py",
    "plans/v0.0.1/directional_derivative_v2_admission_20261029.md",
    "tests/test_directional_derivative_v2.py",
)
QA_SOURCES = (*V1_QA_SOURCES, *SOURCES)
QA_TESTS = (*V1_QA_TESTS, SOURCES[-1])
V1_QA_REF = dict(
    path=str(REPO / "outputs/directional_derivative_native_qa_20261029_vpffywr0/result.json"),
    sha256="e420c1b206ce19bf2e47b1d1ed5a7434dcfc1b60f8cdc6d2df4112f2e345796c",
)
V1_ADMISSION_REF = dict(
    path=str(REPO / "outputs/directional_derivative_admission_20261029/result.json"),
    sha256="56827f760b11d25f913d3993d7441daa7fc068f94b217c52d62f6cd6e48dfbda",
)


def case_root(*, check=False, owner=False):
    suffix = ("_check" if check else "") + ("_execution" if owner else "")
    return REPO / f"outputs/tracking_v5_directional_derivative_v2{suffix}_20261029"


def load_prerequisites():
    original, registration, sqp, closure, refs = load_v1_prerequisites()
    qa = read(remember(V1_QA_REF, refs))
    failed = read(remember(V1_ADMISSION_REF, refs))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is True
        and qa["returncode"] == 0
        and failed["complete"] is failed["passed"] is False
        and failed["native_component_qa_ref"] == V1_QA_REF
        and failed["source_sha256"] == qa["source_sha256"]
        and set(qa["source_sha256"]) == set(V1_QA_SOURCES)
        and len(failed["commands"]) == 3
        and [row["returncode"] for row in failed["commands"]] == [0, 0, 1]
        and all(
            row["actual_wait"] is row["reaped"] is True and row["signal_sent"] is False for row in failed["commands"]
        )
        and all(
            failed[key] == 0
            for key in ("new_responses", "new_queries", "physics_steps", "world_model_calls", "fresh_cases")
        ),
        "Preserved closed v1 metadata failure with no response or model execution",
    )
    require(
        failed["commands"][2]["command"]
        == [str(NATIVE_PYTHON), "scripts/probe_tracking_v5_directional_derivative.py", "--dry"],
        "Exact failed v1 dry command",
    )
    remember(failed["driver_ref"], refs)
    remember(dict(path=str((REPO / V1_QA_REF["path"]).parent / "pytest.log"), sha256=qa["log_sha256"]), refs)
    for command in failed["commands"]:
        remember(command["log_ref"], refs)
    require(
        "Frozen solver start clipping must preserve exact warm bytes"
        in checked(failed["commands"][2]["log_ref"]).read_text(),
        "The recorded failure is the excess warm-clipping contract",
    )
    for path, digest in qa["source_sha256"].items():
        remember(dict(path=str(REPO / path), sha256=digest), refs)
    for path in SOURCES:
        remember(reference(REPO / path), refs)
    finish(refs)
    return original, registration, sqp, closure, refs


def admit_anchor(source, commands, interface):
    """Admit original command slacks, preserving every float64 bit without clipping."""
    require(not np.iscomplexobj(commands), "Real diagnostic anchor")
    value = np.asarray(commands, dtype=np.float64)
    previous = np.asarray(source["previous_command"], dtype=np.float64)[:6]
    require(
        value.shape == (12, 6) and previous.shape == (6,) and np.isfinite(value).all() and np.isfinite(previous).all(),
        "Complete finite H12 anchor and previous command",
    )
    lower, upper = np.asarray(interface.input_min, dtype=float), np.asarray(interface.input_max, dtype=float)
    require(
        lower.shape == upper.shape == (6,)
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.all(lower < upper),
        "Finite strict original command bounds",
    )
    increments = np.diff(np.vstack((previous, value)), axis=0)
    require(
        all(np.all(slack >= -1e-12) for slack in (value - lower, upper - value, 0.25 + increments, 0.25 - increments)),
        "Original complete command box and rate admission; no clipping",
    )
    return value.copy()
