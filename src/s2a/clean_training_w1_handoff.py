"""Metadata-only, scoped handoff from qualified045 W1 to the original041 owner.

This module cannot authorize a model launch. The046 execution envelope must do
that separately. It never rewrites the failed041 owner or converts receipts to
an older schema; the actual045 durable receipts are checked in their own form.
"""

import copy
import inspect
import threading
from contextlib import contextmanager
from pathlib import Path

from s2a.durable_stage_supervisor import REPO, check_sources, read, reference, require

PREPARED_ROOT = REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023"
W1_ROOT = PREPARED_ROOT / "runtime_fit_20261024"
OWNER_PATH = PREPARED_ROOT / "source/entrypoints/run_wam_clean_training.py"
ADMITTED_REF = dict(
    path=str(W1_ROOT / "execution/admitted.json"),
    sha256="6eb0d233c05b3859f1bd50ddeaa6679089df0f7e186b9cf85a463030b238436a",
)
TERMINAL_REF = dict(
    path=str(W1_ROOT / "qa/terminal_audit.json"),
    sha256="480154904dac3a9133c095ae7e093e107ba942039bebe0edbe0b89330a556dc2",
)
_LOCK = threading.Lock()


def _complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "incomplete or wrong-schema W1 handoff evidence",
    )


def _owned_role(entry, role, admitted_ref, delegated_ref):
    owner = read(entry["owner_ref"])
    _complete(owner, "s2a.durable-stage-owner.v1")
    require(owner["timed_out"] is False and owner["automatic_retry"] is False, "W1 role timed out or retried")
    launch, exited, released = (read(owner[key]) for key in ("launch_ref", "exit_ref", "release_ref"))
    require(exited["actual_wait"] is True and exited["returncode"] == 0, "W1 role lacks real wait0")
    require(
        released["actual_wait"] is released["reaped"] is True
        and released["returncode"] == 0
        and released["signal_sent"] is released["kill_sent"] is False
        and released["original_error"] is None
        and all(released[key] == launch["child"][key] for key in ("pid", "start_ticks")),
        "W1 role lacks its normally reaped child birth",
    )
    worker = read(owner["stage_result_ref"])
    _complete(worker, "s2a.clean-training-runtime-fit-worker.v1")
    require(
        worker["repair_ref"] == admitted_ref
        and worker["role"] == role
        and worker["delegate_result_ref"] == delegated_ref
        and worker["delegate_calls_attempted"] == 1
        and worker["producer_fits"] == (1 if role == "producer" else 0)
        and worker["independent_reconstructions"] == (1 if role == "independent" else 0)
        and worker["source_before_after_exact"] is True,
        "W1 role/producer provenance differs",
    )
    proof = worker["source_proof"]
    require(
        proof["callback_corrected_count"] == 1 and proof["source_guard_completed"] is True,
        "W1 source-proof callback was not restored",
    )
    require(
        all(
            worker[key] == 0
            for key in ("new_model_initializations", "new_denoise_calls", "new_physics_steps", "new_fresh_cases")
        ),
        "W1 exceeded its CPU-only scope",
    )


def check_handoff(spec):
    """Validate the one fixed045 W1 and original D0, with no array decoding."""
    require(
        spec["w1_admitted_ref"] == ADMITTED_REF and spec["w1_terminal_ref"] == TERMINAL_REF,
        "045 fixed handoff refs differ",
    )
    admitted, terminal = read(ADMITTED_REF), read(TERMINAL_REF)
    _complete(admitted, "s2a.clean-training-runtime-fit-registration.v1")
    require(terminal["complete"] is terminal["passed"] is True, "045 terminal audit failed")
    require(
        terminal["admitted_ref"] == ADMITTED_REF
        and terminal["source_before_after_exact"] is True
        and terminal["source_files"] == terminal["snapshot_files"] == 508
        and terminal["cumulative_w1_attempts"] == 2
        and terminal["new_successful_fits"] == terminal["independent_reconstructions"] == 1,
        "045 terminal scope/count differs",
    )
    for name, terminal_key in (
        ("w1_owner_ref", "supervisor_ref"),
        ("w1_fit_ref", "fit_ref"),
        ("w1_verification_ref", "independent_ref"),
        ("w1_params_ref", "params_ref"),
    ):
        require(spec[name] == terminal[terminal_key], "fixed045 artifact differs: " + name)
    for name in (
        "prepared_ref",
        "execution_registration_ref",
        "d0_owner_ref",
        "d0_collection_ref",
        "d0_capture_audit_ref",
    ):
        require(spec[name] == admitted[name], "original041/D0 lineage differs: " + name)
    require(read(spec["prepared_ref"])["root"] == str(PREPARED_ROOT), "original prepared root differs")
    check_sources(admitted["source_sha256"])
    check_sources(admitted["snapshot_sha256"])
    supervisor = read(spec["w1_owner_ref"])
    _complete(supervisor, "s2a.clean-training-runtime-fit-owner.v1")
    require(
        supervisor["repair_ref"] == ADMITTED_REF
        and supervisor["producer_ref"] == spec["w1_fit_ref"]
        and supervisor["verification_ref"] == spec["w1_verification_ref"]
        and supervisor["d1_authorized"]
        is supervisor["fresh_performance_authorized"]
        is supervisor["automatic_retry"]
        is False,
        "045 supervisor scope or refs differ",
    )
    commands = supervisor["owned_commands"]
    require([entry["role"] for entry in commands] == ["producer", "independent"], "045 role order differs")
    for entry in commands:
        role = entry["role"]
        _owned_role(
            entry, role, ADMITTED_REF, spec["w1_fit_ref"] if role == "producer" else spec["w1_verification_ref"]
        )
    fit, verified = read(spec["w1_fit_ref"]), read(spec["w1_verification_ref"])
    _complete(fit, "s2a.clean-training-fit.v1")
    _complete(verified, "s2a.clean-training-fit-verification.v1")
    require(
        fit["stage"] == verified["stage"] == "W1"
        and fit["prepared_ref"] == verified["prepared_ref"] == spec["prepared_ref"]
        and fit["execution_registration_ref"]
        == verified["execution_registration_ref"]
        == spec["execution_registration_ref"]
        and fit["actual_rows"] == verified["actual_rows"] == 420000
        and fit["fit_attempts"] == fit["producer_fits"] == verified["independent_reconstructions"] == 1
        and fit["rounds"] == ["D0"]
        and fit["collection_refs"] == [spec["d0_collection_ref"]]
        and fit["capture_audit_refs"] == [spec["d0_capture_audit_ref"]]
        and verified["fit_ref"] == spec["w1_fit_ref"]
        and fit["params_ref"] == verified["params_ref"] == spec["w1_params_ref"]
        and verified["full_training_row_reconstruction"] is True
        and verified["rtol"] == 1e-9
        and verified["atol"] == 1e-11,
        "045 full independent reconstruction or lineage differs",
    )
    read(spec["w1_params_ref"])
    d0 = read(spec["d0_owner_ref"])
    _complete(d0, "s2a.clean-training-owner.v1")
    require(
        d0["stage"] == "D0"
        and d0["producer_ref"] == spec["d0_collection_ref"]
        and d0["verification_ref"] == spec["d0_capture_audit_ref"],
        "retained D0 owner differs",
    )
    check_sources(admitted["source_sha256"])
    check_sources(admitted["snapshot_sha256"])
    return supervisor


def _check_owner(owner, spec):
    original = owner.prior
    require(
        Path(owner.__file__).resolve() == OWNER_PATH
        and inspect.isfunction(original)
        and original.__globals__ is vars(owner)
        and Path(inspect.getsourcefile(original)).resolve() == OWNER_PATH
        and original.__module__ == owner.__name__,
        "handoff requires the actual unchanged041 owner/prior",
    )
    old = read(spec["execution_registration_ref"])
    require(old["source_sha256"].get(str(OWNER_PATH)) == reference(OWNER_PATH)["sha256"], "041 owner source changed")
    return original


@contextmanager
def qualified_w1_prior(owner, spec):
    """Temporarily replace only W1 metadata lookup, always restoring the owner."""
    require(_LOCK.acquire(blocking=False), "W1 handoff is already active")
    try:
        frozen = copy.deepcopy(spec)
        original = _check_owner(owner, frozen)
        qualified = check_handoff(frozen)
        receipt = dict(w1_lookups=0, original_lookups=0, restored=False, source_guard_completed=False)

        def lookup(prepared, prepared_ref, registration_ref, stage):
            require(
                prepared_ref == frozen["prepared_ref"]
                and registration_ref == frozen["execution_registration_ref"]
                and prepared["root"] == str(PREPARED_ROOT),
                "handoff lookup belongs to another preparation",
            )
            if stage == "W1":
                require(check_handoff(frozen) == qualified, "045 qualified owner changed during lookup")
                receipt["w1_lookups"] += 1
                return copy.deepcopy(qualified)
            receipt["original_lookups"] += 1
            return original(prepared, prepared_ref, registration_ref, stage)

        owner.prior = lookup
        try:
            yield receipt
        finally:
            intact = owner.prior is lookup
            owner.prior = original
            receipt["restored"] = True
            require(intact, "W1 handoff callback was replaced while active")
            require(spec == frozen, "W1 handoff specification changed during scope")
            _check_owner(owner, frozen)
            check_handoff(frozen)
            receipt["source_guard_completed"] = True
    finally:
        _LOCK.release()
