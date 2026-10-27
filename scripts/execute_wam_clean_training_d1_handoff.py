"""046 one-shot D1 owner wrapper; original041 collection and audit stay intact."""

import argparse
import importlib.util
import json
import signal
import sys
from pathlib import Path

from s2a.clean_training_d1_admission import PREFIX, admit
from s2a.clean_training_w1_handoff import OWNER_PATH, PREPARED_ROOT, qualified_w1_prior
from s2a.durable_stage_supervisor import check_sources, publish, read, reference, require


def load_original(path, name):
    bundle = OWNER_PATH.parent
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(bundle))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    require(Path(module.__file__).resolve() == path, "actual041 delegate path differs")
    runtime = sys.modules["continuation_runtime"]
    require(
        Path(runtime.__file__).resolve() == bundle / "continuation_runtime.py", "actual041 metadata runtime differs"
    )
    return module


def check_stage(registration, stage_ref):
    require(stage_ref == reference(PREPARED_ROOT / "orchestration/D1/result.json"), "original D1 owner output differs")
    owner = read(stage_ref)
    require(
        owner["complete"] is owner["passed"] is True
        and owner["stage"] == "D1"
        and owner["prepared_ref"] == registration["prepared_ref"]
        and owner["execution_registration_ref"] == registration["execution_registration_ref"],
        "original D1 owner did not complete",
    )
    producer, verified = read(owner["producer_ref"]), read(owner["verification_ref"])
    require(
        producer["complete"] is producer["passed"] is verified["complete"] is verified["passed"] is True,
        "D1 collection/audit did not complete",
    )
    require(
        producer["schema_version"] == "s2a.clean-training-capture.v1"
        and verified["schema_version"] == "s2a.clean-training-capture-audit.v1"
        and producer["round_name"] == verified["round_name"] == "D1"
        and producer["prepared_ref"] == verified["prepared_ref"] == registration["prepared_ref"]
        and producer["execution_registration_ref"]
        == verified["execution_registration_ref"]
        == registration["execution_registration_ref"]
        and producer["d0_audit_ref"] == verified["d0_audit_ref"] == registration["d0_capture_audit_ref"],
        "D1 capture/audit stage or input lineage differs",
    )
    require(
        producer["round_name"] == "D1"
        and producer["actual_requests"] == 300
        and producer["actual_denoise_calls"] == producer["actual_head_calls"] == 10500
        and producer["model_initialization_attempts"] == producer["model_initializations"] == 1
        and producer["future_label_reads"]
        == producer["producer_fits"]
        == producer["physics_steps"]
        == producer["fresh_cases"]
        == 0,
        "D1 actual budget differs",
    )
    require(
        producer["w1_fit_ref"] == registration["w1_fit_ref"]
        and producer["w1_verification_ref"] == registration["w1_verification_ref"]
        and producer["readout_ref"] == registration["w1_params_ref"],
        "D1 used another W1",
    )
    require(
        verified["collection_ref"] == owner["producer_ref"]
        and verified["actual_requests"] == 300
        and verified["actual_denoise_calls"] == verified["actual_online_points_verified"] == 10500,
        "D1 full independent audit differs",
    )
    return owner


def execute(registration_ref, *, dry_run=False):
    registration = admit(registration_ref)
    root = Path(registration["root"])
    for path in (
        PREPARED_ROOT / "capture/D1",
        PREPARED_ROOT / "orchestration/D1",
        PREPARED_ROOT / "review/D1_verified.json",
        root / "worker",
    ):
        require(not path.exists(), "D1 output exists; no retry")
    owner = load_original(OWNER_PATH, "_s2a046_original_owner")
    collector = load_original(OWNER_PATH.parent / "collect_wam_clean_training.py", "_s2a046_collector_admission")
    prepared, _, _ = collector.admit_execution(
        registration["prepared_ref"], registration["execution_registration_ref"], required_sources=[collector.__file__]
    )
    head, params, _ = collector.admit_round(
        prepared,
        round_name="D1",
        d0_audit_ref=registration["d0_capture_audit_ref"],
        w1_fit_ref=registration["w1_fit_ref"],
        w1_verification_ref=registration["w1_verification_ref"],
        prepared_ref=registration["prepared_ref"],
        registration_ref=registration["execution_registration_ref"],
    )
    require(len(head.center) == 1290 and params == registration["w1_params_ref"], "D1 original head admission differs")
    require(
        "torch" not in sys.modules and "mujoco" not in sys.modules,
        "046 metadata preflight imported a model/physics module",
    )
    if dry_run:
        with qualified_w1_prior(owner, registration) as handoff:
            planned = owner.execute(
                registration["prepared_ref"], registration["execution_registration_ref"], stage="D1", dry_run=True
            )
        require(
            handoff["w1_lookups"] == 1 and handoff["restored"] is handoff["source_guard_completed"] is True,
            "046 dry handoff incomplete",
        )
        require(
            Path(planned["worker"][1]).resolve() == Path(registration["collector_ref"]["path"]),
            "D1 planned collector differs",
        )
        check_sources(registration["source_sha256"])
        return dict(
            schema_version=PREFIX + "dry.v1",
            complete=True,
            passed=True,
            registration_ref=registration_ref,
            planned=planned,
            handoff=handoff,
            actual_model_calls=0,
            actual_array_decodes=0,
            actual_fit_calls=0,
        )
    directory = root / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=PREFIX + "worker.v1",
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        prepared_ref=registration["prepared_ref"],
        execution_registration_ref=registration["execution_registration_ref"],
        w1_owner_ref=registration["w1_owner_ref"],
        w1_params_ref=registration["w1_params_ref"],
        delegate_source_ref=registration["original_owner_ref"],
        delegate_calls_attempted=0,
        handoff=None,
        source_before_after_exact=False,
        automatic_retry=False,
        d1_only=True,
        fresh_performance_authorized=False,
    )
    error = None
    previous = {sig: signal.signal(sig, owner.interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with qualified_w1_prior(owner, registration) as handoff:
            result["delegate_calls_attempted"] = 1
            stage_ref = owner.execute(
                registration["prepared_ref"], registration["execution_registration_ref"], stage="D1"
            )
        result["handoff"] = dict(handoff)
        require(
            handoff["w1_lookups"] == 1 and handoff["restored"] is handoff["source_guard_completed"] is True,
            "046 handoff was not uniquely used/restored",
        )
        stage = check_stage(registration, stage_ref)
        result.update(
            delegate_result_ref=stage_ref,
            collection_ref=stage["producer_ref"],
            verification_ref=stage["verification_ref"],
            complete=True,
            passed=True,
        )
    except BaseException as failure:
        error = failure
        result["error"] = dict(type=type(failure).__name__, message=str(failure))
    finally:
        if "handoff" in locals():
            result["handoff"] = dict(handoff)
        try:
            check_sources(registration["source_sha256"])
            check_sources(registration["snapshot_sha256"])
            result["source_before_after_exact"] = True
        except BaseException as failure:
            result.update(complete=False, passed=False, source_guard_error=str(failure))
            if error is None:
                error = failure
        for name, leaf in (
            ("available_stage_ref", PREPARED_ROOT / "orchestration/D1/result.json"),
            ("available_capture_ref", PREPARED_ROOT / "capture/D1/result.json"),
        ):
            if leaf.is_file():
                result[name] = reference(leaf)
        try:
            publish(directory / "result.json", result)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if error is not None:
        raise error
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            execute(dict(path=str(args.registration.resolve()), sha256=args.registration_sha256), dry_run=args.dry_run)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
