"""048 invoke the unchanged041 STATE owner through an explicit metadata bridge."""

import argparse
import contextlib
import json
import os
import signal
from pathlib import Path

from s2a.infrastructure.lifecycle.durable_stage_supervisor import check_sources, identity, publish, reference, require
from s2a.research.wam.training.clean.clean_training_import_bridge import load_frozen_delegate, metadata_import_scope
from s2a.research.wam.training.clean.clean_training_state_handoff import (
    BRIDGE_DIRECTORY,
    BUNDLE,
    DRY_ENV,
    PREFIX,
    ROOT,
    admit,
    check_lookup_receipts,
    dry_receipts,
    reset_dry_receipts,
)
from s2a.research.wam.training.clean.clean_training_w1_handoff import PREPARED_ROOT
from s2a.research.wam.training.state_stage_checks import check_stage


@contextlib.contextmanager
def dry_environment(enabled):
    old = os.environ.get(DRY_ENV)
    if enabled:
        os.environ[DRY_ENV] = "1"
    else:
        os.environ.pop(DRY_ENV, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DRY_ENV, None)
        else:
            os.environ[DRY_ENV] = old


def invoke(a, registration_ref, *, dry_run):
    reset_dry_receipts()
    with (
        dry_environment(dry_run),
        metadata_import_scope(
            bridge_directory=BRIDGE_DIRECTORY,
            bundle_directory=BUNDLE,
            module_refs=a["module_refs"],
            registration_ref=registration_ref,
        ) as (_, scope),
    ):
        owner = load_frozen_delegate(a["original_owner_ref"], name="_s2a048_original_state_owner")
        if dry_run:
            import sys

            import numpy as np

            attempts = []
            old_load = np.load

            def refuse_arrays(*args, **kwargs):
                attempts.append(True)
                raise AssertionError("048 dry cannot decode arrays")

            np.load = refuse_arrays
            try:
                planned = owner.execute(a["prepared_ref"], a["execution_registration_ref"], dry_run=True)
                prepared, _, _ = owner.admit_execution(
                    a["prepared_ref"], a["execution_registration_ref"], required_sources=[owner.__file__]
                )
                sampler = load_frozen_delegate(a["original_sampler_ref"], name="_s2a048_original_sampler_metadata")
                head, params, _, _, entries = sampler.admit_sampling(
                    prepared,
                    a["prepared_ref"],
                    a["execution_registration_ref"],
                    a["w2_fit_ref"],
                    a["w2_verification_ref"],
                )
                require(
                    len(head.center) == 1290 and params == a["w2_params_ref"] and len(entries) == 99,
                    "048 actual sampler metadata admission differs",
                )
                require(
                    not attempts and "torch" not in sys.modules and "mujoco" not in sys.modules,
                    "048 dry imported model/physics or decoded arrays",
                )
            finally:
                np.load = old_load
            receipts = dry_receipts()
            require(
                len(receipts) == 2 and all(v["dry_run"] for v in receipts), "048 owner/sampler dry lookups incomplete"
            )
            result = dict(
                schema_version=PREFIX + "dry.v1",
                complete=True,
                passed=True,
                registration_ref=registration_ref,
                planned=planned,
                dry_lookup_receipts=receipts,
                import_scope=scope,
                actual_model_calls=0,
                actual_array_decode_attempts=0,
                actual_fit_calls=0,
            )
        else:
            previous = {sig: signal.signal(sig, owner.interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                result = owner.execute(a["prepared_ref"], a["execution_registration_ref"])
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    require(scope["restored"] is scope["source_guard_completed"] is True, "048 import scope did not restore")
    return result, scope


def execute(registration_ref, *, dry_run=False):
    a = admit(registration_ref, role="owner")
    for path in (
        PREPARED_ROOT / "state_sampling",
        PREPARED_ROOT / "orchestration/STATE",
        PREPARED_ROOT / "review/STATE_verified.json",
        ROOT / "worker",
        ROOT / "role_receipts",
    ):
        require(not path.exists(), "048 outputs exist; no retry")
    if dry_run:
        result, _ = invoke(a, registration_ref, dry_run=True)
        check_sources(a["source_sha256"])
        check_sources(a["snapshot_sha256"])
        return result
    directory = ROOT / "worker"
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=PREFIX + "worker.v1",
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        identity=identity(os.getpid()),
        w2_owner_ref=a["w2_owner_ref"],
        w2_params_ref=a["w2_params_ref"],
        delegate_source_ref=a["original_owner_ref"],
        delegate_calls_attempted=0,
        import_scope=None,
        source_before_after_exact=False,
        automatic_retry=False,
        fresh_performance_authorized=False,
    )
    error = None
    try:
        result["delegate_calls_attempted"] = 1
        stage_ref, scope = invoke(a, registration_ref, dry_run=False)
        result["import_scope"] = scope
        stage = check_stage(a, stage_ref)
        result.update(
            delegate_result_ref=stage_ref,
            producer_ref=stage["producer_ref"],
            verification_ref=stage["verification_ref"],
            lookup_refs=check_lookup_receipts(a, registration_ref, stage),
            state_gate_passed=stage["state_gate_passed"],
            complete=True,
            passed=True,
        )
    except BaseException as failure:
        error = failure
        result["error"] = dict(type=type(failure).__name__, message=str(failure))
    finally:
        try:
            check_sources(a["source_sha256"])
            check_sources(a["snapshot_sha256"])
            result["source_before_after_exact"] = True
        except BaseException as failure:
            result.update(complete=False, passed=False, source_guard_error=str(failure))
            if error is None:
                error = failure
        for key, path in (
            ("available_stage_ref", PREPARED_ROOT / "orchestration/STATE/result.json"),
            ("available_sampling_ref", PREPARED_ROOT / "state_sampling/result.json"),
        ):
            if path.is_file():
                result[key] = reference(path)
        publish(directory / "result.json", result)
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
