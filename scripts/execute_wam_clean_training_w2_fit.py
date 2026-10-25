"""047 W2-only CPU wrapper around the unchanged fit and independent QR entries.

The new envelope authorizes this narrowly scoped invocation. Original041 JSON,
fit equations, output names and result payloads are never rewritten. Dry runs
exercise the complete original metadata admission without array decoding.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRATION_SCHEMA = "s2a.clean-training-w2-runtime-fit-registration.v1"
PROPOSAL_SCHEMA = "s2a.clean-training-w2-runtime-fit-proposal.v1"
VERIFICATION_SCHEMA = "s2a.clean-training-w2-runtime-fit-verification.v1"
WORKER_SCHEMA = "s2a.clean-training-w2-runtime-fit-worker.v1"
DRY_SCHEMA = "s2a.clean-training-w2-runtime-fit-dry.v1"
BUDGET = dict(
    producer_fits=1,
    independent_reconstructions=1,
    new_model_initializations=0,
    new_denoise_calls=0,
    new_physics_steps=0,
    new_fresh_cases=0,
    automatic_retry=False,
)
MODULES = {
    "producer": ("fit_wam_clean_training", "run_fit"),
    "independent": ("verify_wam_clean_training_fit", "verify_fit"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def path(value):
    result = Path(value)
    require(
        result.is_absolute() and result.resolve() == result and not result.is_symlink(),
        "canonical absolute nonsymlink path required",
    )
    return result


def cpu_environment():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
        and all(os.environ.get(key) == "8" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "047 needs main S2A cwd, empty CUDA/LD and three CPU thread limits8",
    )


class Evidence:
    def __init__(self):
        self.consumed = {}

    def file(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact path/SHA reference required")
        digest = ref["sha256"]
        require(
            isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "malformed SHA256",
        )
        actual = path(ref["path"])
        require(
            actual.is_relative_to(REPO) and actual.is_file() and sha(actual) == digest,
            "source/artifact SHA or root differs",
        )
        previous = self.consumed.setdefault(str(actual), digest)
        require(previous == digest, "conflicting source/artifact SHA")
        return actual

    def json(self, ref):
        actual = self.file(ref)
        require(actual.suffix == ".json", "metadata reader refuses binary arrays")
        data = actual.read_bytes()
        require(hashlib.sha256(data).hexdigest() == ref["sha256"], "JSON changed during read")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, "duplicate JSON metadata key")
                result[key] = value
            return result

        value = json.loads(
            data,
            object_pairs_hook=unique,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + v)),
        )
        require(isinstance(value, dict), "JSON metadata must be an object")
        return value

    def mapping(self, mapping, *, snapshots=False):
        require(isinstance(mapping, dict) and bool(mapping), "source/snapshot closure is absent")
        for filename, digest in mapping.items():
            actual = self.file(dict(path=filename, sha256=digest))
            if snapshots:
                require(actual.is_relative_to(REPO / "outputs"), "snapshot escaped S2A outputs")

    def finish(self):
        require(all(sha(p) == digest for p, digest in self.consumed.items()), "source or metadata changed during047")


def _complete(value, schema, message):
    require(value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True, message)


def _admit(repair_ref):
    from s2a.clean_training_w2_admission import worker_admit

    return worker_admit(repair_ref, runtime=True)


def admit(repair_registration_ref):
    """CPU/metadata-only047 admission, reusable by the separate owning process."""
    registration, _, _ = _admit(repair_registration_ref)
    return registration


def _load_modules(role, registration):
    name, function_name = MODULES[role]
    delegate = importlib.import_module(name)
    function = getattr(delegate, function_name)
    expected = REPO / "scripts" / (name + ".py")
    require(
        Path(delegate.__file__).resolve() == expected
        and Path(inspect.getsourcefile(function)).resolve() == expected
        and function.__module__ == delegate.__name__,
        "delegate must be the original main-repository function",
    )
    helper = importlib.import_module("s2a.clean_training_source_proof")
    helper_path = REPO / "src/s2a/clean_training_source_proof.py"
    require(
        Path(helper.__file__).resolve() == helper_path
        and Path(inspect.getsourcefile(inspect.unwrap(helper.execution_source_proof))).resolve() == helper_path,
        "source correction helper is not the actual main-repository helper",
    )
    contract = importlib.import_module("s2a.clean_training_fit_contract")
    require(
        Path(contract.__file__).resolve() == REPO / "src/s2a/clean_training_fit_contract.py",
        "shared fit metadata source differs",
    )
    for actual in (expected, helper_path, Path(contract.__file__).resolve()):
        require(registration["source_sha256"].get(str(actual)) == sha(actual), "loaded delegate/helper source drift")
    return function, helper, contract, reference(expected), reference(helper_path)


def _caller_sources(role):
    if role == "producer":
        names = (
            "clean_training_fit_contract.py",
            "clean_training_data.py",
            "wam_clean_streaming.py",
            "wam_clean_readout.py",
            "wam_clean_compact.py",
            "wam_clean_capture.py",
            "geometry.py",
        )
    else:
        names = ("clean_training_fit_contract.py", "streaming_ridge_qr.py", "geometry.py")
    return (REPO / "scripts" / (MODULES[role][0] + ".py"), *(REPO / "src/s2a" / name for name in names))


def _proof_record(proof):
    return dict(
        registration_ref=proof.registration_ref,
        source_verifier_ref=proof.source_verifier_ref,
        callback_corrected_count=proof.callback_corrected_count,
        source_guard_completed=proof.source_guard_completed,
    )


def _check_proof(proof, old_verifier, old_ref):
    require(
        proof.registration_ref == old_ref
        and proof.source_verifier_ref == old_verifier
        and proof.callback_corrected_count == 1
        and proof.source_guard_completed is True,
        "unique original source callback repair or restored before/after guard is incomplete",
    )


def _fit_result(evidence, result_ref, registration):
    require(
        path(result_ref["path"]) == path(registration["fit_output_dir"]) / "result.json",
        "delegate W2 result path differs",
    )
    value = evidence.json(result_ref)
    _complete(value, "s2a.clean-training-fit.v1", "original W2 producer failed")
    require(
        value.get("stage") == "W2"
        and value.get("prepared_ref") == registration["prepared_ref"]
        and value.get("execution_registration_ref") == registration["execution_registration_ref"]
        and value.get("collection_refs") == [registration["d0_collection_ref"], registration["d1_collection_ref"]]
        and value.get("capture_audit_refs")
        == [registration["d0_capture_audit_ref"], registration["d1_capture_audit_ref"]]
        and value.get("rounds") == ["D0", "D1"]
        and value.get("producer_fits") == 1
        and value.get("fit_attempts") == 1
        and value.get("actual_rows") == 840000,
        "original W2 fit lineage/denominator differs",
    )
    evidence.file(value["params_ref"])
    return value


def execute(repair_ref, *, role, dry_run=False):
    require(role in MODULES, "047 role must be producer or independent")
    registration, evidence, old_verifier = _admit(repair_ref)
    delegate, helper, contract, delegate_ref, helper_ref = _load_modules(role, registration)
    fit_path = path(registration["fit_output_dir"]) / "result.json"
    verification_path = path(registration["verification_output"])
    require(not verification_path.exists(), "independent output already exists; no retry")
    if role == "producer":
        require(not path(registration["fit_output_dir"]).exists(), "W2 fit output already exists; no retry")
    if dry_run:
        with helper.execution_source_proof(registration["execution_registration_ref"]) as proof:
            context = contract.load_context(
                registration["prepared_ref"],
                registration["execution_registration_ref"],
                stage="W2",
                collection_refs=[registration["d0_collection_ref"], registration["d1_collection_ref"]],
                capture_audit_refs=[registration["d0_capture_audit_ref"], registration["d1_capture_audit_ref"]],
                caller_sources=_caller_sources(role),
            )
            require(
                context.stage == "W2" and context.rounds == ("D0", "D1") and context.expected_rows == 840000,
                "dry context differs",
            )
            context.evidence.finish()
        _check_proof(proof, old_verifier, registration["execution_registration_ref"])
        evidence.finish()
        return dict(
            schema_version=DRY_SCHEMA,
            complete=True,
            passed=True,
            role=role,
            repair_ref=repair_ref,
            dry_run=True,
            metadata_only=True,
            expected_rows=840000,
            source_proof=_proof_record(proof),
            delegate_source_ref=delegate_ref,
            source_proof_helper_ref=helper_ref,
            runtime=registration["runtime"],
            d1_owner_ref=registration["d1_owner_ref"],
            d1_handoff_owner_ref=registration["d1_handoff_owner_ref"],
            source_before_after_exact=True,
            producer_fits=0,
            independent_reconstructions=0,
            new_model_initializations=0,
            new_denoise_calls=0,
            new_physics_steps=0,
            new_fresh_cases=0,
            automatic_retry=False,
        )
    if role == "independent":
        prior_ref = reference(path(registration["root"]) / "workers/producer/result.json")
        prior = evidence.json(prior_ref)
        _complete(prior, WORKER_SCHEMA, "047 producer wrapper did not complete")
        require(
            prior.get("role") == "producer" and prior.get("repair_ref") == repair_ref,
            "producer wrapper belongs to another047",
        )
        fit_ref = reference(fit_path)
        require(prior.get("delegate_result_ref") == fit_ref, "producer wrapper binds another fit")
        produced = _fit_result(evidence, fit_ref, registration)
    directory = path(registration["root"]) / "workers" / role
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(
        schema_version=WORKER_SCHEMA,
        complete=False,
        passed=False,
        role=role,
        repair_ref=repair_ref,
        prepared_ref=registration["prepared_ref"],
        execution_registration_ref=registration["execution_registration_ref"],
        delegate_source_ref=delegate_ref,
        source_proof_helper_ref=helper_ref,
        runtime=registration["runtime"],
        d1_owner_ref=registration["d1_owner_ref"],
        d1_handoff_owner_ref=registration["d1_handoff_owner_ref"],
        delegate_result_ref=None,
        source_proof=None,
        source_before_after_exact=False,
        producer_fits=0,
        independent_reconstructions=0,
        delegate_calls_attempted=0,
        new_model_initializations=0,
        new_denoise_calls=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        automatic_retry=False,
    )
    proof = None
    error = None
    try:
        with helper.execution_source_proof(registration["execution_registration_ref"]) as proof:
            try:
                result["delegate_calls_attempted"] = 1
                if role == "producer":
                    delegated = delegate(
                        registration["prepared_ref"],
                        registration["execution_registration_ref"],
                        stage="W2",
                        collection_refs=[registration["d0_collection_ref"], registration["d1_collection_ref"]],
                        capture_audit_refs=[registration["d0_capture_audit_ref"], registration["d1_capture_audit_ref"]],
                        output_dir=registration["fit_output_dir"],
                    )
                else:
                    delegated = delegate(
                        registration["prepared_ref"],
                        registration["execution_registration_ref"],
                        fit_ref,
                        stage="W2",
                        output=registration["verification_output"],
                    )
            except BaseException as failure:
                result["delegate_error"] = dict(type=type(failure).__name__, message=str(failure))
                raise
        _check_proof(proof, old_verifier, registration["execution_registration_ref"])
        expected = fit_path if role == "producer" else verification_path
        require(delegated == reference(expected), "delegate did not return its exact original output reference")
        result["delegate_result_ref"] = delegated
        if role == "producer":
            _fit_result(evidence, delegated, registration)
            result["producer_fits"] = 1
        else:
            verified = evidence.json(delegated)
            _complete(verified, "s2a.clean-training-fit-verification.v1", "original independent QR failed")
            require(
                verified.get("stage") == "W2"
                and verified.get("fit_ref") == fit_ref
                and verified.get("prepared_ref") == registration["prepared_ref"]
                and verified.get("execution_registration_ref") == registration["execution_registration_ref"]
                and verified.get("params_ref") == produced["params_ref"]
                and verified.get("actual_rows") == 840000
                and verified.get("independent_reconstructions") == 1
                and verified.get("full_training_row_reconstruction") is True,
                "independent QR lineage/denominator differs",
            )
            result["independent_reconstructions"] = 1
        evidence.finish()
        result.update(complete=True, passed=True, source_before_after_exact=True)
    except BaseException as failure:
        error = failure
        result["error"] = dict(type=type(failure).__name__, message=str(failure))
    finally:
        if proof is not None:
            result["source_proof"] = _proof_record(proof)
        try:
            evidence.finish()
            result["source_before_after_exact"] = True
        except BaseException as failure:
            result.update(
                complete=False,
                passed=False,
                source_before_after_exact=False,
                final_guard_error=dict(type=type(failure).__name__, message=str(failure)),
            )
            if error is None:
                error = failure
        expected = fit_path if role == "producer" else verification_path
        if expected.is_file():
            result["available_delegate_result_ref"] = reference(expected)
        result["guarded_source_and_metadata_sha256"] = dict(evidence.consumed)
        with (directory / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    if error is not None:
        raise error
    require(result["passed"], "047 wrapper failed")
    return reference(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-registration", type=Path, required=True)
    parser.add_argument("--repair-registration-sha256", required=True)
    parser.add_argument("--role", choices=tuple(MODULES), required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    value = execute(
        dict(path=str(args.repair_registration.resolve()), sha256=args.repair_registration_sha256),
        role=args.role,
        dry_run=args.dry_run,
    )
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
