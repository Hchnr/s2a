"""043 W1-only CPU wrapper around the unchanged fit and independent QR entries.

The new envelope authorizes this narrowly scoped invocation. Original041 JSON,
fit equations, output names and result payloads are never rewritten. Dry runs
exercise the complete original metadata admission without array decoding.
"""

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRATION_SCHEMA = "s2a.clean-training-postcapture-registration.v1"
PROPOSAL_SCHEMA = "s2a.clean-training-postcapture-proposal.v1"
VERIFICATION_SCHEMA = "s2a.clean-training-postcapture-verification.v1"
WORKER_SCHEMA = "s2a.clean-training-postcapture-worker.v1"
DRY_SCHEMA = "s2a.clean-training-postcapture-dry.v1"
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
        "043 needs main S2A cwd, empty CUDA/LD and three CPU thread limits8",
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
        require(all(sha(p) == digest for p, digest in self.consumed.items()), "source or metadata changed during043")


def _complete(value, schema, message):
    require(value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True, message)


def _admit(repair_ref):
    cpu_environment()
    evidence = Evidence()
    registration = evidence.json(repair_ref)
    _complete(registration, REGISTRATION_SCHEMA, "new043 registration is required;041 cannot authorize this repair")
    require(registration.get("execution_authorized") is True, "043 execution is not authorized")
    root = path(registration["root"])
    require(root.is_relative_to(REPO / "outputs"), "043 output root escaped S2A outputs")
    require(path(repair_ref["path"]).is_relative_to(root), "043 registration is outside its declared root")
    proposal_ref = registration["proposal_ref"]
    independent_ref = registration["independent_verification_ref"]
    proposal, independent = evidence.json(proposal_ref), evidence.json(independent_ref)
    _complete(proposal, PROPOSAL_SCHEMA, "043 proposal is incomplete")
    _complete(independent, VERIFICATION_SCHEMA, "043 independent verification is incomplete")
    require(proposal.get("execution_authorized") is False, "proposal must remain unauthorized")
    expected = copy.deepcopy(proposal)
    expected.update(
        schema_version=REGISTRATION_SCHEMA,
        execution_authorized=True,
        proposal_ref=proposal_ref,
        independent_verification_ref=independent_ref,
    )
    require(canonical(expected) == canonical(registration), "043 registration differs from its exact proposal envelope")
    require(
        independent.get("eligible_for_execution") is True and independent.get("proposal_ref") == proposal_ref,
        "043 independent verification does not qualify this exact proposal",
    )
    require(canonical(registration["budget"]) == canonical(BUDGET), "043 finite budget differs")
    require(registration.get("automatic_retry", False) is False, "043 retry is forbidden")
    sources, snapshots = registration["source_sha256"], registration["snapshot_sha256"]
    evidence.mapping(sources)
    evidence.mapping(snapshots, snapshots=True)
    require(set(sources.values()) <= set(snapshots.values()), "043 snapshots do not cover every frozen source content")
    verifier = independent["verifier_ref"]
    evidence.file(verifier)
    require(sources.get(verifier["path"]) == verifier["sha256"], "043 independent verifier source is not frozen")
    for relative in (
        "scripts/execute_wam_clean_training_postcapture_fit.py",
        "tests/test_clean_training_postcapture_fit.py",
        "src/s2a/clean_training_source_proof.py",
        "scripts/fit_wam_clean_training.py",
        "scripts/verify_wam_clean_training_fit.py",
    ):
        actual = REPO / relative
        require(sources.get(str(actual)) == sha(actual), "actual wrapper/helper/delegate/test source is not043-frozen")
    prepared = evidence.json(registration["prepared_ref"])
    old = evidence.json(registration["execution_registration_ref"])
    _complete(old, "s2a.clean-training-execution-registration.v1", "original041 registration is incomplete")
    require(
        old["prepared_ref"] == registration["prepared_ref"]
        and old.get("execution_authorized") is True
        and old.get("automatic_retry") is False
        and old.get("fresh_performance_authorized") is False,
        "original041 preparation/authorization differs",
    )
    old_sources = old["source_sha256"]
    require(len(old_sources) == 484, "original041 must retain all484 frozen sources")
    require(
        all(sources.get(p) == digest for p, digest in old_sources.items()),
        "043 changed or omitted an original041 source",
    )
    evidence.mapping(old_sources)
    old_verified = evidence.json(old["execution_verification_ref"])
    old_verifier = old_verified["verifier_ref"]
    evidence.file(old_verifier)
    require(
        old_sources.get(old_verifier["path"]) == old_verifier["sha256"], "original041 source verifier binding differs"
    )
    prepared_root = path(prepared["root"])
    require(prepared_root.is_relative_to(REPO / "outputs"), "original prepared root escaped outputs")
    require(path(registration["fit_output_dir"]) == prepared_root / "fit/W1", "W1 output path differs")
    require(
        path(registration["verification_output"]) == prepared_root / "review/W1_verified.json",
        "W1 audit output path differs",
    )
    for key, relative in (
        ("d0_collection_ref", "capture/D0/result.json"),
        ("d0_capture_audit_ref", "review/D0_verified.json"),
        ("d0_owner_ref", "orchestration/D0/result.json"),
    ):
        require(path(registration[key]["path"]) == prepared_root / relative, "D0 evidence escaped its original stage")
    owner = evidence.json(registration["d0_owner_ref"])
    _complete(owner, "s2a.clean-training-owner.v1", "D0 owner did not complete")
    require(
        owner.get("stage") == "D0"
        and owner.get("prepared_ref") == registration["prepared_ref"]
        and owner.get("execution_registration_ref") == registration["execution_registration_ref"]
        and owner.get("producer_ref") == registration["d0_collection_ref"]
        and owner.get("verification_ref") == registration["d0_capture_audit_ref"],
        "D0 owner does not bind the fixed collection/audit",
    )
    for key, schema in (
        ("d0_collection_ref", "s2a.clean-training-capture.v1"),
        ("d0_capture_audit_ref", "s2a.clean-training-capture-audit.v1"),
    ):
        value = evidence.json(registration[key])
        _complete(value, schema, "fixed D0 capture or audit is incomplete")
        require(
            value.get("round_name") == "D0"
            and value.get("prepared_ref") == registration["prepared_ref"]
            and value.get("execution_registration_ref") == registration["execution_registration_ref"],
            "D0 capture/audit metadata lineage differs",
        )
        if key == "d0_capture_audit_ref":
            require(
                value.get("collection_ref") == registration["d0_collection_ref"]
                and value.get("actual_requests") == 300
                and value.get("actual_denoise_calls") == 10500
                and value.get("actual_online_points_verified") == 10500,
                "D0 independent audit denominator differs",
            )
    failure = evidence.json(registration["failure_ref"])
    _complete(failure, "s2a.clean-training-w1-pre-fit-failure-audit.v1", "original failure audit is incomplete")
    require(
        failure.get("failure_before_load_context_return") is True
        and failure.get("producer_fits")
        == failure.get("independent_reconstructions")
        == failure.get("future_array_decodes")
        == 0
        and failure.get("new_model_calls") == 0
        and failure.get("d1_started") is False
        and failure.get("original_failure_preserved") is True
        and failure.get("source_before_after_exact") is True,
        "original failure is not proven pre-fit/pre-decode",
    )
    evidence.finish()
    return registration, evidence, old_verifier


def admit(repair_registration_ref):
    """CPU/metadata-only043 admission, reusable by the separate owning process."""
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
        "delegate W1 result path differs",
    )
    value = evidence.json(result_ref)
    _complete(value, "s2a.clean-training-fit.v1", "original W1 producer failed")
    require(
        value.get("stage") == "W1"
        and value.get("prepared_ref") == registration["prepared_ref"]
        and value.get("execution_registration_ref") == registration["execution_registration_ref"]
        and value.get("collection_refs") == [registration["d0_collection_ref"]]
        and value.get("capture_audit_refs") == [registration["d0_capture_audit_ref"]]
        and value.get("rounds") == ["D0"]
        and value.get("producer_fits") == 1
        and value.get("fit_attempts") == 1
        and value.get("actual_rows") == 420000,
        "original W1 fit lineage/denominator differs",
    )
    evidence.file(value["params_ref"])
    return value


def execute(repair_ref, *, role, dry_run=False):
    require(role in MODULES, "043 role must be producer or independent")
    registration, evidence, old_verifier = _admit(repair_ref)
    delegate, helper, contract, delegate_ref, helper_ref = _load_modules(role, registration)
    fit_path = path(registration["fit_output_dir"]) / "result.json"
    verification_path = path(registration["verification_output"])
    require(not verification_path.exists(), "independent output already exists; no retry")
    if role == "producer":
        require(not path(registration["fit_output_dir"]).exists(), "W1 fit output already exists; no retry")
    if dry_run:
        with helper.execution_source_proof(registration["execution_registration_ref"]) as proof:
            context = contract.load_context(
                registration["prepared_ref"],
                registration["execution_registration_ref"],
                stage="W1",
                collection_refs=[registration["d0_collection_ref"]],
                capture_audit_refs=[registration["d0_capture_audit_ref"]],
                caller_sources=_caller_sources(role),
            )
            require(
                context.stage == "W1" and context.rounds == ("D0",) and context.expected_rows == 420000,
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
            expected_rows=420000,
            source_proof=_proof_record(proof),
            delegate_source_ref=delegate_ref,
            source_proof_helper_ref=helper_ref,
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
        _complete(prior, WORKER_SCHEMA, "043 producer wrapper did not complete")
        require(
            prior.get("role") == "producer" and prior.get("repair_ref") == repair_ref,
            "producer wrapper belongs to another043",
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
                        stage="W1",
                        collection_refs=[registration["d0_collection_ref"]],
                        capture_audit_refs=[registration["d0_capture_audit_ref"]],
                        output_dir=registration["fit_output_dir"],
                    )
                else:
                    delegated = delegate(
                        registration["prepared_ref"],
                        registration["execution_registration_ref"],
                        fit_ref,
                        stage="W1",
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
                verified.get("stage") == "W1"
                and verified.get("fit_ref") == fit_ref
                and verified.get("prepared_ref") == registration["prepared_ref"]
                and verified.get("execution_registration_ref") == registration["execution_registration_ref"]
                and verified.get("params_ref") == produced["params_ref"]
                and verified.get("actual_rows") == 420000
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
    require(result["passed"], "043 wrapper failed")
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
