"""Process-local correction of the execution-verifier source path policy.

The frozen fit contract sends its execution verifier to an outputs-only artifact
reader. This context changes only that callback, after the original immutable
proposal/audit/envelope proof has identified the exact registered Python source.
It grants no execution permission and performs no fitting or array decoding.
"""

import copy
import threading
from contextlib import contextmanager
from pathlib import Path

from s2a import clean_execution_proof as generic
from s2a import clean_training_fit_contract as contract
from s2a.artifacts import REPO

_PROOF = generic.check_execution_proof
_EVIDENCE = contract.Evidence
_METHODS = {name: getattr(_EVIDENCE, name) for name in ("path", "json", "source_map", "finish")}
_LOCK = threading.Lock()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _contract_unchanged():
    _require(
        contract.Evidence is _EVIDENCE
        and generic.check_execution_proof is _PROOF
        and all(getattr(_EVIDENCE, name) is method for name, method in _METHODS.items()),
        "source proof requires the original evidence and generic proof functions",
    )


def _source_ref(ref, registration):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "source needs exact path/SHA reference")
    filename, digest = ref["path"], ref["sha256"]
    _require(isinstance(filename, str), "source path must be a string")
    path = Path(filename)
    _require(
        path.is_absolute()
        and str(path.resolve()) == filename
        and path.is_relative_to(REPO)
        and path.suffix == ".py"
        and path.is_file(),
        "execution verifier must be canonical in-repository Python source",
    )
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        and registration["source_sha256"].get(filename) == digest,
        "execution verifier source SHA is not frozen in this registration",
    )
    return path


def _reader_owner(read_json, check_file):
    owner = getattr(read_json, "__self__", None)
    _require(
        type(owner) is _EVIDENCE
        and getattr(read_json, "__func__", None) is _METHODS["json"]
        and getattr(check_file, "__self__", None) is owner
        and getattr(check_file, "__func__", None) is _METHODS["path"]
        and not any(name in owner.__dict__ for name in _METHODS),
        "source proof requires json/path methods of the same original Evidence instance",
    )
    return owner


class SourceProof:
    """Read-only observations for a separately registered execution wrapper."""

    def __init__(self, registration_ref, verifier_ref):
        self._registration_ref = copy.deepcopy(registration_ref)
        self._verifier_ref = copy.deepcopy(verifier_ref)
        self._count = 0
        self._guard_completed = False

    @property
    def registration_ref(self):
        return copy.deepcopy(self._registration_ref)

    @property
    def source_verifier_ref(self):
        return copy.deepcopy(self._verifier_ref)

    @property
    def callback_corrected_count(self):
        return self._count

    @property
    def source_guard_completed(self):
        return self._guard_completed


@contextmanager
def execution_source_proof(registration_ref):
    """Temporarily correct only the frozen fit module's proof source callback.

    All JSON evidence remains under outputs. Every registered source is hashed at
    entry, before each delegated proof and at exit. The original callback is always
    restored, including when the fit/checker or a source guard raises. Nested and
    concurrent uses in the same process are refused rather than sharing a patch.
    """
    contract.cpu_environment()
    _require(_LOCK.acquire(blocking=False), "execution source proof is already active; nesting is forbidden")
    installed = False
    try:
        _contract_unchanged()
        _require(contract.check_execution_proof is _PROOF, "fit execution proof was already replaced")
        evidence = _EVIDENCE()
        registration_ref = copy.deepcopy(registration_ref)
        registration = evidence.json(registration_ref)
        _require(
            registration.get("schema_version") == contract.REGISTRATION_SCHEMA
            and registration.get("complete") is True
            and registration.get("passed") is True
            and registration.get("execution_authorized") is True
            and registration.get("automatic_retry") is False
            and registration.get("fresh_performance_authorized") is False
            and registration.get("readiness", {}).get("ready_for_execution") is True,
            "source proof requires an exact complete execution registration",
        )
        prepared_ref = registration["prepared_ref"]
        evidence.json(prepared_ref)
        sources = registration["source_sha256"]
        for module in (contract, generic):
            path = str(Path(module.__file__).resolve())
            _require(sources.get(path) == contract.sha256(path), "actual fit/proof source is outside frozen closure")
        evidence.source_map(sources)
        verifier_refs = []

        def initial_source(ref):
            _source_ref(ref, registration)
            evidence.source_map({ref["path"]: ref["sha256"]})
            verifier_refs.append(copy.deepcopy(ref))

        _PROOF(registration, prepared_ref=prepared_ref, read_json=evidence.json, check_file=initial_source)
        _require(len(verifier_refs) == 1, "original execution proof did not identify exactly one verifier")
        verifier_ref = verifier_refs[0]
        evidence.finish()
        state = SourceProof(registration_ref, verifier_ref)

        def corrected(value, *, prepared_ref: dict, read_json, check_file):
            _contract_unchanged()
            contract.cpu_environment()
            _require(
                value == registration and prepared_ref == registration["prepared_ref"],
                "delegated proof differs from the bound registration/preparation",
            )
            owner = _reader_owner(read_json, check_file)
            evidence.finish()
            # Bind the reader to the actual admitted bytes, not merely an equal
            # caller-supplied dictionary. No source path enters Evidence.path.
            _require(owner.json(registration_ref) == registration, "registration bytes differ during delegated proof")
            checked = []

            def source_only(ref):
                _require(ref == verifier_ref and not checked, "source callback is not the unique admitted verifier")
                _source_ref(ref, registration)
                owner.source_map({ref["path"]: ref["sha256"]})
                checked.append(copy.deepcopy(ref))

            verified = _PROOF(value, prepared_ref=prepared_ref, read_json=read_json, check_file=source_only)
            _require(checked == [verifier_ref], "delegated proof skipped its source verification")
            state._count += 1
            return verified

        contract.check_execution_proof = corrected
        installed = True
        try:
            yield state
        finally:
            hook_unchanged = contract.check_execution_proof is corrected
            contract.check_execution_proof = _PROOF
            installed = False
            _require(hook_unchanged, "fit proof hook changed while the source context was active")
            _contract_unchanged()
            evidence.finish()
            contract.cpu_environment()
            state._guard_completed = True
    finally:
        if installed:
            contract.check_execution_proof = _PROOF
        _LOCK.release()
