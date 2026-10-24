"""038 fit artifact/phase admission; no trajectory decoding or numerical fitting.

Only metadata, exact references, source snapshots and normal exit receipts are
shared between the producer and independent checker. Each owns its own label,
normalization, capture-array and row-weight implementation.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_execution_proof import check_execution_proof
from s2a.clean_training_seed_plan import apply_seed_plan
from s2a.relocated_artifacts import RelocatedArtifacts
from s2a.worker_handshake import write_once

PREPARATION_SCHEMA = "s2a.clean-training-preparation.v1"
REGISTRATION_SCHEMA = "s2a.clean-training-execution-registration.v1"
COLLECTION_SCHEMA = "s2a.clean-training-capture.v1"
CAPTURE_AUDIT_SCHEMA = "s2a.clean-training-capture-audit.v1"
FIT_SCHEMA = "s2a.clean-training-fit.v1"
VERIFICATION_SCHEMA = "s2a.clean-training-fit-verification.v1"
FEATURE_COLUMNS = tuple(f"hidden_{i:04d}" for i in range(1280)) + tuple(f"c0_{i:02d}" for i in range(10))
GPU_UUID = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def cpu_environment():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LD_LIBRARY_PATH") == "",
        "038 fit requires S2A cwd and explicitly empty CUDA/LD",
    )
    require(
        all(os.environ.get(name) == "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "038 CPU fits require explicit eight-thread BLAS/OMP limits",
    )


class Evidence:
    def __init__(self):
        self.locator = RelocatedArtifacts(REPO, "outputs/external_deletion_recovery_20261022")
        self.inputs = {}

    def path(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "artifact needs exact path/SHA reference")
        digest = ref["sha256"]
        require(
            isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "artifact SHA256 is malformed",
        )
        path = self.locator.resolve_path(ref["path"])
        require(path.is_relative_to(REPO / "outputs"), "fit artifact escaped S2A outputs")
        require(sha256(path) == digest, f"fit artifact SHA changed: {ref['path']}")
        self.inputs[str(path)] = digest
        return path

    def json(self, ref):
        path = self.path(ref)
        require(path.suffix == ".json", "metadata reader refuses future binary arrays")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "metadata changed while reading")
        return json.loads(raw)

    def source_map(self, mapping, *, snapshots=False):
        require(isinstance(mapping, dict) and mapping, "source/snapshot mapping is missing")
        for filename, expected in mapping.items():
            path = Path(filename).resolve()
            require(path.is_relative_to(REPO) and path.is_file(), "fit source escaped S2A or is absent")
            require(sha256(path) == expected, "fit execution source/snapshot SHA changed")
            self.inputs[str(path)] = expected

    def finish(self):
        require(
            all(sha256(path) == digest for path, digest in self.inputs.items()), "fit input/source changed during run"
        )


def complete(value, schema, message):
    require(
        value.get("schema_version") == schema and value.get("complete") is True and value.get("passed") is True, message
    )


def _normal_exit(evidence, audit):
    launch = evidence.json(audit["worker_launch_ref"])
    exit_record = evidence.json(audit["worker_exit_ref"])
    release = evidence.json(audit["worker_release_ref"])
    gpu = evidence.json(audit["gpu_after_ref"])
    require(
        exit_record.get("actual_wait") is True and exit_record.get("returncode") == 0,
        "complete capture requires actual normal worker wait0",
    )
    require(
        release.get("reaped") is True
        and release.get("returncode") == 0
        and release.get("signal_sent") is False
        and release.get("only_owned_child_signalled") is True
        and release.get("original_error") is None
        and all(launch["child"][key] == release[key] for key in ("pid", "start_ticks")),
        "capture worker was not normally reaped with its owned birth",
    )
    require(
        gpu.get("released") is True
        and gpu.get("memory_mib") == 0
        and gpu.get("compute_apps") == []
        and gpu.get("uuid") == GPU_UUID,
        "capture GPU normal release is missing",
    )


@dataclass
class FitContext:
    prepared_ref: dict
    registration_ref: dict
    prepared: dict
    registration: dict
    stage: str
    rounds: tuple[str, ...]
    collection_refs: tuple[dict, ...]
    capture_audit_refs: tuple[dict, ...]
    collections: tuple[dict, ...]
    evidence: Evidence

    @property
    def expected_rows(self):
        return 420000 * len(self.rounds)


def load_context(prepared_ref, registration_ref, *, stage, collection_refs, capture_audit_refs, caller_sources):
    """Admit all capture exits before returning any opportunity to decode GT."""
    require(stage in ("W1", "W2"), "fit stage must be fixed W1 or W2")
    rounds = ("D0",) if stage == "W1" else ("D0", "D1")
    collection_refs, capture_audit_refs = tuple(collection_refs), tuple(capture_audit_refs)
    require(len(collection_refs) == len(capture_audit_refs) == len(rounds), "fit needs all complete ordered rounds")
    evidence = Evidence()
    prepared, registration = evidence.json(prepared_ref), evidence.json(registration_ref)
    require(prepared.get("schema_version") == PREPARATION_SCHEMA, "038 preparation schema differs")
    require(Path(prepared["root"]).resolve().is_relative_to(REPO / "outputs"), "prepared fit root escaped S2A outputs")
    complete(registration, REGISTRATION_SCHEMA, "final execution registration is incomplete")
    require(
        registration["prepared_ref"] == prepared_ref
        and registration["automatic_retry"] is False
        and registration["readiness"]["ready_for_execution"] is True,
        "metadata-only preparation cannot admit a fit",
    )
    check_execution_proof(registration, prepared_ref=prepared_ref, read_json=evidence.json, check_file=evidence.path)
    seed_ref = registration["training_seed_plan_ref"]
    prepared = apply_seed_plan(prepared, prepared_ref, evidence.json(seed_ref), seed_ref)
    metadata = evidence.json(registration["metadata_verification_ref"])
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("complete") is metadata.get("passed") is metadata.get("metadata_qualified") is True
        and metadata.get("execution_ready") is metadata.get("execution_authorized") is False
        and metadata.get("consumed_before_after_exact") is True
        and metadata["prepared_ref"] == prepared_ref,
        "independent metadata preparation did not complete",
    )
    evidence.source_map(prepared["snapshot_sha256"], snapshots=True)
    evidence.source_map(registration["source_sha256"])
    evidence.source_map(registration["snapshot_sha256"], snapshots=True)
    verifier = metadata["verifier_ref"]
    require(
        set(verifier) == {"path", "sha256"}
        and registration["source_sha256"].get(str(Path(verifier["path"]).resolve())) == verifier["sha256"]
        and sha256(verifier["path"]) == verifier["sha256"],
        "actual independent metadata verifier was not admitted",
    )
    require(
        all(registration["source_sha256"].get(str(Path(path).resolve())) == sha256(path) for path in caller_sources),
        "actual fit/checker/shared source was not admitted",
    )
    require(
        prepared["config"]["ridge"] == 0.1
        and prepared["config"]["feature_terms"] == []
        and prepared["resources"]["maximum_fit_block_rows"] == 1400,
        "fixed ridge/features/block contract changed",
    )
    requests = prepared["training_requests"]
    require(
        len(requests) == 300
        and [r["index"] for r in requests] == list(range(300))
        and len({r["name"] for r in requests}) == len({r["sampling_seed"] for r in requests}) == 300
        and not ({r["sampling_seed"] for r in requests} & {42, 43})
        and len(prepared["training_anchors"]) == 150,
        "complete training-only roster differs",
    )
    collections = []
    for round_name, collection_ref, audit_ref in zip(rounds, collection_refs, capture_audit_refs, strict=True):
        collection, audit = evidence.json(collection_ref), evidence.json(audit_ref)
        complete(collection, COLLECTION_SCHEMA, "training capture did not finish")
        complete(audit, CAPTURE_AUDIT_SCHEMA, "full capture audit did not finish before labels")
        require(
            collection["prepared_ref"] == audit["prepared_ref"] == prepared_ref
            and collection["execution_registration_ref"] == audit["execution_registration_ref"] == registration_ref
            and audit["collection_ref"] == collection_ref
            and collection["round_name"] == audit["round_name"] == round_name,
            "capture stage/preparation/registration identity differs",
        )
        require(
            collection["actual_requests"] == audit["actual_requests"] == 300
            and collection["actual_denoise_calls"] == audit["actual_denoise_calls"] == 10500
            and collection["future_label_reads"] == 0,
            "capture budget or future isolation differs",
        )
        _normal_exit(evidence, audit)
        evidence.source_map(audit["source_sha256"])
        entries = collection["requests"]
        require(len(entries) == 300, "capture request ledger is incomplete")
        for expected, entry in zip(requests, entries, strict=True):
            require(
                all(entry[key] == expected[key] for key in ("index", "name", "anchor_name", "sampling_seed")),
                "capture request order/identity differs from the entire training roster",
            )
            for key in (
                "compact_ledger_ref",
                "initial_ref",
                "request_ref",
                "response_ref",
                "trace_ref",
                "conditioning_ref",
            ):
                evidence.path(entry[key])
        normalization = evidence.json(collection["normalization_ref"])
        require(
            normalization["normalization_id"] == 31 and normalization["projection_id"] == 0,
            "capture normalization identity differs",
        )
        if round_name == "D0":
            require(
                collection["readout_ref"] is None
                and not collection.get("w1_fit_ref")
                and not collection.get("w1_verification_ref"),
                "D0 must use the unadapted raw model",
            )
        else:
            fit, verified = evidence.json(collection["w1_fit_ref"]), evidence.json(collection["w1_verification_ref"])
            complete(fit, FIT_SCHEMA, "D1 lacks its complete fixed W1 fit")
            complete(verified, VERIFICATION_SCHEMA, "D1 lacks its independent W1 verification")
            require(
                fit["stage"] == verified["stage"] == "W1"
                and fit["producer_fits"] == 1
                and verified["fit_ref"] == collection["w1_fit_ref"]
                and fit["params_ref"] == verified["params_ref"] == collection["readout_ref"]
                and fit["prepared_ref"] == verified["prepared_ref"] == prepared_ref
                and fit["execution_registration_ref"] == verified["execution_registration_ref"] == registration_ref
                and fit["collection_refs"] == [collection_refs[0]]
                and fit["capture_audit_refs"] == [capture_audit_refs[0]],
                "D1 W1 fit/parameter lineage differs",
            )
            require(
                collection["normalization_ref"] == collections[0]["normalization_ref"],
                "D0/D1 normalization transform changed",
            )
            evidence.path(collection["readout_ref"])
        collections.append(collection)
    evidence.finish()
    return FitContext(
        prepared_ref,
        registration_ref,
        prepared,
        registration,
        stage,
        rounds,
        collection_refs,
        capture_audit_refs,
        tuple(collections),
        evidence,
    )


def publish(path, value):
    path = Path(path)
    write_once(path, value)
    return reference(path)


def fresh_output(directory, context):
    directory = Path(directory).resolve()
    require(directory.is_relative_to(Path(context.prepared["root"]).resolve()), "fit output escaped registered038 root")
    directory.mkdir(parents=True, exist_ok=False)
    return directory
