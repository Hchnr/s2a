"""Independently qualify a closed040 execution proposal without executing it.

Only JSON, source bytes, safe YAML, OS resource metadata and opaque file hashes
are consumed. No producer validators, array readers, models or physics imports
are used. The successful receipt qualifies the proposal for a separate admitted
envelope; it does not launch a worker or grant fresh performance permission.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIER = Path(__file__).resolve()
OLD_ROOT = Path("/share/project/eai_pwm/home/hcr/repos/s2a")
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
GPU_PCI = "00000000:BA:00.0"
GPU_MINOR = 6
SEED_DOMAIN = "s2a.clean-training-request-seed31.v1"
SEED_ALGORITHM = (
    "payload=json.dumps([DOMAIN,anchor_name,root_seed,original_sampling_seed],"
    "ensure_ascii=True,separators=(',',':')).encode(); "
    "sampling_seed=int.from_bytes(SHA256(payload).digest()[:4],'big') & ((1<<31)-1); "
    "DOMAIN='s2a.clean-training-request-seed31.v1'; reject collisions or42/43; no retry"
)
DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
DISK_REQUIRED = 200 * 1024**3
RAM_REQUIRED = 128 * 1024**3
BUDGET = dict(
    raw_training_requests=300,
    w1_training_requests=300,
    state_zero_requests=3,
    state_paired_requests=96,
    training_and_state_requests=699,
    training_and_state_denoise=24465,
    development_requests=900,
    development_denoise=31500,
    total_requests=1599,
    total_denoise=55965,
    model_initializations=4,
    producer_fits=2,
    independent_reconstructions=2,
    development_physics_steps=7200,
    new_teacher_calls=0,
    future_rgb_frames=0,
    fresh_cases=0,
)
RESOURCES = dict(
    d0_timeout_s=86400,
    d1_timeout_s=86400,
    state_timeout_s=43200,
    development_timeout_s=172800,
    fit_timeout_s=7200,
    independent_fit_timeout_s=7200,
    audit_timeout_s=1800,
    http_timeout_s=300,
    gpu_release_timeout_s=120,
    blas_threads=8,
    maximum_fit_block_rows=1400,
)
STAGES = dict(
    D0="scripts/collect_wam_clean_training.py",
    W1="scripts/fit_wam_clean_training.py",
    D1="scripts/collect_wam_clean_training.py",
    W2="scripts/fit_wam_clean_training.py",
    STATE="scripts/sample_wam_clean_training_state.py",
    DEVELOPMENT="scripts/run_wam_clean_training_development.py",
)
REQUIRED_TESTS = (
    "tests/test_clean_training_preparation.py",
    "tests/test_clean_training_preparation_verification.py",
    "tests/test_clean_training_data.py",
    "tests/test_streaming_ridge_qr.py",
    "tests/test_wam_clean_compact.py",
    "tests/test_clean_training_collection.py",
    "tests/test_clean_training_capture_verification.py",
    "tests/test_clean_training_fit.py",
    "tests/test_clean_training_owner.py",
    "tests/test_clean_training_state_sampling.py",
    "tests/test_clean_training_state_verification.py",
    "tests/test_clean_training_state_owner.py",
    "tests/test_clean_training_development.py",
    "tests/test_clean_training_service.py",
    "tests/test_clean_execution_proof.py",
    "tests/test_clean_training_execution_preparation.py",
    "tests/test_clean_training_execution_verification.py",
    "tests/test_native_environment.py",
    "tests/test_worker_handshake.py",
)
REQUIRED_SOURCES = (
    "scripts/prepare_wam_clean_training.py",
    "scripts/verify_wam_clean_training_preparation.py",
    "scripts/collect_wam_clean_training.py",
    "scripts/verify_wam_clean_training_capture.py",
    "scripts/fit_wam_clean_training.py",
    "scripts/verify_wam_clean_training_fit.py",
    "scripts/run_wam_clean_training.py",
    "scripts/sample_wam_clean_training_state.py",
    "scripts/run_wam_clean_training_state.py",
    "scripts/verify_wam_clean_training_state.py",
    "scripts/run_wam_clean_training_development.py",
    "scripts/serve_wam_clean_training.py",
    "scripts/execute_wam_clean_training_case.py",
    "scripts/verify_wam_clean_training_development.py",
    "scripts/prepare_wam_clean_training_execution.py",
    "scripts/verify_wam_clean_training_execution.py",
    "src/s2a/clean_training_runtime.py",
    "src/s2a/clean_training_fit_contract.py",
    "src/s2a/clean_training_data.py",
    "src/s2a/streaming_ridge_qr.py",
    "src/s2a/wam_clean_compact.py",
    "src/s2a/wam_clean_capture.py",
    "src/s2a/wam_clean_readout.py",
    "src/s2a/wam_clean_streaming.py",
    "src/s2a/native_environment.py",
    "src/s2a/worker_handshake.py",
    "src/s2a/relocated_artifacts.py",
    "src/s2a/clean_execution_proof.py",
    "src/s2a/clean_training_seed_plan.py",
    "configs/wam_clean_training_execution_20261023.json",
    "plans/v0.0.1/040_clean_training_execution_20261023.md",
    *REQUIRED_TESTS,
)
THREADS = dict(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
EXTRA_SOURCES = (
    "configs/wam_clean_training_20261023.json",
    "plans/v0.0.1/038_clean_residual_training_20261023.md",
    "configs/wam_clean_training_execution_20261023.json",
    "plans/v0.0.1/040_clean_training_execution_20261023.md",
    "pyproject.toml",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def resolve(path):
    value = Path(path)
    require(value.is_absolute() and ".." not in value.parts, "absolute safe artifact path required")
    if value.is_relative_to(OLD_ROOT):
        relative = value.relative_to(OLD_ROOT)
        candidates = [REPO / relative]
        if relative.parts[:2] == ("outputs", "wam_stability_20261021"):
            candidates.insert(0, REPO / "outputs/external_deletion_recovery_20261022" / Path(*relative.parts[2:]))
    else:
        candidates = [value]
    for candidate in candidates:
        actual = candidate.resolve()
        require(not actual.is_relative_to(OLD_ROOT), "historical deleted workspace cannot be consumed")
        if actual.is_file():
            return actual
    raise FileNotFoundError(f"safe artifact is missing: {path}")


class Evidence:
    def __init__(self):
        self.consumed = {}

    def check(self, ref):
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact path/SHA reference required")
        require(valid_sha(ref["sha256"]), "invalid SHA256")
        path = resolve(ref["path"])
        require(sha(path) == ref["sha256"], f"artifact SHA differs: {path}")
        self.consumed[str(path)] = ref["sha256"]
        return path

    def json(self, ref):
        path = self.check(ref)
        require(path.suffix == ".json", "JSON parser refuses opaque binary/future archives")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "JSON changed while reading")
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def mapping(self, values):
        require(isinstance(values, dict) and bool(values), "empty source/input/snapshot closure")
        for path, digest in values.items():
            self.check(dict(path=path, sha256=digest))

    def unchanged(self):
        require(
            all(sha(path) == digest for path, digest in self.consumed.items()), "consumed files changed during audit"
        )


def case_order():
    return [
        dict(
            case_index=t * 4 + (seed - 42) * 2 + (variant == "w2"),
            case_id=f"task{t}_init0_s{seed}_{variant}",
            task_index=t,
            record_name=f"task{t}_init0",
            sampling_seed=seed,
            variant=variant,
            expected_requests=75,
        )
        for t in range(3)
        for seed in (42, 43)
        for variant in ("raw", "w2")
    ]


def proposal_contract(proposal, prepared):
    require(
        proposal.get("schema_version") == "s2a.clean-training-execution-proposal.v1",
        "execution proposal schema differs",
    )
    require(
        proposal.get("complete") is True
        and proposal.get("passed") is True
        and proposal.get("readiness", {}).get("ready_for_execution") is False
        and proposal.get("execution_authorized") is False,
        "proposal must remain unadmitted",
    )
    require(
        proposal.get("automatic_retry") is False and proposal.get("fresh_performance_authorized") is False,
        "automatic retry/fresh performance must remain forbidden",
    )
    require(proposal.get("budget") == prepared.get("budget") == BUDGET, "fixed complete1599/55965 budget differs")
    require(
        proposal.get("resources") == prepared.get("resources") == RESOURCES,
        "fixed resource/time/thread contract differs",
    )
    require(
        proposal.get("master_port") == 31338
        and proposal.get("development") == dict(port=31339, case_order=case_order()),
        "ports or fixed12 development cases differ",
    )
    require(proposal.get("stage_entrypoints") == STAGES, "fixed stage entrypoint mapping differs")
    require(
        prepared.get("schema_version") == "s2a.clean-training-preparation.v1"
        and prepared.get("execution_authorized") is False
        and prepared.get("readiness", {}).get("ready_for_execution") is False
        and prepared.get("config", {}).get("gpu_uuid") == GPU,
        "metadata preparation was changed or unauthorized",
    )


def sources(proposal, prepared, evidence):
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        evidence.mapping(proposal[key])
        evidence.mapping(prepared[key])
    evidence.mapping(prepared["qualified_model_source_sha256"])
    current = proposal["source_sha256"]
    expected = {str(path.resolve()) for folder in ("src", "scripts", "tests") for path in (REPO / folder).rglob("*.py")}
    expected.update(str((REPO / name).resolve()) for name in EXTRA_SOURCES)
    require(set(current) == expected, "complete src/scripts/tests Python and fixed metadata source roster differs")
    for path in REQUIRED_SOURCES:
        absolute = str((REPO / path).resolve())
        require(current.get(absolute) == sha(absolute), f"required final source is not frozen: {path}")
    require(current.get(str(VERIFIER)) == sha(VERIFIER), "independent execution verifier is absent from source closure")
    require(set(proposal["entrypoint_refs"]) == set(STAGES), "entrypoint reference stages differ")
    for stage, relative in STAGES.items():
        ref = proposal["entrypoint_refs"][stage]
        require(
            ref == dict(path=str((REPO / relative).resolve()), sha256=current[str((REPO / relative).resolve())]),
            "entrypoint path/SHA does not bind its registered source",
        )
        evidence.check(ref)
    snapshots = proposal["snapshot_refs"]
    require(isinstance(snapshots, dict) and set(snapshots) == set(current), "source/snapshot one-to-one roster differs")
    declared_snapshots = {}
    for path, digest in current.items():
        source = Path(path).resolve()
        require(
            source.is_relative_to(REPO) and not source.is_relative_to(REPO / "outputs"),
            "current executable source must be a repository source; external dependencies are inputs",
        )
        snapshot = snapshots[path]
        actual = evidence.check(snapshot)
        relative = source.relative_to(REPO)
        require(
            snapshot["sha256"] == digest
            and actual == (Path(prepared["root"]) / "execution/source" / relative).resolve()
            and actual.is_relative_to(REPO / "outputs"),
            "snapshot differs from corresponding source",
        )
        require(str(actual) not in declared_snapshots, "two sources share one snapshot")
        declared_snapshots[str(actual)] = digest
    require(declared_snapshots == proposal["snapshot_sha256"], "snapshot map omits/adds a source snapshot")
    for path, digest in prepared["source_sha256"].items():
        require(current.get(path) == digest, "metadata preparation source changed after its independent qualification")
    return dict(required_sources=len(REQUIRED_SOURCES), final_sources=len(current), source_snapshots=len(snapshots))


def execution_config(proposal, prepared, evidence):
    ref = proposal["config_ref"]
    require(
        ref["path"] == str(REPO / "configs/wam_clean_training_execution_20261023.json")
        and proposal["source_sha256"].get(ref["path"]) == ref["sha256"],
        "execution config is not source-frozen",
    )
    value = evidence.json(ref)
    expected = dict(
        schema_version="s2a.clean-training-execution-config.v1",
        seed=42,
        output_dir="outputs/wam_stability_20261021/clean_residual_training_20261023/execution",
        plan_path="plans/v0.0.1/040_clean_training_execution_20261023.md",
        prepared_ref=proposal["prepared_ref"],
        metadata_verification_ref=proposal["metadata_verification_ref"],
        development_control_ref=proposal["development_control_ref"],
        disk_required_bytes=DISK_REQUIRED,
        ram_required_bytes=RAM_REQUIRED,
        gpu_uuid=GPU,
        master_port=31338,
        development_port=31339,
        automatic_retry=False,
        fresh_performance_authorized=False,
        required_test_paths=list(REQUIRED_TESTS),
    )
    require(
        value == expected and ("config" not in proposal or proposal["config"] == expected),
        "constant execution config refs/thresholds/tests differ from audited proposal",
    )
    require(
        Path(prepared["root"]).resolve() == REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023",
        "metadata preparation output root differs",
    )
    return dict(execution_config_exact=True)


def independent_metadata(proposal, evidence):
    metadata = evidence.json(proposal["metadata_verification_ref"])
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("prepared_ref") == proposal["prepared_ref"]
        and all(
            metadata.get(k) is True for k in ("complete", "passed", "metadata_qualified", "consumed_before_after_exact")
        )
        and metadata.get("execution_ready") is False
        and metadata.get("execution_authorized") is False,
        "independent metadata qualification is incomplete, unrelated or improperly authorized",
    )
    verifier = metadata["verifier_ref"]
    evidence.check(verifier)
    require(
        proposal["source_sha256"].get(verifier["path"]) == verifier["sha256"],
        "metadata verifier source is not execution-frozen",
    )
    for key in (
        "new_model_calls",
        "new_fit_calls",
        "new_physics_steps",
        "new_fresh_cases",
        "future_label_archives_decoded",
    ):
        require(metadata.get(key) == 0, "metadata qualification consumed forbidden execution/data")
    require(
        metadata.get("weight_stat_files") == 163 and metadata.get("full_weight_content_rehash") is False,
        "metadata qualification weight evidence boundary differs",
    )
    evidence.mapping(metadata["consumed_sha256"])
    return metadata


def training_seed_plan(proposal, prepared, evidence):
    """Rebuild the explicit040 seed revision; never reinterpret old63 seeds.

    The original metadata remains independently bound and unchanged. The new
    manifest names every original identity and actual31 seed before sampling.
    """
    plan = evidence.json(proposal["training_seed_plan_ref"])
    require(
        set(plan)
        == {
            "schema_version",
            "complete",
            "passed",
            "prepared_ref",
            "algorithm",
            "entries",
            "evaluation_seeds",
            "actual_seed_bits",
            "original_seed_bits",
            "model_calls",
            "fits",
            "physics_steps",
        }
        and "training_seed_plan_ref" not in prepared,
        "seed plan must bind the unmodified original metadata",
    )
    require(
        plan.get("schema_version") == "s2a.clean-training-seed-plan.v1"
        and plan.get("prepared_ref") == proposal["prepared_ref"]
        and plan.get("complete") is True
        and plan.get("passed") is True
        and type(plan.get("actual_seed_bits")) is int
        and plan.get("actual_seed_bits") == 31
        and type(plan.get("original_seed_bits")) is int
        and plan.get("original_seed_bits") == 63
        and plan.get("evaluation_seeds") == [42, 43]
        and all(type(seed) is int for seed in plan["evaluation_seeds"]),
        "explicit pre-run31 seed plan identity/status differs",
    )
    require(plan["algorithm"] == SEED_ALGORITHM, "registered seed31 derivation algorithm differs")
    require(
        all(type(plan.get(key)) is int and plan[key] == 0 for key in ("model_calls", "fits", "physics_steps")),
        "seed plan consumed execution budget",
    )
    anchors = {anchor["name"]: anchor for anchor in prepared["training_anchors"]}
    anchor_names = [anchor["name"] for anchor in prepared["training_anchors"]]
    requests, entries = prepared["training_requests"], plan["entries"]
    require(
        len(anchors) == 150 and len(requests) == len(entries) == 300, "full150-anchor/300-request seed plan required"
    )
    derived, originals, absent = [], [], 0
    for index, (request, entry) in enumerate(zip(requests, entries, strict=True)):
        require(
            type(request["index"]) is int
            and request["index"] == index
            and request["anchor_name"] == anchor_names[index // 2]
            and type(request["root_seed"]) is int
            and request["root_seed"] == (101, 102)[index % 2],
            "seed plan original request order/anchor/root differs",
        )
        anchor = anchors[request["anchor_name"]]
        identity = anchor["source_identity"]
        require(
            type(request["task_index"]) is int
            and request["task_index"] in (0, 1, 2)
            and re.fullmatch(r"demo_(0|[1-9][0-9]*)", request["demo"]) is not None
            and type(request["anchor_frame"]) is int
            and request["anchor_frame"] >= 0
            and request["anchor_frame"] == anchor["anchor_frame"]
            and all(identity[key] == request[key] for key in ("task_index", "demo", "initial_state_sha256"))
            and valid_sha(identity["initial_state_sha256"]),
            "seed plan native source identity differs",
        )
        original_payload = json.dumps(
            dict(
                task=request["task_index"],
                demo=request["demo"],
                anchor_frame=request["anchor_frame"],
                root_seed=request["root_seed"],
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        original_hash = hashlib.sha256(original_payload.encode("utf-8")).digest()
        original = int.from_bytes(original_hash[:8], "big") & ((1 << 63) - 1)
        require(
            type(request["sampling_seed"]) is int
            and type(request["request"]["seed"]) is int
            and request["sampling_seed"] == request["request"]["seed"] == original
            and request["canonical_json"] == original_payload
            and request["identity_sha256"] == original_hash.hex(),
            "original63 metadata identity/derivation changed",
        )
        payload = json.dumps(
            [SEED_DOMAIN, request["anchor_name"], request["root_seed"], original],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        actual = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
        expected = dict(
            index=index,
            name=request["name"],
            anchor_name=request["anchor_name"],
            root_seed=request["root_seed"],
            original_sampling_seed=original,
            sampling_seed=actual,
            payload_sha256=digest.hex(),
        )
        require(
            entry == expected
            and json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
            == json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "seed plan row/source/actual31/payload SHA differs",
        )
        derived.append(actual)
        originals.append(original)
        draw = (
            int(hashlib.blake2b(json.dumps(actual, sort_keys=True).encode(), digest_size=16).hexdigest(), 16) / 2**128
        )
        absent += draw >= 0.9
    require(
        len(set(derived)) == len(set(originals)) == 300 and not ({42, 43} & set(derived)),
        "seed collision/evaluation overlap must fail without retry",
    )
    require(
        prepared["noise_roster"] == dict(training_seeds=originals, evaluation_seeds=[42, 43])
        and json.dumps(prepared["noise_roster"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        == json.dumps(dict(training_seeds=originals, evaluation_seeds=[42, 43]), sort_keys=True, separators=(",", ":")),
        "original63 noise roster/order differs",
    )
    return dict(
        training_seed_plan_independently_rebuilt=True,
        actual_training_seed_bits=31,
        original_training_seed_bits=63,
        training_requests_with_revised_seeds=300,
        training_captions_without_task_plan=absent,
    )


def metadata_and_model(proposal, prepared, evidence):
    independent_metadata(proposal, evidence)
    bridge = evidence.json(prepared["prerequisite_refs"]["bridge_prepared"])
    for key in ("source_sha256", "snapshot_sha256", "input_sha256"):
        evidence.mapping(bridge[key])
    require(
        bridge["source_sha256"] == prepared["qualified_model_source_sha256"], "qualified model source closure differs"
    )
    small = bridge["checkpoint_small_metadata_sha256"]
    require(
        len(small) == 3 and {Path(path).name for path in small} == {"config.yaml", ".metadata", ".save_complete"},
        "qualified checkpoint three small metadata identities differ",
    )
    evidence.mapping(small)
    weights = proposal["weight_files"]
    require(
        weights == prepared["weight_files"] == bridge["weight_files"] and len(weights) == 163,
        "qualified163 weight-stat roster changed",
    )
    for path, expected in weights.items():
        stat = resolve(path).stat()
        require(
            stat.st_size == expected["bytes"] and stat.st_mtime_ns == expected["mtime_ns"], "weight file stat differs"
        )
    require(
        prepared["model_config"] == bridge["config"] and prepared["model_config"]["gpu_uuid"] == GPU,
        "model config/source/UUID differs from qualified037",
    )
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    require(
        qualified.get("complete") is True
        and qualified.get("bridge_qualified") is True
        and qualified.get("actual_requests") == 7
        and qualified.get("actual_denoise_calls") == 245,
        "complete037 model qualification is missing",
    )
    require(
        proposal["sampler_trace_ref"] == qualified["generations"][0]["trace_ref"],
        "sampler trace is not the qualified first raw",
    )
    prerequisite_chain(prepared, evidence)
    return bridge


def normal_lifecycle(command, evidence):
    launch, exit_record, release = (evidence.json(command[k]) for k in ("launch", "exit", "release"))
    require(
        exit_record["actual_wait"] is True
        and exit_record["returncode"] == 0
        and release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and (release["pid"], release["start_ticks"]) == (launch["child"]["pid"], launch["child"]["start_ticks"]),
        "qualified worker lacks normal owned PID/birth wait/reap/release",
    )


def prerequisite_chain(prepared, evidence):
    refs = prepared["prerequisite_refs"]
    require(refs == prepared["config"]["refs"], "historical qualification refs differ from metadata config")
    documents = {name: evidence.json(ref) for name, ref in refs.items()}
    for name in (
        "settled_collection",
        "settled_audit",
        "settled_owner",
        "settled_peer",
        "bridge_qualification",
        "native_mask_review",
        "native_mask_owner",
        "native_mask_peer",
    ):
        require(
            documents[name].get("complete") is True and documents[name].get("passed") is True,
            "required completed034/037/039 qualification is missing",
        )
    peer, audit, owner = (documents[name] for name in ("settled_peer", "settled_audit", "settled_owner"))
    require(
        peer["prepared_ref"] == audit["prepared_ref"] == refs["settled_prepared"]
        and peer["verified_ref"] == owner["verified_ref"] == refs["settled_audit"]
        and peer["owner_ref"] == refs["settled_owner"]
        and peer["new_quality"]["total_cases"]
        == peer["new_quality"]["stable_native_cases"]
        == peer["native_prefix_exact_cases"]
        == 30
        and peer["new_quality"]["stable_native_by_task"] == {"0": 10, "1": 10, "2": 10}
        and peer["new_quality"]["eligible_for_model_training_registration"] is True
        and audit["eligible_for_model_training_registration"] is True
        and peer["sources_and_inputs_before_after_exact"] is True,
        "complete034 source/quality30/30 evidence differs",
    )
    require(
        peer["preserved_original034_owner_passed"] is False
        and evidence.json(peer["original_failed_owner_ref"])["passed"] is False,
        "original034 False was rewritten",
    )
    for name in ("worker_lifecycle", "audit_lifecycle"):
        lifecycle = peer[name]
        require(
            lifecycle["actual_wait"] is True
            and lifecycle["returncode"] == 0
            and lifecycle["reaped"] is True
            and lifecycle["signal_sent"] is False,
            "034 lifecycle is incomplete",
        )
    bridge, failed, gpu = (
        documents[name] for name in ("bridge_qualification", "bridge_failed_owner", "bridge_gpu_after")
    )
    require(
        bridge["prepared_ref"] == refs["bridge_prepared"]
        and bridge["parameters_before"] == bridge["parameters_after"]
        and failed["passed"] is False
        and failed["gpu_after_ref"] == refs["bridge_gpu_after"],
        "qualified037 base/source binding or original False differs",
    )
    normal_lifecycle(failed["owned_commands"][0], evidence)
    require(
        gpu["released"] is True and gpu["uuid"] == GPU and gpu["memory_mib"] == 0 and gpu["compute_apps"] == [],
        "037 allowed GPU normal release is missing",
    )
    registration, review, corrected, final = (
        documents[name]
        for name in ("native_mask_prepared", "native_mask_review", "native_mask_owner", "native_mask_peer")
    )
    require(
        all(item["bridge_qualified"] is True for item in (review, corrected, final))
        and corrected["registration_ref"] == refs["native_mask_prepared"]
        and corrected["review_ref"] == refs["native_mask_review"]
        and corrected["preserved_original_owner_passed"] is False,
        "joint039 qualification was not completed under the original evidence",
    )
    for name, target in (
        ("original_failed_owner_ref", "bridge_failed_owner"),
        ("original_prepared_ref", "bridge_prepared"),
        ("qualification_ref", "bridge_qualification"),
    ):
        require(corrected[name] == registration[name] == refs[target], "039 original source/False binding differs")
    require(
        review["actual_online_points_verified"] == final["counts"]["points"] == 245
        and review["nonzero_changed_active_values"] == final["counts"]["changed_values"] == 102
        and final["counts"]["native_initial_masks"] == 7
        and final["consumed_source_and_artifacts_before_after_exact"] is True,
        "039 complete245-point/102-value evidence differs",
    )
    for name, target in (
        ("owner039_ref", "native_mask_owner"),
        ("formal_ref", "native_mask_review"),
        ("worker_ref", "bridge_qualification"),
        ("prepared_ref", "bridge_prepared"),
    ):
        require(final[name] == refs[target], "039 independent peer source binding differs")
    require(len(corrected["owned_commands"]) == 1, "039 supplementary CPU audit lifecycle budget differs")
    normal_lifecycle(corrected["owned_commands"][0], evidence)


def normalization(proposal, bridge, evidence):
    import yaml

    receipt = evidence.json(proposal["normalization_ref"])
    require(
        receipt.get("schema_version") == "s2a.clean-training-normalization.v1", "normalization receipt schema differs"
    )
    config_ref = receipt["config_ref"]
    require(
        Path(config_ref["path"]).name == "config.yaml"
        and bridge["checkpoint_small_metadata_sha256"].get(config_ref["path"]) == config_ref["sha256"],
        "normalization source is not the qualified checkpoint config",
    )
    path = evidence.check(config_ref)
    config = yaml.safe_load(path.read_text())
    spec = config["model"]["config"]["action128_spec"]
    require(
        "${" not in json.dumps(spec),
        "unresolved Action128 interpolation cannot be qualified without model initialization",
    )
    registry, transforms = spec["registry"], spec["transforms"]
    require(
        isinstance(transforms, list)
        and len(transforms) == 67
        and len(registry["domains"]) == 67
        and len(registry["domain_projection_ids"]) == 67
        and registry["domains"][31] == DOMAIN
        and registry["domain_projection_ids"][31] == 0,
        "Action128 list67 normalization31/projection0/domain routing differs",
    )
    transform = transforms[31]
    require(
        set(transform) == {"normalization_mask", "scale", "bias", "clip_min", "clip_max"}
        and all(
            isinstance(transform[k], list) and len(transform[k]) == 128 for k in ("normalization_mask", "scale", "bias")
        ),
        "normalization128 transform schema differs",
    )
    require(
        all(type(x) is bool for x in transform["normalization_mask"])
        and transform["normalization_mask"] == [True] * 3 + [False] * 125
        and all(type(x) in (int, float) and math.isfinite(x) for k in ("scale", "bias") for x in transform[k])
        and transform["clip_min"] == -1.0
        and transform["clip_max"] == 1.0,
        "qualified FP32 affine/clipping transform differs",
    )
    expected = dict(
        schema_version="s2a.clean-training-normalization.v1",
        config_ref=config_ref,
        domain=DOMAIN,
        normalization_id=31,
        projection_id=0,
        transform=transform,
    )
    require(receipt == expected, "normalization receipt does not exactly describe published normalization31")
    return dict(normalization_id=31, projection_id=0, transforms=67, action_columns=128)


def sampler(proposal, prepared, evidence):
    trace = evidence.json(proposal["sampler_trace_ref"])
    require(
        trace.get("complete") is True
        and trace["purpose"] == "evaluation"
        and trace["seed"] == 42
        and trace["sampler_calls"] == 1
        and trace["parameters_before"] == trace["parameters_after"]
        and trace["source_sha256"] == prepared["qualified_model_source_sha256"],
        "qualified sampler trace/source/parameters differ",
    )
    steps, sigmas, timesteps = trace["steps"], trace["sigma_values"], trace["scheduler_timesteps"]
    require(
        len(steps) == 35
        and len(sigmas) == 36
        and len(timesteps) == 35
        and sigmas[-1] == trace["terminal_sigma"] == 0.0
        and trace["terminal_neural_calls"] == 0
        and trace["solver_sigma_dtype"] == "torch.float32",
        "actual35 positive calls/terminal contract differs",
    )
    require(
        trace["actual_sampler_arguments"] == dict(num_steps=35, shift=5.0, seed=[42]), "actual sampler arguments differ"
    )
    for index, point in enumerate(steps):
        require(
            point["step_index"] == index
            and point["solver_sigma"] == sigmas[index]
            and point["solver_timestep"] == timesteps[index]
            and type(timesteps[index]) is int
            and 0 < sigmas[index] < 1
            and math.isfinite(sigmas[index])
            and (index == 0 or sigmas[index] < sigmas[index - 1])
            and all(point[k] == 1 for k in ("head_calls", "denoise_calls", "conversion_calls", "velocity_delegations")),
            "actual observed sigma/time grid or single delegation differs",
        )
    return dict(actual_positive_sigma_calls=35, terminal_neural_calls=0)


def control(proposal, evidence):
    value = evidence.json(proposal["development_control_ref"])
    require(
        value.get("schema_version") == "s2a.clean-training-development-preflight-addendum.v1"
        and all(
            value.get(k) is True
            for k in ("complete", "passed", "control_contract_reuse_eligible", "original_failed_verdict_preserved")
        )
        and all(
            value.get(k) is False
            for k in (
                "new_physics_control_qualification",
                "execution_ready",
                "execution_authorized",
                "fresh_performance_authorized",
            )
        ),
        "conditional control reuse evidence differs",
    )
    guard = value["cv2_guard_registration"]
    helper = guard["helper_ref"]
    require(
        helper["path"] == str(REPO / "src/s2a/native_environment.py")
        and proposal["source_sha256"].get(helper["path"]) == helper["sha256"],
        "exact qualified cv2 helper is not source-frozen",
    )
    evidence.check(helper)
    probe = evidence.json(guard["exact_probe_ref"])
    environment = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO))
    require(
        probe["schema_version"] == "s2a.native-training-import-environment-probe.v1"
        and probe["before"] == environment
        and probe["after_import_cv2"]["cwd"] == str(REPO)
        and probe["after_import_cv2"]["CUDA_VISIBLE_DEVICES"] == ""
        and bool(probe["after_import_cv2"]["LD_LIBRARY_PATH"])
        and probe["reproduces_empty_ld_guard_rejection"] is True,
        "exact cv2 environment probe is not qualified",
    )
    runtime = guard["runtime_source_refs"]
    loader = Path(probe["cv2_loader_ref"]["path"])
    require(
        len(runtime) == 3
        and probe["cv2_loader_ref"] in runtime
        and {r["path"] for r in runtime}
        == {str(loader), str(loader.parent / "config.py"), str(loader.parent / "config-3.py")},
        "OpenCV loader/config source identities differ",
    )
    for ref in [guard["exact_probe_ref"], *runtime]:
        evidence.check(ref)
        require(
            proposal["input_sha256"].get(ref["path"]) == ref["sha256"], "exact cv2 runtime/probe is not input-frozen"
        )
    facts = value["actual_facts"]
    require(
        facts["source_sha256_before_after_exact"] is True
        and facts["requirements_eight_versions_exact"] is True
        and facts["assets_all_locatable"] is True
        and facts["control_import_pyarrow_loaded"] is False,
        "actual control dependency qualification is incomplete",
    )
    closure = evidence.json(facts["source_closure_ref"])
    for path, digest in closure["s2a_and_entry_source_sha256"].items():
        evidence.check(dict(path=path, sha256=digest))
        require(proposal["source_sha256"].get(path) == digest, "qualified control source is not final-source-frozen")
    guarded = evidence.json(facts["guarded_actual_import_ref"])
    require(
        guarded["passed"] is True
        and guarded["final_environment"] == environment
        and guarded["pyarrow_loaded"] is False
        and guarded["cv2_readiness"]["passed"] is True,
        "actual guarded control imports did not pass",
    )
    fixture = facts["fixture_contract_and_three_expert_byte_identity"]
    require(fixture["passed"] is True and len(fixture["records"]) == 3, "qualified fixed three-case fixture differs")
    f = fixture["fixture"]
    evidence.check(dict(path=f["actual"], sha256=f["sha256"]))
    require(
        f["sha256"] == "f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e",
        "qualified development fixture identity differs",
    )
    require(
        [r["name"] for r in fixture["records"]] == [f"task{t}_init0" for t in range(3)]
        and [r["task_index"] for r in fixture["records"]] == [0, 1, 2]
        and all(valid_sha(r["initial_state_sha256"]) and valid_sha(r["model_xml_sha256"]) for r in fixture["records"]),
        "fixed development initial/source identity metadata differs",
    )
    worker = evidence.check(
        dict(
            path=str(REPO / "scripts/execute_wam_clean_training_case.py"),
            sha256=proposal["source_sha256"][str(REPO / "scripts/execute_wam_clean_training_case.py")],
        )
    )
    guarded_worker(worker.read_text())
    return dict(qualified_control_cases=3, development_cases=12, cv2_runtime_sources=3, new_physics_qualification=False)


def guarded_worker(source):
    tree = ast.parse(source)
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    heavy = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "s2a.stability_position_simulation"
    ]
    guards = [
        node.lineno
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "restore_cv2_environment"
    ]
    strict = [
        node.lineno
        for node in ast.walk(execute)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "strict_native_environment"
    ]
    require(
        len(heavy) == len(guards) == 1
        and execute.lineno < guards[0] < heavy[0]
        and any(line > heavy[0] for line in strict),
        "development worker does not enforce guard before heavy imports",
    )
    for node in tree.body:
        if isinstance(node, ast.Import):
            require(
                not any(alias.name.split(".")[0] in ("cv2", "libero", "mujoco", "robosuite") for alias in node.names),
                "unguarded heavy top-level development import",
            )
        elif isinstance(node, ast.ImportFrom):
            require(
                not node.module or node.module.split(".")[0] not in ("cv2", "libero", "mujoco", "robosuite"),
                "unguarded heavy top-level development import",
            )


def memory_observation():
    entries = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host = int(entries["MemAvailable"].split()[0]) * 1024
    maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    limit = None if maximum == "max" else int(maximum)
    return dict(
        host_MemAvailable=host,
        cgroup_memory_max=limit,
        cgroup_memory_current=current,
        MemAvailable=host if limit is None else min(host, max(0, limit - current)),
    )


def gpu_observation():
    def query(arguments):
        result = subprocess.run(["nvidia-smi", *arguments], check=True, capture_output=True, text=True, timeout=15)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    rows = query(["--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    selected = [row.split(",") for row in rows if row.split(",")[0].strip() == GPU]
    require(len(selected) == 1, "allowed fixed UUID is not uniquely visible")
    row = selected[0]
    pci = row[1].strip().upper()
    require(pci == GPU_PCI, "fixed allowed GPU PCI identity changed")
    information = Path("/proc/driver/nvidia/gpus/0000:ba:00.0/information").read_text()
    device = dict(line.split(":", 1) for line in information.splitlines() if ":" in line)
    require(device["GPU UUID"].strip() == GPU, "driver/probed GPU UUID differs")
    minor = int(device["Device Minor"].strip())
    require(minor == GPU_MINOR, "fixed allowed GPU driver minor differs")
    apps = query(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    return dict(
        uuid=GPU,
        pci_bus_id=pci,
        device_minor=minor,
        memory_mib=int(row[2]),
        utilization_gpu=int(row[3]),
        compute_apps=[line for line in apps if line.split(",")[0].strip() == GPU],
    )


def port_available(port):
    try:
        with socket.socket() as stream:
            stream.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def machine_observation(root):
    return dict(
        disk_free=shutil.disk_usage(root).free,
        **memory_observation(),
        CPU=len(os.sched_getaffinity(0)),
        GPU=gpu_observation(),
        ports={str(p): port_available(p) for p in (31338, 31339)},
    )


def resources(proposal, prepared, evidence):
    receipt = evidence.json(proposal["resource_observation_ref"])
    require(
        receipt.get("complete") is True and receipt.get("passed") is True, "actual resource observation did not pass"
    )
    require(
        receipt.get("required_disk_bytes") == DISK_REQUIRED and receipt.get("required_ram_bytes") == RAM_REQUIRED,
        "conservative200GiB/128GiB resource thresholds differ",
    )
    observed_at = datetime.fromisoformat(receipt["observed_utc"].replace("Z", "+00:00"))
    require(
        observed_at.tzinfo is not None and observed_at <= datetime.now(timezone.utc),
        "actual observation UTC is invalid",
    )
    require(receipt["threads"] == THREADS, "actual worker thread configuration differs")
    for actual in (receipt, machine_observation(Path(prepared["root"]))):
        require(
            type(actual["disk_free"]) is int
            and actual["disk_free"] >= DISK_REQUIRED
            and type(actual["MemAvailable"]) is int
            and actual["MemAvailable"] >= RAM_REQUIRED
            and type(actual["CPU"]) is int
            and actual["CPU"] >= 8,
            "actual disk/effective RAM/CPU headroom insufficient",
        )
        host, maximum, current = (
            actual["host_MemAvailable"],
            actual["cgroup_memory_max"],
            actual["cgroup_memory_current"],
        )
        effective = host if maximum is None else min(host, max(0, maximum - current))
        require(
            actual["MemAvailable"] == effective and host > 0 and current >= 0,
            "effective cgroup/host memory evidence differs",
        )
        require(
            actual["GPU"]
            == dict(
                uuid=GPU, pci_bus_id=GPU_PCI, device_minor=GPU_MINOR, memory_mib=0, compute_apps=[], utilization_gpu=0
            ),
            "fixed allowed GPU is occupied/different",
        )
        require(actual["ports"] == {"31338": True, "31339": True}, "registered ports are unavailable")
    return dict(
        resource_observation_recorded=True,
        resources_reobserved=True,
        required_disk_bytes=DISK_REQUIRED,
        required_ram_bytes=RAM_REQUIRED,
    )


def qa(proposal, evidence):
    value = evidence.json(proposal["qa_ref"])
    require(
        value.get("complete") is True and value.get("passed") is True and value.get("exit_code") == 0,
        "final actual CPU QA did not pass",
    )
    require(type(value.get("tests_passed")) is int and value["tests_passed"] > 0, "actual passed-test count missing")
    require(
        value["source_sha256"] == proposal["source_sha256"],
        "QA did not freeze the exact final executable source closure",
    )
    require(
        set(REQUIRED_TESTS) <= set(value["test_paths"]) and len(value["test_paths"]) == len(set(value["test_paths"])),
        "required final CPU tests were omitted or duplicated",
    )
    kinds = set()
    for command in value["commands"]:
        argv = command["argv"]
        require(
            isinstance(argv, list) and argv and all(isinstance(arg, str) for arg in argv) and command["exit_code"] == 0,
            "final QA command is missing or failed",
        )
        text = evidence.check(command["log_ref"]).read_text()
        tools = [Path(arg).name for arg in argv]
        if "pytest" in tools:
            test_arguments = argv[tools.index("pytest") + 1 :]
            match = re.findall(r"\b(\d+) passed\b", text)
            require(bool(match) and int(match[-1]) == value["tests_passed"], "actual pytest log/count differs")
            require(
                all(path in argv for path in value["test_paths"]) or "tests" in argv,
                "pytest command omitted listed tests",
            )
            require(
                not any(arg in ("-k", "-m", "--ignore") or arg.startswith("--ignore=") for arg in test_arguments),
                "final relevant QA cannot selectively deselect required tests",
            )
            kinds.add("pytest")
        elif "ruff" in tools and "format" in argv:
            require(
                "--check" in argv and ("All done!" in text or "already formatted" in text),
                "actual Ruff format check did not complete",
            )
            kinds.add("format")
        elif "ruff" in tools and "check" in argv:
            require("All checks passed!" in text, "actual Ruff check did not complete")
            kinds.add("ruff")
    require(kinds == {"pytest", "ruff", "format"}, "final QA lacks pytest, Ruff check or format --check")
    return dict(actual_cpu_tests_passed=value["tests_passed"], final_source_qa_exact=True)


def audit(proposal_ref):
    evidence = Evidence()
    proposal = evidence.json(proposal_ref)
    prepared = evidence.json(proposal["prepared_ref"])
    proposal_contract(proposal, prepared)
    checks = sources(proposal, prepared, evidence)
    checks.update(execution_config(proposal, prepared, evidence))
    bridge = metadata_and_model(proposal, prepared, evidence)
    checks.update(training_seed_plan(proposal, prepared, evidence))
    checks.update(normalization(proposal, bridge, evidence))
    checks.update(sampler(proposal, prepared, evidence))
    checks.update(control(proposal, evidence))
    checks.update(resources(proposal, prepared, evidence))
    checks.update(qa(proposal, evidence))
    evidence.check(proposal["plan_ref"])
    require(
        proposal["plan_ref"]["path"] == str(REPO / "plans/v0.0.1/040_clean_training_execution_20261023.md")
        and proposal["source_sha256"].get(proposal["plan_ref"]["path"]) == proposal["plan_ref"]["sha256"],
        "final execution plan is not exactly source-frozen",
    )
    admitted = {
        **proposal["input_sha256"],
        **proposal["source_sha256"],
        **proposal["snapshot_sha256"],
        proposal_ref["path"]: proposal_ref["sha256"],
    }
    require(
        all(admitted.get(path) == digest for path, digest in evidence.consumed.items()),
        "proposal closure omits a consumed independent evidence/source/input",
    )
    evidence.unchanged()
    return dict(
        prepared_ref=proposal["prepared_ref"],
        training_seed_plan_ref=proposal["training_seed_plan_ref"],
        checks=checks,
        input_sha256=evidence.consumed,
        consumed_before_after_exact=True,
        weight_stat_files=163,
        full_weight_content_rehash=False,
    )


def verify(args):
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "independent execution verifier requires seed42 and explicit empty CUDA/LD",
    )
    require(Path.cwd().resolve() == REPO, "independent execution verifier must run from S2A")
    output = Path(args.output).absolute()
    require(
        output.resolve() == output and output.is_relative_to(REPO / "outputs") and not output.exists(),
        "new output must stay inside S2A outputs and cannot overwrite evidence",
    )
    proposal_ref = dict(path=str(Path(args.proposal).absolute()), sha256=args.proposal_sha256)
    result = dict(
        schema_version="s2a.clean-training-execution-verification.v1",
        complete=False,
        passed=False,
        proposal_ref=proposal_ref,
        verifier_ref=dict(path=str(VERIFIER), sha256=sha(VERIFIER)),
        eligible_for_execution=False,
        execution_authorized=False,
        fresh_performance_authorized=False,
        new_model_initializations=0,
        new_model_calls=0,
        new_fits=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        future_label_archives_decoded=0,
    )
    try:
        result.update(audit(proposal_ref))
        result.update(complete=True, passed=True, eligible_for_execution=True)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    print(json.dumps(verify(parser.parse_args())))


if __name__ == "__main__":
    main()
