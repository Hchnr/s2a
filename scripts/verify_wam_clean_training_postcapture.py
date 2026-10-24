"""Independent metadata admission for the single043 post-capture W1 repair.

This checker imports no producer, repair helper or training implementation. It
inherits the completed D0 numerical audit, checks its immutable metadata and
normal owned lifecycle, and never opens an array or runs a fit.
"""

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIER = Path(__file__).resolve()
OLD_REL = "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023"
NEW_REL = "outputs/clean_training_postcapture_repair_20261023"
PLAN_REL = "plans/v0.0.1/043_clean_training_postcapture_20261023.md"
CONFIG_REL = "configs/wam_clean_training_postcapture_20261023.json"
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
BUDGET = dict(
    producer_fits=1,
    independent_reconstructions=1,
    new_model_initializations=0,
    new_denoise_calls=0,
    new_physics_steps=0,
    new_fresh_cases=0,
    automatic_retry=False,
)
FIXED = (
    "root",
    "prepared_ref",
    "execution_registration_ref",
    "d0_collection_ref",
    "d0_capture_audit_ref",
    "d0_owner_ref",
    "failure_ref",
    "d0_readonly_audit_ref",
    "fit_output_dir",
    "verification_output",
    "budget",
)
NEW_SOURCES = (
    "src/s2a/clean_training_source_proof.py",
    "tests/test_clean_training_source_proof.py",
    "scripts/execute_wam_clean_training_postcapture_fit.py",
    "scripts/run_wam_clean_training_postcapture.py",
    "scripts/verify_wam_clean_training_postcapture.py",
    "tests/test_clean_training_postcapture_verification.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def same(left, right):
    return canonical(left) == canonical(right)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def actual_path(value, *, allow_external=False):
    require(isinstance(value, str), "path must be a string")
    path = Path(value)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and path.resolve() == path
        and not path.is_symlink()
        and not path.is_relative_to("/share/project/eai_pwm/home/hcr/repos/s2a")
        and (allow_external or path.is_relative_to(REPO)),
        "actual path must be canonical/nonsymlink, outside the deleted root and inside the permitted scope",
    )
    return path


def parse_json(raw):
    def unique(pairs):
        value = {}
        for key, entry in pairs:
            require(key not in value, "duplicate JSON key")
            value[key] = entry
        return value

    return json.loads(
        raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))
    )


class Evidence:
    def __init__(self):
        self.consumed = {}

    def path(self, ref, expected=None, *, allow_external=False):
        require(
            isinstance(ref, dict)
            and set(ref) == {"path", "sha256"}
            and isinstance(ref["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
            "exact path/SHA reference required",
        )
        path = actual_path(ref["path"], allow_external=allow_external)
        require(path.is_file() and (expected is None or path == expected), "missing or misplaced evidence")
        require(sha256(path) == ref["sha256"], "evidence SHA differs: " + str(path))
        require(self.consumed.setdefault(str(path), ref["sha256"]) == ref["sha256"], "conflicting SHA")
        return path

    def json(self, ref, expected=None):
        require(Path(ref["path"]).suffix == ".json", "metadata checker refuses binary/non-JSON decode")
        path = self.path(ref, expected)
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "evidence changed while reading")
        value = parse_json(raw)
        require(isinstance(value, dict), "JSON object required")
        return value

    def mapping(self, values, *, text_only=False):
        require(isinstance(values, dict) and values, "nonempty SHA closure required")
        for path, digest in values.items():
            if text_only and Path(path).suffix in {".npz", ".npy", ".png", ".safetensors", ".pt"}:
                continue
            self.path(dict(path=path, sha256=digest), allow_external=text_only)

    def unchanged(self):
        require(
            all(sha256(path) == digest for path, digest in self.consumed.items()),
            "consumed evidence changed during independent review",
        )


def complete(value, schema):
    require(
        value.get("schema_version") == schema and value.get("complete") is value.get("passed") is True,
        "incomplete or wrong-schema evidence: " + schema,
    )


def lifecycle(evidence, launch_ref, exit_ref, release_ref, *, code=0, normal_error=True):
    require(len({r["path"] for r in [launch_ref, exit_ref, release_ref]}) == 3, "lifecycle receipts reused")
    launch, exit_, release = [evidence.json(ref) for ref in [launch_ref, exit_ref, release_ref]]
    child = launch.get("child", {})
    require(all(type(child.get(k)) is int and child[k] > 0 for k in ["pid", "start_ticks"]), "PID/birth absent")
    require(
        exit_.get("actual_wait") is True and type(exit_.get("returncode")) is int and exit_["returncode"] == code,
        "actual owned wait/exit missing",
    )
    require(
        release.get("reaped") is True
        and type(release.get("returncode")) is int
        and release["returncode"] == code
        and release.get("signal_sent") is False
        and release.get("kill_sent", False) is False
        and release.get("only_owned_child_signalled") is True
        and all(type(release.get(k)) is int and release[k] == child[k] for k in ["pid", "start_ticks"]),
        "normal same-birth reaped/no-signal receipt missing",
    )
    require(
        "original_error" in release
        and (
            release["original_error"] is None
            if normal_error
            else isinstance(release["original_error"], str) and bool(release["original_error"])
        ),
        "owned release error does not match the original outcome",
    )
    return launch


def old_envelope(e, proposal):
    root = REPO / OLD_REL
    prepared = e.json(proposal["prepared_ref"], root / "preparation/prepared.json")
    old = e.json(proposal["execution_registration_ref"], root / "execution/admitted.json")
    complete(old, "s2a.clean-training-execution-registration.v1")
    original = e.json(old["proposal_ref"], root / "execution/proposal.json")
    verified = e.json(old["execution_verification_ref"], root / "execution/verified.json")
    complete(original, "s2a.clean-training-execution-proposal.v1")
    complete(verified, "s2a.clean-training-execution-verification.v1")
    require(
        original.get("execution_authorized") is False
        and original.get("readiness", {}).get("ready_for_execution") is False
        and old.get("automatic_retry") is False
        and old.get("fresh_performance_authorized") is False
        and verified.get("eligible_for_execution") is True
        and verified.get("prepared_ref") == proposal["prepared_ref"]
        and verified.get("proposal_ref") == old["proposal_ref"]
        and old["prepared_ref"] == proposal["prepared_ref"],
        "original041 admission chain differs",
    )
    expected = copy.deepcopy(original)
    expected.update(
        schema_version="s2a.clean-training-execution-registration.v1",
        execution_authorized=True,
        proposal_ref=old["proposal_ref"],
        execution_verification_ref=old["execution_verification_ref"],
    )
    expected["readiness"]["ready_for_execution"] = True
    require(same(old, expected), "original041 registration changed its audited proposal")
    require(
        prepared.get("schema_version") == "s2a.clean-training-preparation.v1" and prepared.get("root") == str(root),
        "original prepared root/schema differs",
    )
    sources, snapshots, refs = old["source_sha256"], old["snapshot_sha256"], old["snapshot_refs"]
    require(
        len(sources) == len(snapshots) == len(refs) == 484 and set(refs) == set(sources), "original484 closure differs"
    )
    e.mapping(sources)
    e.mapping(snapshots)
    require(len({r["path"] for r in refs.values()}) == 484, "old snapshots are not one-to-one")
    for source, ref in refs.items():
        e.path(ref)
        require(sources[source] == ref["sha256"] == snapshots.get(ref["path"]), "old source/snapshot differs")
    verifier = verified["verifier_ref"]
    e.path(verifier, REPO / "scripts/verify_wam_clean_training_continuation.py")
    require(sources.get(verifier["path"]) == verifier["sha256"], "old independent verifier is not frozen")
    metadata = e.json(old["metadata_verification_ref"])
    complete(metadata, "s2a.clean-training-preparation-independent.v1")
    require(
        metadata.get("prepared_ref") == proposal["prepared_ref"]
        and metadata.get("metadata_qualified") is True
        and metadata.get("execution_authorized") is metadata.get("execution_ready") is False
        and metadata.get("consumed_before_after_exact") is True,
        "original metadata qualification differs",
    )
    e.mapping(prepared["snapshot_sha256"])
    return prepared, old, verifier


def d0_evidence(e, proposal, prepared, old):
    root = REPO / OLD_REL
    collection = e.json(proposal["d0_collection_ref"], root / "capture/D0/result.json")
    audit = e.json(proposal["d0_capture_audit_ref"], root / "review/D0_verified.json")
    owner = e.json(proposal["d0_owner_ref"], root / "orchestration/D0/result.json")
    complete(collection, "s2a.clean-training-capture.v1")
    complete(audit, "s2a.clean-training-capture-audit.v1")
    complete(owner, "s2a.clean-training-owner.v1")
    for value in [collection, audit, owner]:
        require(
            value["prepared_ref"] == proposal["prepared_ref"]
            and value["execution_registration_ref"] == proposal["execution_registration_ref"],
            "D0 belongs to another preparation/admission",
        )
    require(
        collection.get("round_name") == audit.get("round_name") == owner.get("stage") == "D0"
        and collection.get("readout_ref") is None
        and collection.get("d0_audit_ref") is None
        and audit["collection_ref"] == proposal["d0_collection_ref"]
        and owner["producer_ref"] == proposal["d0_collection_ref"]
        and owner["verification_ref"] == proposal["d0_capture_audit_ref"],
        "D0 raw capture/audit/owner chain differs",
    )
    require(
        collection["actual_requests"] == audit["actual_requests"] == len(collection["requests"]) == 300
        and collection["actual_denoise_calls"]
        == collection["actual_head_calls"]
        == audit["actual_denoise_calls"]
        == audit["actual_online_points_verified"]
        == 10500
        and collection["model_initialization_attempts"] == collection["model_initializations"] == 1
        and audit["changed_active_values"] == 0
        and collection["parameters_before"] == collection["parameters_after"],
        "D0 full300/10500 raw denominator differs",
    )
    require(
        all(
            type(value.get(k)) is int and value[k] == 0
            for value in [collection, audit]
            for k in ["future_label_reads", "producer_fits", "physics_steps", "fresh_cases"]
        ),
        "D0 read labels or fitted/executed",
    )
    require(
        collection["source_sha256"] == audit["source_sha256"]
        and all(audit["source_sha256"].get(p) == s for p, s in old["source_sha256"].items()),
        "D0 source closure differs from041",
    )
    e.mapping(audit["source_sha256"])
    e.mapping(audit["input_sha256"], text_only=True)
    require(collection["training_seed_plan_ref"] == old["training_seed_plan_ref"], "D0 seed-plan ref differs")
    seeds = e.json(old["training_seed_plan_ref"])["entries"]
    require(len(seeds) == len(prepared["training_requests"]) == 300, "training seed count differs")
    for i, (entry, seed, original) in enumerate(
        zip(collection["requests"], seeds, prepared["training_requests"], strict=True)
    ):
        payload = json.dumps(
            [
                "s2a.clean-training-request-seed31.v1",
                seed["anchor_name"],
                seed["root_seed"],
                seed["original_sampling_seed"],
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(payload).digest()
        require(
            entry["index"] == seed["index"] == original["index"] == i
            and entry["complete"] is True
            and entry["name"] == seed["name"] == original["name"]
            and entry["sampling_seed"] == seed["sampling_seed"]
            and seed["sampling_seed"] == int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
            and seed["payload_sha256"] == digest.hex()
            and original["sampling_seed"] == seed["original_sampling_seed"],
            "D0 request identity or independent31-bit seed differs",
        )
    labels = [entry["name"] for entry in collection["requests"] for _ in range(35)]
    require(
        collection["boundary_calls"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "D0 boundary ledger differs",
    )
    require(len(owner["owned_commands"]) == 2, "D0 model/independent lifecycle pair missing")
    for command in owner["owned_commands"]:
        lifecycle(e, command["launch"], command["exit"], command["release"])
    model = owner["owned_commands"][0]
    require(
        all(audit["worker_" + k + "_ref"] == model[k] for k in ["launch", "exit", "release"]),
        "D0 audit lifecycle binding differs",
    )
    require(
        e.json(model["launch"])["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""),
        "D0 model GPU differs",
    )
    gpu = e.json(owner["gpu_after_ref"])
    require(
        audit["gpu_after_ref"] == owner["gpu_after_ref"]
        and gpu.get("released") is True
        and gpu["uuid"] == GPU
        and gpu["device_minor"] == 6
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == [],
        "D0 GPU normal release missing",
    )
    readonly = e.json(proposal["d0_readonly_audit_ref"])
    complete(readonly, "s2a.clean-training-d0-readonly-review.v1")
    require(
        readonly.get("D0_data_reuse_eligible_with_original_frozen_contract") is True
        and readonly.get("fit_rerun_authorized") is False
        and readonly["prior_independent_math_audit_ref"] == proposal["d0_capture_audit_ref"]
        and readonly["D0_owner_ref"] == proposal["d0_owner_ref"]
        and readonly.get("new_array_decodes") == 0
        and readonly.get("consumed_metadata_before_after_exact") is True,
        "D0 read-only qualification differs",
    )
    durable = e.json(readonly["D0_durable_stage_owner_ref"], root / "supervisor/D0/result.json")
    complete(durable, "s2a.durable-stage-owner.v1")
    require(
        durable["stage_result_ref"] == proposal["d0_owner_ref"] and durable["source_sha256"] == old["source_sha256"],
        "D0 durable owner did not bind the original source/owner",
    )
    lifecycle(e, durable["launch_ref"], durable["exit_ref"], durable["release_ref"])
    return audit


def failure_evidence(e, proposal, old):
    root = REPO / OLD_REL
    failure = e.json(proposal["failure_ref"])
    complete(failure, "s2a.clean-training-w1-pre-fit-failure-audit.v1")
    require(
        failure.get("exception") == "ValueError: fit artifact escaped S2A outputs"
        and failure.get("failure_before_load_context_return") is True
        and failure.get("original_failure_preserved") is True
        and failure.get("source_before_after_exact") is True
        and failure.get("source_sha256") == old["source_sha256"],
        "pre-fit failure diagnosis/source differs",
    )
    require(
        all(
            type(failure.get(k)) is int and failure[k] == 0
            for k in ["producer_fits", "independent_reconstructions", "future_array_decodes", "new_model_calls"]
        )
        and failure.get("d1_started") is False,
        "failure incident admits unrecorded work",
    )
    supervisor = e.json(failure["supervisor_ref"], root / "supervisor/result.json")
    stage = e.json(failure["stage_owner_ref"], root / "supervisor/W1/result.json")
    require(
        supervisor.get("complete") is supervisor.get("passed") is False
        and supervisor.get("active_stage") == "W1"
        and stage.get("complete") is stage.get("passed") is False
        and supervisor["failed_stage_ref"] == failure["stage_owner_ref"],
        "original supervisor W1 failure differs",
    )
    for k in ["launch", "exit", "release"]:
        require(stage[k + "_ref"] == failure["stage_" + k + "_ref"], "failed stage lifecycle differs")
    for prefix, subdir in [("stage", "supervisor/W1"), ("producer", "orchestration/W1/producer")]:
        for key, relative in [
            ("launch", "launch.json"),
            ("exit", "exit.json"),
            ("release", "release.json" if prefix == "stage" else "ownership/release.json"),
        ]:
            e.path(failure[prefix + "_" + key + "_ref"], root / subdir / relative)
    stage_launch = lifecycle(
        e,
        failure["stage_launch_ref"],
        failure["stage_exit_ref"],
        failure["stage_release_ref"],
        code=1,
        normal_error=False,
    )
    producer = lifecycle(
        e,
        failure["producer_launch_ref"],
        failure["producer_exit_ref"],
        failure["producer_release_ref"],
        code=1,
        normal_error=False,
    )
    require(
        producer["child"].get("ppid") == stage_launch["child"]["pid"]
        and len(producer.get("command", [])) > 1
        and producer["command"][1] == "scripts/fit_wam_clean_training.py",
        "failure belongs to another fit worker",
    )
    require(
        producer["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""), "failed fit was not CPU"
    )
    log = e.path(failure["producer_log_ref"], root / "orchestration/W1/producer/command.log").read_text()
    for text in [
        "context = load_context(",
        "check_execution_proof(",
        "check_file(verifier)",
        "ValueError: fit artifact escaped S2A outputs",
    ]:
        require(text in log, "original failure traceback is not the pre-fit source-proof gate")
    fit, verification = actual_path(proposal["fit_output_dir"]), actual_path(proposal["verification_output"])
    require(
        fit == root / "fit/W1"
        and verification == root / "review/W1_verified.json"
        and not fit.exists()
        and not verification.exists(),
        "W1 fit/verification already exists or target changed",
    )
    require(
        not (root / "supervisor/D1/launch.json").exists() and not (root / "orchestration/D1").exists(),
        "D1 was already started",
    )


def new_sources(e, proposal, old):
    sources, snapshots, refs = [proposal[k] for k in ["source_sha256", "snapshot_sha256", "source_snapshot_refs"]]
    require(
        set(refs) == set(sources) and len(snapshots) == len(sources) == len({v["path"] for v in refs.values()}),
        "new source/snapshot closure is not one-to-one",
    )
    require(
        all(sources.get(p) == s and refs.get(p) == old["snapshot_refs"][p] for p, s in old["source_sha256"].items())
        and all(snapshots.get(p) == s for p, s in old["snapshot_sha256"].items()),
        "original484 source/snapshot changed",
    )
    required = {REPO / rel for rel in [*NEW_SOURCES, PLAN_REL, CONFIG_REL]}
    required.update(
        path.resolve() for directory in ["src", "scripts", "tests"] for path in (REPO / directory).rglob("*.py")
    )
    require(
        all(str(path) in sources for path in required),
        "new helper/wrapper/owner/verifier/tests/config/plan closure missing",
    )
    e.mapping(sources)
    e.mapping(snapshots)
    for path, ref in refs.items():
        expected = (
            Path(old["snapshot_refs"][path]["path"])
            if path in old["source_sha256"]
            else REPO / NEW_REL / "source/repo" / Path(path).relative_to(REPO)
        )
        e.path(ref, expected)
        require(sources[path] == ref["sha256"] == snapshots.get(ref["path"]), "source/snapshot bytes differ")
    return sources


def dry_and_qa(e, proposal, source_verifier, sources):
    dry = e.json(proposal["metadata_dry_ref"])
    require(
        dry.get("complete") is dry.get("passed") is True
        and dry.get("zero_array_decodes") is True
        and type(dry.get("array_decode_attempts")) is int
        and dry["array_decode_attempts"] == 0
        and type(dry.get("expected_rows")) is int
        and dry["expected_rows"] == 420000
        and type(dry.get("callback_corrected_count")) is int
        and dry["callback_corrected_count"] == 1
        and dry.get("source_guard_completed") is True
        and dry.get("source_before_after_exact") is True,
        "actual original load_context dry proof incomplete or decoded arrays",
    )
    for key in ["prepared_ref", "execution_registration_ref", "d0_collection_ref", "d0_capture_audit_ref"]:
        require(dry.get(key) == proposal[key], "metadata dry proof binds another input")
    require(
        dry.get("source_verifier_ref") == source_verifier and dry.get("source_sha256") == sources,
        "dry actual source/verifier closure differs",
    )
    qa = e.json(proposal["qa_ref"])
    require(
        qa.get("complete") is qa.get("passed") is True and qa.get("source_sha256") == sources,
        "QA source/pass closure differs",
    )
    require(
        all(type(qa.get(k)) is int and qa[k] == 0 for k in ["tests_exit_code", "ruff_exit_code", "format_exit_code"])
        and type(qa.get("tests_passed")) is int
        and qa["tests_passed"] > 0,
        "actual QA exits/count absent",
    )
    require(
        isinstance(qa.get("test_paths"), list) and str(REPO / NEW_SOURCES[-1]) in qa["test_paths"],
        "independent verifier tests absent from QA",
    )
    logs = qa.get("log_refs", {})
    require(isinstance(logs, dict) and {"pytest", "ruff", "format"} <= set(logs), "actual QA logs missing")
    for ref in logs.values():
        e.path(ref)
    pytest_log = e.path(logs["pytest"]).read_text()
    match = re.search(r"(?m)^(\d+) passed(?:,.*?)? in [0-9.]+s", pytest_log)
    require(
        match is not None and int(match[1]) == qa["tests_passed"] and " FAILED " not in pytest_log,
        "QA pytest log does not show the reported actual success count",
    )
    require(
        "All checks passed!" in e.path(logs["ruff"]).read_text()
        and "already formatted" in e.path(logs["format"]).read_text(),
        "QA lint/format logs do not show success",
    )


def verify_postcapture(proposal_ref):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "metadata admission requires explicit empty CUDA/LD",
    )
    e = Evidence()
    proposal = e.json(proposal_ref)
    complete(proposal, "s2a.clean-training-postcapture-proposal.v1")
    require(
        proposal.get("execution_authorized") is False
        and proposal.get("root") == str(REPO / NEW_REL)
        and same(proposal.get("budget"), BUDGET),
        "043 scope/budget/authorization differs",
    )
    config = e.json(proposal["config_ref"], REPO / CONFIG_REL)
    e.path(proposal["plan_ref"], REPO / PLAN_REL)
    require(
        config.get("protocol") == "043"
        and config.get("stage_scope") == "W1_only"
        and config.get("automatic_retry") is False
        and all(same(config.get(k), proposal[k]) for k in FIXED),
        "043 fixed config inputs/output/budget differ",
    )
    prepared, old, verifier = old_envelope(e, proposal)
    audit = d0_evidence(e, proposal, prepared, old)
    failure_evidence(e, proposal, old)
    sources = new_sources(e, proposal, old)
    dry_and_qa(e, proposal, verifier, sources)
    verifier_ref = dict(path=str(VERIFIER), sha256=sha256(VERIFIER))
    e.path(verifier_ref)
    require(sources.get(verifier_ref["path"]) == verifier_ref["sha256"], "actual independent verifier is not frozen")
    e.unchanged()
    return dict(
        schema_version="s2a.clean-training-postcapture-verification.v1",
        complete=True,
        passed=True,
        eligible_for_execution=True,
        execution_authorized=False,
        proposal_ref=proposal_ref,
        verifier_ref=verifier_ref,
        prepared_ref=proposal["prepared_ref"],
        execution_registration_ref=proposal["execution_registration_ref"],
        source_sha256=sources,
        consumed_sha256=e.consumed,
        consumed_before_after_exact=True,
        source_verifier_ref=verifier,
        actual_requests=300,
        actual_denoise_calls=10500,
        inherited_D0_numerical_audit_ref=proposal["d0_capture_audit_ref"],
        evidence_boundary=audit["evidence_boundary"],
        expected_rows=420000,
        new_model_initializations=0,
        new_model_calls=0,
        new_fits=0,
        new_fit_calls=0,
        new_physics_steps=0,
        new_fresh_cases=0,
        future_array_decodes=0,
        automatic_retry=False,
        fresh_performance_authorized=False,
        goal_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--proposal-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=[42], required=True)
    args = parser.parse_args()
    output = actual_path(str(Path(args.output).absolute()))
    require(output.is_relative_to(REPO / NEW_REL) and not output.exists(), "new exclusive043 review output required")
    value = verify_postcapture(dict(path=str(Path(args.proposal).absolute()), sha256=args.proposal_sha256))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
