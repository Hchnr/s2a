"""Prepare the single registered021 zero-step continuation; starts no process."""

import argparse
import json
from pathlib import Path

from prepare_wam_raw8_seed43 import merge, unused_root
from run_wam_long_service import weight_stats
from run_wam_short_replan import cpu_environment, process_record, verify_hashes, write_new
from run_wam_temporal_continuation_matrix import CONFIG, EXTRA_SOURCES, PLAN, execution_inputs, protocol, registered

from s2a.artifacts import REPO, sha256

BASE = REPO / "outputs/wam_stability_20261021"
OLD_GATE = BASE / "temporal_equal_launch_gates_v1/partition2.json"
OLD_GATE_SHA = "c2faf8c27e64f064e3d9860c28791672c69edfaa4c90ab2bee910ec2fdc048db"
ZERO = BASE / "temporal_equal_continuation_preflight/original_task2_zero_effect.json"
ZERO_SHA = "bc117d00bcf2ed79f28356faf78c1cc8ce6a2744b6bfd54e2158e39f876d1528"
BUDGET = dict(
    continuation_wam_trials=1, physics_steps=600, maximum_h40_queries=75, new_oracle_trials=0, fresh_initial_states=0
)
SOURCES = (
    *EXTRA_SOURCES,
    "scripts/run_wam_temporal_continuation_service.py",
    "scripts/prepare_wam_temporal_continuation.py",
    "scripts/capture_wam_temporal_continuation_ready.py",
    "scripts/capture_wam_long_ready.py",
)


def no_original_worker():
    """Only inspect cmdlines matching the exact old entry/config/record."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command = (entry / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if str(REPO / "scripts/run_wam_temporal_matrix.py") not in command or "--record" not in command:
                continue
            if (
                "--config" in command
                and Path(command[command.index("--config") + 1]).resolve()
                == REPO / "configs/wam_temporal_equal_task2_20261021.json"
            ):
                matches.append(dict(pid=int(entry.name), command=command))
        except (OSError, UnicodeError, IndexError):
            continue
    if matches:
        raise ValueError("original unregistered worker is still present")
    return matches


def original_evidence(signatures):
    if sha256(ZERO) != ZERO_SHA:
        raise ValueError("zero-effect evidence differs")
    zero = json.loads(ZERO.read_text())
    if (
        zero["schema_version"] != "s2a.temporal-zero-step-original-failure.v1"
        or zero["scored_physics_steps"] != 0
        or zero["model_generations"] != 0
        or zero["original_failed_task_is_scientific_failure"]
    ):
        raise ValueError("original task2 is not an established zero-effect infrastructure failure")
    merge(signatures, zero["input_sha256"])
    merge(signatures, {str(ZERO): ZERO_SHA})
    references = {label: zero["original_evidence"][label] for label in ("launch", "partial_ready", "incident")}
    references["zero_effect"] = dict(path=str(ZERO), sha256=ZERO_SHA)
    for reference in references.values():
        if sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("original evidence reference changed")
        merge(signatures, {reference["path"]: reference["sha256"]})
    case, service = Path(zero["case_path"]), Path(zero["service_path"])
    if (
        not case.is_dir()
        or list(case.iterdir())
        or sorted(p.name for p in service.iterdir()) != sorted(zero["service_entries"])
    ):
        raise ValueError("old failed case/service acquired additional outputs")
    launch = json.loads(Path(references["launch"]["path"]).read_text())
    part = next(part for part in launch["partitions"] if part["partition"] == 2)
    directory = BASE / "temporal_equal_development/task2/orchestration"
    model = json.loads((directory / "model_process.json").read_text())
    matrix = json.loads((directory / "matrix_process.json").read_text())
    identities = [dict(part["coordinator"], cuda=""), dict(model, cuda="6"), dict(matrix, cuda="")]
    absent = []
    for item in identities:
        status = process_record(item["pid"], item["cuda"], start_ticks=item["start_ticks"])
        if status["present"]:
            raise ValueError("original task2 owned process is still present")
        absent.append(dict(pid=item["pid"], start_ticks=item["start_ticks"], present=False))
    no_original_worker()
    return references, absent


def prepare(source_review, source_review_sha):
    cpu_environment()
    config, root, _, checkpoint = protocol(CONFIG)
    if not unused_root(root):
        raise ValueError("continuation output already contains a prior attempt")
    if sha256(OLD_GATE) != OLD_GATE_SHA:
        raise ValueError("original020 gate changed")
    old = json.loads(OLD_GATE.read_text())
    if not old["all_prerequisites_passed"] or old["partition"] != 2:
        raise ValueError("wrong original020 partition gate")
    signatures = dict(old["input_sha256"])
    merge(signatures, {str(OLD_GATE): OLD_GATE_SHA})
    references, absent = original_evidence(signatures)
    source_review = Path(source_review).resolve()
    if not source_review.is_relative_to(REPO / "outputs") or sha256(source_review) != source_review_sha:
        raise ValueError("source review path/SHA differs")
    review = json.loads(source_review.read_text())
    if review.get("review_passed") is not True:
        raise ValueError("new source review is not passed")
    merge(signatures, review["input_sha256"])
    merge(signatures, {str(source_review): source_review_sha})
    merge(signatures, execution_inputs(config, REPO / CONFIG))
    for relative in (CONFIG, PLAN):
        merge(signatures, {str(REPO / relative): registered(relative)})
    for relative in SOURCES:
        path = REPO / relative
        if review["input_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"new executable source not bound by source review: {relative}")
        merge(signatures, {str(path): sha256(path)})
    verify_hashes(signatures)
    return dict(
        schema_version="s2a.temporal-zero-step-continuation-gate.v1",
        all_prerequisites_passed=True,
        partition=2,
        physical_gpu=6,
        port=9946,
        registered_budget=BUDGET,
        config_path=str(REPO / CONFIG),
        config_sha256=sha256(REPO / CONFIG),
        original_evidence=references,
        source_review=dict(path=str(source_review), sha256=source_review_sha),
        original020_gate=dict(path=str(OLD_GATE), sha256=OLD_GATE_SHA),
        qualification=old["qualification"],
        prior_seed43=old["prior_seed43"],
        original_task2_registered_processes_absent=absent,
        input_sha256=signatures,
        actual_checkpoint_stat=weight_stats(checkpoint),
        model_started=False,
        physics_started=False,
        new_model_requests=0,
        new_physics_steps=0,
        automatic_retry=False,
    )


def validate_gate(path, expected_sha):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO / "outputs") or sha256(path) != expected_sha:
        raise ValueError("continuation gate path/SHA differs")
    gate = json.loads(path.read_text())
    expected = prepare(gate["source_review"]["path"], gate["source_review"]["sha256"])
    if gate != expected:
        raise ValueError("continuation gate differs from independently rechecked prerequisites")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO / "outputs") or output == REPO / "outputs" or output.exists():
        raise ValueError("preparation output must be a new repository outputs directory")
    gate = prepare(args.source_review, args.source_review_sha256)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "partition2.json"
    write_new(path, gate)
    prepared = dict(
        schema_version="s2a.temporal-zero-step-continuation-preparation.v1",
        all_prerequisites_passed=True,
        gate=dict(path=str(path), sha256=sha256(path)),
        input_sha256=gate["input_sha256"],
        new_model_requests=0,
        new_physics_steps=0,
    )
    write_new(output / "prepared.json", prepared)
    print(
        json.dumps(
            dict(
                path=str(output / "prepared.json"),
                sha256=sha256(output / "prepared.json"),
                all_prerequisites_passed=True,
            )
        )
    )


if __name__ == "__main__":
    main()
