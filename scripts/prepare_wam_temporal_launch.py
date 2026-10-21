"""Prepare020 gates after actual qualification and019 final release; launches nothing."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from prepare_wam_raw8_seed43 import merge, unused_root
from run_wam_long_service import weight_stats
from run_wam_short_replan import cpu_environment, process_record, verify_hashes, write_new

from s2a.artifacts import REPO, sha256


def registered(relative):
    original = subprocess.check_output(["git", "show", f"7025c45:{relative}"], cwd=REPO)
    expected = hashlib.sha256(original).hexdigest()
    assert sha256(REPO / relative) == expected
    return expected


def prepare(qualification_path, qualification_sha, prior_path, prior_sha):
    from run_wam_temporal_matrix import protocol

    from s2a.temporal_ensemble_simulation import execution_inputs

    cpu_environment()
    assert Path.cwd().resolve() == REPO
    signatures, sources = {}, {}
    for label, path, digest in (
        ("qualification", qualification_path, qualification_sha),
        ("prior_seed43", prior_path, prior_sha),
    ):
        path = Path(path).resolve()
        assert path.is_relative_to(REPO / "outputs") and sha256(path) == digest
        value = json.loads(path.read_text())
        assert value["complete"] and value["normal_process_release"]
        if label == "qualification":
            assert value["schema_version"] == "s2a.temporal-equal-oracle-qualification.v1"
            assert value["registered_budget"] == dict(
                oracle_trials=3, physical_steps=1800, model_requests=0, fresh_initial_states=0
            )
            assert value["errors"] == [] and value["source_oracle_only"]
            assert value["oracle_is_wam_success_evidence"] is False
            assert value["qualified"] and value["verified_oracle_cases"] == 3
            assert value["verified_physics_log_endpoints"] == 1800 and value["all1800_common_step_fields_exact_to_007"]
        else:
            assert value["schema_version"] == "s2a.raw8-seed43-independent-audit.v1"
            assert value["all_cases_independently_verified"] and value["verified_cases"] == 3
            assert value["registered_budget"] == dict(
                wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0
            )
            assert (
                value["verified_physics_steps"] == 1800
                and value["verified_queries"] == value["verified_http_queries"] == 225
            )
            assert value["errors"] == []
        merge(signatures, value["input_sha256"])
        merge(signatures, {str(path): digest})
        sources[label] = dict(path=str(path), sha256=digest)
    plan = "plans/v0.0.1/020_causal_temporal_ensemble_20261021.md"
    merge(signatures, {str(REPO / plan): registered(plan)})
    ready_path = REPO / "outputs/wam_stability_20261021/raw8_seed43_review/actual_ready_snapshot.json"
    assert signatures[str(ready_path)] == sha256(ready_path)
    ready = json.loads(ready_path.read_text())
    assert ready["schema_version"] == "s2a.raw8-seed43-live-ready-snapshot.v1" and ready["ready"]
    assert [part["partition"] for part in ready["partitions"]] == [0, 1, 2]
    absent = []
    for part in ready["partitions"]:
        assert len(part["local_registered_processes"]) == 4
        assert {identity["role"] for identity in part["local_registered_processes"]} == {
            "coordinator",
            "model",
            "matrix",
            "physics",
        }
        for identity in part["local_registered_processes"]:
            state = process_record(
                identity["pid"], identity["environment"]["CUDA_VISIBLE_DEVICES"], start_ticks=identity["start_ticks"]
            )
            assert not state["present"]
            absent.append(dict(pid=identity["pid"], start_ticks=identity["start_ticks"], present=False))
    assert len(absent) == 12 and len({p["pid"] for p in absent}) == 12
    partitions = []
    for part in range(3):
        relative = f"configs/wam_temporal_equal_task{part}_20261021.json"
        merge(signatures, {str(REPO / relative): registered(relative)})
        config, root, records, ck = protocol(
            relative, source="wam", variant_name="temporal_equal", checkpoint="8400_masked"
        )
        assert config["records"] == [f"task{part}_init0"] and config["sampling_seeds"] == [42] and len(records) == 1
        assert ck["gpu"] == 4 + part and ck["port"] == 9844 + part and unused_root(root)
        merge(signatures, execution_inputs(config, relative))
        stats = weight_stats(ck)
        partitions.append(
            dict(
                partition=part,
                physical_gpu=ck["gpu"],
                port=ck["port"],
                config_path=str(REPO / relative),
                config_sha256=sha256(REPO / relative),
                output_dir=str(root),
                actual_checkpoint_stat=stats,
            )
        )
    for name in (
        __file__,
        "scripts/prepare_wam_raw8_seed43.py",
        "scripts/run_wam_temporal_service.py",
        "scripts/capture_wam_temporal_ready.py",
        "scripts/serve_wam_acceptance.py",
        "src/s2a/wam_runtime.py",
    ):
        path = Path(name).resolve()
        merge(signatures, {str(path): sha256(path)})
    verify_hashes(signatures)
    return dict(
        schema_version="s2a.temporal-equal-launch-preparation.v1",
        all_prerequisites_passed=True,
        qualification=sources["qualification"],
        prior_seed43=sources["prior_seed43"],
        prior12_owned_processes_absent=absent,
        partitions=partitions,
        input_sha256=signatures,
        new_model_requests=0,
        new_physics_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--prior-seed43", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(REPO / "outputs") and output != REPO / "outputs" and not output.exists()
    value = prepare(args.qualification, args.qualification_sha256, args.prior_seed43, args.prior_sha256)
    output.mkdir(parents=True, exist_ok=False)
    gates = []
    for part in value["partitions"]:
        gate = dict(
            schema_version="s2a.temporal-equal-launch-gate.v1",
            all_prerequisites_passed=True,
            qualification=value["qualification"],
            prior_seed43=value["prior_seed43"],
            partition=part["partition"],
            input_sha256=value["input_sha256"],
        )
        path = output / f"partition{part['partition']}.json"
        write_new(path, gate)
        gates.append(dict(partition=part["partition"], path=str(path), sha256=sha256(path)))
    value["gates"] = gates
    path = output / "prepared.json"
    write_new(path, value)
    print(json.dumps(dict(path=str(path), sha256=sha256(path), all_prerequisites_passed=True)))


if __name__ == "__main__":
    main()
