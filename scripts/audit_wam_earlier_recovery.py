"""Audit only the frozen12-case8400 recovery batch using the original full auditor.

No model or simulator is started. The six existing oracle qualifications remain
an explicitly referenced prior artifact, outside this WAM batch's denominator.
"""

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_recovery import Evidence, verify_case

from s2a.artifacts import REPO

CONFIG = "configs/wam_recovery_pilot_8400_g10_20261021.json"
ORIGINAL_AUDITOR = "scripts/audit_wam_recovery.py"
ORIGINAL_AUDITOR_SHA256 = "ee20bbc41688a16f590370c5a1a90f7e30b78d2b24f6f15bb0be57086cc0e4b0"
QUALIFICATION = "outputs/wam_stability_20261021/recovery_full_progress_20261021_07.json"
VARIANTS = (
    "intent_hold",
    "recovering_intent_hold",
    "recovering_intent_hold_alpha05",
    "recovering_intent_hold_alpha0",
)
SCHEMA = "s2a.recovery-earlier-readonly-audit.v1"


def process_environment(pid, expected_cuda, *, start_ticks=None):
    proc = Path(f"/proc/{pid}")
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        environment_bytes = (proc / "environ").read_bytes()
    except FileNotFoundError:
        return dict(pid=pid, present=False)
    actual_start = fields[19]
    if start_ticks is not None:
        assert actual_start == str(start_ticks), "owned coordinator PID was reused"
    environ = dict(item.split(b"=", 1) for item in environment_bytes.split(b"\0") if b"=" in item)
    assert environ[b"CUDA_VISIBLE_DEVICES"].decode() == expected_cuda
    assert environ[b"LD_LIBRARY_PATH"] == b""
    return dict(pid=pid, present=True, start_ticks=actual_start, cuda_visible_devices=expected_cuda, ld_library_path="")


def service_snapshot(config, launch, evidence):
    root = Path(config["output_dir"])
    label = "8400_masked_" + "_".join(VARIANTS)
    directory = root / "orchestration" / label
    manifest = evidence.json(directory / "manifest.json")
    evidence.check(CONFIG, manifest["config_sha256"])
    assert manifest["config"] == config and manifest["physical_gpu"] == launch["physical_gpu"] == 4
    assert manifest["coordinator_pid"] == launch["pid"]
    expected_jobs = [
        [
            variant,
            [
                str((REPO / ".venv/bin/python").absolute()),
                "scripts/run_wam_recovery_matrix.py",
                "--config",
                CONFIG,
                "--seed",
                "42",
                "--source",
                "wam",
                "--variant",
                variant,
                "--checkpoint",
                "8400_masked",
            ],
        ]
        for variant in VARIANTS
    ]
    assert manifest["jobs"] == expected_jobs
    evidence.check(
        "outputs/wam_independent_20261020/checkpoint_fingerprints.json",
        manifest["fingerprint_manifest_sha256"],
    )
    fingerprints = evidence.json("outputs/wam_independent_20261020/checkpoint_fingerprints.json")
    weights = next(c for c in fingerprints["checkpoints"] if c["name"] == "8400_masked")
    checkpoint = config["checkpoints"][0]
    assert Path(weights["checkpoint"]) == Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"
    for filename, signature in weights["files"].items():
        stat = Path(filename).stat()
        assert stat.st_size == signature["bytes"] and stat.st_mtime_ns == signature["mtime_ns"]
    model = evidence.json(directory / "model_process.json")
    assert model["gpu"] == 4 and model["port"] == 9244
    processes = dict(
        coordinator=process_environment(launch["pid"], "", start_ticks=launch["start_ticks"]),
        model=process_environment(model["pid"], "4"),
        matrix_workers={},
    )
    for variant in VARIANTS:
        path = directory / f"{variant}_process.json"
        if path.exists():
            process = evidence.json(path)
            assert process["command"] == next(command for name, command in expected_jobs if name == variant)
            processes["matrix_workers"][variant] = process_environment(process["pid"], "")
    info_path = directory / "server_info.json"
    if info_path.exists():
        info = evidence.json(info_path)
        assert Path(info["checkpoint_path"]).resolve() == Path(weights["checkpoint"]).resolve()
        assert not info["ema_weights"] and info["acceptance_intervention"]["active_slot_mask"]
    error_path = directory / "error.json"
    recorded_error = evidence.json(error_path) if error_path.exists() else None
    complete_path, release_path = directory / "complete.json", directory / "release.json"
    released = False
    if complete_path.exists() and release_path.exists():
        complete, release = evidence.json(complete_path), evidence.json(release_path)
        assert complete == dict(complete=True, variants=list(VARIANTS))
        assert release["model_pid"] == model["pid"] and release["model_returncode"] in (0, -15)
        released = not processes["model"]["present"] and not any(
            p["present"] for p in processes["matrix_workers"].values()
        )
    return dict(
        processes=processes,
        recorded_execution_error=recorded_error,
        owned_model_and_matrix_workers_released_normally=released and recorded_error is None,
        reused_complete_weight_fingerprint_files=len(weights["files"]),
        full_weight_bytes_rehashed_now=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-verified", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO / "outputs"):
        raise ValueError("audit snapshot must be a new file inside repository outputs")
    evidence, cases, reusable = Evidence(), [], {}
    evidence.read(__file__)
    evidence.check(ORIGINAL_AUDITOR, ORIGINAL_AUDITOR_SHA256)
    original = subprocess.check_output(["git", "show", f"cd1580b:{ORIGINAL_AUDITOR}"])
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_AUDITOR_SHA256
    config = evidence.json(CONFIG)
    frozen_config = subprocess.check_output(["git", "show", f"fd17d78:{CONFIG}"])
    evidence.check(CONFIG, hashlib.sha256(frozen_config).hexdigest())
    reference_config = evidence.json("configs/wam_recovery_pilot_8800_g10_20261021.json")
    omitted = {"checkpoints", "output_dir", "notes"}
    assert {k: v for k, v in config.items() if k not in omitted} == {
        k: v for k, v in reference_config.items() if k not in omitted
    }
    assert config["records"] == ["task0_init0", "task1_init0", "task2_init0"]
    assert config["sampling_seeds"] == [42] and config["online_max_steps"] == 600
    assert tuple(v["name"] for v in config["variants"]) == VARIANTS
    ck = config["checkpoints"][0]
    assert len(config["checkpoints"]) == 1 and ck["name"] == "8400_masked"
    assert ck["gpu"] == 4 and ck["port"] == 9244 and ck["iteration"] == "iter_000008400"
    launch = evidence.json("outputs/wam_stability_20261021/earlier_recovery_coordinator/launch.json")
    assert launch["additional_wam_trials"] == 12 and launch["physics_steps_budget"] == 7200
    assert launch["model_query_budget_upper_bound"] == 180 and launch["budget_introducing_commit"] == "fd17d78"
    subprocess.run(["git", "merge-base", "--is-ancestor", "fd17d78", launch["launch_head"]], check=True)
    qualification = evidence.json(QUALIFICATION)
    assert qualification["schema_version"] == "s2a.recovery-readonly-audit.v2" and qualification["scope"] == "full"
    assert qualification["oracle_full_independent_gate_passed"] and qualification["oracle_basic_gate_passed"]
    assert (
        qualification["source_before_after_hashes"][str((REPO / ORIGINAL_AUDITOR).resolve())] == ORIGINAL_AUDITOR_SHA256
    )
    oracle_cases = [c for c in qualification["cases"] if c["source"] == "oracle"]
    assert len(oracle_cases) == 6 and all(c["status"] == "verified" and c["outcome"]["stable"] for c in oracle_cases)
    if args.reuse_verified:
        previous_path = args.reuse_verified.resolve()
        assert previous_path.is_relative_to(REPO / "outputs")
        previous = evidence.json(previous_path)
        assert previous["schema_version"] == SCHEMA
        own_path = str(Path(__file__).resolve())
        assert previous["source_before_after_hashes"][own_path] == evidence.hashes[own_path]
        for path, signature in previous["source_before_after_hashes"].items():
            evidence.check(path, signature)
        reusable = {str(Path(c["case"]).resolve()): c for c in previous["cases"] if c["status"] == "verified"}
    evidence.check(config["fixture"], config["fixture_sha256"])
    fixtures = {r["name"]: r for r in evidence.json(config["fixture"])["records"]}
    observed_requests = 0
    for variant in config["variants"]:
        for name in config["records"]:
            case = Path(config["output_dir"]) / "simulation" / f"8400_masked_{variant['name']}" / f"{name}_s42"
            requests = len(list(case.glob("request_*.json")))
            observed_requests += requests
            assert requests <= 15, "single trial exceeded registered request upper bound"
            identity = dict(
                case=str(case), source="wam", checkpoint="8400_masked", variant=variant["name"], record=name
            )
            marker = case / "result.json"
            if not marker.exists():
                cases.append(dict(identity, status="partial" if case.exists() else "missing"))
                continue
            try:
                json.loads(marker.read_bytes())
            except json.JSONDecodeError:
                cases.append(dict(identity, status="partial", reason="result marker being written"))
                continue
            result = evidence.json(marker)
            if result["error"]:
                cases.append(dict(identity, status="execution_error", error=result["error"]))
                continue
            assert requests == result["chunks"], "complete trial has unreferenced/missing request records"
            cached = reusable.get(str(case.resolve()))
            if cached is not None:
                cases.append(dict(cached, reused_from=str(args.reuse_verified.resolve())))
                continue
            try:
                cases.append(
                    verify_case(
                        config,
                        fixtures[name],
                        variant,
                        case,
                        evidence,
                        source="wam",
                        checkpoint="8400_masked",
                        full=True,
                    )
                )
            except Exception as exc:
                cases.append(dict(identity, status="audit_error", error=repr(exc), traceback=traceback.format_exc()))
    assert len(cases) == 12 and observed_requests <= 180
    resources = service_snapshot(config, launch, evidence)
    groups = []
    for variant in config["variants"]:
        selected = [c for c in cases if c["variant"] == variant["name"]]
        verified = [c for c in selected if c["status"] == "verified"]
        complete = len(verified) == 3
        groups.append(
            dict(
                variant=variant["name"],
                model_rotation_alpha=variant["model_rotation_alpha"],
                expected_cases=3,
                verified_cases=len(verified),
                complete=complete,
                stable_successes_among_verified=sum(c["outcome"]["stable"] for c in verified),
                ever_successes_among_verified=sum(c["outcome"]["ever"] for c in verified),
                stable_rate=sum(c["outcome"]["stable"] for c in verified) / 3 if complete else None,
                raw_checkpoint_reference=variant["model_rotation_alpha"] == 1,
                component_ablation_only=variant["model_rotation_alpha"] == 0,
                tracking_pooled_equal_length_cases={
                    ref: {
                        metric: float(np.sqrt(np.mean([c["tracking"][ref][metric] ** 2 for c in verified])))
                        for metric in ("position_rmse_m", "so3_rmse_rad")
                    }
                    for ref in verified[0]["tracking"]
                }
                if complete
                else None,
            )
        )
    all_cases_verified = all(c["status"] == "verified" for c in cases)
    evidence.read(__file__)
    evidence.recheck()
    value = dict(
        schema_version=SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="only12_frozen_8400_wam_development_trials",
        complete=all_cases_verified and resources["owned_model_and_matrix_workers_released_normally"],
        all12_cases_independently_verified=all_cases_verified,
        reused_verified_cases=sum("reused_from" in c for c in cases),
        registered_budget=dict(wam_trials=12, physics_steps=7200, model_queries_upper_bound=180),
        verified_physics_steps=sum(c.get("steps", 0) for c in cases if c["status"] == "verified"),
        observed_request_files=observed_requests,
        cases=cases,
        groups=groups,
        resources=resources,
        existing_qualification_reference=dict(
            path=QUALIFICATION,
            sha256=evidence.hashes[str((REPO / QUALIFICATION).resolve())],
            oracle_cases=6,
            stable_oracle_cases=6,
            full_independent_gate_passed=True,
            oracle_cases_replayed_by_this_wrapper=0,
            added_oracle_trials=0,
            oracle_excluded_from_wam_denominator=True,
        ),
        source_before_after_hashes=evidence.hashes,
        warnings=[
            "Only the registered12 old-development WAM trials are traversed; no fresh holdout or new oracle run.",
            "Native outcome/contact/object fields remain metrics, never causal replay inputs.",
            "Alpha.5 changes generated references; alpha0 is constant-orientation component diagnosis, not raw full-checkpoint success.",
            "Small admitted error does not establish following raw/effective reference on the nominal physical clock.",
            "Full governor/query/control/gripper reconstruction is imported from unchanged cd1580b auditor; its30-case protocol is untouched.",
            "Prior artifact07 is referenced as historical six-case oracle qualification, not rerun or counted as these12 WAM successes.",
            "Weights reuse prior complete byte SHA with current size/mtime checks; no new full-weight byte hash is claimed.",
            "Partial/error cases and pending resource release prevent complete=True; no partial winner is selected.",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                complete=value["complete"],
                statuses={s: sum(c["status"] == s for c in cases) for s in {c["status"] for c in cases}},
            )
        )
    )


if __name__ == "__main__":
    main()
