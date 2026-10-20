"""Independent013 state/backend audit; reads saved evidence without inference."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from audit_wam_sampler_steps import (
    Evidence,
    check_saved_metrics,
    expected_schedule,
    integrate_and_score,
    json_sha,
    pool,
    tensor_header,
    value_identity,
)
from run_wam_short_replan import process_record
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/wam_stability_20261021"
STUDY = BASE / "backend_state/8400_flash3_video"
SDPA = BASE / "sampler_steps_state"
HELPER = ROOT / "scripts/audit_wam_sampler_steps.py"
HELPER_SHA = "12329768297939fe6b15b4127aa6b65e144578fd72b0e896dd306f9e88b10069"
READY = BASE / "backend_runtime_review/services_ready_resource_snapshot.json"
READY_SHA = "f126e8bf79c6fed8b9c44f2128bcadc51ae44fe38b015a0ec83f9f8341bc6634"
PREFIXES = (8, 16, 40)
SHUTDOWN_SUPPLEMENT = BASE / "backend_runtime_review/shutdown_observability.json"
SHUTDOWN_SHA = "4c74e6c27e197ec50a4e1fbc3ca20ba9cfb23b0453531c929a5c20ff21c12c17"
SIGNAL_PROBE_SHA = "ef8b5284017d429c678bd614c0b4a899c87bfe584c00b1f3cdfa7be239dbe46a"
CONFIG_UTILS = Path(
    "/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm/.venv/lib/python3.13/site-packages/diffusers/configuration_utils.py"
)


def eligibility(candidate, baseline):
    """Apply the registered six comparisons in SI; display positions remain mm."""
    checks = []
    for horizon in (8, 16):
        for key, scale, unit in (
            ("position_rmse_mm", 0.001, "m"),
            ("rotation_rmse_rad", 1, "rad"),
            ("opening_mae", 1, "normalized opening"),
        ):
            actual = candidate[str(horizon)]["wam"][key] * scale
            reference = baseline[str(horizon)]["wam"][key] * scale
            checks.append(
                dict(
                    horizon=horizon,
                    metric=key,
                    unit=unit,
                    candidate=actual,
                    reference=reference,
                    nonregression=actual <= reference + 1e-10,
                )
            )
    improved = any(
        baseline[str(h)]["wam"]["position_rmse_mm"] > 0
        and candidate[str(h)]["wam"]["position_rmse_mm"] <= 0.9 * baseline[str(h)]["wam"]["position_rmse_mm"]
        for h in (8, 16)
    )
    return dict(
        checks=checks,
        all_six_nonregression=all(c["nonregression"] for c in checks),
        at_least_one_position_improves_ten_percent=improved,
        passed=all(c["nonregression"] for c in checks) and improved,
        absolute_tolerance=1e-10,
        physical_trial_authorized=False,
        fresh_evaluation_authorized=False,
    )


def scheduler_identity(config):
    """Normalize only Diffusers' unordered default-name bookkeeping.

    configuration_utils.register_to_config builds _use_default_values via
    list(set(...)); separate Python services can serialize different orders.
    Keep the complete key set and every effective scheduler value exact.
    """
    defaults = config["_use_default_values"]
    if (
        not isinstance(defaults, list)
        or any(not isinstance(key, str) or key not in config for key in defaults)
        or len(set(defaults)) != len(defaults)
    ):
        raise ValueError("invalid scheduler default-name bookkeeping")
    return dict(config, _use_default_values=sorted(defaults))


def verify_trace(candidate, baseline, protocol, prepared_sha):
    if (
        not candidate["complete"]
        or not candidate["actual_state_and_condition_pair_with_sdpa35"]
        or candidate["prepared_sha256"] != prepared_sha
        or candidate["source_sha256"] != protocol["source_sha256"]
        or candidate["num_steps"] != 35
        or candidate["sampler_calls"] != 1
        or candidate["velocity_evaluation_count"] != 35
        or candidate["derived_network_forward_count"] != 35
    ):
        raise ValueError("013 trace identity/source/call-count differs")
    if (
        candidate["name"] != baseline["name"]
        or candidate["attention_backend"] != "flash3_video"
        or candidate["baseline_sdpa35_trace"] != protocol["baseline_traces"][baseline["name"]]
    ):
        raise ValueError("013 paired trace name/backend/provenance differs")
    if candidate["parameters_before"] != candidate["parameters_after"]:
        raise ValueError("013 model parameter identity changed")
    if candidate["parameters_before"]["count"] != baseline["parameters_before"]["count"]:
        raise ValueError("backend parameter count differs from the same checkpoint")
    if (
        candidate["observer_parameter_rng_identity_checks"]
        != "passed around each observation, separately from computation"
    ):
        raise ValueError("observer parameter/RNG guard missing")
    for field in (
        "initial_state",
        "clean_condition_inputs",
        "condition_reference",
        "condition_mask",
        "conditional_token_ids_sha256",
        "conditioning",
        "actual_sampler_arguments",
        "actual_generation_branch",
    ):
        if candidate[field] != baseline[field]:
            raise ValueError(f"013 non-backend input/branch differs: {field}")
    branch = candidate["actual_generation_branch"]
    if (
        branch["guidance"] != 1
        or branch["velocity_postprocess"] is not None
        or branch["velocity_postprocess_builder"] is not None
        or branch["_extra_num_steps"] != 0
        or branch["scheduler_type"] != "unipc"
    ):
        raise ValueError("source-derived single-forward NFE branch differs")
    state = candidate["initial_state"]
    headers = (
        list(state.values())
        + list(candidate["clean_condition_inputs"].values())
        + [candidate["condition_reference"], candidate["condition_mask"], candidate["final_sampler_state"]]
    )
    for header in headers:
        tensor_header(header, dtype="torch.float32")
        if header["device"] != "cuda:0" or header["inference_tensor"] == header["version_counter_available"]:
            raise ValueError("actual sampler tensor metadata differs")
    nvision = int(np.prod(state["vision"]["shape"]))
    if (
        state["action"]["shape"] != [40, 128]
        or state["flat"]["shape"] != [nvision + 5120]
        or state["action"]["storage_offset"] != nvision
    ):
        raise ValueError("actual sampler tensor layout differs")
    pair = dict(
        initial_state={k: value_identity(v) for k, v in state.items()},
        clean_condition_inputs={k: value_identity(v) for k, v in candidate["clean_condition_inputs"].items()},
        condition_reference=value_identity(candidate["condition_reference"]),
        condition_mask=value_identity(candidate["condition_mask"]),
        conditional_token_ids_sha256=candidate["conditional_token_ids_sha256"],
        request_conditioning=candidate["conditioning"],
    )
    if (
        candidate["pair_identity"] != pair
        or pair != baseline["pair_identity"]
        or candidate["pair_identity_sha256"] != json_sha(pair)
    ):
        raise ValueError("actual input digest witnesses do not pair with SDPA35")
    sigmas = np.asarray(candidate["sigma_values"], dtype=np.float32)
    times = np.asarray(candidate["scheduler_timestep_values"], dtype=np.int64)
    tensor_header(candidate["sigmas"], shape=(36,), dtype="torch.float32", bytes_value=sigmas.tobytes())
    tensor_header(candidate["scheduler_timesteps"], shape=(35,), dtype="torch.int64", bytes_value=times.tobytes())
    expected_sigmas, expected_times = expected_schedule(35)
    if not np.array_equal(sigmas, expected_sigmas) or not np.array_equal(times, expected_times):
        raise ValueError("actual sigma/timestep schedule differs from independent reconstruction")
    for field in ("sigma_values", "scheduler_timestep_values", "timesteps", "step_indexes"):
        if candidate[field] != baseline[field]:
            raise ValueError(f"013 sampler schedule differs from SDPA35: {field}")
    if scheduler_identity(candidate["scheduler_config"]) != scheduler_identity(baseline["scheduler_config"]):
        raise ValueError("013 sampler effective configuration/default-name set differs from SDPA35")
    if candidate["timesteps"] != times.tolist() or candidate["step_indexes"] != [None] + list(range(1, 35)):
        raise ValueError("actual velocity evaluations do not follow the original schedule")


def read_case(case, sample, seed, protocol, config, transform, trace, evidence):
    result, raw, request = (evidence.json(case / p) for p in ("result.json", "response.json", "request.json"))
    name = f"{sample['name']}_h40_task_only_s{seed}"
    cached = protocol["cached_cases"][name]
    expected = evidence.json(cached["request_path"])
    if (
        request != expected
        or raw["status"] != "success"
        or result["response_sha256"] != evidence.hashes[str(case / "response.json")]
    ):
        raise ValueError("request/response differs from fixed01135 input")
    for key, value in dict(
        name=name,
        checkpoint="8400_masked",
        sample=sample["name"],
        episode=sample["episode"],
        phase=sample["phase"],
        split="diagnostic",
        anchor_frame=sample["anchor_frame"],
        sampling_seed=seed,
        horizon=40,
        prompt_mode="task_only",
    ).items():
        if result[key] != value:
            raise ValueError(f"result identity differs: {key}")
    content, args = raw["outputs"][0]["content"], raw["args"]
    action, normalized = (
        np.asarray(x, dtype=np.float32) for x in (content["action"], content["action128"]["normalized_action"])
    )
    if any(x.shape != (40, 128) or not np.isfinite(x).all() or np.any(x[:, 10:]) for x in (action, normalized)):
        raise ValueError("invalid full40x128 raw/normalized arrays")
    meta = content["action128"]
    if (
        meta["checkpoint_path"] != protocol["checkpoint"]
        or type(meta["seed"]) is not int
        or meta["seed"] != seed
        or meta["action_fps"] != 20
        or meta["action_domain"] != config["action_domain"]
        or meta["action_units"] != "raw_physical_or_source_units"
        or meta["action_dim_mask"] != [True] * 10 + [False] * 118
        or not all(type(v) is bool for v in meta["action_dim_mask"])
    ):
        raise ValueError("raw physical contract/weight/seed/mask differs")
    np.testing.assert_array_equal(meta["action_timestamps_seconds"], np.arange(40) / 20)
    decoded = normalized.copy()
    mask = np.asarray(transform["normalization_mask"], dtype=bool)
    decoded[:, mask] = (decoded[:, mask] - np.asarray(transform["bias"], np.float32)[mask]) / np.asarray(
        transform["scale"], np.float32
    )[mask]
    np.testing.assert_allclose(decoded, action, atol=2e-7, rtol=0)
    for key in (
        "seed",
        "fps",
        "num_frames",
        "action_chunk_size",
        "guidance",
        "num_steps",
        "shift",
        "prompt",
        "model_mode",
    ):
        if args[key] != request[key]:
            raise ValueError(f"actual resolved model request differs: {key}")
    if (
        args["condition_frame_indexes_vision"] != [0]
        or args["action_path"] is not None
        or args["native_prompt_upsampling"]
    ):
        raise ValueError("unexpected actual model conditioning")
    conditioning = trace["conditioning"]
    if (
        conditioning["original_request_without_steps"] != {k: v for k, v in request.items() if k != "num_steps"}
        or conditioning["image_sha256"] != sample["image_sha256"]
        or meta["conditioning_caption_sha256"] != hashlib.sha256(conditioning["caption"][0].encode()).hexdigest()
    ):
        raise ValueError("actual image/text/request conditioning differs")
    if trace["response_arrays_sha256"] != [
        json_sha(np.asarray(x).tolist()) for x in (content["action"], meta["normalized_action"])
    ]:
        raise ValueError("trace is not bound to exact returned full128 values")
    output_dir = Path(args["output_dir"])
    if (
        not output_dir.name.startswith("generate_")
        or output_dir.parent != (STUDY if case.is_relative_to(STUDY) else SDPA) / "service"
    ):
        raise ValueError("server-local response directory differs")
    if (
        evidence.json(output_dir / "sample_outputs.json") != raw
        or evidence.json(output_dir / "sample_args.json") != args
    ):
        raise ValueError("HTTP response differs from original server-local output")
    if evidence.json(output_dir / "acceptance_input.json") != dict(
        caption=conditioning["caption"], exact_published_caption=False, active_slot_mask=True, seed=seed
    ):
        raise ValueError("per-request receipt differs")
    evidence.check(sample["arrays_path"], sample["arrays_sha256"])
    evidence.check(sample["image_path"], sample["image_sha256"])
    with np.load(sample["arrays_path"]) as data:
        count, errors, generated = integrate_and_score(
            np.asarray(data["states"], float), sample, np.asarray(content["action"], float)
        )
    if count != result["valid_future_steps"]:
        raise ValueError("valid future denominator differs")
    check_saved_metrics(errors["wam"], result["metrics"])
    for method in ("hold", "constant_velocity"):
        check_saved_metrics(errors[method], result["baselines"][method])
    return dict(
        name=name,
        sample=sample,
        seed=seed,
        errors=errors,
        generated=generated,
        parameters=trace["parameters_before"],
        server_request_dir=str(output_dir),
        response_sha256=result["response_sha256"],
    )


def state_distance(pairs):
    values = {str(n): dict(position=[], rotation=[], opening=[]) for n in PREFIXES}
    for old, new in pairs:
        for h in PREFIXES:
            for i in range(min(h, len(old["errors"]["wam"]["position"]))):
                values[str(h)]["position"].append(
                    np.linalg.norm(np.asarray(old["generated"]["position"][i]) - new["generated"]["position"][i])
                )
                values[str(h)]["rotation"].append(
                    Rotation.from_matrix(
                        np.asarray(old["generated"]["rotation"][i]) @ np.asarray(new["generated"]["rotation"][i]).T
                    ).magnitude()
                )
                values[str(h)]["opening"].append(abs(old["generated"]["opening"][i] - new["generated"]["opening"][i]))
    return {
        h: dict(
            valid_endpoints=len(v["position"]),
            position_rmse_mm=float(np.sqrt(np.mean(np.square(v["position"]))) * 1000),
            rotation_rmse_rad=float(np.sqrt(np.mean(np.square(v["rotation"])))),
            opening_mae=float(np.mean(v["opening"])),
        )
        for h, v in values.items()
    }


def validate_shutdown_supplement(artifact, probe, names):
    """Validate only finite saved evidence; never infer missing memory values."""
    if (
        artifact["schema_version"] != "s2a.readonly-shutdown-observability.v1"
        or not artifact["normal_owned_process_release"]
        or artifact["closed_json_exists"]
        or artifact["mask_evidence_json_exists"]
        or artifact["poisoned_final_memory_state_observed"] is not None
        or artifact["initial_noise_rms_ledger_observed"] is not None
        or artifact["completed_results_and_trace_pairs"] != 48
        or len(artifact["cases"]) != 48
        or [c["name"] for c in artifact["cases"]] != names
        or artifact["new_model_requests"] != 0
        or artifact["new_physics_steps"] != 0
        or not probe["cpu_only"]
        or probe["new_model_requests"] != 0
        or probe["new_physics_steps"] != 0
        or probe["returncode"] != -15
        or not probe["entered_exists"]
        or probe["finally_exists"]
    ):
        raise ValueError("finite shutdown supplement/probe identity differs")


def shutdown_observability(evidence, args, protocol, release, processes):
    closed_path, mask_path = STUDY / "service/closed.json", STUDY / "service/mask_evidence.json"
    if args.shutdown_supplement is None:
        closed, masks = evidence.json(closed_path), evidence.json(mask_path)
        if (
            closed != dict(completed=48, pending=None, poisoned=False)
            or len(masks) != 48
            or any(m["inactive_rms_after"] != 0 for m in masks)
        ):
            raise ValueError("final service receipts do not cover48 unpoisoned initial-noise masks")
        return dict(
            mode="strict_saved_shutdown_receipts",
            closed_json_available=True,
            initial_noise_rms_ledger_available=True,
            final_poisoned_memory_state=closed["poisoned"],
            full_internal_ledger_audit=False,
        )
    if (
        args.shutdown_supplement.resolve() != SHUTDOWN_SUPPLEMENT
        or args.shutdown_supplement_sha256 != SHUTDOWN_SHA
        or closed_path.exists()
        or mask_path.exists()
    ):
        raise ValueError("only the explicit pre-reviewed absent-ledger supplement is accepted")
    evidence.check(SHUTDOWN_SUPPLEMENT, SHUTDOWN_SHA)
    artifact = evidence.json(SHUTDOWN_SUPPLEMENT)
    for path, digest in artifact["input_sha256"].items():
        evidence.check(path, digest)
    evidence.check(artifact["cpu_signal_probe_path"], SIGNAL_PROBE_SHA)
    probe = evidence.json(artifact["cpu_signal_probe_path"])
    for path, digest in probe["source_sha256"].items():
        evidence.check(path, digest)
    validate_shutdown_supplement(artifact, probe, protocol["request_order"])
    if artifact["release"] != release:
        raise ValueError("shutdown supplement belongs to another release")
    for role, witness in artifact["processes"].items():
        actual = processes["state" if role == "worker" else role]
        if (
            witness["pid"] != actual["pid"]
            or witness["registered_start_ticks"] != actual["registered_start_ticks"]
            or witness["present"]
            or actual["present"]
        ):
            raise ValueError("shutdown supplement belongs to another process lifetime")
    for case in artifact["cases"]:
        for label in ("trace", "result", "response"):
            record = case[label]
            evidence.check(record["path"], record["sha256"])
    log = evidence.read(STUDY / "orchestration/run/service.log").decode()
    statuses = re.findall(r'"POST /generate HTTP/1\.1" (\d{3})', log)
    if statuses != ["200"] * 48:
        raise ValueError("finite service log does not prove exactly48 successful HTTP requests")
    return dict(
        mode="explicit_sigterm_shutdown_supplement",
        supplement_path=str(SHUTDOWN_SUPPLEMENT),
        supplement_sha256=SHUTDOWN_SHA,
        cpu_signal_probe_sha256=SIGNAL_PROBE_SHA,
        closed_json_available=False,
        initial_noise_rms_ledger_available=False,
        final_poisoned_memory_state=None,
        full_internal_ledger_audit=False,
        successful_http_requests=48,
        missing_values_reconstructed=False,
        limitation="Final in-memory poison/counter/pending state and initial-noise before/after RMS ledger were not saved; per-request trace/raw/mask/source and normal lifetime evidence support only the registered state-quality comparison.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--shutdown-supplement", type=Path)
    parser.add_argument("--shutdown-supplement-sha256")
    args = parser.parse_args()
    if bool(args.shutdown_supplement) != bool(args.shutdown_supplement_sha256):
        raise ValueError("explicit shutdown supplement requires its exact reviewed SHA")
    if (
        Path.cwd().resolve() != ROOT
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
        or sys.byteorder != "little"
    ):
        raise ValueError("audit requires repository cwd, little-endian CPU and empty CUDA/LD")
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("audit output must be a new repository artifact")
    evidence, rows, pairs, errors = Evidence(), [], [], []
    evidence.check(HELPER, HELPER_SHA)
    evidence.read(ROOT / "scripts/audit_wam_earlier_screen.py")
    evidence.read(CONFIG_UTILS)
    evidence.read(__file__)
    evidence.check(READY, READY_SHA)
    ready = evidence.json(READY)
    prepared_path = STUDY / "orchestration/prepared.json"
    protocol, launch, root_launch = (
        evidence.json(p)
        for p in (prepared_path, STUDY / "orchestration/run/launch.json", STUDY / "orchestration/root_launch.json")
    )
    prepared_sha = evidence.hashes[str(prepared_path)]
    if (
        root_launch["prepared_sha256"] != prepared_sha
        or launch["prepared_sha256"] != prepared_sha
        or launch["head"] != root_launch["launch_git_commit"]
        or root_launch["budget"] != dict(requests=48, physics_steps=0)
        or launch["coordinator"]["pid"] != root_launch["coordinator_pid"]
        or launch["coordinator"]["start_ticks"] != int(root_launch["coordinator_start_ticks"])
    ):
        raise ValueError("root/actual launch identity or budget differs")
    for commit in ("75c7cf1", "c08d433"):
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, root_launch["launch_git_commit"]], check=True)
    for field in ("source_sha256", "input_sha256", "snapshot_sha256"):
        for path, digest in protocol[field].items():
            evidence.check(path, digest)
    prior = evidence.json(protocol["guidance_audit_path"])
    if (
        not prior["complete"]
        or prior["verified_queries"] != 48
        or not prior["resources"]["owned_service_released_normally"]
    ):
        raise ValueError("012 complete independent release gate differs")
    for path, digest in prior["input_sha256"].items():
        evidence.check(path, digest)
    for path, frozen in protocol["weight_files"].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (frozen["bytes"], frozen["mtime_ns"]):
            raise ValueError("weight differs from prior full byte fingerprint")
    config = evidence.json(protocol["config_path"])
    manifest_path = STUDY / "samples/manifest.json"
    samples = evidence.json(manifest_path)["cases"]
    expected = {f"{s['name']}_h40_task_only_s{seed}": (s, seed) for s in samples for seed in (42, 43)}
    if len(expected) != 48 or list(expected) != protocol["request_order"] or len({s["episode"] for s in samples}) != 6:
        raise ValueError("registered sample/seed coverage differs")
    template = evidence.json(config["policy_template"])
    config = dict(config, action_domain=template["action_domain"])
    saved = yaml.safe_load(evidence.read(Path(config["checkpoints"][0]["run"]) / "config.yaml"))
    spec = saved["model"]["config"]["action128_spec"]
    transform = spec["transforms"][spec["registry"]["domains"].index(config["action_domain"])]
    info = evidence.json(STUDY / "orchestration/run/server_info.json")
    backend = evidence.json(STUDY / "service/actual_backend.json")
    if (
        info["checkpoint_path"] != protocol["checkpoint"]
        or info["ema_weights"]
        or info["attention_backend"] != "flash3_video"
        or info["compile"]
        or info["reuse_small_action_metadata"]
        or info["reuse_small_action_und"]
        or any(info[k] != 1 for k in ("world_size", "dp_shard_size", "cp_size", "cfgp_size"))
        or backend != info["acceptance_intervention"]["actual_backend_probe"]
        or backend != ready["actual_backend"]
        or backend["network_backend"] != "flash3_video"
        or backend["language_backend"] != "flash3_video"
        or not backend["flash3_supported"]
        or backend["provider"] != "nv"
        or backend["small_action_fixed_lengths"] is not None
        or backend["small_action_und_short_tokens"] != 0
        or backend["flash3_function_module"] != "cosmos_framework.model.attention.flash3.functions"
        or backend["cuda_device_uuid"] != launch["gpu_uuid"].lower().removeprefix("gpu-")
    ):
        raise ValueError("actual backend/provider/device/reuse/padding differs")
    for field in ("provider_module", "provider_interface"):
        evidence.check(backend[field], backend[field + "_sha256"])
    model, worker = (
        evidence.json(STUDY / "orchestration/run" / p) for p in ("model_process.json", "state_process.json")
    )
    if [(p["pid"], p["start_ticks"]) for p in ready["local_registered_processes"]] != [
        (root_launch["coordinator_pid"], int(root_launch["coordinator_start_ticks"])),
        (model["pid"], model["start_ticks"]),
        (worker["pid"], worker["start_ticks"]),
    ]:
        raise ValueError("historical live environment witness belongs to another process")
    state_dir = STUDY / "state_queries/8400_masked"
    query = evidence.optional(state_dir / "manifest.json")
    if query is not None and query["immutable"] != dict(
        config={k: v for k, v in config.items() if k != "action_domain"},
        samples_sha256=evidence.hashes[str(manifest_path)],
        checkpoint=protocol["checkpoint"],
    ):
        raise ValueError("state query immutable source differs")
    for name, (sample, seed) in expected.items():
        case, trace_path = state_dir / name, STUDY / "service/traces35" / f"{name}.json"
        if evidence.optional(case / "result.json") is None or evidence.optional(trace_path) is None:
            continue
        try:
            trace = evidence.json(trace_path)
            baseline_record = protocol["baseline_traces"][name]
            evidence.check(baseline_record["path"], baseline_record["sha256"])
            baseline = evidence.json(baseline_record["path"])
            verify_trace(trace, baseline, protocol, prepared_sha)
            candidate = read_case(case, sample, seed, protocol, config, transform, trace, evidence)
            reference = read_case(
                SDPA / "8400_steps35/state_queries/8400_masked" / name,
                sample,
                seed,
                protocol,
                config,
                transform,
                baseline,
                evidence,
            )
            cached = evidence.json(protocol["cached_cases"][name]["response_path"])["outputs"][0]["content"]
            old = evidence.json(SDPA / "8400_steps35/state_queries/8400_masked" / name / "response.json")["outputs"][0][
                "content"
            ]
            np.testing.assert_array_equal(cached["action"], old["action"])
            np.testing.assert_array_equal(
                cached["action128"]["normalized_action"], old["action128"]["normalized_action"]
            )
            rows.append(candidate)
            pairs.append((reference, candidate))
        except (ValueError, AssertionError, OSError, KeyError) as exc:
            errors.append(dict(case=name, error=repr(exc)))
    processes = dict(
        coordinator=process_record(
            root_launch["coordinator_pid"], "", start_ticks=int(root_launch["coordinator_start_ticks"])
        ),
        model=process_record(model["pid"], "7", start_ticks=model["start_ticks"]),
        state=process_record(worker["pid"], "", start_ticks=worker["start_ticks"]),
    )
    for role, ticks in (
        ("coordinator", int(root_launch["coordinator_start_ticks"])),
        ("model", model["start_ticks"]),
        ("state", worker["start_ticks"]),
    ):
        processes[role]["registered_start_ticks"] = ticks
    complete_marker = evidence.optional(STUDY / "orchestration/run/complete.json")
    state_complete = evidence.optional(state_dir / "complete.json")
    release = evidence.optional(STUDY / "orchestration/run/release.json")
    execution_error = evidence.optional(STUDY / "orchestration/run/error.json")
    if execution_error:
        errors.append(dict(execution_error=execution_error))
    if list((STUDY / "service/traces35").glob("*.error.json")):
        errors.append(dict(trace_error_files=[str(p) for p in (STUDY / "service/traces35").glob("*.error.json")]))
    released = release is not None and not any(p["present"] for p in processes.values())
    if released and (
        release["model_pid"] != model["pid"]
        or release["model_start_ticks"] != model["start_ticks"]
        or release["job_pid"] != worker["pid"]
        or release["model_returncode"] not in (0, -15)
        or release["job_returncode"] != 0
        or release["cleanup_errors"]
        or not release["only_owned_process_groups_considered"]
    ):
        raise ValueError("owned model/state release differs")
    if (
        released
        and release["model_stop"].get("signal_sent")
        and (
            release["model_stop"]["pid"] != model["pid"]
            or release["model_stop"]["start_ticks"] != model["start_ticks"]
            or release["model_stop"]["cuda_visible_devices"] != "7"
            or release["model_stop"]["ld_library_path"] != ""
        )
    ):
        raise ValueError("controlled service release lacks the registered live environment")
    complete = (
        len(rows) == 48 and released and complete_marker is not None and state_complete is not None and not errors
    )
    shutdown = None
    if complete:
        if (
            complete_marker["requests"] != 48
            or complete_marker["physics_steps"] != 0
            or not complete_marker["complete"]
            or state_complete["cases"] != 48
            or set(state_complete["result_files"]) != set(expected)
        ):
            raise ValueError("complete coverage/budget differs")
        evidence.check(state_dir / "complete.json", complete_marker["state_complete_sha256"])
        evidence.check(state_dir / "manifest.json", state_complete["manifest_sha256"])
        for name in expected:
            evidence.check(state_dir / name / "result.json", state_complete["result_files"][name])
            evidence.check(STUDY / "service/traces35" / f"{name}.json", complete_marker["trace_sha256"][name])
        shutdown = shutdown_observability(evidence, args, protocol, release, processes)
        if len({r["parameters"]["metadata_sha256"] for r in rows}) != 1:
            raise ValueError("parameter identity changed across013 requests")
        if {str(p) for p in (STUDY / "service").glob("generate_*") if p.is_dir()} != {
            r["server_request_dir"] for r in rows
        }:
            raise ValueError("unregistered extra server request directory")
        if {p.parent.name for p in state_dir.glob("*/result.json")} != set(expected) or {
            p.stem for p in (STUDY / "service/traces35").glob("*.json")
        } != set(expected):
            raise ValueError("extra or missing result/trace")
    metrics = dict(flash3_video=pool(rows), sdpa35=pool([p[0] for p in pairs])) if complete else None
    if complete and any(
        [metrics[group][str(h)]["wam"]["valid_endpoints"] for h in PREFIXES] != [384, 768, 1778] for group in metrics
    ):
        raise ValueError("valid unpadded endpoint denominators differ")
    evidence.recheck()
    for path, frozen in protocol["weight_files"].items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (frozen["bytes"], frozen["mtime_ns"]):
            raise ValueError("weight metadata changed during independent audit")
    if (args.require_complete or args.output.name == "complete48_verified.json") and not complete:
        raise ValueError("complete48 plus normal resource release required; canonical output not written")
    result = dict(
        schema_version="s2a.backend-state-readonly-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        quality_complete=complete,
        status="error" if errors else "complete" if complete else "partial",
        errors=errors,
        verified_queries=len(rows),
        registered_queries=48,
        registered_physics_steps=0,
        historical_episodes=6,
        phase_cases=24,
        distinct_anchors=22,
        unique_anchor_seed_pairs=44,
        metrics=metrics,
        by_episode={
            e: {
                label: pool([r for r in group if r["sample"]["episode"] == e])
                for label, group in (("flash3_video", rows), ("sdpa35", [p[0] for p in pairs]))
            }
            for e in sorted({s["episode"] for s in samples})
        }
        if complete
        else None,
        by_phase={
            phase: {
                label: pool([r for r in group if r["sample"]["phase"] == phase])
                for label, group in (("flash3_video", rows), ("sdpa35", [p[0] for p in pairs]))
            }
            for phase in sorted({s["phase"] for s in samples})
        }
        if complete
        else None,
        cases=rows,
        generated_state_distance=state_distance(pairs) if complete else None,
        promotion=eligibility(metrics["flash3_video"], metrics["sdpa35"]) if complete else None,
        actual_backend=backend,
        processes=processes,
        release=release,
        owned_service_released_normally=released,
        shutdown_observability=shutdown,
        observer_evidence_boundary="Actual initial/clean/mask/text value bytes are digest witnesses from frozen tested observer, not saved tensors for independent byte replay. Actual sigma/timestep values ARE independently reconstructed and byte-hashed. Network NFE is source-derived; velocity-call count is observed.",
        scheduler_bookkeeping_comparison="Only _use_default_values name order is sorted: frozen Diffusers configuration_utils.py creates list(set(...)). Every name, effective scheduler value, sigma and timestep remains exact; development_partial01 retained the initial strict-list-order audit failure.",
        input_sha256=evidence.hashes,
        frozen_helper_sha256=HELPER_SHA,
        full_weight_bytes_rehashed=False,
        new_model_requests=0,
        new_physics_steps=0,
        limitations=[
            "Six old development episodes only; no fresh, pretraining-unseen or stable task success claim.",
            "Single hybrid attention backend comparison retains disabled inference padding, EMA=false and model/decoder compile=false; internal Flex kernel compilation is included.",
            "Eligibility does not authorize physics, task-specific planning or automatic candidate promotion.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), complete=complete, verified=len(rows), errors=errors)))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
