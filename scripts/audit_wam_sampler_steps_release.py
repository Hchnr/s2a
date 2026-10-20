"""Explicit fork of frozen cb79290 sampler audit, with narrow shutdown evidence.

All raw-state/pair/schedule/metric/promotion checks are retained. Only the exit
receipt check adds a separately reported, fail-closed alternative for actual
Uvicorn SIGTERM re-raise. No old source, runtime or artifact is modified.
"""

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
from audit_wam_earlier_screen import decode_rotation
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/wam_stability_20261021/sampler_steps_state"
READY = ROOT / "outputs/wam_stability_20261021/short_replan_preflight/sampler_services_ready_resource_snapshot.json"
READY_SHA = "aab01a6a8b5acea2000e6e50143cdb4c5ad51f7a8551ec818e58a597323ea34e"
PREFIXES = (8, 16, 40)
STEPS = (35, 70, 140)
METRIC_KEYS = ("position_rmse_mm", "rotation_rmse_rad", "opening_mae")
STRICT_SOURCE = ROOT / "scripts/audit_wam_sampler_steps.py"
STRICT_SHA = "12329768297939fe6b15b4127aa6b65e144578fd72b0e896dd306f9e88b10069"
EXIT_REVIEW = ROOT / "outputs/wam_stability_20261021/sampler_steps_exit_semantics_review_20261021.json"
EXIT_REVIEW_SHA = "573730a8522f4c72b0fbbf08365acfc0c7156a149df112a1833dbb60482e3bbe"
CPU_PROBE = ROOT / "outputs/wam_stability_20261021/backend_runtime_review/uvicorn_sigterm_finally_cpu_contract.json"
CPU_PROBE_SHA = "ef8b5284017d429c678bd614c0b4a899c87bfe584c00b1f3cdfa7be239dbe46a"


def check_shutdown_log(log, model_pid, expected_requests=144):
    """Require exact successful HTTP budget before owned normal Uvicorn shutdown."""
    statuses = re.findall(r'"POST /generate HTTP/1\.1" ([0-9]{3})', log)
    markers = (
        f"Started server process [{model_pid}]",
        "Shutting down",
        "Application shutdown complete.",
        f"Finished server process [{model_pid}]",
    )
    if statuses != ["200"] * expected_requests or any(log.count(m) != 1 for m in markers):
        raise ValueError("shutdown log does not cover exact HTTP budget and owned normal lifecycle")
    positions = [log.index(m) for m in markers]
    last_post = log.rfind('"POST /generate HTTP/1.1"')
    if positions != sorted(positions) or not positions[0] < last_post < positions[1]:
        raise ValueError("owned shutdown lifecycle is out of order or requests continued after shutdown")
    if "ERROR:" in log or "Traceback (most recent call last):" in log:
        raise ValueError("service log contains an error or traceback")
    return dict(successful_http_requests=len(statuses), owned_lifecycle_in_order=True)


def shutdown_observability(evidence, study, release, model_pid, *, full_coverage):
    """Do not reconstruct missing in-memory data; validate preserved alternatives."""
    closed_path, mask_path = study / "service/closed.json", study / "service/mask_evidence.json"
    closed, masks = evidence.optional(closed_path), evidence.optional(mask_path)
    # A present malformed/truncated receipt is an error, never an allowed absence.
    if closed_path.exists() and closed is None or mask_path.exists() and masks is None:
        raise ValueError("present shutdown receipt is incomplete")
    if closed is not None and closed != dict(counters={str(n): 48 for n in STEPS}, poisoned=False):
        raise ValueError("present close ledger differs from exact144 unpoisoned protocol")
    if masks is not None and (len(masks) != 144 or any(m["inactive_rms_after"] != 0 for m in masks)):
        raise ValueError("present initialization-mask ledger differs")
    if closed is not None and masks is not None:
        return dict(
            mode="persisted_shutdown_ledgers",
            exit_observability_complete=True,
            closed_json_observed=True,
            initial_noise_rms_ledger_observed=True,
            final_in_memory_poisoned=False,
        )
    if not full_coverage or release["model_returncode"] != -15 or release["last_job_returncode"] != 0:
        raise ValueError("limited shutdown evidence requires complete144 and controlled SIGTERM release")
    evidence.check(EXIT_REVIEW, EXIT_REVIEW_SHA)
    review = evidence.json(EXIT_REVIEW)
    for path, digest in review["input_sha256"].items():
        evidence.check(path, digest)
    evidence.check(CPU_PROBE, CPU_PROBE_SHA)
    probe = evidence.json(CPU_PROBE)
    if (
        not probe["cpu_only"]
        or probe["new_model_requests"] != 0
        or probe["new_physics_steps"] != 0
        or probe["returncode"] != -15
        or not probe["entered_exists"]
        or probe["finally_exists"]
    ):
        raise ValueError("actual-Uvicorn CPU SIGTERM causal probe differs")
    for path, digest in probe["source_sha256"].items():
        evidence.check(path, digest)
    log = evidence.read(study / "orchestration/run/service.log").decode()
    log_check = check_shutdown_log(log, model_pid)
    return dict(
        mode="limited_saved_request_and_owned_shutdown_evidence",
        exit_observability_complete=False,
        closed_json_observed=closed is not None,
        initial_noise_rms_ledger_observed=masks is not None,
        final_in_memory_poisoned=None if closed is None else False,
        initial_noise_rms_values=None if masks is None else "present in original ledger",
        **log_check,
        actual_uvicorn_cpu_probe=dict(path=str(CPU_PROBE), sha256=CPU_PROBE_SHA),
        source_exit_review=dict(path=str(EXIT_REVIEW), sha256=EXIT_REVIEW_SHA),
        boundary="No final-only receipt fabricated. Initial-noise RMS ledger and final in-memory poison flag remain unavailable when absent. Full144 raw final inactive slots are independently zero; mask implementation and per-request witnesses are source-bound, not reconstructed in-memory diagnostics.",
    )


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_sha(value):
    return sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = str(Path(path).resolve())
        data = Path(path).read_bytes()
        digest = sha_bytes(data)
        if path in self.hashes and self.hashes[path] != digest:
            raise ValueError(f"evidence changed during audit: {path}")
        self.hashes[path] = digest
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def optional(self, path):
        if not Path(path).exists():
            return None
        # Existing state writer is not atomic. A currently truncated new JSON
        # is pending, never an accepted completed case or scientific failure.
        data = Path(path).read_bytes()
        try:
            result = json.loads(data)
        except json.JSONDecodeError:
            return None
        self.read(path)
        return result

    def check(self, path, expected):
        self.read(path)
        if self.hashes[str(Path(path).resolve())] != expected:
            raise ValueError(f"SHA differs: {path}")

    def recheck(self):
        for path, digest in self.hashes.items():
            if sha_bytes(Path(path).read_bytes()) != digest:
                raise ValueError(f"evidence changed after audit: {path}")


def pool(rows):
    if not rows:
        return None
    result = {}
    for prefix in PREFIXES:
        result[str(prefix)] = {}
        for method in ("wam", "hold", "constant_velocity"):
            values = {
                key: np.concatenate([np.asarray(r["errors"][method][key][:prefix]) for r in rows])
                for key in ("position", "rotation", "opening")
            }
            result[str(prefix)][method] = dict(
                valid_endpoints=len(values["position"]),
                position_rmse_mm=float(np.sqrt(np.mean(values["position"] ** 2)) * 1000),
                rotation_rmse_rad=float(np.sqrt(np.mean(values["rotation"] ** 2))),
                opening_mae=float(np.mean(np.abs(values["opening"]))),
            )
    return result


def integrate_and_score(states, sample, actions):
    """Explicit fork of audit_metrics' independent local integration, plus baselines."""
    frame = sample["anchor_frame"]
    count = min(40, sample["valid_future_steps"], len(states) - frame - 1)
    if count < 1:
        raise ValueError("no valid future state")
    p0, r0 = states[frame, :3].copy(), decode_rotation(states[frame, 3:9])
    position, rotation = p0.copy(), r0.copy()
    history = min(frame, 4)
    dp = np.zeros(3) if not history else (p0 - states[frame - history, :3]) / history
    angular = (
        np.zeros(3)
        if not history
        else Rotation.from_matrix(r0 @ decode_rotation(states[frame - history, 3:9]).T).as_rotvec() / history
    )
    errors = {method: dict(position=[], rotation=[], opening=[]) for method in ("wam", "hold", "constant_velocity")}
    generated = dict(position=[], rotation=[], opening=[])
    for index, action in enumerate(actions, start=1):
        position = position + rotation @ action[:3]
        rotation = rotation @ decode_rotation(action[3:9])
        generated["position"].append(position.copy().tolist())
        generated["rotation"].append(rotation.copy().tolist())
        generated["opening"].append(float(np.clip(action[9], 0, 1)))
        if index > count:
            continue
        truth = states[frame + index]
        truth_rotation = decode_rotation(truth[3:9])
        candidates = dict(
            wam=(position, rotation, generated["opening"][-1]),
            hold=(p0, r0, states[frame, 9]),
            constant_velocity=(
                p0 + index * dp,
                Rotation.from_rotvec(index * angular).as_matrix() @ r0,
                states[frame, 9],
            ),
        )
        for method, (p, r, opening) in candidates.items():
            errors[method]["position"].append(float(np.linalg.norm(truth[:3] - p)))
            errors[method]["rotation"].append(float(Rotation.from_matrix(truth_rotation @ r.T).magnitude()))
            errors[method]["opening"].append(float(opening - truth[9]))
    return count, errors, generated


def check_saved_metrics(errors, saved):
    for key, field in (
        ("position", "position_errors_m"),
        ("rotation", "rotation_errors_rad"),
        ("opening", "opening_errors"),
    ):
        np.testing.assert_allclose(errors[key], saved[field], atol=1e-12, rtol=0, err_msg=field)
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(errors["position"]))), saved["position_rmse_m"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        np.sqrt(np.mean(np.square(errors["rotation"]))), saved["rotation_rmse_rad"], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(np.mean(np.abs(errors["opening"])), saved["opening_mae"], atol=1e-12, rtol=0)


def expected_schedule(steps):
    """Independent NumPy reconstruction of the frozen actual UniPC schedule.

    Constructor first stores training sigmas as FP32. set_timesteps then builds
    its linspace in FP64, applies request shift5, and quantizes time to int64.
    """
    initial = (1 - np.linspace(1, 1 / 1000, 1000)[::-1]).astype(np.float32)
    grid = np.linspace(float(initial[0]), float(initial[-1]), steps + 1)[:-1]
    shifted = 5 * grid / (1 + 4 * grid)
    return np.r_[shifted, 0].astype(np.float32), (shifted * 1000).astype(np.int64)


def tensor_header(header, *, shape=None, dtype=None, bytes_value=None):
    sizes = {"torch.float32": 4, "torch.bfloat16": 2, "torch.int64": 8, "torch.bool": 1}
    if header["dtype"] not in sizes or len(header["value_bytes_sha256"]) != 64:
        raise ValueError("invalid tensor value-byte witness")
    int(header["value_bytes_sha256"], 16)
    if shape is not None and header["shape"] != list(shape):
        raise ValueError("tensor witness shape differs")
    if dtype is not None and header["dtype"] != dtype:
        raise ValueError("tensor witness dtype differs")
    if header["value_bytes"] != int(np.prod(header["shape"])) * sizes[header["dtype"]]:
        raise ValueError("tensor witness byte count differs")
    if bytes_value is not None and sha_bytes(bytes_value) != header["value_bytes_sha256"]:
        raise ValueError("saved actual scheduler values do not reproduce their raw-byte SHA")


def value_identity(header):
    return {k: header[k] for k in ("dtype", "shape", "value_bytes_sha256", "value_bytes")}


def verify_trace(trace, protocol, prepared_sha, name, stage, request, baseline):
    if (
        not trace["complete"]
        or trace["name"] != name
        or trace["num_steps"] != stage
        or trace["sampler_calls"] != 1
        or trace["velocity_evaluation_count"] != stage
        or trace["prepared_manifest_sha256"] != prepared_sha
        or trace["source_sha256"] != protocol["source_sha256"]
        or trace["actual_sampler_arguments"] != dict(num_steps=stage, shift=5.0, seed=[request["seed"]])
    ):
        raise ValueError("trace identity/call-count/source differs")
    branch = dict(
        guidance=1.0,
        velocity_postprocess=None,
        velocity_postprocess_builder=None,
        _extra_num_steps=0,
        scheduler_type="unipc",
        reuse_pack_templates=True,
        reuse_text_kv=False,
    )
    if trace["actual_generation_branch"] != branch or trace["derived_network_forward_count"] != stage:
        raise ValueError("derived denoising-network NFE branch is not qualified")
    if trace["parameters_before"] != trace["parameters_after"] or trace["parameters_before"]["count"] <= 0:
        raise ValueError("parameter identity/storage/version changed")
    if trace["observer_parameter_rng_identity_checks"] != "passed around each observation, separately from computation":
        raise ValueError("observer RNG/parameter guards did not complete")
    expected_conditioning = dict(
        original_request_without_steps={k: v for k, v in request.items() if k != "num_steps"},
        caption=trace["conditioning"]["caption"],
        image_sha256=protocol["cached_cases"][name]["image_sha256"],
    )
    if trace["conditioning"] != expected_conditioning:
        raise ValueError("trace original request/image conditioning differs")
    state = trace["initial_state"]
    for header in (
        list(state.values())
        + list(trace["clean_condition_inputs"].values())
        + [trace["condition_mask"], trace["condition_reference"], trace["final_sampler_state"]]
    ):
        tensor_header(header, dtype="torch.float32")
        # Dummy clean action is constructed/moved before execute enters
        # inference_mode. It legitimately has a version counter; sampler noise
        # is created inside that context. Verify the recorded distinction.
        if header["device"] != "cuda:0" or header["inference_tensor"] == header["version_counter_available"]:
            raise ValueError("inconsistent actual inference/version tensor metadata")
    if state["action"]["shape"] != [40, 128] or state["flat"]["shape"] != [
        np.prod(state["vision"]["shape"]).item() + 5120
    ]:
        raise ValueError("actual initial action/vision layout differs")
    if state["action"]["storage_offset"] != int(np.prod(state["vision"]["shape"])):
        raise ValueError("actual initial action offset differs")
    pair = dict(
        initial_state={k: value_identity(v) for k, v in state.items()},
        clean_condition_inputs={k: value_identity(v) for k, v in trace["clean_condition_inputs"].items()},
        condition_reference=value_identity(trace["condition_reference"]),
        condition_mask=value_identity(trace["condition_mask"]),
        conditional_token_ids_sha256=trace["conditional_token_ids_sha256"],
        request_conditioning=trace["conditioning"],
    )
    if trace["pair_identity"] != pair or trace["pair_identity_sha256"] != json_sha(pair):
        raise ValueError("pair signature is not consistent with actual trace fields")
    if stage == 35:
        if trace["cached35_full128_exact"] is not True:
            raise ValueError("35 cached exact gate absent")
    else:
        if (
            baseline is None
            or trace["actual_state_and_condition_pair_with35"] is not True
            or pair != baseline["pair_identity"]
        ):
            raise ValueError("actual sampler state/conditioning does not pair with traced35")
        # This also compares stride/device/storage and inference metadata, which
        # the numerical pairing signature intentionally does not include.
        for field in (
            "initial_state",
            "clean_condition_inputs",
            "condition_reference",
            "condition_mask",
            "conditional_token_ids_sha256",
            "conditioning",
            "parameters_before",
            "actual_generation_branch",
        ):
            if trace[field] != baseline[field]:
                raise ValueError(f"cross-step observed field changed: {field}")
    cfg = trace["scheduler_config"]
    expected_cfg = dict(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        shift=1,
        use_dynamic_shifting=False,
        thresholding=False,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        disable_corrector=[],
        solver_p=None,
        timestep_spacing="linspace",
        steps_offset=0,
        final_sigmas_type="zero",
    )
    if any(cfg[k] != v for k, v in expected_cfg.items()):
        raise ValueError("actual scheduler configuration differs")
    sigmas, times = (
        np.asarray(trace["sigma_values"], dtype=np.float32),
        np.asarray(trace["scheduler_timestep_values"], dtype=np.int64),
    )
    tensor_header(trace["sigmas"], shape=(stage + 1,), dtype="torch.float32", bytes_value=sigmas.tobytes())
    tensor_header(trace["scheduler_timesteps"], shape=(stage,), dtype="torch.int64", bytes_value=times.tobytes())
    expected_sigmas, expected_times = expected_schedule(stage)
    if not np.array_equal(sigmas, expected_sigmas) or not np.array_equal(times, expected_times):
        raise ValueError("actual UniPC sigma/timestep grid differs from independent reconstruction")
    if trace["timesteps"] != times.tolist() or trace["step_indexes"] != [None] + list(range(1, stage)):
        raise ValueError("actual velocity evaluations do not follow actual scheduler timepoints")


def promotion(metrics):
    """Pre-registered six non-regression checks plus one10% position improvement."""

    def qualify(candidate, reference):
        comparisons = {
            f"H{h}:{'position_rmse_m' if k == 'position_rmse_mm' else k}": dict(
                candidate=candidate[str(h)]["wam"][k] / (1000 if k == "position_rmse_mm" else 1),
                reference=reference[str(h)]["wam"][k] / (1000 if k == "position_rmse_mm" else 1),
                unit="m" if k == "position_rmse_mm" else "rad" if k == "rotation_rmse_rad" else "normalized opening",
            )
            for h in (8, 16)
            for k in METRIC_KEYS
        }
        nonregression = all(v["candidate"] <= v["reference"] + 1e-10 for v in comparisons.values())
        improved = any(
            reference[str(h)]["wam"]["position_rmse_mm"] > 0
            and candidate[str(h)]["wam"]["position_rmse_mm"] <= 0.9 * reference[str(h)]["wam"]["position_rmse_mm"]
            for h in (8, 16)
        )
        return dict(
            eligible=nonregression and improved,
            all_six_nonregression=nonregression,
            at_least_one_position_improves10pct=improved,
            comparisons=comparisons,
        )

    candidates = {str(n): qualify(metrics[str(n)], metrics["35"]) for n in (70, 140)}
    eligible = [n for n in (70, 140) if candidates[str(n)]["eligible"]]
    selected = None if not eligible else eligible[0]
    stronger140 = qualify(metrics["140"], metrics["70"])
    if eligible == [70, 140] and stronger140["eligible"]:
        selected = 140
    return dict(
        candidates=candidates,
        extra140_vs70=stronger140,
        selected_steps=selected,
        absolute_nonregression_tolerance=1e-10,
        nonregression_units=dict(position="m", rotation="rad", opening="normalized"),
        unit_clarification="Position converted from display mm to production primary SI m before applying1e-10. Confirmed before complete70/140 results or any promotion decision; frozen runtime/plan unchanged.",
        new_physics_authorized=False,
        interpretation="Eligibility for separately registered development physics only; no task capability claim",
    )


def process_snapshot(registration, *, cpu):
    pid = registration["pid"]
    expected = str(registration.get("start_ticks", registration.get("startticks")))
    directory = Path(f"/proc/{pid}")
    result = dict(pid=pid, registered_startticks=expected, owned_live=False)
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        result.update(observed_startticks=fields[19], state=fields[0])
        if fields[19] != expected:
            result["pid_reused"] = True
            return result
        result["owned_live"] = fields[0] not in ("Z", "X")
        if result["owned_live"]:
            environment = dict(
                x.split("=", 1) for x in (directory / "environ").read_bytes().decode().split("\0") if "=" in x
            )
            result.update(
                command=(directory / "cmdline").read_bytes().decode().replace("\0", " "),
                cuda_visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
                ld_library_path=environment.get("LD_LIBRARY_PATH"),
            )
            if result["cuda_visible_devices"] != ("" if cpu else "6") or result["ld_library_path"] != "":
                raise ValueError("registered process violates CPU/GPU6 environment contract")
    except FileNotFoundError:
        result["absent"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
        or sys.byteorder != "little"
    ):
        raise ValueError("audit requires little-endian CPU with empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("audit must be a new file within outputs")
    evidence = Evidence()
    evidence.check(STRICT_SOURCE, STRICT_SHA)
    exit_observation = dict(mode="pending_owned_release", exit_observability_complete=False)
    errors, groups, cases_by_stage, traces35, process = [], {}, {}, {}, {}
    prepared_path = STUDY / "orchestration/prepared.json"
    protocol = evidence.json(prepared_path)
    prepared_sha = evidence.hashes[str(prepared_path)]
    root_launch = evidence.json(STUDY / "orchestration/root_launch.json")
    if root_launch["prepared_sha256"] != prepared_sha or root_launch["budget"] != dict(
        requests=144, requests_per_stage=48, physics_steps=0
    ):
        raise ValueError("root launch preparation/budget differs")
    launch = evidence.json(STUDY / "orchestration/run/launch.json")
    if (
        launch["prepared_sha256"] != prepared_sha
        or launch["head"] != root_launch["launch_git_commit"]
        or launch["coordinator_pid"] != root_launch["coordinator_pid"]
        or launch["coordinator_startticks"] != root_launch["coordinator_start_ticks"]
        or launch["physical_gpu"] != 6
        or launch["port"] != 9566
        or launch["budget"] != root_launch["budget"]
    ):
        raise ValueError("actual coordinator launch differs from root registration")
    for commit in ("7a9e571", "1ad56d4"):
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, root_launch["launch_git_commit"]], check=True)
    for field in ("source_sha256", "input_sha256", "snapshot_sha256"):
        for path, digest in protocol[field].items():
            evidence.check(path, digest)
    for path, item in protocol["weight_files"].items():
        stat = Path(path).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("weight stat differs from existing byte fingerprint")
    evidence.check(READY, READY_SHA)
    archive_index_path = STUDY.parent / "sampler_steps_prepare_v1_archive.json"
    archive = evidence.json(archive_index_path)
    if not archive["all_original_bytes_preserved"] or archive["model_processes_or_queries_or_physics_started"] != 0:
        raise ValueError("superseded pure-CPU preparation scope differs")
    archive_root = ROOT / archive["archive_path"]
    for relative, h in archive["file_sha256_relative_to_archived_root"].items():
        evidence.check(archive_root / relative, h)
    samples = evidence.json(STUDY / "8400_steps35/samples/manifest.json")["cases"]
    expected = {f"{s['name']}_h40_task_only_s{seed}": (s, seed) for s in samples for seed in (42, 43)}
    if len(expected) != 48 or list(expected) != protocol["request_order"]:
        raise ValueError("registered48 order differs")
    config = evidence.json(protocol["configs"]["35"]["path"])
    spec = yaml.safe_load(evidence.read(Path(config["checkpoints"][0]["run"]) / "config.yaml"))["model"]["config"][
        "action128_spec"
    ]
    domain = json.loads(evidence.read(config["policy_template"]))["action_domain"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    for stage in STEPS:
        rows, cases_by_stage[str(stage)] = [], {}
        directory = STUDY / f"8400_steps{stage}/state_queries/8400_masked"
        config = evidence.json(protocol["configs"][str(stage)]["path"])
        sample_manifest = STUDY / f"8400_steps{stage}/samples/manifest.json"
        query_manifest = evidence.optional(directory / "manifest.json")
        if query_manifest is not None:
            if query_manifest["immutable"] != dict(
                config=config, samples_sha256=evidence.hashes[str(sample_manifest)], checkpoint=protocol["checkpoint"]
            ):
                raise ValueError("actual state-query configuration/sample/checkpoint differs")
            info = query_manifest["server_info"]
            if (
                info["checkpoint_path"] != protocol["checkpoint"]
                or info["ema_weights"]
                or info["sampler_trace_protocol"]["source_sha256"] != protocol["source_sha256"]
                or not info["sampler_trace_protocol"]["active_slot_mask"]
            ):
                raise ValueError("actual state-query service identity differs")
            for relative in (
                "scripts/run_wam_state_matrix.py",
                "src/s2a/checkpoint_audit.py",
                "src/s2a/geometry.py",
                "src/s2a/prediction_audit.py",
            ):
                evidence.check(ROOT / relative, query_manifest["code"]["files_sha256"][relative])
        complete = evidence.optional(directory / "complete.json")
        admission = evidence.optional(STUDY / f"orchestration/run/admission{stage}.json")
        for name, (sample, seed) in expected.items():
            case = directory / name
            trace_path = STUDY / f"service/traces{stage}/{name}.json"
            if not (case / "result.json").exists() or not trace_path.exists():
                continue
            try:
                result, trace = evidence.optional(case / "result.json"), evidence.optional(trace_path)
                if result is None or trace is None:
                    continue
                request, raw = evidence.json(case / "request.json"), evidence.json(case / "response.json")
                cached = protocol["cached_cases"][name]
                expected_request = evidence.json(cached["request_path"])
                expected_request["num_steps"] = stage
                if request != expected_request or raw["status"] != "success":
                    raise ValueError("raw request/response identity differs")
                if result["response_sha256"] != evidence.hashes[str(case / "response.json")]:
                    raise ValueError("saved raw response SHA differs")
                if any(
                    result[k] != v
                    for k, v in dict(
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
                    ).items()
                ):
                    raise ValueError("saved scoring result identity differs")
                content = raw["outputs"][0]["content"]
                action = np.asarray(content["action"], dtype=float)
                normalized = np.asarray(content["action128"]["normalized_action"], dtype=float)
                meta = content["action128"]
                if (
                    action.shape != (40, 128)
                    or normalized.shape != (40, 128)
                    or not np.isfinite(action).all()
                    or not np.isfinite(normalized).all()
                ):
                    raise ValueError("invalid full128 values")
                if (
                    meta["checkpoint_path"] != protocol["checkpoint"]
                    or meta["seed"] != seed
                    or meta["action_domain"] != domain
                    or meta["action_fps"] != 20
                    or meta["action_dim_mask"] != [True] * 10 + [False] * 118
                    or np.any(action[:, 10:])
                    or np.any(normalized[:, 10:])
                ):
                    raise ValueError("raw checkpoint/seed/domain/mask identity differs")
                if not np.array_equal(np.asarray(meta["action_timestamps_seconds"]), np.arange(40, dtype=float) / 20):
                    raise ValueError("raw action timepoints differ")
                decoded = normalized.copy()
                mask = np.asarray(transform["normalization_mask"], dtype=bool)
                decoded[:, mask] = (decoded[:, mask] - np.asarray(transform["bias"])[mask]) / np.asarray(
                    transform["scale"]
                )[mask]
                if np.max(np.abs(decoded - action)) > 2e-7:
                    raise ValueError("independent de-normalization mismatch")
                verify_trace(trace, protocol, prepared_sha, name, stage, request, traces35.get(name))
                if trace["response_arrays_sha256"] != [json_sha(x.tolist()) for x in (action, normalized)]:
                    raise ValueError("trace does not bind the raw full128 arrays")
                if meta["conditioning_caption_sha256"] != sha_bytes(trace["conditioning"]["caption"][0].encode()):
                    raise ValueError("raw text conditioning differs from actual trace")
                generated_dir = Path(raw["args"]["output_dir"])
                if generated_dir.parent != STUDY / "service" or not generated_dir.name.startswith("generate_"):
                    raise ValueError("raw server request directory differs")
                if (
                    evidence.json(generated_dir / "sample_outputs.json") != raw
                    or evidence.json(generated_dir / "sample_args.json") != raw["args"]
                ):
                    raise ValueError("HTTP response differs from server-local original output")
                acceptance = evidence.json(generated_dir / "acceptance_input.json")
                if acceptance != dict(
                    caption=trace["conditioning"]["caption"],
                    exact_published_caption=False,
                    active_slot_mask=True,
                    seed=seed,
                ):
                    raise ValueError("per-request acceptance input differs")
                if stage == 35:
                    old_raw = evidence.json(cached["response_path"])["outputs"][0]["content"]
                    if not np.array_equal(action, np.asarray(old_raw["action"])) or not np.array_equal(
                        normalized, np.asarray(old_raw["action128"]["normalized_action"])
                    ):
                        raise ValueError("independent full35 cached regression is not exact")
                    traces35[name] = trace
                with np.load(sample["arrays_path"]) as data:
                    states = np.asarray(data["states"], dtype=float)
                count, errors_by_method, generated = integrate_and_score(states, sample, action)
                if count != result["valid_future_steps"]:
                    raise ValueError("GT valid future count differs")
                check_saved_metrics(errors_by_method["wam"], result["metrics"])
                for method in ("hold", "constant_velocity"):
                    check_saved_metrics(errors_by_method[method], result["baselines"][method])
                row = dict(
                    name=name,
                    sample=sample,
                    seed=seed,
                    errors=errors_by_method,
                    generated=generated,
                    parameters=trace["parameters_before"],
                    trace_sha256=evidence.hashes[str(trace_path)],
                    response_sha256=result["response_sha256"],
                    server_request_dir=str(generated_dir),
                )
                rows.append(row)
                cases_by_stage[str(stage)][name] = row
            except (ValueError, KeyError, AssertionError, OSError) as exc:
                errors.append(dict(stage=stage, case=name, error=repr(exc)))
        data_complete = len(rows) == 48
        if complete is not None:
            if complete["cases"] != 48 or set(complete["result_files"]) != set(expected):
                raise ValueError("stage completion does not cover exact48 requests")
            for name, h in complete["result_files"].items():
                evidence.check(directory / name / "result.json", h)
            evidence.check(directory / "manifest.json", complete["manifest_sha256"])
        if admission is not None:
            if not admission["passed"] or admission["full_cases"] != 48 or complete is None:
                raise ValueError("stage admission exists without its full48 completion")
            evidence.check(directory / "complete.json", admission["complete_sha256"])
            for name in expected:
                evidence.check(STUDY / f"service/traces{stage}/{name}.json", admission["trace_sha256"][name])
                evidence.check(directory / name / "result.json", admission["result_sha256"][name])
        groups[str(stage)] = dict(
            expected_requests=48,
            verified_requests=len(rows),
            data_complete=data_complete,
            stage_admission_present=admission is not None,
            metrics=pool(rows) if data_complete else None,
            by_episode={
                e: pool([r for r in rows if r["sample"]["episode"] == e])
                for e in sorted({s["episode"] for s in samples})
            }
            if data_complete
            else None,
            by_phase={
                p: pool([r for r in rows if r["sample"]["phase"] == p]) for p in sorted({s["phase"] for s in samples})
            }
            if data_complete
            else None,
        )
    root_record = dict(pid=root_launch["coordinator_pid"], start_ticks=root_launch["coordinator_start_ticks"])
    process["coordinator"] = process_snapshot(root_record, cpu=True)
    for name, filename, cpu in [("model", "model_process.json", False)] + [
        (f"steps{n}", f"steps{n}_process.json", True) for n in STEPS
    ]:
        registration = evidence.optional(STUDY / "orchestration/run" / filename)
        process[name] = None if registration is None else process_snapshot(registration, cpu=cpu)
    release = evidence.optional(STUDY / "orchestration/run/release.json")
    done = evidence.optional(STUDY / "orchestration/run/complete.json")
    error = evidence.optional(STUDY / "orchestration/run/error.json")
    if error:
        errors.append(dict(orchestration_error=error))
    all_data = all(g["data_complete"] and g["stage_admission_present"] for g in groups.values())
    released = release is not None and not any(p is None or p["owned_live"] for p in process.values())
    if released:
        if release["model_returncode"] not in (0, -15) or release["last_job_returncode"] != 0:
            raise ValueError("service or final state worker release is abnormal")
        model_registration = evidence.json(STUDY / "orchestration/run/model_process.json")
        last_job = evidence.json(STUDY / "orchestration/run/steps140_process.json")
        if (
            release["model_pid"] != model_registration["pid"]
            or release["model_startticks"] != model_registration["start_ticks"]
            or release["last_job_pid"] != last_job["pid"]
        ):
            raise ValueError("released process identities differ from registered owners")
        exit_observation = shutdown_observability(
            evidence,
            STUDY,
            release,
            model_registration["pid"],
            full_coverage=all_data and done is not None and not errors,
        )
    complete = all_data and released and done is not None and not errors
    if complete:
        if done["requests"] != 144 or done["physics_steps"] != 0 or not done["complete"]:
            raise ValueError("coordinator completion budget differs")
        for n in STEPS:
            evidence.check(STUDY / f"orchestration/run/admission{n}.json", done["admissions"][str(n)])
        if len({r["parameters"]["metadata_sha256"] for rows in cases_by_stage.values() for r in rows.values()}) != 1:
            raise ValueError("one fixed model's parameter identity changed across requests")
        for group in groups.values():
            if [group["metrics"][str(n)]["wam"]["valid_endpoints"] for n in PREFIXES] != [384, 768, 1778]:
                raise ValueError("registered unpadded endpoint denominator differs")
        if (STUDY / "service/acceptance_input.json").exists():
            raise ValueError("acceptance input unexpectedly written at constant service root")
        actual_request_dirs = {str(p) for p in (STUDY / "service").glob("generate_*") if p.is_dir()}
        expected_request_dirs = {r["server_request_dir"] for rows in cases_by_stage.values() for r in rows.values()}
        if len(expected_request_dirs) != 144 or actual_request_dirs != expected_request_dirs:
            raise ValueError("service request directories do not correspond exactly to144 registered raw responses")
        for stage in STEPS:
            actual_traces = {p.name for p in (STUDY / f"service/traces{stage}").glob("*.json")}
            actual_results = {
                p.parent.name for p in (STUDY / f"8400_steps{stage}/state_queries/8400_masked").glob("*/result.json")
            }
            if actual_traces != {n + ".json" for n in expected} or actual_results != set(expected):
                raise ValueError("unexpected or missing registered trace/result files")
    distances = None
    if complete:
        distances = {}
        for a, b in ((35, 70), (70, 140)):
            values = {str(n): dict(position=[], rotation=[], opening=[]) for n in PREFIXES}
            for name in expected:
                ra, rb = cases_by_stage[str(a)][name], cases_by_stage[str(b)][name]
                for n in PREFIXES:
                    valid = min(n, len(ra["errors"]["wam"]["position"]))
                    for i in range(valid):
                        values[str(n)]["position"].append(
                            np.linalg.norm(np.asarray(ra["generated"]["position"][i]) - rb["generated"]["position"][i])
                        )
                        values[str(n)]["rotation"].append(
                            Rotation.from_matrix(
                                np.asarray(ra["generated"]["rotation"][i])
                                @ np.asarray(rb["generated"]["rotation"][i]).T
                            ).magnitude()
                        )
                        values[str(n)]["opening"].append(
                            abs(ra["generated"]["opening"][i] - rb["generated"]["opening"][i])
                        )
            distances[f"{a}_vs{b}"] = {
                h: dict(
                    valid_endpoints=len(v["position"]),
                    position_rmse_mm=float(np.sqrt(np.mean(np.square(v["position"]))) * 1000),
                    rotation_rmse_rad=float(np.sqrt(np.mean(np.square(v["rotation"])))),
                    opening_mae=float(np.mean(v["opening"])),
                )
                for h, v in values.items()
            }
    evidence.read(__file__)
    evidence.read(ROOT / "scripts/audit_wam_earlier_screen.py")
    evidence.recheck()
    for path, item in protocol["weight_files"].items():
        stat = Path(path).stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("weight stat changed after independent audit")
    output = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        quality_evidence_complete=complete,
        exit_observability=exit_observation,
        explicit_fork=dict(
            original_commit="cb79290",
            original_source=str(STRICT_SOURCE),
            original_sha256=STRICT_SHA,
            scope="Only shutdown observability replaced; all raw/state/pair/schedule/metric/promotion checks retained.",
        ),
        status="error" if errors else "complete" if complete else "partial",
        errors=errors,
        scope=dict(
            requests=144,
            requests_per_stage=48,
            physics_steps=0,
            phase_cases=24,
            historical_episodes=6,
            unique_anchors=22,
            unique_anchor_seed_pairs=44,
            fresh=False,
        ),
        groups=groups,
        cases=cases_by_stage,
        processes=process,
        release=release,
        launch=launch,
        registration=protocol["registration"],
        source_tree_correction=protocol["source_tree_correction"],
        pure_cpu_preparation_v1_archive=dict(
            path=str(archive_index_path),
            sha256=evidence.hashes[str(archive_index_path)],
            files_preserved=len(archive["file_sha256_relative_to_archived_root"]),
        ),
        weight_stat_checks=dict(
            files=len(protocol["weight_files"]), before_after_match=True, full_weight_bytes_rehashed=False
        ),
        generated_state_distance=distances,
        promotion=promotion({n: g["metrics"] for n, g in groups.items()}) if complete else None,
        tensor_evidence_boundary="Initial/clean/mask/text tensors are observed digest witnesses from frozen observer; raw tensor values are not saved and cannot be independently rehashed. Scheduler values ARE saved and raw-byte SHA and grid are independently reconstructed. RNG/parameter guard outcome is bound to frozen tested observer, not independent tensor replay.",
        model_calls=0,
        physics_steps=0,
        input_sha256=evidence.hashes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                complete=complete,
                status=output["status"],
                verified={n: g["verified_requests"] for n, g in groups.items()},
            )
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
