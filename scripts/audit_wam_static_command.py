"""Independent022 caption-pair and raw-state audit; never requests a model/physics.

SciPy state reconstruction is the unchanged, independently implemented011
helper. Registration, actual-caption evidence, groups and022 eligibility are
separate from the inference coordinator and its pooled metrics.
"""

import argparse
import hashlib
import json
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from audit_wam_sampler_steps import Evidence, check_saved_metrics, integrate_and_score, pool
from run_wam_short_replan import cpu_environment, process_record, write_new

from s2a.artifacts import REPO, sha256

STUDY = REPO / "outputs/wam_stability_20261021/static_command_state"
BASELINE = REPO / "outputs/wam_stability_20261021/earlier_screening/8400_g10"
PLAN = "plans/v0.0.1/022_static_command_state_20261021.md"
HELPER_SHA = "12329768297939fe6b15b4127aa6b65e144578fd72b0e896dd306f9e88b10069"
PREFIXES = (8, 16, 40)
METRICS = ("position_rmse_mm", "rotation_rmse_rad", "opening_mae")
MODES = ("reproduction", "expanded")
ADDENDUM = "plans/v0.0.1/022_static_command_condition_addendum_20261021.md"


def caption_pair(short, expanded, *, task, command):
    """Compare actual batch captions, including empty renderer schema fields."""
    assert len(short) == len(expanded) == 1 and isinstance(short[0], str) and isinstance(expanded[0], str)
    left, right = json.loads(short[0]), json.loads(expanded[0])
    assert isinstance(left, dict) and isinstance(right, dict)
    assert left["task_title"] == right["task_title"] == task and left["command"] == task
    assert right["command"] == command
    assert {k: v for k, v in left.items() if k != "command"} == {k: v for k, v in right.items() if k != "command"}
    assert left.get("task_plan", []) == right.get("task_plan", []) == []
    assert "current_subtask" not in left and "current_subtask" not in right
    assert left["action_frequency_hz"] == 20 and left["duration"] == "2s"
    assert left["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    assert json.dumps(dict(left, command=command), ensure_ascii=False) == expanded[0]
    return dict(
        only_command_differs=True,
        task=task,
        command=command,
        renderer_empty_task_plan_preserved=("task_plan" in left),
        short_sha256=hashlib.sha256(short[0].encode()).hexdigest(),
        expanded_sha256=hashlib.sha256(expanded[0].encode()).hexdigest(),
    )


def full_actions(response, transform, *, checkpoint, seed, domain):
    """Check every raw/normalized slot and independently invert the saved affine."""
    assert response["status"] == "success" and len(response["outputs"]) == 1
    content = response["outputs"][0]["content"]
    meta = content["action128"]
    raw, normalized = np.asarray(content["action"], float), np.asarray(meta["normalized_action"], float)
    assert all(a.shape == (40, 128) and np.isfinite(a).all() and not np.any(a[:, 10:]) for a in (raw, normalized))
    assert meta["checkpoint_path"] == checkpoint and type(meta["seed"]) is int and meta["seed"] == seed
    assert meta["action_domain"] == domain and meta["action_units"] == "raw_physical_or_source_units"
    assert meta["action_fps"] == 20 and meta["action_dim_mask"] == [True] * 10 + [False] * 118
    assert all(type(v) is bool for v in meta["action_dim_mask"])
    np.testing.assert_array_equal(meta["action_timestamps_seconds"], np.arange(40) / 20)
    decoded = np.asarray(normalized, np.float32).copy()
    mask = np.asarray(transform["normalization_mask"], bool)
    scale, bias = np.asarray(transform["scale"], np.float32), np.asarray(transform["bias"], np.float32)
    assert mask.shape == scale.shape == bias.shape == (128,) and np.isfinite(scale).all() and np.isfinite(bias).all()
    assert np.all(scale[mask] != 0)
    decoded[:, mask] = (decoded[:, mask] - bias[mask]) / scale[mask]
    np.testing.assert_allclose(decoded, raw, atol=2e-7, rtol=0)
    return raw, normalized


def checkpoint_stat(config, evidence):
    filename = REPO / "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
    evidence.check(filename, "624ba2dd93b49fd76629f7a6c6905b2813b6605d48da8fc7c3f05a026da413a8")
    value = next(c for c in evidence.json(filename)["checkpoints"] if c["name"] == "8400_masked")
    ck = config["checkpoints"][0]
    assert (
        Path(value["checkpoint"]).resolve() == (Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model").resolve()
    )
    assert len(value["files"]) == 163
    for path, saved in value["files"].items():
        stat = Path(path).stat()
        assert stat.st_size == saved["bytes"] and stat.st_mtime_ns == saved["mtime_ns"] and len(saved["sha256"]) == 64
    return dict(
        actual_checkpoint="8400_masked",
        files_stat_rechecked=163,
        weight_bytes_rehashed=False,
        unrelated_checkpoint_stat_checked=False,
    )


def exact_reproduction(current, baseline):
    """No tolerance, active-slot slicing, horizon cropping or float32 recasting."""
    for field in ("action", "normalized_action"):

        def values(response):
            content = response["outputs"][0]["content"]
            return np.asarray(content[field] if field == "action" else content["action128"][field])

        left, right = values(current), values(baseline)
        assert left.shape == right.shape == (40, 128) and np.array_equal(left, right), field
    return True


def receipt_pair(prepared, predicted, request, sample, response):
    """Require the persisted actual prepare/predict handoff, not just request text."""
    assert prepared["stage"] == "actual_prepare_return_before_predict"
    assert predicted == dict(prepared, stage="actual_predict_entry_same_prepared_and_batch")
    assert prepared["request"] == request and prepared["name"] == request["name"]
    assert type(prepared["seed"]) is int and prepared["seed"] == request["seed"]
    assert prepared["image_sha256"] == sample["image_sha256"] and prepared["active_slot_mask"] is True
    mask = prepared["actual_action_dim_mask"]
    assert mask == [True] * 10 + [False] * 118 and all(type(v) is bool for v in mask)
    caption = prepared["caption"]
    assert len(caption) == 1 and isinstance(caption[0], str)
    assert (
        response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"]
        == hashlib.sha256(caption[0].encode()).hexdigest()
    )
    return caption


def read_case(item, sample, config, transform, domain, evidence):
    request = item["request"]
    mode, name, seed = item["mode"], request["name"], request["seed"]
    case = REPO / config["output_dir"] / "state_queries/8400_masked" / name
    result, raw = evidence.json(case / "result.json"), evidence.json(case / "response.json")
    assert evidence.json(case / "request.json") == request
    assert result["response_sha256"] == evidence.hashes[str(case / "response.json")]
    for key, value in dict(
        name=name,
        checkpoint="8400_masked",
        sample=sample["name"],
        episode=sample["episode"],
        task=sample["task"],
        phase=sample["phase"],
        split="diagnostic",
        anchor_frame=sample["anchor_frame"],
        sampling_seed=seed,
        horizon=40,
        prompt_mode="task_only" if mode == "reproduction" else "published_caption",
    ).items():
        assert result[key] == value, key
    checkpoint = str(Path(config["checkpoints"][0]["run"]) / "checkpoints/iter_000008400/model")
    actions, _ = full_actions(raw, transform, checkpoint=checkpoint, seed=seed, domain=domain)
    args = raw["args"]
    for key in (
        "name",
        "seed",
        "prompt",
        "vision_path",
        "extra",
        "fps",
        "num_frames",
        "num_steps",
        "guidance",
        "shift",
        "action_chunk_size",
        "model_mode",
        "image_size",
        "domain_name",
    ):
        assert args[key] == request[key], key
    assert args["condition_frame_indexes_vision"] == [0] and args["action_path"] is None
    assert not args["native_prompt_upsampling"] and not args["negative_prompt"]
    output_dir = Path(args["output_dir"])
    assert output_dir.parent == STUDY / "service" and output_dir.name.startswith("generate_")
    assert len(output_dir.name) == len("generate_") + 32
    int(output_dir.name.removeprefix("generate_"), 16)
    assert evidence.json(output_dir / "sample_args.json") == args
    assert evidence.json(output_dir / "sample_outputs.json") == raw
    if mode == "reproduction":
        assert request == evidence.json(item["original_request"])
        exact_reproduction(raw, evidence.json(item["original_response"]))
    receipt = evidence.json(STUDY / "text_input_receipts" / f"{name}.prepare.json")
    assert receipt["empty_plan_schema_alignment"] is (mode == "expanded")
    caption = receipt_pair(
        receipt, evidence.json(STUDY / "text_input_receipts" / f"{name}.predict.json"), request, sample, raw
    )
    evidence.check(sample["arrays_path"], sample["arrays_sha256"])
    evidence.check(sample["image_path"], sample["image_sha256"])
    with np.load(sample["arrays_path"]) as data:
        count, errors, generated = integrate_and_score(np.asarray(data["states"], float), sample, actions)
    assert count == result["valid_future_steps"]
    check_saved_metrics(errors["wam"], result["metrics"])
    for method in ("hold", "constant_velocity"):
        check_saved_metrics(errors[method], result["baselines"][method])
    return dict(
        name=name,
        mode=mode,
        sample=sample,
        seed=seed,
        valid_future_steps=count,
        errors=errors,
        generated=generated,
        actual_caption=caption,
        service_request_dir=str(output_dir),
        response_sha256=result["response_sha256"],
        original_full128_reproduction_exact=mode == "reproduction",
    )


def preparation(source, expected_sha, evidence):
    """Reconstruct the full fixed96 manifest without importing its producer."""
    evidence.check(REPO / "scripts/audit_wam_sampler_steps.py", HELPER_SHA)
    source = Path(source).resolve()
    assert source.parent == STUDY and source.name.startswith("prepared") and source.suffix == ".json"
    evidence.check(source, expected_sha)
    protocol = evidence.json(source)
    assert protocol["schema_version"] == "s2a.static-command-preparation.v1"
    assert protocol["registration_commit"] == "e117243" and protocol["root"] == str(STUDY)
    assert protocol["model_started"] is False and protocol["physics_started"] is False
    assert protocol["query_budget"] == dict(reproduction=48, expanded=48, physics_steps=0, fresh_initial_states=0)
    for path, digest in protocol["input_sha256"].items():
        evidence.check(path, digest)
    for commit, filename in (("e117243", PLAN), ("d4277dd", ADDENDUM)):
        digest = hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{filename}"], cwd=REPO)).hexdigest()
        evidence.check(REPO / filename, digest)
        assert protocol["input_sha256"][str(REPO / filename)] == digest
    for previous in protocol["previous_immutable_preparations"]:
        evidence.check(previous["path"], previous["sha256"])
        value = evidence.json(previous["path"])
        assert value["model_started"] is False and value["physics_started"] is False
        # Historical source-only manifests can intentionally bind superseded
        # source bytes. Keep those original bytes; never reinterpret them as a run.
        assert value["commands"] == protocol["commands"] and value["requests"] == protocol["requests"]
    baseline = evidence.json(BASELINE / "samples/manifest.json")
    cases = baseline["cases"]
    assert len(cases) == len({c["name"] for c in cases}) == 24
    assert len({c["episode"] for c in cases}) == 6 and len({(c["episode"], c["anchor_frame"]) for c in cases}) == 22
    assert all(c["split"] == "diagnostic" for c in cases)
    commands = {}
    for sample in cases:
        if sample["task"] not in commands:
            commands[sample["task"]] = json.loads(sample["published_prompts"]["40"])["command"]
        for kind in ("arrays", "image"):
            evidence.check(sample[f"{kind}_path"], sample[f"{kind}_sha256"])
    assert len(commands) == 3 and set(commands) == set(protocol["commands"])
    for task, command in commands.items():
        assert isinstance(command, str) and command
        assert json.loads(protocol["commands"][task]) == dict(task_title=task, command=command)
    configs, samples, expected_requests = {}, {}, []
    for mode in MODES:
        filename = f"configs/wam_static_command_{mode}_20261021.json"
        expected = hashlib.sha256(subprocess.check_output(["git", "show", f"e117243:{filename}"], cwd=REPO)).hexdigest()
        assert protocol["configs"][mode] == dict(path=str(REPO / filename), sha256=expected)
        evidence.check(REPO / filename, expected)
        config = evidence.json(REPO / filename)
        assert config["checkpoints"][0] == dict(baseline["config"]["checkpoints"][0], gpu=7, port=10047)
        assert config["offline_prompt_modes"] == ["task_only" if mode == "reproduction" else "published_caption"]
        configs[mode] = config
        expected_manifest = json.loads(json.dumps(baseline))
        expected_manifest["config"] = config
        if mode == "expanded":
            for sample in expected_manifest["cases"]:
                sample["published_prompts"]["40"] = protocol["commands"][sample["task"]]
        manifest_path = REPO / config["output_dir"] / "samples/manifest.json"
        actual = evidence.json(manifest_path)
        assert actual == expected_manifest
        assert protocol["input_sha256"][str(manifest_path)] == evidence.hashes[str(manifest_path)]
        samples[mode] = actual["cases"]
        for sample in cases:
            for seed in (42, 43):
                name = f"{sample['name']}_h40_task_only_s{seed}"
                original = BASELINE / "state_queries/8400_masked" / name
                request = evidence.json(original / "request.json")
                assert request["seed"] == seed and request["prompt"] == sample["task"]
                assert request["num_steps"] == 35 and request["guidance"] == 1 and request["shift"] == 5
                assert request["fps"] == 20 and request["action_chunk_size"] == 40 and request["num_frames"] == 41
                if mode == "expanded":
                    request = dict(
                        request,
                        name=name.replace("_task_only_", "_published_caption_"),
                        prompt=protocol["commands"][sample["task"]],
                    )
                expected_requests.append(
                    dict(
                        mode=mode,
                        request=request,
                        original_response=str(original / "response.json"),
                        original_request=str(original / "request.json"),
                    )
                )
    assert protocol["requests"] == expected_requests and len(expected_requests) == 96
    for relative in (
        "scripts/run_wam_static_command.py",
        "scripts/serve_wam_static_command.py",
        "scripts/run_wam_state_matrix.py",
        "src/s2a/worker_handshake.py",
        "src/s2a/wam_runtime.py",
    ):
        assert (
            protocol["input_sha256"][str(REPO / relative)] == hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        )
    return protocol, configs, samples


def normalized_scores(rows):
    """Mean of the registered per-endpoint sum in m, radians, normalized opening."""
    result = {}
    for horizon in PREFIXES:
        values = []
        for row in rows:
            errors = row["errors"]["wam"]
            p, r, g = (np.asarray(errors[key][:horizon], float) for key in ("position", "rotation", "opening"))
            assert p.shape == r.shape == g.shape and len(p) and all(np.isfinite(v).all() for v in (p, r, g))
            values.extend(((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2).tolist())
        result[str(horizon)] = (
            None if not values else dict(valid_endpoints=len(values), mean_squared_sum=float(np.mean(values)))
        )
    return result


def summarize(rows):
    return dict(metrics=pool(rows), normalized_scores=normalized_scores(rows))


def group_summaries(rows):
    return {
        field: {
            str(value): summarize([r for r in rows if r["sample"][field] == value])
            for value in sorted({r["sample"][field] for r in rows}, key=str)
        }
        for field in ("task", "episode", "phase")
    }


def eligibility(short, expanded, short_groups, expanded_groups):
    """022 fixed gates; no added numerical slack, no automatic physical promotion."""
    comparisons, short_improvement = [], []
    for h in PREFIXES:
        reference, candidate = (v["metrics"][str(h)]["wam"] for v in (short, expanded))
        for metric in METRICS:
            factor = 1.02 if h == 40 else 1.0
            comparisons.append(
                dict(
                    horizon=h,
                    metric=metric,
                    reference=reference[metric],
                    candidate=candidate[metric],
                    maximum_ratio=factor,
                    passed=candidate[metric] <= factor * reference[metric],
                )
            )
        if h != 40:
            short_improvement.append(
                dict(
                    horizon=h,
                    passed=any(reference[key] > 0 and candidate[key] <= 0.9 * reference[key] for key in METRICS),
                )
            )
    tasks, episodes = [], []
    assert set(short_groups["task"]) == set(expanded_groups["task"]) and len(short_groups["task"]) == 3
    assert set(short_groups["episode"]) == set(expanded_groups["episode"]) and len(short_groups["episode"]) == 6
    for field, factor, destination in (("task", 1.0, tasks), ("episode", 1.02, episodes)):
        for name in sorted(short_groups[field]):
            checks = []
            for h in (8, 16):
                left, right = (v[field][name]["normalized_scores"][str(h)] for v in (short_groups, expanded_groups))
                assert left["valid_endpoints"] == right["valid_endpoints"]
                checks.append(
                    dict(
                        horizon=h,
                        reference=left["mean_squared_sum"],
                        candidate=right["mean_squared_sum"],
                        passed=right["mean_squared_sum"] <= factor * left["mean_squared_sum"],
                    )
                )
            destination.append(dict(name=name, checks=checks, passed=all(c["passed"] for c in checks)))
    passed = (
        all(c["passed"] for c in comparisons + short_improvement + tasks) and sum(c["passed"] for c in episodes) >= 5
    )
    return dict(
        eligible_for_separately_registered_development_physics=passed,
        global_metric_checks=comparisons,
        each_short_horizon_has_ten_percent_improvement=short_improvement,
        per_task_nonregression=tasks,
        episodes_with_both_short_horizons_within102_percent=episodes,
        passing_episodes=sum(c["passed"] for c in episodes),
        required_passing_episodes=5,
        added_numerical_tolerance=0,
        new_physics_authorized=False,
        fresh_evaluation_authorized=False,
        interpretation="State-only eligibility is not task success or authorization to run physics.",
    )


def http_lifetime(text, model_pid):
    markers = [
        f"Started server process [{model_pid}]",
        "Uvicorn running on http://127.0.0.1:10047",
        "Shutting down",
        "Waiting for application shutdown.",
        "Application shutdown complete.",
        f"Finished server process [{model_pid}]",
    ]
    assert all(text.count(marker) == 1 for marker in markers)
    offsets = [text.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    requests = list(re.finditer(r'"([A-Z]+) ([^ ]+) HTTP/1\.[01]" (\d{3})', text))
    posts = [r for r in requests if r[1] == "POST"]
    assert len(posts) == text.count('"POST ') == 96
    assert all((r[2], r[3]) == ("/generate", "200") and offsets[1] < r.start() < offsets[2] for r in posts)
    return dict(
        successful_http_requests=96, owned_pid_ordered_lifetime=True, marker_offsets_decoded_text_characters=offsets
    )


def launch_contract(args, protocol, configs, evidence):
    evidence.check(args.launch, args.launch_sha256)
    launch = evidence.json(args.launch)
    assert launch["schema_version"] == "s2a.static-command-root-launch.v1"
    assert launch["prepared"] == dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256)
    assert launch["registered_budget"] == protocol["query_budget"]
    assert (launch["physical_gpu"], launch["port"], launch["strict_zero_gpu_memory_at_launch"]) == (7, 10047, 0)
    assert launch["only_new_port_bindable_before_launch"] and not launch["automatic_retry"]
    assert launch["external_sources_readonly"] and launch["new_physics_steps"] == 0
    for key, commit in (("registration_commit", "e117243"), ("condition_addendum_commit", "d4277dd")):
        assert launch[key] == subprocess.check_output(["git", "rev-parse", commit], cwd=REPO, text=True).strip()
    assert launch["command"] == [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_static_command.py",
        "--seed",
        "42",
        "--prepared",
        str(args.prepared.resolve()),
        "--prepared-sha256",
        args.prepared_sha256,
    ]
    assert launch["model_command"] == [
        str(Path(configs["reproduction"]["cosmos_repo"]) / ".venv/bin/python"),
        "scripts/serve_wam_static_command.py",
        "--prepared",
        str(args.prepared.resolve()),
        "--prepared-sha256",
        args.prepared_sha256,
    ]
    for label in ("cpu_condition_proof", "independent_preflight"):
        ref = launch[label]
        evidence.check(ref["path"], ref["sha256"])
        value = evidence.json(ref["path"])
        for name, digest in value["input_sha256"].items():
            evidence.check(name, digest)
    cpu = evidence.json(launch["cpu_condition_proof"]["path"])
    assert cpu["schema_version"] == "s2a.static-command-actual-cpu-preparation.v1"
    assert cpu["complete"] and cpu["verified_pairs"] == len(cpu["rows"]) == 48
    assert cpu["prepared"] == launch["prepared"] and not cpu["model_weights_loaded"]
    assert cpu["new_model_requests"] == cpu["new_physics_steps"] == 0
    for left, right, row in zip(protocol["requests"][:48], protocol["requests"][48:], cpu["rows"], strict=True):
        assert (row["reproduction_name"], row["expanded_name"], row["seed"]) == (
            left["request"]["name"],
            right["request"]["name"],
            left["request"]["seed"],
        )
        assert (
            row["only_command_different"]
            and row["first_frame_source_pixels_exact"]
            and row["complete_packed_video_equal"]
        )
        assert row["domain_mask_fps_shape_equal"] and row["schema_alignment"] == [False, True]
    peer = evidence.json(launch["independent_preflight"]["path"])
    assert peer["review_passed"] and peer["prepared"] == launch["prepared"]
    assert peer["actual_cpu_preparation"] == launch["cpu_condition_proof"]
    return launch, cpu


def resources(args, launch, protocol, evidence):
    """Two historical phase snapshots contain four distinct processes, not six concurrent."""
    snapshots = {}
    assert len(args.ready) == len(args.ready_sha256) <= 2
    for path, digest in zip(args.ready, args.ready_sha256, strict=True):
        evidence.check(path, digest)
        value = evidence.json(path)
        assert value["schema_version"] == "s2a.static-command-phase-live-ready.v1" and value["ready"]
        mode = value["phase"]
        assert mode in MODES and mode not in snapshots
        assert value["launch"] == dict(path=str(args.launch.resolve()), sha256=args.launch_sha256)
        assert value["physical_gpu"] == 7 and value["port"] == 10047
        assert value["concurrent_scope"] == f"coordinator, model and {mode} query worker only"
        assert value["other_phase_concurrently_claimed"] is False
        assert value["local_network_operations"] == "none" and value["nvml_to_local_pid_mapping_claimed"] is False
        assert value["new_model_requests"] == value["new_physics_steps"] == 0
        for name, signature in value["input_sha256"].items():
            evidence.check(name, signature)
        snapshots[mode] = value
    if not snapshots:
        return dict(
            normal_process_release=False,
            actual_phase_snapshots=[],
            processes={},
            http_budget=None,
            scope="Source-only review; no actual live identity is claimed.",
        )
    directory = STUDY / "orchestration"
    manifest, saved = evidence.json(directory / "manifest.json"), evidence.json(directory / "model_ready.json")
    assert manifest["prepared"] == launch["prepared"] and manifest["model_command"] == launch["model_command"]
    assert manifest["coordinator_pid"] == launch["coordinator"]["pid"]
    assert (manifest["physical_gpu"], manifest["port"]) == (7, 10047) and not manifest["automatic_retry"]
    assert manifest["new_physics_steps"] == 0
    info = saved["info"]
    assert saved["present"] and saved["cuda_visible_devices"] == "7" and saved["ld_library_path"] == ""
    assert (
        info["static_command_protocol"] == launch["prepared"]
        and Path(info["actual_caption_ledger"]) == STUDY / "text_input_receipts"
    )
    assert info["port"] == 10047 and info["attention_backend"] == "sdpa" and info["ema_weights"] is False
    config = evidence.json(protocol["configs"]["reproduction"]["path"])
    ck = config["checkpoints"][0]
    assert Path(info["checkpoint_path"]) == Path(ck["run"]) / "checkpoints/iter_000008400/model"
    assert Path(info["config_file"]) == Path(ck["run"]) / "config.yaml"
    assert Path(info["output_dir"]) == STUDY / "service"
    assert info["acceptance_intervention"] == dict(active_slot_mask=True, exact_published_caption=False)
    assert all(info[k] == 1 for k in ("world_size", "dp_shard_size", "cp_size", "cfgp_size"))
    assert not any(
        info[k]
        for k in (
            "compile",
            "fuse_qkv",
            "cuda_graphs",
            "reuse_small_action_und",
            "reuse_small_action_metadata",
            "fixed_training_buckets",
        )
    )
    roles = {}
    for mode, snapshot in snapshots.items():
        entries = snapshot["local_registered_processes"]
        assert len(entries) == len({(p["pid"], p["start_ticks"]) for p in entries}) == 3
        assert {p["role"] for p in entries} == {"coordinator", "model", "query"} and snapshot[
            "actual_saved_info"
        ] == info
        for entry in entries:
            role = f"{mode}_query" if entry["role"] == "query" else entry["role"]
            if role in roles:
                for key in ("pid", "start_ticks", "ppid", "command", "cwd", "environment"):
                    assert entry[key] == roles[role][key]
            roles[role] = entry
            assert entry["cwd"] == str(REPO) and entry["proc_state"] not in ("Z", "X")
            assert entry["environment"]["CUDA_VISIBLE_DEVICES"] == ("7" if role == "model" else "")
            assert entry["environment"]["LD_LIBRARY_PATH"] == ""
            if role != "coordinator":
                assert entry["ppid"] == launch["coordinator"]["pid"]
            if role == "coordinator":
                assert entry["command"] == launch["command"]
                assert {k: entry[k] for k in ("pid", "start_ticks")} == launch["coordinator"]
            else:
                folder = "model" if role == "model" else mode
                pending = evidence.json(directory / f"{folder}_ownership/pending.json")
                assert all(pending[k] == entry[k] for k in ("pid", "start_ticks", "ppid", "command"))
                expected = (
                    launch["model_command"]
                    if role == "model"
                    else [
                        str(REPO / ".venv/bin/python"),
                        "scripts/run_wam_state_matrix.py",
                        "--config",
                        protocol["configs"][mode]["path"],
                        "--seed",
                        "42",
                        "--checkpoint",
                        "8400_masked",
                    ]
                )
                assert entry["command"] == expected and pending["automatic_retry"] is False
        assert manifest["model"]["pid"] == saved["pid"] == roles["model"]["pid"]
        assert manifest["model"]["start_ticks"] == saved["start_ticks"] == roles["model"]["start_ticks"]
    assert len({(p["pid"], p["start_ticks"]) for p in roles.values()}) == len(roles)
    processes = {
        role: process_record(p["pid"], "7" if role == "model" else "", start_ticks=p["start_ticks"])
        for role, p in roles.items()
    }
    complete = evidence.optional(directory / "complete.json")
    released, http = False, None
    if len(snapshots) == 2 and complete is not None and not any(p["present"] for p in processes.values()):
        assert complete == dict(
            complete=True,
            reproduction_queries=48,
            expanded_queries=48,
            physics_steps=0,
            fresh_initial_states=0,
            automatic_retry=False,
        )
        for role in ("model", *MODES):
            released_role = "model" if role == "model" else f"{role}_query"
            release = evidence.json(directory / f"{role}_ownership/release.json")
            assert (
                release["pid"] == roles[released_role]["pid"]
                and release["start_ticks"] == roles[released_role]["start_ticks"]
            )
            assert release["reaped"] and release["original_error"] is None and release["only_owned_child_signalled"]
            assert release["returncode"] in ((0, -15) if role == "model" else (0,)) and not release.get(
                "kill_sent", False
            )
        http = http_lifetime(evidence.read(directory / "service.log").decode(), roles["model"]["pid"])
        released = True
    return dict(
        normal_process_release=released,
        actual_phase_snapshots=list(snapshots),
        processes=processes,
        complete_record=complete,
        http_budget=http,
        scope="Same coordinator/model and two sequential query workers: four distinct historical births, not six concurrently ready.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "launch", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("prepared-sha256", "launch-sha256"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--ready", type=Path, action="append", default=[])
    parser.add_argument("--ready-sha256", action="append", default=[])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    cpu_environment()
    assert not args.output.exists() and args.output.resolve().is_relative_to(REPO / "outputs")
    assert not (args.full and args.source_only)
    evidence = Evidence()
    evidence.read(__file__)
    protocol, configs, samples = preparation(args.prepared, args.prepared_sha256, evidence)
    launch, cpu = launch_contract(args, protocol, configs, evidence)
    resource = resources(args, launch, protocol, evidence)
    template = evidence.json(REPO / configs["reproduction"]["policy_template"])
    ck = configs["reproduction"]["checkpoints"][0]
    spec = yaml.safe_load(evidence.read(Path(ck["run"]) / "config.yaml"))["model"]["config"]["action128_spec"]
    domain = template["action_domain"]
    transform = spec["transforms"][spec["registry"]["domains"].index(domain)]
    weights_before = checkpoint_stat(configs["reproduction"], evidence)
    rows = {mode: [] for mode in MODES}
    errors = []
    if not args.source_only:
        for item in protocol["requests"]:
            mode, name = item["mode"], item["request"]["name"]
            if (
                evidence.optional(
                    REPO / configs[mode]["output_dir"] / "state_queries/8400_masked" / name / "result.json"
                )
                is None
            ):
                continue
            try:
                sample = next(
                    s
                    for s in samples[mode]
                    if name
                    == f"{s['name']}_h40_{'task_only' if mode == 'reproduction' else 'published_caption'}_s{item['request']['seed']}"
                )
                rows[mode].append(read_case(item, sample, configs[mode], transform, domain, evidence))
            except Exception:
                errors.append(dict(mode=mode, name=name, error=traceback.format_exc()))
    quality = len(rows["reproduction"]) == len(rows["expanded"]) == 48 and not errors
    captions = []
    if quality:
        assert len({r["service_request_dir"] for values in rows.values() for r in values}) == 96
        assert {Path(r["service_request_dir"]) for values in rows.values() for r in values} == set(
            (STUDY / "service").glob("generate_*")
        )
        names = {r["name"] for values in rows.values() for r in values}
        for suffix in ("prepare", "predict"):
            assert {p.name for p in (STUDY / "text_input_receipts").glob(f"*.{suffix}.json")} == {
                f"{n}.{suffix}.json" for n in names
            }
        assert not list((STUDY / "text_input_receipts").glob("*.error.json"))
        for left, right, proof in zip(rows["reproduction"], rows["expanded"], cpu["rows"], strict=True):
            assert (left["sample"]["name"], left["seed"]) == (right["sample"]["name"], right["seed"])
            task = left["sample"]["task"]
            captions.append(
                caption_pair(
                    left["actual_caption"],
                    right["actual_caption"],
                    task=task,
                    command=json.loads(protocol["commands"][task])["command"],
                )
            )
            assert json.loads(left["actual_caption"][0]) == proof["baseline_caption"]
            assert json.loads(right["actual_caption"][0]) == proof["expanded_caption"]
        admission = evidence.json(STUDY / "reproduction_admission.json")
        assert (
            admission["passed"]
            and admission["verified_queries"] == 48
            and admission["full128_raw_and_normalized_exact"]
        )
        for name, digest in admission["input_sha256"].items():
            evidence.check(name, digest)
        for mode in MODES:
            dest = REPO / configs[mode]["output_dir"] / "state_queries/8400_masked"
            complete, manifest = evidence.json(dest / "complete.json"), evidence.json(dest / "manifest.json")
            assert complete["cases"] == 48 and complete["manifest_sha256"] == sha256(dest / "manifest.json")
            assert set(complete["result_files"]) == {r["name"] for r in rows[mode]}
            for name, digest in complete["result_files"].items():
                evidence.check(dest / name / "result.json", digest)
            assert manifest["immutable"] == dict(
                config=configs[mode],
                samples_sha256=sha256(REPO / configs[mode]["output_dir"] / "samples/manifest.json"),
                checkpoint=str(Path(ck["run"]) / "checkpoints/iter_000008400/model"),
            )
            assert manifest["server_info"] == evidence.json(STUDY / "orchestration/model_ready.json")["info"]
            for path in ("scripts/run_wam_state_matrix.py", "src/s2a/checkpoint_audit.py", "src/s2a/geometry.py"):
                assert manifest["code"]["files_sha256"][path] == protocol["input_sha256"][str(REPO / path)]
    aggregate = {mode: summarize(values) for mode, values in rows.items()} if quality else None
    groups = {mode: group_summaries(values) for mode, values in rows.items()} if quality else None
    gate = (
        None
        if not quality
        else eligibility(aggregate["reproduction"], aggregate["expanded"], groups["reproduction"], groups["expanded"])
    )
    weights_after = checkpoint_stat(configs["reproduction"], evidence)
    assert weights_before == weights_after
    evidence.recheck()
    complete = quality and resource["normal_process_release"]
    output = dict(
        schema_version="s2a.static-command-state-independent-audit.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=complete,
        quality_complete=quality,
        normal_process_release=resource["normal_process_release"],
        source_only=args.source_only,
        registered_budget=protocol["query_budget"],
        verified_queries={mode: len(values) for mode, values in rows.items()},
        aggregate=aggregate,
        groups=groups,
        eligibility=gate,
        caption_pairs=captions,
        cases=rows,
        resources=resource,
        errors=errors,
        sample_scope=dict(
            phase_records=24, unique_anchors=22, unique_anchor_seed_pairs=44, episodes=6, queries_per_mode=48
        ),
        source_only_superseded_preparations=protocol["previous_immutable_preparations"],
        actual_checkpoint_stat=weights_after,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_initial_states=0,
        limitations=[
            "Existing development episodes/overlapping windows are not independent fresh episodes.",
            "Actual caption-only conditioning changes task instructions; no raw short-task score is revised.",
            "Same seed is not a recorded initial-noise tensor witness; this experiment does not observe sampler noise.",
            "No S2A, task success, physical trial or fresh evaluation follows automatically from state eligibility.",
            "All96 successful actual handoffs are checked; no unavailable final in-memory poison counter is inferred.",
        ],
    )
    write_new(args.output, output)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=sha256(args.output),
                complete=complete,
                quality_complete=quality,
                verified_queries=output["verified_queries"],
                errors=len(errors),
            )
        )
    )
    if errors or (args.full and not complete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
