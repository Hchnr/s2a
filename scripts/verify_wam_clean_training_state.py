"""Independent038 state audit: all99 captures and real exits precede six GT reads.

The shared capture checker is itself an independent NumPy implementation. No
model, sampling producer, normalizer or control/physics implementation is used.
State proximity is the unchanged028 diagnostic, separate from development
eligibility under038. Omitted video values remain online/hash witnesses only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import verify_wam_clean_training_capture as CAPTURE
from scipy.spatial.transform import Rotation

REPO = CAPTURE.REPO
GPU = CAPTURE.GPU
EPISODES = (1254, 1126, 868, 1139, 1209, 1176)
SCHEMA = "s2a.clean-training-state-verification.v1"
require = CAPTURE.require


def numeric(value, message):
    array = np.asarray(value)
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), message)
    return array


def values_equal(left, right, message):
    left, right = numeric(left, message), numeric(right, message)
    require(left.shape == right.shape and np.array_equal(left, right), message)


def rotation(values):
    """Column-rot6d Gram-Schmidt, independently matching the frozen011 rule."""
    value = numeric(values, "nonfinite rotation").astype(np.float64)
    require(value.shape == (6,), "column-rot6d geometry")
    first, second = value[:3], value[3:]
    norm = np.linalg.norm(first)
    require(norm >= 1e-8, "degenerate first rotation column")
    first = first / norm
    second = second - np.dot(first, second) * first
    norm = np.linalg.norm(second)
    require(norm >= 1e-8, "degenerate second rotation column")
    second = second / norm
    return np.column_stack((first, second, np.cross(first, second)))


def integrate(states, anchor, actions):
    """Right-local SE3 increments at fixed20Hz; opening is an absolute endpoint."""
    states = numeric(states, "nonfinite validation state").astype(np.float64)
    actions = numeric(actions, "nonfinite generated action").astype(np.float64)
    frame = anchor["anchor_frame"]
    require(states.ndim == 2 and states.shape[1] == 128, "state128 shape")
    require(type(frame) is int and 0 <= frame and frame + 40 < len(states), "incomplete H40 endpoints")
    require(actions.shape == (40, 128) and not actions[:, 10:].any(), "complete masked generated action128")
    p0, r0 = states[frame, :3].copy(), rotation(states[frame, 3:9])
    position, attitude = p0.copy(), r0.copy()
    history = min(frame, 4)
    dt = 0.05
    linear = np.zeros(3) if not history else (p0 - states[frame - history, :3]) / (history * dt)
    angular = (
        np.zeros(3)
        if not history
        else Rotation.from_matrix(r0 @ rotation(states[frame - history, 3:9]).T).as_rotvec() / (history * dt)
    )
    errors = {method: dict(position=[], rotation=[], opening=[]) for method in ("wam", "hold", "constant_velocity")}
    for endpoint, action in enumerate(actions, start=1):
        position = position + attitude @ action[:3]
        attitude = attitude @ rotation(action[3:9])
        truth = states[frame + endpoint]
        truth_rotation = rotation(truth[3:9])
        candidates = dict(
            wam=(position, attitude, float(np.clip(action[9], 0, 1))),
            hold=(p0, r0, states[frame, 9]),
            constant_velocity=(
                p0 + endpoint * dt * linear,
                Rotation.from_rotvec(endpoint * dt * angular).as_matrix() @ r0,
                states[frame, 9],
            ),
        )
        for method, (p, r, opening) in candidates.items():
            errors[method]["position"].append(float(np.linalg.norm(truth[:3] - p)))
            errors[method]["rotation"].append(float(Rotation.from_matrix(truth_rotation @ r.T).magnitude()))
            errors[method]["opening"].append(float(opening - truth[9]))
    return errors


def pool(rows):
    require(bool(rows), "empty state metric group")
    result = {}
    for horizon in (8, 16, 40):
        methods = {}
        for method in ("raw", "w2", "hold", "constant_velocity"):
            selected = [r for r in rows if r["variant"] == (method if method in ("raw", "w2") else "raw")]
            require(bool(selected), "missing raw/W2 metric rows")
            field = "wam" if method in ("raw", "w2") else method
            errors = {}
            for name in ("position", "rotation", "opening"):
                sequences = [numeric(r["errors"][field][name], "nonfinite endpoint errors") for r in selected]
                require(all(a.shape == (40,) for a in sequences), "state scoring must retain every H40 endpoint")
                errors[name] = np.concatenate([a[:horizon] for a in sequences])
            p, r, g = (errors[k] for k in ("position", "rotation", "opening"))
            methods[method] = dict(
                valid_endpoints=len(p),
                position_rmse_m=float(np.sqrt(np.mean(p**2))),
                rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
                opening_mae=float(np.abs(g).mean()),
                joint_score=float(np.mean((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2)),
            )
            require(all(np.isfinite(value) for value in methods[method].values()), "state metrics overflow")
        result[str(horizon)] = methods
    return result


def state_gate(groups):
    """Original028 thresholds, with the sole final candidate named w2."""
    pooled = groups["pooled"]
    require(
        len(groups["by_episode"]) == 6 and set(groups["by_seed"]) == {"42", "43"},
        "state gate needs all six episodes and both seeds",
    )
    components = {}
    for horizon in (8, 16, 40):
        raw, w2 = pooled[str(horizon)]["raw"], pooled[str(horizon)]["w2"]
        for name, additive in (("position_rmse_m", 0.002), ("rotation_rmse_rad", 0.01), ("opening_mae", 0.01)):
            components[f"h{horizon}_{name}"] = w2[name] <= 1.05 * raw[name] and w2[name] <= raw[name] + additive
    episodes = {
        episode: all(g[str(h)]["w2"]["joint_score"] <= 1.05 * g[str(h)]["raw"]["joint_score"] for h in (8, 16))
        for episode, g in groups["by_episode"].items()
    }
    checks = dict(
        short_joint_improvement=all(
            pooled[str(h)]["w2"]["joint_score"] <= 0.90 * pooled[str(h)]["raw"]["joint_score"] for h in (8, 16)
        ),
        strictly_beats_hold_and_constant_velocity=all(
            pooled[str(h)]["w2"]["joint_score"] < pooled[str(h)][b]["joint_score"]
            for h in (8, 16)
            for b in ("hold", "constant_velocity")
        ),
        all_components_protected=all(components.values()),
        at_least_five_episodes=sum(episodes.values()) >= 5,
        no_episode_short_outlier=all(
            g[str(h)]["w2"]["joint_score"] <= 1.25 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_episode"].values()
            for h in (8, 16)
        ),
        both_seeds_protected=all(
            g[str(h)]["w2"]["joint_score"] <= 1.05 * g[str(h)]["raw"]["joint_score"]
            for g in groups["by_seed"].values()
            for h in (8, 16)
        ),
    )
    return dict(passed=all(checks.values()), checks=checks, component_checks=components, episode_passes=episodes)


def diagnosis(rows):
    require(len(rows) == 96, "full96 state metric denominator")
    groups = dict(pooled=pool(rows))
    for label, field in (("by_episode", "episode"), ("by_task", "task_index"), ("by_seed", "sampling_seed")):
        groups[label] = {str(v): pool([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows})}
    require(set(groups["by_episode"]) == {str(e) for e in EPISODES}, "six original validation episodes")
    require(set(groups["by_task"]) == {"0", "1", "2"}, "all three tasks must be reported")
    return groups, state_gate(groups)


def diagnostic_verdict(gate, *, numerical_contract_passed):
    require(
        numerical_contract_passed is True and type(gate.get("passed")) is bool,
        "complete numerical contract and scored state diagnosis required",
    )
    return dict(
        state_gate=gate,
        state_gate_passed=gate["passed"],
        numerical_contract_passed=True,
        eligible_for_development=True,
        physics_authorized=False,
        fresh_authorized=False,
    )


def weight_stats(prepared):
    require(len(prepared["weight_files"]) == 163, "all163 qualified weight file stats required")
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]),
            "qualified weight stat changed during state audit",
        )


def validation_labels(evidence, prepared, *, numerical_contract_passed):
    # This is the only validation NPZ decoder. The caller invokes it only after
    # every capture, source, parameter and externally supplied exit/release gate.
    require(numerical_contract_passed is True, "future states require the complete numerical/exit gate")
    anchors = prepared["validation_anchors"]
    require(len(anchors) == 24 and {a["episode"] for a in anchors} == set(EPISODES), "fixed24 validation anchors")
    labels, opened = {}, set()
    for episode in EPISODES:
        selected = [a for a in anchors if a["episode"] == episode]
        require(len(selected) == 4 and [a["anchor_index"] for a in selected] == list(range(4)), "four ordered anchors")
        reference = selected[0]["arrays_ref"]
        path = evidence.path(reference)
        require(path.suffix == ".npz" and path not in opened, "six unique validation state archives")
        opened.add(path)
        with np.load(path, allow_pickle=False) as archive:
            # No source actions, camera arrays, native training or fresh data.
            states, mask = archive["states"].copy(), archive["state_mask"].copy()
        numeric(states, "nonfinite validation states")
        require(
            states.ndim == 2
            and states.shape[1] == 128
            and len(states) >= 44
            and mask.shape == states.shape
            and mask.dtype == np.bool_,
            "validation state/mask geometry",
        )
        frames = [j * (len(states) - 41) // 3 for j in range(4)]
        require(len(set(frames)) == 4 and [a["anchor_frame"] for a in selected] == frames, "full original anchor clock")
        for anchor in selected:
            frame = anchor["anchor_frame"]
            require(
                anchor["arrays_ref"] == reference and anchor["initial_sync_certified"] is False,
                "validation provenance/synchronization boundary",
            )
            values_equal(states[frame], anchor["measured_state128"], "current anchor does not match state source")
            values_equal(
                mask[max(0, frame - 4) : frame + 41],
                np.broadcast_to(CAPTURE.MASK, (frame + 41 - max(0, frame - 4), 128)),
                "H40 and causal CV-history common valid mask",
            )
            require(anchor["state_mask128"] == CAPTURE.MASK, "recorded current active mask")
            require(anchor["task_index"] == EPISODES.index(episode) // 2, "validation task/episode binding")
        labels[episode] = states
    require(len(opened) == 6, "exact six validation future archive reads")
    return labels


def response_actions(response, request, checkpoint, transform, caption_sha):
    require(response["status"] == "success" and len(response["outputs"]) == 1, "successful single generated response")
    content = response["outputs"][0]["content"]
    metadata = content["action128"]
    raw, normalized = (
        numeric(x, "nonfinite full action response") for x in (content["action"], metadata["normalized_action"])
    )
    require(
        raw.shape == normalized.shape == (40, 128) and not raw[:, 10:].any() and not normalized[:, 10:].any(),
        "full40x128 response and inactive118",
    )
    for action in raw:
        rotation(action[3:9])
    require(
        metadata["checkpoint_path"] == checkpoint
        and metadata["seed"] == request["seed"]
        and metadata["action_domain"] == CAPTURE.DOMAIN
        and metadata["action_dim_mask"] == CAPTURE.MASK
        and metadata["action_units"] == "raw_physical_or_source_units"
        and metadata["action_fps"] == 20
        and metadata["conditioning_caption_sha256"] == caption_sha,
        "response checkpoint/domain/seed/clock/caption",
    )
    values_equal(metadata["action_timestamps_seconds"], np.arange(40) / 20, "20Hz generated action clock")
    scale, bias = (
        numeric(transform[k], "invalid recorded affine transform").astype(np.float32) for k in ("scale", "bias")
    )
    mask = np.asarray(transform["normalization_mask"])
    require(mask.dtype == np.bool_ and mask.shape == scale.shape == bias.shape == (128,), "normalization geometry")
    require(np.all(scale[mask] != 0), "noninvertible normalization")
    decoded = normalized.astype(np.float32)
    decoded[:, mask] = (decoded[:, mask] - bias[mask]) / scale[mask]
    require(np.allclose(decoded, raw, rtol=0, atol=2e-7), "independent FP32 denormalization")
    require(all(response["args"][k] == v for k, v in request.items()), "actual current-only request binding")
    require(
        response["args"]["condition_frame_indexes_vision"] == [0]
        and response["args"]["action_path"] is None
        and response["args"]["native_prompt_upsampling"] is False
        and not response["args"]["negative_prompt"],
        "no privileged/future/upsampling response branch",
    )
    return raw


def conditioning_check(conditioning, request):
    caption = json.loads(conditioning["caption"])
    require(
        set(conditioning) == {"caption", "caption_sha256", "context", "inference_only_current_image"}
        and conditioning["inference_only_current_image"] is True
        and conditioning["caption_sha256"] == hashlib.sha256(conditioning["caption"].encode()).hexdigest(),
        "actual current-image caption identity",
    )
    require(
        caption
        == dict(
            task_title=request["prompt"],
            command=request["prompt"],
            embodiment="franka_panda__parallel_gripper",
            duration="2s",
            fps=20.0,
            action_frequency_hz=20.0,
            resolution=dict(W=640, H=960),
            aspect_ratio="2:3",
            camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
            task_plan=[],
        ),
        "state caption contains unregistered/future conditioning",
    )
    require(
        conditioning["context"]
        == dict(domain=CAPTURE.DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=CAPTURE.MASK),
        "actual state domain",
    )


def lifecycle(evidence, refs, prepared_ref, registration_ref, actual, prepared):
    launch, exit_record, release, gpu = [
        evidence.json(refs[k]) for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")
    ]
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False), "real normal state worker wait0"
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "real owned state reap/noSignal",
    )
    require(all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")), "owned state PID/starttick")
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""), "state worker UUID/LD"
    )
    command = launch["command"]
    require(
        command[:2] == [prepared["model_config"]["model_python"], "scripts/sample_wam_clean_training_state.py"],
        "actual state worker interpreter/entrypoint",
    )
    for flag, value in (
        ("--prepared", prepared_ref["path"]),
        ("--prepared-sha256", prepared_ref["sha256"]),
        ("--execution-registration", registration_ref["path"]),
        ("--execution-registration-sha256", registration_ref["sha256"]),
        ("--w2-fit", actual["w2_fit_ref"]["path"]),
        ("--w2-fit-sha256", actual["w2_fit_ref"]["sha256"]),
        ("--w2-verification", actual["w2_verification_ref"]["path"]),
        ("--w2-verification-sha256", actual["w2_verification_ref"]["sha256"]),
        ("--gpu-uuid", GPU),
        ("--seed", "42"),
    ):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, "state worker admission flags")
    require(
        gpu["released"] is True
        and gpu["uuid"] == GPU
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "actual state GPU release",
    )
    require(
        all(
            Path(refs[k]["path"]).resolve().is_relative_to(Path(prepared["root"]))
            for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")
        ),
        "state receipt escaped registered root",
    )


def fixed_w2(evidence, prepared_ref, registration_ref, actual):
    fit, verification = evidence.json(actual["w2_fit_ref"]), evidence.json(actual["w2_verification_ref"])
    require(
        fit["schema_version"] == "s2a.clean-training-fit.v1"
        and verification["schema_version"] == "s2a.clean-training-fit-verification.v1",
        "W2 fixed fit schemas",
    )
    require(
        all(
            v["complete"] is v["passed"] is True
            and v["stage"] == "W2"
            and v["prepared_ref"] == prepared_ref
            and v["execution_registration_ref"] == registration_ref
            and v["actual_rows"] == 840000
            for v in (fit, verification)
        ),
        "full840000-row W2 qualification",
    )
    require(
        verification["fit_ref"] == actual["w2_fit_ref"]
        and verification["independent_reconstructions"] == 1
        and verification["full_training_row_reconstruction"] is True
        and fit["rounds"] == ["D0", "D1"]
        and fit["producer_fits"] == fit["fit_attempts"] == 1
        and len(fit["collection_refs"]) == len(fit["capture_audit_refs"]) == 2
        and fit["params_ref"] == verification["params_ref"] == actual["readout_ref"],
        "unique final W2 parameter origin",
    )
    primary, independent = fit["fit_metadata"]["replay_sha256"], verification["qr_metadata"]["replay_sha256"]
    require(
        primary["features"] == independent["features"]
        and primary["weights"] == independent["weights"]
        and primary["target_clean_residual"] == independent["targets"],
        "independent W2 all-row replay identities",
    )
    for v in (fit, verification):
        evidence.mapping(v["input_sha256"])
    for round_name, collection_ref, audit_ref in zip(
        ("D0", "D1"), fit["collection_refs"], fit["capture_audit_refs"], strict=True
    ):
        audit, collection = evidence.json(audit_ref), evidence.json(collection_ref)
        require(
            audit["schema_version"] == "s2a.clean-training-capture-audit.v1"
            and collection["schema_version"] == "s2a.clean-training-capture.v1"
            and all(
                v["complete"] is v["passed"] is True
                and v["prepared_ref"] == prepared_ref
                and v["execution_registration_ref"] == registration_ref
                and v["round_name"] == round_name
                and v["actual_requests"] == 300
                and v["actual_denoise_calls"] == 10500
                for v in (audit, collection)
            )
            and audit["collection_ref"] == collection_ref,
            "W2 exact fully audited D0/D1 ancestry",
        )
        evidence.mapping(audit["input_sha256"])
        prepared = evidence.json(prepared_ref)
        CAPTURE.lifecycle(
            evidence,
            {k: audit[k] for k in ("worker_launch_ref", "worker_exit_ref", "worker_release_ref", "gpu_after_ref")},
            prepared_ref,
            registration_ref,
            collection,
            Path(prepared["root"]),
            model_python=prepared["model_config"]["model_python"],
        )
    return CAPTURE.head_parameters(evidence.json(actual["readout_ref"]))


def point_math(record, arrays, header, step, *, head, vision_shape=CAPTURE.VISION, time_vision_shape=(6000,)):
    require(
        record["metadata"]["purpose"] == header["request"]["purpose"] == "evaluation", "state point evaluation identity"
    )
    return CAPTURE.verify_point(
        record,
        arrays,
        header,
        step,
        head=head,
        vision_shape=vision_shape,
        time_vision_shape=time_vision_shape,
        purpose="evaluation",
    )


def zero_sources(evidence, record, historical_ref, expected_directory, index):
    metadata = evidence.json(historical_ref)
    require(
        Path(historical_ref["path"]).resolve() == expected_directory / f"point_{index:02d}.json"
        and metadata["step_index"] == index,
        "037 raw fullpoint order/path",
    )
    arrays = evidence.arrays(metadata["arrays"], expected_directory / f"point_{index:02d}.npz")
    require(set(arrays) == CAPTURE.SOURCE_KEYS and set(metadata["array_index"]) == set(arrays), "037 fullpoint roster")
    dtypes = dict(metadata["original_dtypes"])
    summaries = {}
    for name, array in arrays.items():
        require(
            metadata["array_index"][name] == dict(shape=list(array.shape), dtype=str(array.dtype)),
            "037 fullpoint index",
        )
        witness = dict(CAPTURE.summary(array), original_dtype=dtypes[name])
        require(witness == record["source_arrays"][name], "zero fullpoint source-array bits differ from037 raw")
        summaries[name] = witness
    return summaries


def compact_points(
    evidence,
    ledger,
    header,
    directory,
    trace,
    *,
    head,
    historical=None,
    vision_shape=CAPTURE.VISION,
    time_vision_shape=(6000,),
):
    require(ledger["nodes"] == len(ledger["points"]) == len(trace["steps"]) == 35, "full35 state points")
    chain = hashlib.sha256(CAPTURE.canonical(header))
    first, changed = None, 0
    for i, (ref, step) in enumerate(zip(ledger["points"], trace["steps"], strict=True)):
        record = evidence.json(ref)
        require(
            Path(ref["path"]).resolve() == directory / f"compact/step{i:02d}.json"
            and record["request_ref"] == ledger["request_ref"],
            "state point path/order/request",
        )
        arrays = evidence.arrays(record["arrays_ref"], directory / f"compact/step{i:02d}.npz")
        chain.update(CAPTURE.canonical({k: record[k] for k in ("metadata", "source_arrays", "compact_arrays")}))
        require(chain.hexdigest() == record["source_stream_sha256"], "state compact chain")
        changed += point_math(
            record, arrays, header, step, head=head, vision_shape=vision_shape, time_vision_shape=time_vision_shape
        )
        if historical is not None:
            summaries = zero_sources(
                evidence, record, historical["points"][i], Path(historical["points"][0]["path"]).parent, i
            )
            if "source_summary_witnesses" in historical:
                require(
                    summaries == historical["source_summary_witnesses"][i],
                    "zero producer sidecar differs from actual037 arrays",
                )
        if i == 0:
            first = dict(record, _sample_active=arrays["sample_active10_fp32"])
    require(chain.hexdigest() == ledger["source_stream_sha256"], "state final source hash chain")
    return first, changed


def planned_generations(prepared):
    planned = []
    for i, case in enumerate(prepared["zero_identity_cases"]):
        planned.append(
            dict(
                index=i,
                name="zero_" + case["name"],
                variant="zero",
                zero_case_name=case["name"],
                sampling_seed=42,
                request=case["request"],
            )
        )
    require(len(planned) == 3 and len(prepared["validation_generations"]) == 96, "fixed3zero+96 roster")
    for item in prepared["validation_generations"]:
        planned.append(dict(item, index=item["index"] + 3, validation_index=item["index"]))
    require([p["index"] for p in planned] == list(range(99)), "complete ordered99 state requests")
    return planned


def verify_generations(evidence, prepared, registration, sources, actual, head):
    root = Path(prepared["root"]) / "state_sampling"
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    qualified_trace = evidence.json(qualified["generations"][0]["trace_ref"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in qualified_trace["steps"]]
    require(len(grid) == 35, "qualified35 grid")
    normalizer = evidence.json(actual["normalization_ref"])
    require(
        normalizer["domain"] == CAPTURE.DOMAIN
        and normalizer["normalization_id"] == 31
        and normalizer["projection_id"] == 0,
        "state normalization31/projection0",
    )
    checkpoint = str(
        Path(prepared["model_config"]["checkpoint"]["run"])
        / "checkpoints"
        / prepared["model_config"]["checkpoint"]["iteration"]
        / "model"
    )
    anchors = {a["name"]: a for a in prepared["validation_anchors"]}
    roster = planned_generations(prepared)
    require(len(actual["generations"]) == 99, "all99 state outputs required")
    require(
        actual["boundary_calls"]
        == [dict(index=i * 35 + j, label=p["name"], completed=True) for i, p in enumerate(roster) for j in range(35)],
        "3465 denoise boundary calls",
    )
    generated, pairs, changed = [], {}, 0
    for planned, entry in zip(roster, actual["generations"], strict=True):
        identity = (
            ("index", "name", "variant", "sampling_seed", "zero_case_name")
            if planned["variant"] == "zero"
            else ("index", "validation_index", "name", "variant", "sampling_seed", "anchor_name")
        )
        require(
            entry["complete"] is True and all(entry[k] == planned[k] for k in identity), "exact99 state identity/order"
        )
        directory = root / f"{entry['index']:03d}_{entry['name']}"
        historical = None
        if entry["variant"] == "zero":
            case = prepared["zero_identity_cases"][entry["index"]]
            anchor = dict(image_ref=case["image_ref"], task=planned["request"]["prompt"])
            require(
                entry["qualified_generation_ref"] == prepared["prerequisite_refs"]["bridge_qualification"]
                and entry["qualified_generation_index"] == entry["index"],
                "zero037 qualified source identity",
            )
            historical = qualified["generations"][entry["index"]]
            require(
                historical["variant"] == "raw"
                and historical["case_id"] == case["name"]
                and entry["expected_response_ref"] == case["expected_response_ref"]
                and entry["qualified_response_ref"] == historical["response_ref"],
                "zero raw/source response origin",
            )
            gold = evidence.json(entry["zero_source_hashes_ref"])
            require(
                Path(entry["zero_source_hashes_ref"]["path"]).resolve() == directory / "historical_source_hashes.json"
                and gold["qualified_generation_ref"] == entry["qualified_generation_ref"]
                and gold["qualified_generation_index"] == entry["qualified_generation_index"]
                and gold["point_refs"] == historical["points"]
                and len(gold["source_arrays"]) == 35,
                "zero source sidecar exact qualified origin",
            )
            historical = dict(historical, source_summary_witnesses=gold["source_arrays"])
        else:
            anchor = anchors[entry["anchor_name"]]
            require(
                entry["sampling_seed"] in (42, 43) and entry["variant"] in ("raw", "w2"), "evaluation noise/variant"
            )
        CAPTURE.current_request(planned["request"], dict(planned, name=planned["request"]["name"]), anchor)
        evidence.path(anchor["image_ref"])
        request = evidence.json(entry["request_ref"])
        require(
            Path(entry["request_ref"]["path"]).resolve() == directory / "request.json"
            and request == planned["request"],
            "exact saved state request",
        )
        conditioning = evidence.json(entry["conditioning_ref"])
        conditioning_check(conditioning, request)
        response = evidence.json(entry["response_ref"])
        actions = response_actions(
            response, request, checkpoint, normalizer["transform"], conditioning["caption_sha256"]
        )
        trace = evidence.json(entry["trace_ref"])
        require(
            trace["purpose"] == "evaluation" and trace["parameters_before"] == actual["parameters_before"],
            "actual evaluation base parameter identity",
        )
        CAPTURE.verify_trace(trace, planned, grid, sources, prepared["noise_roster"], purpose="evaluation")
        ledger = evidence.json(entry["compact_ledger_ref"])
        require(
            Path(entry["compact_ledger_ref"]["path"]).resolve() == directory / "compact/ledger.json"
            and ledger["schema_version"] == "s2a.clean-compact-ledger.v1"
            and ledger["complete"] is True
            and ledger["future_labels_read"] == 0
            and ledger["omitted_vision_reconstructible"] is False
            and ledger["qualification_claim"] is False,
            "complete label-free state compact ledger",
        )
        header = evidence.json(ledger["request_ref"])
        require(
            Path(ledger["request_ref"]["path"]).resolve() == directory / "compact/request.json"
            and header["schema_version"] == "s2a.clean-compact-request.v1"
            and header["expected_vision_shape"] == list(CAPTURE.VISION)
            and header["expected_nodes"] == 35
            and header["expected_grid"] == [list(x) for x in grid]
            and header["source_hash_scope"] == "C-order NumPy snapshot bytes"
            and header["labels_accepted"] is header["omitted_vision_reconstructible"] is False,
            "state compact header",
        )
        require(
            header["request"]
            == dict(
                request_name=entry["name"],
                noise_seed=entry["sampling_seed"],
                purpose="evaluation",
                round_name="state",
                anchor_id=anchor.get("name", entry.get("zero_case_name")),
                readout_sha256=actual["readout_ref"]["sha256"] if entry["variant"] == "w2" else None,
                current_image_sha256=anchor["image_ref"]["sha256"],
            ),
            "state compact identity",
        )
        first, mutated = compact_points(
            evidence,
            ledger,
            header,
            directory,
            trace,
            head=head if entry["variant"] == "w2" else None,
            historical=historical,
        )
        changed += mutated
        initial = evidence.json(entry["initial_ref"])
        arrays = evidence.arrays(initial["arrays"], directory / "initial.npz")
        require(
            initial["variant"] == entry["variant"]
            and initial["array_index"] == {k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in arrays.items()},
            "state actual initial metadata",
        )
        CAPTURE.verify_initial(initial, arrays, planned, conditioning["caption"], first)
        if historical is not None:
            original = evidence.json(historical["initial_ref"])
            original_arrays = evidence.arrays(
                original["arrays"], Path(historical["initial_ref"]["path"]).parent / "initial.npz"
            )
            CAPTURE.paired_initial(initial, arrays, original, original_arrays)
            for old_ref in (entry["expected_response_ref"], entry["qualified_response_ref"]):
                old = evidence.json(old_ref)["outputs"][0]["content"]
                content = response["outputs"][0]["content"]
                values_equal(content["action"], old["action"], "zero full128 raw exact regression")
                values_equal(
                    content["action128"]["normalized_action"],
                    old["action128"]["normalized_action"],
                    "zero full128 normalized exact regression",
                )
        else:
            key = (entry["anchor_name"], entry["sampling_seed"])
            if entry["variant"] == "raw":
                require(key not in pairs, "duplicate raw validation pair")
                pairs[key] = entry["initial_ref"]
            else:
                require(key in pairs, "W2 missing corresponding raw initial state")
                original = evidence.json(pairs[key])
                raw_directory = root / f"{entry['index'] - 1:03d}_{roster[entry['index'] - 1]['name']}"
                raw_arrays = evidence.arrays(original["arrays"], raw_directory / "initial.npz")
                CAPTURE.paired_initial(initial, arrays, original, raw_arrays)
            generated.append((planned, anchor, actions))
    require(len(pairs) == 48 and len(generated) == 96, "full48 identical-noise/current-condition pairs")
    return generated, changed


def verify(args):
    require(
        args.seed == 42
        and Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "state checker requires CPU/empty CUDA and LD/S2A cwd/seed42",
    )
    names = (
        "prepared",
        "execution_registration",
        "sampling",
        "worker_launch",
        "worker_exit",
        "worker_release",
        "gpu_after",
    )
    refs = {k: dict(path=str(Path(getattr(args, k)).resolve()), sha256=getattr(args, k + "_sha256")) for k in names}
    evidence = CAPTURE.Evidence()
    prepared, registration, sources = CAPTURE.admission(evidence, refs["prepared"], refs["execution_registration"])
    for path in (Path(__file__).resolve(), REPO / "scripts/sample_wam_clean_training_state.py"):
        require(
            registration["source_sha256"].get(str(path)) == CAPTURE.sha(path), "state source is not execution-frozen"
        )
    actual = evidence.json(refs["sampling"])
    root = Path(prepared["root"])
    require(Path(refs["sampling"]["path"]) == root / "state_sampling/result.json", "registered state result path")
    require(
        actual["schema_version"] == "s2a.clean-training-state-sampling.v1"
        and actual["complete"] is actual["passed"] is True
        and actual["prepared_ref"] == refs["prepared"]
        and actual["execution_registration_ref"] == refs["execution_registration"],
        "complete related state sampling",
    )
    require(
        actual["actual_requests"] == 99
        and actual["actual_denoise_calls"] == actual["actual_head_calls"] == 3465
        and actual["model_initializations"] == actual["model_initialization_attempts"] == 1
        and all(actual[k] == 0 for k in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases")),
        "full99/3465 single model, label-free sampling budget",
    )
    require(
        actual["source_sha256"] == sources
        and len(prepared["weight_files"]) == 163
        and actual["parameters_before"] == actual["parameters_after"]
        and actual["parameters_before"]["count"] > 0
        and actual["gpu_uuid"] == GPU,
        "fixed sources/stat163/base-parameter identity",
    )
    lifecycle(evidence, refs, refs["prepared"], refs["execution_registration"], actual, prepared)
    head = fixed_w2(evidence, refs["prepared"], refs["execution_registration"], actual)
    zero_parameters = CAPTURE.head_parameters(evidence.json(actual["zero_head_ref"]))
    require(
        not zero_parameters[0].any() and not zero_parameters[1].any() and zero_parameters[2] == 1.0,
        "explicit all-zero constant clean head",
    )
    before, after = actual["cosmos_imported_sources_after_load"], actual["cosmos_imported_sources_final"]
    require(before and all(after.get(k) == v for k, v in before.items()), "state imported model source changed")
    for record in after.values():
        if record.get("file") is not None:
            path = Path(record["file"]).resolve()
            require(
                path.is_relative_to(Path(prepared["model_config"]["cosmos_source_root"]))
                and sources.get(str(path)) == record["sha256"],
                "state imported source escaped qualified closure",
            )
            evidence.path(dict(path=str(path), sha256=record["sha256"]))
    generated, changed = verify_generations(evidence, prepared, registration, sources, actual, head)
    # Rehash all consumed source/params/artifacts before the first GT decode.
    evidence.finish()
    weight_stats(prepared)
    labels = validation_labels(evidence, prepared, numerical_contract_passed=True)
    rows = [
        dict(
            name=p["name"],
            episode=a["episode"],
            task_index=a["task_index"],
            sampling_seed=p["sampling_seed"],
            variant=p["variant"],
            errors=integrate(labels[a["episode"]], a, actions),
        )
        for p, a, actions in generated
    ]
    groups, gate = diagnosis(rows)
    evidence.finish()
    weight_stats(prepared)
    result = dict(
        schema_version=SCHEMA,
        complete=True,
        passed=True,
        prepared_ref=refs["prepared"],
        execution_registration_ref=refs["execution_registration"],
        sampling_ref=refs["sampling"],
        w2_fit_ref=actual["w2_fit_ref"],
        w2_verification_ref=actual["w2_verification_ref"],
        readout_ref=actual["readout_ref"],
        actual_requests=99,
        actual_denoise_calls=3465,
        actual_head_calls=3465,
        model_initializations=1,
        zero_requests_verified=3,
        zero_fullpoint_nodes_verified=105,
        paired_initial_conditions=48,
        actual_solver_sigma_traces_verified=99,
        validation_episode_states_read=6,
        state_rows=rows,
        state_metrics=groups,
        **diagnostic_verdict(gate, numerical_contract_passed=True),
        physics_tasks_executed=0,
        fresh_cases=0,
        source_sha256=sources,
        input_sha256=evidence.inputs,
        changed_active_values=changed,
        validation_initial_sync_certified=False,
        validation_clock_seconds=0.05,
        clock_evidence_boundary="Inherited published20Hz endpoints; cache has no per-endpoint physical timestamps. Validation native initial synchronization remains uncertified.",
        evidence_boundary="All3465 saved action nodes and full initial arrays verified. Omitted video/flat/head values retain online/hash witnesses; full-flat qualification remains037/039. State gate is reference proximity, not task success.",
        **{k + "_ref": refs[k] for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")},
    )
    output = Path(args.output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "new state audit output must remain in training root")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared",
        "execution-registration",
        "sampling",
        "worker-launch",
        "worker-exit",
        "worker-release",
        "gpu-after",
    ):
        result.add_argument("--" + name, type=Path, required=True)
        result.add_argument("--" + name + "-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, choices=(42,), required=True)
    return result


if __name__ == "__main__":
    verify(parser().parse_args())
