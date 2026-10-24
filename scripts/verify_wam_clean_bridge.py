"""Independent NumPy audit of seven actual clean-bridge qualification requests.

This program consumes saved online captures only. It loads neither model nor
simulator and imports no producer feature, residual, conversion or rounding
implementation. Historical-output equality is a qualification gate, separate
from complete and internally consistent evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_wam_clean_bridge import publish, read, read_prepared, reference, require, resolve_path

ACTIVE = np.array([True] * 10 + [False] * 118)
LIBERO_DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
POINT_ARRAYS = {
    "sample_fp32_flat",
    "sample_full128_fp32",
    "raw_velocity_flat",
    "raw_velocity_full128",
    "deployed_velocity_flat",
    "deployed_velocity_full128",
    "c0_full128",
    "hidden",
    "projection_ids",
    "packed_action",
    "packed_vision",
    "time_action",
    "time_vision",
    "condition_action",
    "condition_vision",
    "head_base_full128",
    "clean_residual_active10",
    "velocity_delta_active10",
}


def numeric(value, message):
    value = np.asarray(value)
    require(value.dtype.kind in "biuf" and np.isfinite(value).all(), message)
    return value


def bits(actual, expected, message):
    actual, expected = numeric(actual, message), numeric(expected, message)
    require(
        actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
        message,
    )


def close(actual, expected, message, atol=1e-12):
    actual, expected = numeric(actual, message), numeric(expected, message)
    require(actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=0), message)


def array_file(ref):
    require(reference(ref["path"]) == ref, "saved arrays digest changed")
    with np.load(resolve_path(ref["path"]), allow_pickle=False) as archive:
        result = {key: numeric(archive[key], f"invalid saved array: {key}").copy() for key in archive.files}
    require(reference(ref["path"]) == ref, "saved arrays changed during audit")
    return result


def bundle(ref):
    point = read(ref)
    arrays = array_file(point["arrays"])
    require(
        point["array_index"]
        == {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
        "point array index differs from actual archive",
    )
    return point, arrays


def bf16(values):
    """Direct FP64 to BF16, rounding ties to even without an FP32 intermediate."""
    values = numeric(values, "non-finite BF16 input").astype(np.float64)
    _, exponent = np.frexp(np.abs(values))
    spacing = np.ldexp(np.ones(values.shape), np.maximum(exponent - 8, -133))
    magnitude = np.rint(np.abs(values) / spacing) * spacing
    require(np.all(magnitude <= np.ldexp(2 - 2**-7, 127)), "BF16 deployment overflow")
    return np.copysign(magnitude, values).astype(np.float32)


def cast_deployed(values, dtype):
    if dtype == "torch.bfloat16":
        return bf16(values)
    require(dtype in ("torch.float32", "torch.float16"), "unsupported deployed velocity dtype")
    result = np.asarray(values).astype(np.float32 if dtype == "torch.float32" else np.float16)
    require(np.isfinite(result).all(), "deployment cast overflow")
    return result


def saved_dtype(value, original, message):
    expected = {
        "torch.float64": np.float64,
        "numpy.float64": np.float64,
        "torch.float32": np.float32,
        "torch.bfloat16": np.float32,
        "torch.float16": np.float16,
        "torch.int64": np.int64,
        "torch.bool": np.bool_,
    }
    require(original in expected and value.dtype == expected[original], message)
    if original == "torch.bfloat16":
        bits(value, bf16(value), "BF16 export is not losslessly representable")


def read_head(ref, *, residual):
    """The qualification heads are registered constants; no data fit occurs."""
    parameters = read(ref)
    require(
        set(parameters) == {"schema_version", "coefficients", "center", "scale", "ridge"},
        "clean head parameter roster differs",
    )
    require(parameters["schema_version"] == "s2a.clean-residual-readout.v1", "clean head schema differs")
    coefficients = numeric(parameters["coefficients"], "invalid clean coefficients").astype(np.float64)
    center = numeric(parameters["center"], "invalid clean center").astype(np.float64)
    expected = np.zeros((1291, 10), np.float64)
    expected[-1] = residual
    bits(coefficients, expected, "qualification head is not the registered zero/constant-bias head")
    bits(center, np.zeros(1290, np.float64), "qualification head center was fitted/changed")
    require(parameters["scale"] == 1 and parameters["ridge"] == 0.1, "qualification head scale/ridge changed")
    return dict(coefficients=coefficients, center=center, scale=float(parameters["scale"]))


def verify_point(metadata, arrays, *, generation, step, head):
    require(set(arrays) == POINT_ARRAYS, "online capture array roster differs")
    require(
        metadata["request_name"] == generation["name"]
        and metadata["noise_seed"] == generation["sampling_seed"]
        and metadata["purpose"] == "evaluation"
        and metadata["step_index"] == step["step_index"]
        and metadata["sigma_actual"] == step["solver_sigma"]
        and metadata["network_timestep"] == step["solver_timestep"]
        and metadata["feature_terms"] == []
        and metadata["action_dim_mask"] == ACTIVE.tolist(),
        "point request/actual scheduler/domain binding differs",
    )
    sigma = metadata["sigma_actual"]
    require(
        type(sigma) is float and np.isfinite(sigma) and sigma > 0 and float(np.float32(sigma)) == sigma,
        "actual sigma must be positive observed FP32",
    )
    require(
        metadata["clean_semantics"] == "ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0",
        "ideal clean estimate is confused with mixed-precision scheduler x0",
    )
    original = dict(metadata["original_dtypes"])
    require(set(original) == set(arrays), "actual original dtype roster incomplete")
    for key, value in arrays.items():
        saved_dtype(value, original[key], f"actual stored/original dtype differs: {key}")
    sample, raw, deployed = (arrays[key] for key in ("sample_fp32_flat", "raw_velocity_flat", "deployed_velocity_flat"))
    offset = metadata["action_offset"]
    require(
        type(offset) is int
        and offset == arrays["packed_vision"].size
        and offset > 0
        and sample.dtype == np.float32
        and sample.shape == raw.shape == deployed.shape == (offset + 5120,),
        "actual FP32 flat sample/vision/action layout differs",
    )
    require(
        arrays["hidden"].shape == (40, 1280)
        and arrays["head_base_full128"].shape == (40, 128)
        and arrays["projection_ids"].shape == (40,)
        and arrays["projection_ids"].dtype == np.int64
        and np.all(arrays["projection_ids"] == 0),
        "original hidden/head/full-domain projection binding differs",
    )
    for key in (
        "sample_full128_fp32",
        "raw_velocity_full128",
        "deployed_velocity_full128",
        "packed_action",
        "c0_full128",
    ):
        require(
            arrays[key].shape == (40, 128) and not arrays[key][:, 10:].any(),
            f"action layout or inactive118 changed: {key}",
        )
    for key in ("clean_residual_active10", "velocity_delta_active10"):
        require(
            arrays[key].shape == (40, 10) and arrays[key].dtype == np.float64,
            "clean residual/velocity delta must be FP64 active10",
        )
    for key in ("time_action", "time_vision"):
        require(
            arrays[key].size > 0 and original[key] == "torch.float32", "actual packed network time dtype/shape differs"
        )
        bits(
            arrays[key],
            np.full_like(arrays[key], metadata["network_timestep"]),
            "actual packed network time is not observed sampler time",
        )
    require(not arrays["condition_action"].any(), "online action tokens contain privileged clean conditioning")
    bits(
        arrays["sample_full128_fp32"],
        sample[offset:].reshape(40, 128),
        "saved action sample is not actual FP32 latent slice",
    )
    bits(
        arrays["raw_velocity_full128"],
        raw[offset:].reshape(40, 128),
        "saved raw action velocity is not actual original return slice",
    )
    bits(
        arrays["deployed_velocity_full128"],
        deployed[offset:].reshape(40, 128),
        "deployed action is not actual scheduler return slice",
    )
    bits(deployed[:offset], raw[:offset], "clean head changed actual vision velocity bits")
    bits(
        arrays["deployed_velocity_full128"][:, 10:],
        arrays["raw_velocity_full128"][:, 10:],
        "clean head changed inactive118 velocity bits",
    )
    bits(
        arrays["packed_action"],
        cast_deployed(arrays["sample_full128_fp32"], original["packed_action"]),
        "packed action does not match actual FP32 sample cast",
    )
    c0 = arrays["sample_full128_fp32"].astype(np.float64) - sigma * arrays["raw_velocity_full128"].astype(np.float64)
    c0[:, 10:] = 0
    bits(arrays["c0_full128"], c0, "ideal FP64 C0 formula differs from actual FP32 sample/raw velocity")
    features = np.column_stack((arrays["hidden"].astype(np.float64), c0[:, :10]))
    design = np.column_stack(((features - head["center"]) / head["scale"], np.ones(40)))
    residual = design @ head["coefficients"]
    delta = -residual / sigma
    close(arrays["clean_residual_active10"], residual, "independent clean residual differs")
    close(arrays["velocity_delta_active10"], delta, "delta velocity is not minus residual / actual sigma")
    expected = arrays["raw_velocity_full128"].copy()
    if generation["variant"] == "nonzero":
        expected[:, :10] = cast_deployed(expected[:, :10].astype(np.float64) + delta, original["raw_velocity_full128"])
        require(
            step["returned_original_velocity_object"] is False, "nonzero head failed to return the adapted velocity"
        )
    else:
        require(
            step["returned_original_velocity_object"] is True, "raw/zero head must return original velocity objects"
        )
    bits(
        arrays["deployed_velocity_full128"],
        expected,
        "actual deployed velocity differs after registered dtype rounding",
    )
    return dict(
        step_index=step["step_index"],
        sigma_actual=sigma,
        nonzero_deployed_active_values=int(
            np.count_nonzero(expected[:, :10] != arrays["raw_velocity_full128"][:, :10])
        ),
    )


def verify_trace(trace, generation):
    require(
        trace["complete"] is True
        and trace["name"] == generation["name"]
        and trace["seed"] == generation["sampling_seed"]
        and trace["purpose"] == "evaluation"
        and trace["sampler_calls"] == 1
        and trace["parameters_before"] == trace["parameters_after"]
        and all(
            trace["actual_sampler_arguments"].get(key) == value
            for key, value in dict(num_steps=35, shift=5.0, seed=[generation["sampling_seed"]]).items()
        )
        and trace["solver_sigma_dtype"] == "torch.float32"
        and trace["terminal_sigma"] == 0
        and trace["terminal_neural_calls"] == 0,
        "actual single-conditional 35-node sampler trace differs",
    )
    sigmas, timesteps = (
        numeric(trace["sigma_values"], "invalid actual sigma grid"),
        numeric(trace["scheduler_timesteps"], "invalid actual time grid"),
    )
    require(
        sigmas.shape == (36,)
        and timesteps.shape == (35,)
        and sigmas[-1] == 0
        and np.all(sigmas[:-1] > 0)
        and np.all(np.diff(sigmas) < 0),
        "actual positive35 + terminal0 grid differs",
    )
    require(len(trace["steps"]) == 35, "actual35 velocity/denoise/head/conversion denominator missing")
    for index, step in enumerate(trace["steps"]):
        require(
            step["step_index"] == index
            and step["solver_step_index_before"] == (None if index == 0 else index)
            and step["solver_sigma"] == sigmas[index]
            and step["solver_timestep"] == timesteps[index]
            and step["velocity_delegations"]
            == step["denoise_calls"]
            == step["head_calls"]
            == step["conversion_calls"]
            == 1
            and step["conversion_sample_is_original_storage_view"] is True
            and step["conversion_velocity_is_deployed_storage_view"] is True
            and step["conversion_returned_original_object"] is True,
            "actual sampler velocity/conversion delegation or storage binding differs",
        )
    return trace["steps"]


def output_arrays(response):
    content = response["outputs"][0]["content"]
    raw, normalized = (
        numeric(content["action"], "invalid raw response"),
        numeric(content["action128"]["normalized_action"], "invalid normalized response"),
    )
    require(
        raw.shape == normalized.shape == (40, 128) and not raw[:, 10:].any() and not normalized[:, 10:].any(),
        "response full128 or inactive slots differ",
    )
    return raw, normalized


def output_exact(actual, expected):
    actual, expected = output_arrays(actual), output_arrays(expected)
    return all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for a, b in zip(actual, expected, strict=True)
    )


def verify_conditioning(conditioning, request):
    context = conditioning["context"]
    require(
        conditioning["inference_only_current_image"] is True
        and context["domain"] == LIBERO_DOMAIN
        and context["normalization_id"] == 31
        and context["projection_id"] == 0
        and context["action_dim_mask"] == ACTIVE.tolist(),
        "actual generation was not current-image full LIBERO conditioning",
    )
    caption = conditioning["caption"]
    require(
        hashlib.sha256(caption.encode()).hexdigest() == conditioning["caption_sha256"], "actual caption digest differs"
    )
    caption = json.loads(caption)
    require(
        caption["task_title"] == caption["command"] == request["prompt"],
        "actual generation text differs from current task",
    )


def verify_request(case):
    request, historical = case["request"], case["historical_entry"]["request"]
    image = case["image_ref"]
    require(reference(image["path"]) == image, "current image bytes differ from historical declaration")
    require(
        resolve_path(historical["vision_path"]) == resolve_path(image["path"]),
        "current image is not the selected historical current frame",
    )
    migrated = dict(historical, vision_path=str(resolve_path(image["path"])))
    require(request == migrated, "historical request changed beyond current-image path relocation")
    allowed = {
        "action_chunk_size",
        "domain_name",
        "extra",
        "fps",
        "guidance",
        "image_size",
        "model_mode",
        "name",
        "num_frames",
        "num_steps",
        "prompt",
        "seed",
        "shift",
        "vision_path",
    }
    require(set(request) <= allowed and set(request["extra"]) == {"action128"}, "unregistered online request input")
    action = request["extra"]["action128"]
    require(
        set(action)
        == {
            "action_dim_mask",
            "action_domain",
            "action_fps",
            "camera_layout",
            "return_video",
            "selected_roles",
            "task_title",
        },
        "unregistered privileged action conditioning",
    )
    require(
        request["model_mode"] == "wam"
        and request["action_chunk_size"] == 40
        and request["num_frames"] == 41
        and request["num_steps"] == 35
        and request["guidance"] == 1.0
        and request["shift"] == 5.0
        and request["fps"] == action["action_fps"] == 20
        and action["action_domain"] == LIBERO_DOMAIN
        and action["action_dim_mask"] == ACTIVE.tolist()
        and action["return_video"] is False
        and action["task_title"] == request["prompt"]
        and request["seed"] in (42, 43),
        "request differs from current-only masked H40/20Hz/35-node WAM contract",
    )


def verify_initial(metadata, arrays, generation, conditioning, first_point):
    require(
        set(arrays) == {"initial_noise", "condition_mask", "first_condition_latent"},
        "initial capture array roster differs",
    )
    require(
        all(value.dtype == np.float32 for value in arrays.values()), "initial sample/condition must retain actual FP32"
    )
    require(
        metadata["name"] == generation["name"]
        and metadata["sampling_seed"] == generation["sampling_seed"]
        and metadata["variant"] == generation["variant"]
        and metadata["generation_index"] == generation["index"]
        and metadata["inference_only_current_image"] is True
        and metadata["caption"] == conditioning["caption"],
        "actual initial current-image request binding differs",
    )
    require(
        arrays["condition_mask"].shape == arrays["initial_noise"].shape == first_point["sample_fp32_flat"].shape,
        "initial mask/noise layout differs",
    )
    require(
        np.isin(arrays["condition_mask"], [0, 1]).all() and not arrays["condition_mask"][-5120:].any(),
        "initial sample contains privileged action/future conditioning",
    )
    bits(
        arrays["initial_noise"],
        first_point["sample_fp32_flat"],
        "first actual sampler latent differs from saved initial FP32 sample",
    )
    require(
        arrays["first_condition_latent"].shape == (1, 48, 1, 60, 40), "current-frame condition latent geometry differs"
    )


def verify_imports(actual, prepared):
    root = Path(prepared["config"]["cosmos_source_root"])
    before, after = actual["cosmos_imported_sources_after_load"], actual["cosmos_imported_sources_final"]
    require(
        before and all(after.get(name) == record for name, record in before.items()),
        "loaded Cosmos modules changed/disappeared during qualification",
    )
    for records in (before, after):
        for name, record in records.items():
            require(
                name == "cosmos_framework" or name.startswith("cosmos_framework."), "non-Cosmos imported-source record"
            )
            if record.get("file") is None:
                require(record == dict(file=None, namespace_package=True), "invalid namespace-package source record")
            else:
                path = Path(record["file"])
                require(
                    path.is_relative_to(root)
                    and prepared["source_sha256"].get(str(path)) == record["sha256"]
                    and reference(path) == dict(path=str(path), sha256=record["sha256"]),
                    "actual Cosmos import escaped frozen source closure",
                )


def verify_pair(raw, zero):
    """Compare every saved online tensor, not merely the final action chunk."""
    require(
        raw["entry"]["case_id"] == zero["entry"]["case_id"]
        and raw["entry"]["sampling_seed"] == zero["entry"]["sampling_seed"],
        "raw/zero request pair differs",
    )
    require(output_exact(raw["response"], zero["response"]), "zero clean head changed final raw/normalized output")
    if "initial" in raw or "initial" in zero:
        require("initial" in raw and "initial" in zero, "raw/zero initial capture missing")
        for key in raw["initial"]:
            bits(zero["initial"][key], raw["initial"][key], f"zero head changed actual initial condition/noise: {key}")
    require(len(raw["points"]) == len(zero["points"]) == 35, "raw/zero35-point denominator differs")
    for (raw_meta, raw_arrays), (zero_meta, zero_arrays) in zip(raw["points"], zero["points"], strict=True):
        require(set(raw_arrays) == set(zero_arrays), "raw/zero tensor roster differs")
        for key in raw_arrays:
            bits(zero_arrays[key], raw_arrays[key], f"zero head changed actual online tensor: {key}")
        for key in (
            "sigma_actual",
            "network_timestep",
            "action_offset",
            "original_dtypes",
            "feature_terms",
            "action_dim_mask",
            "purpose",
        ):
            require(raw_meta[key] == zero_meta[key], f"zero head changed actual online metadata: {key}")
    return dict(case_id=raw["entry"]["case_id"], online_points_exact=35, final_raw_and_normalized_exact=True)


def verify(args):
    prepared_ref = dict(path=str(Path(args.prepared).absolute()), sha256=args.prepared_sha256)
    prepared = read_prepared(prepared_ref)
    config, root = prepared["config"], Path(prepared["root"])
    require(prepared["schema_version"] == "s2a.clean-bridge-preparation.v1", "clean bridge preparation schema differs")
    budget = dict(requests=7, denoise_calls=245, model_initializations=1, fit_groups=0, physics_steps=0, fresh_cases=0)
    require(
        config["budget"] == budget
        and config["model_training_authorized"]
        is config["fresh_performance_authorized"]
        is config["automatic_retry"]
        is False,
        "qualification budget/authorization contract changed",
    )
    require(config["feature_terms"] == [], "qualification feature contract changed")
    noise = config["noise_roster"]
    require(
        noise["evaluation_seeds"] == [42, 43]
        and noise["training_seeds"]
        and not set(noise["training_seeds"]) & {42, 43},
        "training noise overlaps reserved evaluation seeds",
    )
    require(
        len(noise["training_seeds"]) == len(set(noise["training_seeds"]))
        and all(type(seed) is int and seed >= 0 for seed in noise["training_seeds"]),
        "training noise roster must contain explicit unique valid seeds",
    )
    residual = numeric(config["nonzero_clean_residual_active10"], "invalid registered residual").astype(np.float64)
    require(
        residual.shape == (10,) and np.any(residual) and np.max(np.abs(residual)) <= 1e-4,
        "fixed small nonzero qualification head changed",
    )
    requests = prepared["requests3"]
    require(
        len(requests) == len({entry["name"] for entry in requests}) == 3, "fixed three old development requests missing"
    )
    historical_prepared = read(config["historical_prepared_ref"])
    require(
        config["historical_generation_indices"] == [0, 16, 32]
        and [entry["historical_entry"] for entry in requests]
        == [historical_prepared["free_generations"][index] for index in (0, 16, 32)],
        "fixed three historical development source entries changed",
    )
    for request in requests:
        verify_request(request)
    qualification_ref = reference(args.qualification)
    actual = read(qualification_ref)
    require(
        actual["schema_version"] == "s2a.clean-bridge-qualification.v1"
        and actual["passed"] is actual["complete"] is True
        and actual["prepared_ref"] == prepared_ref,
        "qualification incomplete or wrong origin",
    )
    require(
        actual["gpu_uuid"] == config["gpu_uuid"] and actual["gpu_uuid"] in config["allowed_gpu_uuids"],
        "model used an unauthorized GPU",
    )
    require(
        actual["runtime"]["gpu_uuid"].removeprefix("GPU-").lower() == config["gpu_uuid"].removeprefix("GPU-").lower(),
        "actual model driver GPU UUID differs from allowed device",
    )
    require(actual["parameters_before"] == actual["parameters_after"], "original model parameters changed")
    verify_imports(actual, prepared)
    require(
        actual["budget"] == budget
        and actual["wam_model_initializations"] == 1
        and actual["actual_requests"] == 7
        and actual["actual_denoise_calls"] == actual["actual_head_calls"] == 245
        and all(
            actual[key] == 0
            for key in ("new_candidate_fits", "new_episode_reads", "new_future_reads", "physics_steps", "fresh_cases")
        )
        and actual["python_version"] == ".".join(map(str, config["python_version"]))
        and str(resolve_path(actual["python_executable"])) == str(resolve_path(config["model_python"]))
        and actual["cosmos_source_root"] == config["cosmos_source_root"]
        and actual["mask_policy"] == config["mask_policy"],
        "actual model runtime/budget differs from finite qualification",
    )
    zero = read_head(actual["heads"]["zero_ref"], residual=np.zeros(10))
    nonzero = read_head(actual["heads"]["nonzero_ref"], residual=residual)
    entries = actual["generations"]
    require(
        [entry["variant"] for entry in entries] == ["raw"] * 3 + ["zero"] * 3 + ["nonzero"],
        "seven fixed raw3/zero3/nonzero1 requests differ",
    )
    require(
        len({entry["name"] for entry in entries}) == 7 and all(entry["complete"] is True for entry in entries),
        "seven uniquely named complete requests required",
    )
    expected_ids = [entry["name"] for entry in requests] * 2 + [config["nonzero_request_name"]]
    require(
        [entry["case_id"] for entry in entries] == expected_ids and config["nonzero_request_name"] in expected_ids[:3],
        "fixed request identities changed",
    )
    cases = {entry["name"]: entry for entry in requests}
    labels = [entry["name"] for entry in entries for _ in range(35)]
    require(
        actual["boundary_calls"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "actual245 denoise boundary ledger differs",
    )
    records, historical, points_verified, changed, changed_steps = [], [], 0, 0, 0
    for index, entry in enumerate(entries):
        require(entry["index"] == index, "request order/index changed")
        case = cases[entry["case_id"]]
        require(
            entry["request_ref"] == reference(resolve_path(entry["request_ref"]["path"])), "actual request ref differs"
        )
        request = read(entry["request_ref"])
        require(
            request == case["request"] and entry["sampling_seed"] == request["seed"] and request["seed"] in (42, 43),
            "request differs from registered old development input",
        )
        response = read(entry["response_ref"])
        conditioning = read(entry["conditioning_ref"])
        verify_conditioning(conditioning, request)
        trace = read(entry["trace_ref"])
        steps = verify_trace(trace, entry)
        require(trace["noise_roster"] == noise and trace["feature_terms"] == [], "actual request noise/features differ")
        require(trace["source_sha256"] == prepared["source_sha256"], "actual request frozen source closure differs")
        require(len(entry["points"]) == 35, "actual request must retain all35 online captures")
        points = [bundle(ref) for ref in entry["points"]]
        initial_metadata, initial_arrays = bundle(entry["initial_ref"])
        verify_initial(initial_metadata, initial_arrays, entry, conditioning, points[0][1])
        head = nonzero if entry["variant"] == "nonzero" else zero
        for (metadata, arrays), step in zip(points, steps, strict=True):
            result = verify_point(metadata, arrays, generation=entry, step=step, head=head)
            changed += result["nonzero_deployed_active_values"]
            changed_steps += int(result["nonzero_deployed_active_values"] > 0)
            points_verified += 1
        records.append(dict(entry=entry, response=response, points=points, initial=initial_arrays))
        if entry["variant"] == "raw":
            exact = output_exact(response, read(case["expected_response_ref"]))
            require(
                entry["historical_output_exact"] is exact,
                "worker historical equality flag differs from saved full outputs",
            )
            historical.append(
                dict(
                    case_id=entry["case_id"],
                    historical_output_exact=exact,
                    expected_response_ref=case["expected_response_ref"],
                )
            )
    pairs = [verify_pair(records[index], records[index + 3]) for index in range(3)]
    require(
        all(entries[index]["zero_pair_exact"] is True for index in (3, 4, 5)),
        "zero-pair worker status differs from full independent equality",
    )
    require(points_verified == 245 and changed > 0, "actual nonzero residual never changed any deployed active value")
    require(actual["mutation_steps"] == changed_steps, "actual nonzero mutation-step summary differs")
    read_prepared(prepared_ref)
    output = Path(args.output).absolute()
    require(output == root / "review/verified.json", "independent review output escaped fixed root")
    output.parent.mkdir(parents=True, exist_ok=False)
    qualified = all(row["historical_output_exact"] for row in historical)
    require(
        actual["bridge_qualified"] is qualified,
        "worker bridge qualification differs from independent historical equality",
    )
    publish(
        output,
        dict(
            schema_version="s2a.clean-bridge-independent-verification.v1",
            passed=True,
            complete=True,
            prepared_ref=prepared_ref,
            qualification_ref=qualification_ref,
            bridge_qualified=qualified,
            raw_zero_pairs=pairs,
            historical_regressions=historical,
            actual_online_points_verified=245,
            nonzero_changed_active_values=changed,
            model_training_authorized=False,
            fresh_performance_authorized=False,
            model_initializations=1,
            requests=7,
            denoise_calls=245,
            fit_groups=0,
            physics_steps=0,
            fresh_cases=0,
            evidence_level="real_model_saved_online_sampler_qualification",
            task_success_proven=False,
            interpretation="Qualification checks actual clean-bridge identity and arithmetic only. Historical response changes preserve a False bridge gate; no model fit, physical task or reserved-fresh success is certified.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
