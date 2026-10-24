"""Independent, label-free NumPy audit of a complete D0 or D1 capture.

Only explicitly named capture NPZs are decoded. Omitted vision/flat values are
online hash witnesses, not offline numerical reconstructions. Process receipts
must come from the real owner; this checker never creates lifecycle evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from s2a.clean_execution_proof import check_execution_proof
from s2a.clean_training_seed_plan import apply_seed_plan

REPO = Path(__file__).resolve().parents[1]
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
VISION = (1, 48, 11, 60, 40)
DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
MASK = [True] * 10 + [False] * 118
ACTIVE = {
    "sample_active10_fp32": "sample_full128_fp32",
    "raw_velocity_active10": "raw_velocity_full128",
    "deployed_velocity_active10": "deployed_velocity_full128",
    "c0_active10": "c0_full128",
}
SMALL = {
    "hidden",
    "projection_ids",
    "clean_residual_active10",
    "velocity_delta_active10",
    "time_vision",
    "time_action",
    "condition_vision",
    "condition_action",
}
SOURCE_KEYS = (
    set(ACTIVE.values())
    | SMALL
    | {
        "packed_vision",
        "packed_action",
        "sample_fp32_flat",
        "raw_velocity_flat",
        "deployed_velocity_flat",
        "head_base_full128",
    }
)
META_KEYS = {
    "request_name",
    "noise_seed",
    "purpose",
    "step_index",
    "sigma_actual",
    "network_timestep",
    "action_offset",
    "action_dim_mask",
    "feature_terms",
    "original_dtypes",
    "clean_semantics",
    "returned_original_velocity_object",
}
SEMANTICS = "ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0"
REQUEST_KEYS = {
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
ACTION_KEYS = {
    "action_dim_mask",
    "action_domain",
    "action_fps",
    "camera_layout",
    "return_video",
    "selected_roles",
    "task_title",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def hexsha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Evidence:
    def __init__(self):
        self.inputs = {}

    def path(self, reference):
        require(
            isinstance(reference, dict) and set(reference) == {"path", "sha256"} and hexsha(reference["sha256"]),
            "exact path/SHA reference required",
        )
        path = Path(reference["path"])
        require(path.is_absolute() and path.is_file(), "missing absolute artifact")
        path = path.resolve()
        actual = sha(path)
        require(actual == reference["sha256"] and self.inputs.get(str(path), actual) == actual, "artifact SHA changed")
        self.inputs[str(path)] = actual
        return path

    def json(self, reference):
        path = self.path(reference)
        require(path.suffix == ".json", "JSON reader refuses future arrays")
        return json.loads(path.read_text())

    def arrays(self, reference, expected_path):
        path = self.path(reference)
        require(
            path == Path(expected_path).resolve() and path.suffix == ".npz",
            "capture archive escaped its exact registered path",
        )
        with np.load(path, allow_pickle=False) as archive:
            return {k: np.asarray(archive[k]).copy() for k in archive.files}

    def mapping(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), "empty source/input closure")
        for path, digest in mapping.items():
            self.path({"path": path, "sha256": digest})

    def finish(self):
        require(
            all(sha(path) == digest for path, digest in self.inputs.items()),
            "consumed source/parameter/artifact changed during audit",
        )


def summary(array):
    a = np.asarray(array)
    require(a.dtype.kind in "biuf" and np.isfinite(a).all(), "nonfinite or nonnumeric capture")
    return {
        "shape": list(a.shape),
        "storage_dtype": a.dtype.str,
        "snapshot_bytes": a.nbytes,
        "snapshot_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
    }


def exact(actual, expected, message):
    require(summary(actual) == summary(expected), message)


def deploy(values, dtype):
    """Direct binary64 to BF16 ties-to-even using integer significands."""
    x = np.asarray(values, np.float64)
    require(np.isfinite(x).all(), "nonfinite deployment input")
    if dtype == "torch.float32":
        result = x.astype(np.float32)
    else:
        require(dtype == "torch.bfloat16", "only qualified BF16/FP32 deployment")
        magnitude, out = np.abs(x), np.zeros_like(x)
        normal = magnitude >= 2.0**-126
        bits = magnitude[normal].view(np.uint64)
        exponent = ((bits >> np.uint64(52)) & np.uint64(2047)).astype(np.int64) - 1023
        significand = (bits & np.uint64((1 << 52) - 1)) | np.uint64(1 << 52)
        quotient, remainder = significand >> np.uint64(45), significand & np.uint64((1 << 45) - 1)
        increment = (remainder > np.uint64(1 << 44)) | (
            (remainder == np.uint64(1 << 44)) & ((quotient & np.uint64(1)) != 0)
        )
        out[normal] = np.ldexp((quotient + increment.astype(np.uint64)).astype(np.float64), exponent - 7)
        out[~normal] = np.ldexp(np.rint(np.ldexp(magnitude[~normal], 133)), -133)
        result = np.copysign(out, x).astype(np.float32)
    require(np.isfinite(result).all(), "deployment overflow")
    return result


def head_parameters(value):
    require(
        set(value) == {"schema_version", "coefficients", "center", "scale", "ridge"}
        and value["schema_version"] == "s2a.clean-residual-readout.v1",
        "W1 parameter schema",
    )
    coefficient, center = np.asarray(value["coefficients"], np.float64), np.asarray(value["center"], np.float64)
    scale = value["scale"]
    require(
        coefficient.shape == (1291, 10)
        and center.shape == (1290,)
        and np.isfinite(coefficient).all()
        and np.isfinite(center).all(),
        "W1 parameter layout",
    )
    require(
        type(scale) in (int, float) and np.isfinite(scale) and scale > 0 and value["ridge"] == 0.1,
        "W1 frozen scale/ridge",
    )
    return coefficient, center, float(scale)


def current_request(request, entry, anchor):
    require(
        set(request) == REQUEST_KEYS
        and set(request["extra"]) == {"action128"}
        and set(request["extra"]["action128"]) == ACTION_KEYS,
        "unregistered/future request input",
    )
    action = request["extra"]["action128"]
    require(
        request["name"] == entry["name"]
        and request["seed"] == entry["sampling_seed"]
        and request["vision_path"] == anchor["image_ref"]["path"]
        and request["prompt"] == action["task_title"] == anchor["task"],
        "request current-image/seed/task binding",
    )
    require(
        request["model_mode"] == "wam"
        and request["action_chunk_size"] == 40
        and request["num_frames"] == 41
        and request["num_steps"] == 35
        and request["guidance"] == 1.0
        and request["shift"] == 5.0
        and request["image_size"] == 960
        and request["domain_name"] == "franka_panda__parallel_gripper"
        and request["fps"] == action["action_fps"] == 20
        and action["action_domain"] == DOMAIN
        and action["action_dim_mask"] == MASK
        and action["selected_roles"] == ["auxiliary_0", "left_wrist"]
        and action["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
        and action["return_video"] is False,
        "fixed current-only H40 sampling contract",
    )


def verify_point(
    record, arrays, header, step, *, head=None, vision_shape=VISION, time_vision_shape=(6000,), purpose="training"
):
    """Pure numerical/summary audit; no producer or archive reader is called."""
    require(purpose in ("training", "evaluation"), "explicit training/evaluation audit purpose required")
    meta, sources = record["metadata"], record["source_arrays"]
    request = header["request"]
    require(
        set(record)
        == {
            "schema_version",
            "request_ref",
            "metadata",
            "source_arrays",
            "compact_arrays",
            "arrays_ref",
            "source_stream_sha256",
            "online_checks",
            "omitted_vision_reconstructible",
        },
        "unregistered/future compact record field",
    )
    require(
        record["schema_version"] == "s2a.clean-compact-point.v1" and record["omitted_vision_reconstructible"] is False,
        "compact point schema",
    )
    require(
        set(meta) == META_KEYS
        and set(sources) == SOURCE_KEYS
        and set(arrays) == set(ACTIVE) | SMALL | {"action_dim_mask"},
        "unregistered/future point field",
    )
    require(
        set(record["compact_arrays"]) == set(arrays)
        and all(summary(a) == record["compact_arrays"][k] for k, a in arrays.items()),
        "compact array bytes/index mismatch",
    )
    require(
        meta["request_name"] == request["request_name"]
        and meta["noise_seed"] == request["noise_seed"]
        and meta["purpose"] == request["purpose"] == purpose
        and ((request["noise_seed"] in (42, 43)) == (purpose == "evaluation")),
        "point noise/purpose identity",
    )
    require(
        meta["step_index"] == step["step_index"]
        and [meta["sigma_actual"], meta["network_timestep"]] == list(header["expected_grid"][step["step_index"]]),
        "actual sigma/time grid",
    )
    sigma = meta["sigma_actual"]
    require(
        type(sigma) is float
        and 0 < sigma <= 1
        and float(np.float32(sigma)) == sigma
        and sigma == step["solver_sigma"]
        and meta["network_timestep"] == step["solver_timestep"],
        "positive actual FP32 sigma",
    )
    require(
        meta["action_offset"] == int(np.prod(vision_shape))
        and meta["action_dim_mask"] == MASK
        and meta["feature_terms"] == []
        and meta["clean_semantics"] == SEMANTICS,
        "actual sample geometry/mask/C0 convention",
    )
    dtypes = dict(meta["original_dtypes"])
    require(len(dtypes) == len(meta["original_dtypes"]) and set(dtypes) == SOURCE_KEYS, "original dtype roster")
    storage = {
        "torch.float32": "<f4",
        "torch.bfloat16": "<f4",
        "torch.int64": "<i8",
        "torch.bool": "|b1",
        "numpy.float64": "<f8",
        "torch.float64": "<f8",
    }
    for name, witness in sources.items():
        require(
            set(witness) == {"shape", "storage_dtype", "snapshot_bytes", "snapshot_sha256", "original_dtype"},
            "source witness schema",
        )
        require(
            witness["original_dtype"] == dtypes[name]
            and witness["storage_dtype"] == storage.get(dtypes[name])
            and hexsha(witness["snapshot_sha256"]),
            "source dtype/hash witness",
        )
        require(
            witness["snapshot_bytes"] == int(np.prod(witness["shape"])) * np.dtype(witness["storage_dtype"]).itemsize,
            "source witness byte denominator",
        )
    raw_dtype = dtypes["raw_velocity_full128"]
    require(
        dtypes["hidden"] in ("torch.bfloat16", "torch.float32")
        and dtypes["projection_ids"] == "torch.int64"
        and all(
            dtypes[k] == "torch.float32" for k in ("time_action", "time_vision", "condition_action", "condition_vision")
        )
        and all(
            dtypes[k] == "numpy.float64" for k in ("c0_full128", "clean_residual_active10", "velocity_delta_active10")
        ),
        "original small-array dtype convention",
    )
    require(
        raw_dtype in ("torch.bfloat16", "torch.float32")
        and all(
            dtypes[k] == raw_dtype for k in ("raw_velocity_flat", "deployed_velocity_flat", "deployed_velocity_full128")
        ),
        "raw/deployed dtype contract",
    )
    require(
        dtypes["sample_full128_fp32"] == dtypes["sample_fp32_flat"] == "torch.float32"
        and tuple(sources["packed_vision"]["shape"]) == tuple(vision_shape),
        "actual FP32 sample/vision shape",
    )
    for name in ("sample_fp32_flat", "raw_velocity_flat", "deployed_velocity_flat"):
        require(sources[name]["shape"] == [meta["action_offset"] + 5120], "flat source shape")
    for name in ("packed_action", "head_base_full128", *ACTIVE.values()):
        require(sources[name]["shape"] == [40, 128], "full action witness shape")
    for name, value in arrays.items():
        summary(value)
        if name in SMALL:
            require(
                summary(value) == {k: sources[name][k] for k in summary(value)},
                "small array differs from source witness",
            )
            if dtypes[name] == "torch.bfloat16":
                require(
                    value.dtype == np.float32 and not (value.view(np.uint32) & 65535).any(), "not lossless BF16 capture"
                )
    exact(arrays["action_dim_mask"], np.asarray(MASK, bool), "active10 mask bytes")
    require(
        arrays["hidden"].shape == (40, 1280)
        and arrays["hidden"].dtype == np.float32
        and arrays["projection_ids"].dtype == np.int64
        and arrays["projection_ids"].shape == (40,)
        and not arrays["projection_ids"].any(),
        "hidden/projection layout",
    )
    for name in ("time_action", "time_vision"):
        require(
            arrays[name].dtype == np.float32
            and arrays[name].shape == ((40,) if name == "time_action" else tuple(time_vision_shape)),
            "network-time shape/dtype",
        )
        exact(arrays[name], np.full_like(arrays[name], meta["network_timestep"]), "network-time values")
    cond = np.zeros((vision_shape[2], 1, 1), np.float32)
    cond[0] = 1
    exact(arrays["condition_vision"], cond, "only current vision conditioned")
    exact(arrays["condition_action"], np.zeros((40, 1), np.float32), "no privileged action conditioning")
    for name in ACTIVE:
        require(
            arrays[name].shape == (40, 10)
            and arrays[name].dtype == (np.float64 if name == "c0_active10" else np.float32),
            "compact action layout",
        )
    for name in ("clean_residual_active10", "velocity_delta_active10"):
        require(arrays[name].shape == (40, 10) and arrays[name].dtype == np.float64, "residual FP64 layout")
    sample, raw, deployed = [
        arrays[k] for k in ("sample_active10_fp32", "raw_velocity_active10", "deployed_velocity_active10")
    ]
    if raw_dtype == "torch.bfloat16":
        require(
            not (raw.view(np.uint32) & 65535).any() and not (deployed.view(np.uint32) & 65535).any(),
            "active velocity export is not lossless BF16",
        )
    c0 = sample.astype(np.float64) - sigma * raw.astype(np.float64)
    exact(arrays["c0_active10"], c0, "C0 must use actual FP32 sample and raw velocity")
    adapted = head is not None and bool(np.any(head[0]))
    residual = np.zeros((40, 10), np.float64)
    if adapted:
        coefficient, center, scale = head
        features = np.column_stack((arrays["hidden"].astype(np.float64), c0))
        residual = np.column_stack(((features - center) / scale, np.ones(40))) @ coefficient
    require(
        np.allclose(arrays["clean_residual_active10"], residual, atol=1e-12, rtol=0), "independent W1 clean residual"
    )
    delta = -arrays["clean_residual_active10"] / sigma if adapted else np.zeros((40, 10), np.float64)
    exact(arrays["velocity_delta_active10"], delta, "actual clean-to-velocity sign/sigma")
    expected = deploy(raw.astype(np.float64) + delta, raw_dtype) if adapted else raw
    exact(deployed, expected, "direct target-dtype deployment rounding")
    require(
        meta["returned_original_velocity_object"] is (not adapted)
        and step["returned_original_velocity_object"] is (not adapted),
        "returned velocity object witness",
    )
    for compact_name, full_name in ACTIVE.items():
        full = np.zeros((40, 128), arrays[compact_name].dtype)
        full[:, :10] = arrays[compact_name]
        require(
            summary(full) == {k: sources[full_name][k] for k in summary(full)}, "full128/zero inactive witness binding"
        )
    full = np.zeros((40, 128), np.float32)
    full[:, :10] = sample
    packed = deploy(full, dtypes["packed_action"])
    require(
        summary(packed) == {k: sources["packed_action"][k] for k in summary(packed)},
        "packed action rounding/source binding",
    )
    require(
        record["online_checks"]
        == dict(
            vision_velocity_bits_unchanged=True,
            inactive_velocity_bits_unchanged=True,
            flat_action_binding=True,
            active_clean_and_deployment_math=True,
        ),
        "incomplete online omitted-array witnesses",
    )
    if not adapted:
        require(
            sources["raw_velocity_flat"] == sources["deployed_velocity_flat"], "raw returned full-flat bytes differ"
        )
    return int(np.count_nonzero(deployed != raw))


def verify_initial(metadata, arrays, entry, caption, first_point, *, vision_shape=VISION):
    require(
        set(arrays) == {"initial_noise", "condition_mask", "first_condition_latent"}
        and all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values()),
        "full initial FP32 roster",
    )
    require(
        metadata["name"] == entry["name"]
        and metadata["generation_index"] == entry["index"]
        and metadata["sampling_seed"] == entry["sampling_seed"]
        and metadata["inference_only_current_image"] is True
        and metadata["caption"] == caption,
        "initial request/current condition identity",
    )
    expected_vision = np.zeros(vision_shape, np.float32)
    expected_vision[:, :, :1] = 1
    action_mask = np.ones((40, 128), np.float32)
    action_mask[:, :10] = 0
    exact(
        arrays["condition_mask"],
        np.concatenate((expected_vision.reshape(-1), action_mask.reshape(-1))),
        "full native current/future/active/inactive initial mask",
    )
    offset = int(np.prod(vision_shape))
    noise = arrays["initial_noise"]
    require(noise.shape == (offset + 5120,), "initial full sample shape")
    exact(noise[offset:].reshape(40, 128)[:, 10:], np.zeros((40, 118), np.float32), "initial inactive zero clamp")
    exact(
        noise[:offset].reshape(vision_shape)[:, :, :1], arrays["first_condition_latent"], "current latent prefix bits"
    )
    require(
        summary(noise) == {k: first_point["source_arrays"]["sample_fp32_flat"][k] for k in summary(noise)},
        "actual first sample SHA does not bind full initial noise",
    )
    full = np.zeros((40, 128), np.float32)
    full[:, :10] = first_point["_sample_active"]
    exact(noise[offset:].reshape(40, 128), full, "initial-to-first active sample binding")


def verify_trace(trace, entry, grid, sources, noise_roster, *, purpose="training"):
    require(purpose in ("training", "evaluation"), "explicit training/evaluation audit purpose required")
    require(
        trace["complete"] is True
        and trace["name"] == entry["name"]
        and trace["seed"] == entry["sampling_seed"]
        and trace["purpose"] == purpose
        and ((trace["seed"] in (42, 43)) == (purpose == "evaluation"))
        and trace["sampler_calls"] == 1,
        "single sampler trace with the declared noise/purpose",
    )
    require(
        trace["parameters_before"] == trace["parameters_after"]
        and trace["source_sha256"] == sources
        and trace["noise_roster"] == noise_roster
        and trace["feature_terms"] == [],
        "trace parameters/source/noise closure",
    )
    require(
        trace["solver_sigma_dtype"] == "torch.float32"
        and trace["terminal_sigma"] == 0
        and trace["terminal_neural_calls"] == 0
        and len(trace["steps"]) == 35,
        "terminal/35-node denominator",
    )
    require(
        trace["sigma_values"] == [x[0] for x in grid] + [0.0] and trace["scheduler_timesteps"] == [x[1] for x in grid],
        "qualified35 actual grid",
    )
    require(
        all(
            trace["actual_sampler_arguments"].get(k) == v
            for k, v in dict(num_steps=35, shift=5.0, seed=[entry["sampling_seed"]]).items()
        ),
        "actual sampler arguments",
    )
    for i, step in enumerate(trace["steps"]):
        require(
            step["step_index"] == i
            and step["solver_step_index_before"] == (None if i == 0 else i)
            and (step["solver_sigma"], step["solver_timestep"]) == tuple(grid[i]),
            "actual solver step binding",
        )
        require(
            all(step[k] == 1 for k in ("denoise_calls", "head_calls", "velocity_delegations", "conversion_calls")),
            "one denoise/head/velocity/conversion per node",
        )
        require(
            all(
                step[k] is True
                for k in (
                    "conversion_velocity_is_deployed_storage_view",
                    "conversion_sample_is_original_storage_view",
                    "conversion_returned_original_object",
                )
            ),
            "scheduler consumes actual adapted storage",
        )


def audit_points(
    evidence,
    ledger,
    header,
    directory,
    steps,
    *,
    head=None,
    vision_shape=VISION,
    time_vision_shape=(6000,),
    purpose="training",
):
    require(ledger["nodes"] == len(ledger["points"]) == len(steps) == 35, "complete35 compact points required")
    stream = hashlib.sha256(canonical(header))
    first, changed = None, 0
    for index, (point_ref, step) in enumerate(zip(ledger["points"], steps, strict=True)):
        record = evidence.json(point_ref)
        require(
            Path(point_ref["path"]).resolve() == directory / f"compact/step{index:02d}.json"
            and record["request_ref"] == ledger["request_ref"]
            and step["step_index"] == index,
            "compact step path/request/order",
        )
        arrays = evidence.arrays(record["arrays_ref"], directory / f"compact/step{index:02d}.npz")
        stream.update(canonical({k: record[k] for k in ("metadata", "source_arrays", "compact_arrays")}))
        require(stream.hexdigest() == record["source_stream_sha256"], "compact source-summary chain")
        changed += verify_point(
            record,
            arrays,
            header,
            step,
            head=head,
            vision_shape=vision_shape,
            time_vision_shape=time_vision_shape,
            purpose=purpose,
        )
        if index == 0:
            first = dict(record, _sample_active=arrays["sample_active10_fp32"])
    require(stream.hexdigest() == ledger["source_stream_sha256"], "final compact source-summary chain")
    return first, 35, changed


def paired_initial(metadata, arrays, original, original_arrays):
    for key in ("sampling_seed", "caption", "text_tokens", "inference_only_current_image"):
        require(metadata[key] == original[key], "D0/D1 current condition/text differs")
    require(set(original_arrays) == set(arrays), "D0/D1 initial roster")
    for key in arrays:
        exact(arrays[key], original_arrays[key], "D0/D1 actual initial noise/current condition bits differ")


def lifecycle(evidence, refs, prepared_ref, registration_ref, collection, root, *, model_python):
    launch, exit_record, release, gpu = [
        evidence.json(refs[k]) for k in ("worker_launch_ref", "worker_exit_ref", "worker_release_ref", "gpu_after_ref")
    ]
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False), "real normal worker wait0 required"
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["original_error"] is None
        and release["signal_sent"] is False
        and release["only_owned_child_signalled"] is True,
        "real owned reap/noSignal required",
    )
    require(all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")), "owned PID/starttick binding")
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""), "worker UUID/LD identity"
    )
    command = launch["command"]
    require(
        len(command) > 1 and command[0] == model_python and command[1] == "scripts/collect_wam_clean_training.py",
        "lifecycle is not the collection worker",
    )
    for flag, value in (
        ("--prepared", prepared_ref["path"]),
        ("--prepared-sha256", prepared_ref["sha256"]),
        ("--execution-registration", registration_ref["path"]),
        ("--execution-registration-sha256", registration_ref["sha256"]),
        ("--round", collection["round_name"]),
        ("--gpu-uuid", GPU),
        ("--seed", "42"),
    ):
        require(
            command.count(flag) == 1 and command[command.index(flag) + 1] == value, "worker command admission binding"
        )
    require(
        gpu["released"] is True
        and gpu["uuid"] == GPU
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "actual normal GPU release required",
    )
    require(
        all(Path(refs[k]["path"]).resolve().is_relative_to(root) for k in refs),
        "lifecycle receipt escaped registered training root",
    )


def independent_metadata(evidence, metadata, prepared_ref, registration):
    require(
        metadata.get("schema_version") == "s2a.clean-training-preparation-independent.v1"
        and metadata.get("metadata_qualified") is True
        and metadata.get("execution_ready") is False
        and metadata.get("execution_authorized") is False
        and metadata.get("consumed_before_after_exact") is True
        and metadata.get("passed") is True
        and metadata.get("complete") is True
        and metadata.get("prepared_ref") == prepared_ref,
        "exact independent metadata verification",
    )
    verifier = metadata["verifier_ref"]
    evidence.path(verifier)
    require(
        registration["source_sha256"].get(verifier["path"]) == verifier["sha256"],
        "independent metadata verifier is not execution-frozen",
    )


def admission(evidence, prepared_ref, registration_ref):
    prepared, registration = evidence.json(prepared_ref), evidence.json(registration_ref)
    require(prepared["schema_version"] == "s2a.clean-training-preparation.v1", "metadata preparation schema")
    require(
        registration["schema_version"] == "s2a.clean-training-execution-registration.v1"
        and registration["prepared_ref"] == prepared_ref
        and registration["passed"] is registration["complete"] is True
        and registration["readiness"]["ready_for_execution"] is True
        and registration["automatic_retry"] is registration["fresh_performance_authorized"] is False,
        "independent final execution registration required",
    )
    check_execution_proof(registration, prepared_ref=prepared_ref, read_json=evidence.json, check_file=evidence.path)
    metadata = evidence.json(registration["metadata_verification_ref"])
    independent_metadata(evidence, metadata, prepared_ref, registration)
    sources = {}
    for mapping in (
        prepared["qualified_model_source_sha256"],
        prepared["source_sha256"],
        registration["source_sha256"],
    ):
        evidence.mapping(mapping)
        for path, digest in mapping.items():
            require(sources.get(path, digest) == digest, "conflicting source closures")
            sources[path] = digest
    for mapping in (prepared["snapshot_sha256"], prepared["input_sha256"], registration["snapshot_sha256"]):
        evidence.mapping(mapping)
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_prepared"])
    for key in ("snapshot_sha256", "input_sha256", "checkpoint_small_metadata_sha256"):
        evidence.mapping(qualified[key])
    require(len(qualified["checkpoint_small_metadata_sha256"]) == 3, "three qualified checkpoint metadata SHAs")
    for required in (
        Path(__file__).resolve(),
        REPO / "scripts/collect_wam_clean_training.py",
        REPO / "src/s2a/clean_training_runtime.py",
        REPO / "src/s2a/wam_clean_compact.py",
    ):
        require(
            registration["source_sha256"].get(str(required)) == sha(required),
            "actual capture/checker source is not execution-frozen",
        )
    for path, expected in prepared["weight_files"].items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]), "weight stat changed")
    root = Path(prepared["root"]).resolve()
    require(
        root == REPO / "outputs/wam_stability_20261021/clean_residual_training_20261023"
        and prepared["config"]["gpu_uuid"] == GPU,
        "registered training root/GPU",
    )
    seed_ref = registration["training_seed_plan_ref"]
    prepared = apply_seed_plan(prepared, prepared_ref, evidence.json(seed_ref), seed_ref)
    return prepared, registration, sources


def training_caption(caption, task, seed):
    """Independently enforce the published optional-plan branch for this seed."""
    base = {
        "task_title",
        "command",
        "embodiment",
        "duration",
        "fps",
        "action_frequency_hz",
        "resolution",
        "aspect_ratio",
    }
    draw = int(hashlib.blake2b(json.dumps(seed, sort_keys=True).encode(), digest_size=16).hexdigest(), 16) / 2**128
    required = base | ({"task_plan"} if draw < 0.9 else set())
    require(
        set(caption) - {"camera_layout"} == required
        and caption["task_title"] == caption["command"] == task
        and (caption["task_plan"] == [] if draw < 0.9 else "task_plan" not in caption),
        "current-task caption differs from actual seed's frozen renderer",
    )


def verify(args):
    require(
        args.seed == 42
        and Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "CPU checker requires S2A cwd/seed42/empty CUDA and LD",
    )
    evidence = Evidence()
    names = (
        "prepared",
        "execution_registration",
        "collection",
        "worker_launch",
        "worker_exit",
        "worker_release",
        "gpu_after",
    )
    refs = {
        name: dict(path=str(Path(getattr(args, name)).resolve()), sha256=getattr(args, name + "_sha256"))
        for name in names
    }
    prepared, registration, sources = admission(evidence, refs["prepared"], refs["execution_registration"])
    collection = evidence.json(refs["collection"])
    root = Path(prepared["root"])
    round_name = collection["round_name"]
    require(
        round_name in ("D0", "D1")
        and Path(refs["collection"]["path"]) == root / "capture" / round_name / "result.json",
        "capture round/source path",
    )
    require(
        collection["schema_version"] == "s2a.clean-training-capture.v1"
        and collection["passed"] is collection["complete"] is True
        and collection["prepared_ref"] == refs["prepared"]
        and collection["execution_registration_ref"] == refs["execution_registration"],
        "complete capture producer binding",
    )
    require(
        collection["training_seed_plan_ref"] == registration["training_seed_plan_ref"],
        "capture did not bind the explicit31-bit execution seed plan",
    )
    require(
        collection["actual_requests"] == len(collection["requests"]) == len(prepared["training_requests"]) == 300
        and collection["actual_denoise_calls"] == collection["actual_head_calls"] == 10500
        and collection["model_initializations"] == collection["model_initialization_attempts"] == 1,
        "complete300/10500 single-initialization denominator",
    )
    require(
        all(collection[k] == 0 for k in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases"))
        and collection["source_sha256"] == sources
        and collection["parameters_before"] == collection["parameters_after"]
        and collection["gpu_uuid"] == GPU,
        "label-free unmodified-source/base-parameter capture",
    )
    lifecycle_refs = {k + "_ref": refs[k] for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")}
    lifecycle(
        evidence,
        lifecycle_refs,
        refs["prepared"],
        refs["execution_registration"],
        collection,
        root,
        model_python=prepared["model_config"]["model_python"],
    )
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    qualified_trace = evidence.json(qualified["generations"][0]["trace_ref"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in qualified_trace["steps"]]
    require(len(grid) == 35 and all(a[0] > b[0] > 0 for a, b in zip(grid, grid[1:])), "qualified grid denominator")
    normalizer = evidence.json(collection["normalization_ref"])
    require(
        normalizer["domain"] == DOMAIN and normalizer["normalization_id"] == 31 and normalizer["projection_id"] == 0,
        "recorded normalization identity",
    )
    before, after = collection["cosmos_imported_sources_after_load"], collection["cosmos_imported_sources_final"]
    require(before and all(after.get(k) == v for k, v in before.items()), "imported source closure changed")
    for record in after.values():
        if record.get("file") is not None:
            path = Path(record["file"]).resolve()
            require(
                path.is_relative_to(Path(prepared["model_config"]["cosmos_source_root"]))
                and sources.get(str(path)) == record["sha256"],
                "import escaped frozen Cosmos source",
            )
            evidence.path(dict(path=str(path), sha256=record["sha256"]))
    head, d0 = None, None
    d0_audit_ref = None
    if round_name == "D0":
        require(
            args.d0_audit is None
            and all(
                collection[k] is None for k in ("readout_ref", "d0_audit_ref", "w1_fit_ref", "w1_verification_ref")
            ),
            "D0 must be raw without a fitted head",
        )
    else:
        require(args.d0_audit is not None and args.d0_audit_sha256 is not None, "D1 requires exact D0 audit ref")
        d0_audit_ref = dict(path=str(Path(args.d0_audit).resolve()), sha256=args.d0_audit_sha256)
        d0_audit = evidence.json(d0_audit_ref)
        require(
            d0_audit["schema_version"] == "s2a.clean-training-capture-audit.v1"
            and d0_audit["passed"] is d0_audit["complete"] is True
            and d0_audit["prepared_ref"] == refs["prepared"]
            and d0_audit["execution_registration_ref"] == refs["execution_registration"]
            and d0_audit["round_name"] == "D0"
            and d0_audit["actual_requests"] == 300
            and d0_audit["actual_denoise_calls"] == 10500,
            "D0 complete independent capture qualification",
        )
        d0 = evidence.json(d0_audit["collection_ref"])
        require(
            d0["passed"] is d0["complete"] is True and d0["normalization_ref"] == collection["normalization_ref"],
            "D0 normalizer/source binding",
        )
        fit, checked_fit = evidence.json(collection["w1_fit_ref"]), evidence.json(collection["w1_verification_ref"])
        require(
            collection["d0_audit_ref"] == d0_audit_ref
            and fit["capture_audit_refs"] == [d0_audit_ref]
            and fit["collection_refs"] == [d0_audit["collection_ref"]],
            "W1 exact D0 origin",
        )
        require(
            fit["schema_version"] == "s2a.clean-training-fit.v1"
            and checked_fit["schema_version"] == "s2a.clean-training-fit-verification.v1"
            and all(
                v["passed"] is v["complete"] is True
                and v["stage"] == "W1"
                and v["prepared_ref"] == refs["prepared"]
                and v["execution_registration_ref"] == refs["execution_registration"]
                for v in (fit, checked_fit)
            ),
            "frozen W1 full-fit qualification",
        )
        require(
            checked_fit["fit_ref"] == collection["w1_fit_ref"]
            and checked_fit["params_ref"] == fit["params_ref"] == collection["readout_ref"]
            and fit["producer_fits"] == 1,
            "W1 parameter identity",
        )
        head = head_parameters(evidence.json(collection["readout_ref"]))
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    require(
        len(anchors) == 150
        and len({e["sampling_seed"] for e in prepared["training_requests"]}) == 300
        and prepared["noise_roster"]["evaluation_seeds"] == [42, 43],
        "complete training identity/noise matrix",
    )
    labels = [e["name"] for e in prepared["training_requests"] for _ in range(35)]
    require(
        collection["boundary_calls"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "10500 actual denoise boundary ledger",
    )
    points, changed = 0, 0
    for expected, entry in zip(prepared["training_requests"], collection["requests"], strict=True):
        require(
            entry["complete"] is True
            and all(entry[k] == expected[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
            "registered300 request order/identity",
        )
        require(expected["sampling_seed"] not in (42, 43), "evaluation noise entered training capture")
        directory = root / "capture" / round_name / f"{entry['index']:03d}_{entry['name']}"
        anchor = anchors[entry["anchor_name"]]
        current_request(expected["request"], expected, anchor)
        require(
            evidence.json(entry["request_ref"]) == expected["request"]
            and Path(entry["request_ref"]["path"]).resolve() == directory / "request.json",
            "exact current-only saved request",
        )
        evidence.path(anchor["image_ref"])
        conditioning = evidence.json(entry["conditioning_ref"])
        caption = json.loads(conditioning["caption"])
        training_caption(caption, expected["request"]["prompt"], expected["sampling_seed"])
        require(
            conditioning["inference_only_current_image"] is True
            and hashlib.sha256(conditioning["caption"].encode()).hexdigest() == conditioning["caption_sha256"]
            and caption["command"] == caption["task_title"] == expected["request"]["prompt"],
            "current-image/task-only caption",
        )
        context = conditioning["context"]
        require(
            context["domain"] == DOMAIN
            and context["normalization_id"] == 31
            and context["projection_id"] == 0
            and context["action_dim_mask"] == MASK,
            "actual conditioning domain",
        )
        response = evidence.json(entry["response_ref"])["outputs"][0]["content"]
        for value in (response["action"], response["action128"]["normalized_action"]):
            array = np.asarray(value)
            require(
                array.shape == (40, 128) and np.isfinite(array).all() and not array[:, 10:].any(),
                "generated full128 response/mask",
            )
        trace = evidence.json(entry["trace_ref"])
        verify_trace(trace, expected, grid, sources, prepared["noise_roster"])
        ledger = evidence.json(entry["compact_ledger_ref"])
        require(
            Path(entry["compact_ledger_ref"]["path"]).resolve() == directory / "compact/ledger.json"
            and ledger["schema_version"] == "s2a.clean-compact-ledger.v1"
            and ledger["complete"] is True
            and ledger["nodes"] == len(ledger["points"]) == 35
            and ledger["future_labels_read"] == 0
            and ledger["omitted_vision_reconstructible"] is ledger["qualification_claim"] is False,
            "full label-free compact ledger",
        )
        header = evidence.json(ledger["request_ref"])
        require(
            Path(ledger["request_ref"]["path"]).resolve() == directory / "compact/request.json"
            and header["schema_version"] == "s2a.clean-compact-request.v1"
            and header["expected_vision_shape"] == list(VISION)
            and header["expected_grid"] == [list(x) for x in grid]
            and header["expected_nodes"] == 35
            and header["source_hash_scope"] == "C-order NumPy snapshot bytes"
            and header["labels_accepted"] is header["omitted_vision_reconstructible"] is False,
            "registered compact request/layout",
        )
        require(
            header["request"]
            == dict(
                request_name=entry["name"],
                noise_seed=entry["sampling_seed"],
                purpose="training",
                round_name=round_name,
                anchor_id=entry["anchor_name"],
                readout_sha256=None if head is None else collection["readout_ref"]["sha256"],
                current_image_sha256=anchor["image_ref"]["sha256"],
            ),
            "compact parameter/current-image identity",
        )
        first, count, mutated = audit_points(evidence, ledger, header, directory, trace["steps"], head=head)
        changed += mutated
        points += count
        initial = evidence.json(entry["initial_ref"])
        initial_arrays = evidence.arrays(initial["arrays"], directory / "initial.npz")
        require(
            initial["array_index"]
            == {k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in initial_arrays.items()},
            "initial array index",
        )
        require(initial["variant"] == ("raw" if round_name == "D0" else "w1"), "initial policy binding")
        verify_initial(initial, initial_arrays, expected, conditioning["caption"], first)
        if d0 is not None:
            paired = d0["requests"][entry["index"]]
            require(
                all(paired[k] == entry[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
                "D0/D1 paired request",
            )
            original = evidence.json(paired["initial_ref"])
            original_arrays = evidence.arrays(
                original["arrays"], root / "capture/D0" / f"{entry['index']:03d}_{entry['name']}" / "initial.npz"
            )
            paired_initial(initial, initial_arrays, original, original_arrays)
    require(points == 10500, "full10500 numerical denominator")
    evidence.finish()
    result = dict(
        schema_version="s2a.clean-training-capture-audit.v1",
        passed=True,
        complete=True,
        prepared_ref=refs["prepared"],
        execution_registration_ref=refs["execution_registration"],
        collection_ref=refs["collection"],
        round_name=round_name,
        **lifecycle_refs,
        actual_requests=300,
        actual_denoise_calls=10500,
        actual_online_points_verified=points,
        changed_active_values=changed,
        source_sha256=sources,
        input_sha256=evidence.inputs,
        d0_audit_ref=d0_audit_ref,
        future_label_reads=0,
        producer_fits=0,
        physics_steps=0,
        fresh_cases=0,
        omitted_vision_reconstructed=False,
        evidence_boundary="All saved action math and full initial arrays verified; omitted video/flat/head values retain hash/online witnesses only. Full-flat numerical qualification remains037/039.",
        residual_prediction_atol=1e-12,
        residual_prediction_rtol=0,
    )
    output = Path(args.output).resolve()
    require(
        output.is_relative_to(root) and not output.exists(), "new audit output must remain in registered training root"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared",
        "execution-registration",
        "collection",
        "worker-launch",
        "worker-exit",
        "worker-release",
        "gpu-after",
    ):
        result.add_argument("--" + name, type=Path, required=True)
        result.add_argument("--" + name + "-sha256", required=True)
    result.add_argument("--d0-audit", type=Path)
    result.add_argument("--d0-audit-sha256")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, choices=(42,), required=True)
    return result


if __name__ == "__main__":
    verify(parser().parse_args())
