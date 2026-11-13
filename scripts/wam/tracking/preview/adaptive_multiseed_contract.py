"""Bind task, seed, request and actual response for adaptive WAM sampling."""

import copy
from pathlib import Path

from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveModelAdmission
from s2a.research.wam.tracking.runtime.sampling_seed import require_sampling_seed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from scripts.wam.tracking.preview.reset_multitask_live_requests import TASKS


def validate_registration(registration, output):
    """Validate the explicit seed, task, paths and limits before model loading."""
    output = Path(output).resolve()
    prefix.require(type(registration) is dict, "An explicit adaptive registration is required")
    seed = require_sampling_seed(registration.get("seed"))
    prefix.require(
        output.is_relative_to(prefix.REPO / "outputs")
        and type(registration.get("max_requests")) is int
        and 1 <= registration["max_requests"] <= 75
        and type(registration.get("denoise_steps")) is int
        and registration["denoise_steps"] == 35
        and registration.get("task") in TASKS
        and type(registration.get("max_periods")) is int
        and type(registration.get("initial")) is dict
        and type(registration["initial"].get("step")) is int
        and registration["initial"]["step"] + registration["max_periods"] <= 600
        and type(registration.get("image_root")) is str
        and (prefix.REPO / registration["image_root"]).resolve() == output / "producer/simulation"
        and (output / "producer/simulation").is_dir()
        and registration["seed"] == seed,
        "Explicit bounded request and sampling limits are required before model initialization",
    )
    return AdaptiveModelAdmission(**registration)


def make_task_request(template, image, task, seed):
    """Retain the legacy H40 contract with one explicit registered seed."""
    seed = require_sampling_seed(seed)
    prefix.require(
        task in TASKS and type(image.get("step")) is int and image["step"] >= 0,
        "One registered task and current step are required",
    )
    request = copy.deepcopy(template)
    request["name"] = f"s2a_reset_task{TASKS.index(task)}_t{image['step']}_s{seed}"
    request["prompt"] = task
    request["vision_path"] = str(prefix.REPO / image["path"])
    request["seed"] = seed
    request["extra"]["action128"]["task_title"] = task
    normalized = copy.deepcopy(request)
    normalized["name"] = "s2a_task0_init0_s42_0"
    normalized["prompt"] = normalized["extra"]["action128"]["task_title"] = TASKS[0]
    normalized["seed"] = 42
    repro.validate_request(normalized, {"path": request["vision_path"]})
    extra = request["extra"]["action128"]
    prefix.require(
        extra["selected_roles"] == ["auxiliary_0", "left_wrist"]
        and extra["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
        "The original dual-camera conditioning layout is required",
    )
    return request


def validate_response(response, *, model_request, task, checkpoint, seed):
    """Require the actual model output to identify the registered sampling seed."""
    seed = require_sampling_seed(seed)
    args = response["args"]
    identity = response["outputs"][0]["content"]["action128"]
    expected_checkpoint = (Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model").resolve()
    prefix.require(
        task in TASKS
        and model_request.get("seed") == seed
        and all(key in args and args[key] == value for key, value in model_request.items())
        and args["prompt"] == task
        and type(args["seed"]) is type(identity["seed"]) is int
        and args["seed"] == identity["seed"] == seed
        and args["num_steps"] == 35
        and args["action_chunk_size"] == 40
        and args["fps"] == identity["action_fps"] == 20
        and Path(identity["checkpoint_path"]).resolve() == expected_checkpoint
        and identity["action_units"] == "raw_physical_or_source_units"
        and identity["action_dim_mask"] == [True] * 10 + [False] * 118
        and all(type(item) is bool for item in identity["action_dim_mask"]),
        "Actual WAM response checkpoint, seed, task or mask differs",
    )
