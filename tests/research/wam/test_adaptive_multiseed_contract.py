"""Keep registered sampling identity consistent across tasks and responses."""

import copy
from types import SimpleNamespace

import pytest

from s2a.research.wam.tracking.runtime.sampling_seed import require_sampling_seed, reset_sampling_seed
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.wam.tracking.preview import adaptive_multiseed_contract as contract
from scripts.wam.tracking.preview import reset_multitask_live_requests as legacy


def template():
    return {
        "name": "s2a_task0_init0_s42_0",
        "model_mode": "wam",
        "prompt": contract.TASKS[0],
        "vision_path": str(prefix.REPO / "outputs/old.png"),
        "domain_name": "franka_panda__parallel_gripper",
        "image_size": 960,
        "action_chunk_size": 40,
        "fps": 20,
        "num_frames": 41,
        "seed": 42,
        "num_steps": 35,
        "guidance": 1.0,
        "shift": 5.0,
        "extra": {
            "action128": {
                "action_domain": "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1",
                "action_dim_mask": [True] * 10 + [False] * 118,
                "action_fps": 20,
                "task_title": contract.TASKS[0],
                "selected_roles": ["auxiliary_0", "left_wrist"],
                "camera_layout": [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                "return_video": False,
            }
        },
    }


@pytest.mark.parametrize("task", contract.TASKS)
@pytest.mark.parametrize("seed", [42, 43])
def test_request_retains_full_contract_for_each_task_and_registered_seed(tmp_path, task, seed):
    original = template()
    before = copy.deepcopy(original)
    image = dict(step=16, path=str((tmp_path / "current.png").relative_to(prefix.REPO)), sha256="a" * 64)
    request = contract.make_task_request(original, image, task, seed)
    assert request["seed"] == seed and request["name"].endswith(f"_s{seed}")
    assert request["prompt"] == request["extra"]["action128"]["task_title"] == task
    assert request["vision_path"] == str(prefix.REPO / image["path"])
    assert request["num_steps"] == 35 and request["action_chunk_size"] == 40 and request["fps"] == 20
    assert original == before
    if seed == 42:
        assert request == legacy.make_task_request(original, image, task)


@pytest.mark.parametrize("seed", [True, False, -1, 0, 44, "43", None])
def test_unregistered_seed_is_rejected_before_rng_changes(seed):
    calls = []
    rng = SimpleNamespace(seed=lambda value: calls.append(value))
    torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(value), cuda=SimpleNamespace(manual_seed_all=calls.append)
    )
    with pytest.raises(ValueError, match="42 or 43"):
        reset_sampling_seed(seed, python_random=rng, numpy_random=rng, torch=torch)
    assert calls == []


@pytest.mark.parametrize("seed", [42, 43])
def test_registered_seed_reaches_all_random_streams(seed):
    calls = []
    python_random = SimpleNamespace(seed=lambda value: calls.append(("python", value)))
    numpy_random = SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value)),
        cuda=SimpleNamespace(manual_seed_all=lambda value: calls.append(("cuda", value))),
    )
    reset_sampling_seed(seed, python_random=python_random, numpy_random=numpy_random, torch=torch)
    assert calls == [(name, seed) for name in ("python", "numpy", "torch", "cuda")]
    assert require_sampling_seed(seed) == seed


def test_registration_accepts_only_explicit_seed_and_original_bounds(tmp_path):
    output = tmp_path / "trial"
    (output / "producer/simulation").mkdir(parents=True)
    registration = dict(
        session="multiseed-test",
        initial=dict(step=107, sim_time_s=5.6, state_sha256="a" * 64),
        image_root=str((output / "producer/simulation").relative_to(prefix.REPO)),
        task=contract.TASKS[0],
        seed=43,
        model_sha256="b" * 64,
        max_requests=2,
        max_periods=8,
        denoise_steps=35,
    )
    assert contract.validate_registration(registration, output).seed == 43
    for key, value in (("seed", 44), ("max_requests", 76), ("max_periods", 494), ("denoise_steps", 34)):
        changed = {**registration, key: value}
        with pytest.raises(ValueError):
            contract.validate_registration(changed, output)


@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("change", [None, "request_seed", "response_seed", "identity_seed", "task", "mask"])
def test_actual_response_must_match_registered_seed_and_fixed_identity(tmp_path, seed, change):
    image = dict(step=16, path=str((tmp_path / "current.png").relative_to(prefix.REPO)), sha256="a" * 64)
    task = contract.TASKS[1]
    request = contract.make_task_request(template(), image, task, seed)
    checkpoint = dict(run=str(tmp_path / "run"), iteration="iter_8400")
    identity = dict(
        seed=seed,
        action_fps=20,
        checkpoint_path=str(tmp_path / "run/checkpoints/iter_8400/model"),
        action_units="raw_physical_or_source_units",
        action_dim_mask=[True] * 10 + [False] * 118,
    )
    response = {"args": copy.deepcopy(request), "outputs": [{"content": {"action128": identity}}]}
    if change == "request_seed":
        request["seed"] = 42 if seed == 43 else 43
    elif change == "response_seed":
        response["args"]["seed"] = 42 if seed == 43 else 43
    elif change == "identity_seed":
        identity["seed"] = 42 if seed == 43 else 43
    elif change == "task":
        response["args"]["prompt"] = contract.TASKS[2]
    elif change == "mask":
        identity["action_dim_mask"][10] = True
    if change is None:
        contract.validate_response(response, model_request=request, task=task, checkpoint=checkpoint, seed=seed)
    else:
        with pytest.raises(ValueError, match="checkpoint, seed, task or mask"):
            contract.validate_response(response, model_request=request, task=task, checkpoint=checkpoint, seed=seed)


def test_camera_and_sampling_contract_cannot_be_smuggled_through_seed_normalization(tmp_path):
    image = dict(step=8, path=str((tmp_path / "current.png").relative_to(prefix.REPO)), sha256="a" * 64)
    for key, value in (("num_steps", 34), ("fps", 10), ("guidance", 2.0)):
        changed = template()
        changed[key] = value
        with pytest.raises(ValueError):
            contract.make_task_request(changed, image, contract.TASKS[0], 43)
    changed = template()
    changed["extra"]["action128"]["selected_roles"].reverse()
    with pytest.raises(ValueError, match="camera"):
        contract.make_task_request(changed, image, contract.TASKS[0], 43)
