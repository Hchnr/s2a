"""Exercise all900 causal development requests and irreversible case binding."""

import copy
import importlib

import pytest

from s2a.wam_readout import LIBERO_DOMAIN


@pytest.fixture
def server():
    return importlib.import_module("s2a.terminal_development_policy")


def prepared(root):
    templates = []
    for task in range(3):
        request = dict(
            name="registered",
            model_mode="wam",
            prompt=f"task {task}",
            vision_path="/old/current.png",
            domain_name="franka_panda__parallel_gripper",
            image_size=960,
            action_chunk_size=40,
            fps=20,
            num_frames=41,
            seed=42,
            num_steps=35,
            guidance=1.0,
            shift=5.0,
            extra=dict(
                action128=dict(
                    action_domain=LIBERO_DOMAIN,
                    action_dim_mask=[True] * 10 + [False] * 118,
                    action_fps=20,
                    task_title=f"task {task}",
                    selected_roles=["auxiliary_0", "left_wrist"],
                    camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                    return_video=False,
                )
            ),
        )
        templates.append(dict(request=request))
    return dict(
        root=str(root),
        zero_cases=templates,
        development_tasks=[dict(task_index=i, task=f"task {i}") for i in range(3)],
    )


def current_request(server, policy, *, pixels=b"identical current canvas"):
    case = policy.current
    directory = server.case_directory(policy.root, case)
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / f"input_{policy.case_requests:04d}.png"
    image.write_bytes(pixels)
    return dict(
        copy.deepcopy(policy.templates[policy.tasks[case["task_index"]]]),
        name=f"s2a_{directory.name}_{policy.case_requests}",
        seed=case["sampling_seed"],
        vision_path=str(image),
    )


def complete(policy, request):
    pending = policy.reserve(request)
    policy.complete(dict(pending, complete=True, initial_ref=dict(path="/synthetic", sha256="a" * 64)))


def test_complete_fixed900_policy_has_no_variant_selection_or_early_finish(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    assert len(server.case_order()) == 12
    with pytest.raises(ValueError, match="complete"):
        policy.finish(dict(completed_cases=12, completed_requests=900))
    for case in server.case_order():
        policy.bind(case)
        for _ in range(75):
            complete(policy, current_request(server, policy))
        with pytest.raises(ValueError, match="budget"):
            policy.reserve(current_request(server, policy))
    assert [r["global_request_index"] for r in policy.requests] == list(range(900))
    assert [r["variant"] for r in policy.requests] == [c["variant"] for c in server.case_order() for _ in range(75)]
    policy.finish(dict(completed_cases=12, completed_requests=900))
    assert policy.finish_requested is policy.stopped is True and not policy.poisoned


@pytest.mark.parametrize("fault", ["skip_case", "change_variant", "early_next_case", "duplicate_bind", "wrong_count"])
def test_binding_cannot_skip_mix_or_truncate_cases(tmp_path, server, fault):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    cases = server.case_order()
    if fault in ("early_next_case", "duplicate_bind"):
        policy.bind(cases[0])
        candidate = cases[1] if fault == "early_next_case" else cases[0]
    else:
        candidate = copy.deepcopy(cases[1] if fault == "skip_case" else cases[0])
        if fault == "change_variant":
            candidate["variant"] = "terminal058"
        elif fault == "wrong_count":
            candidate["expected_requests"] = 74
    with pytest.raises(ValueError):
        policy.bind(candidate)


@pytest.mark.parametrize(
    "fault", ["future", "nested_future", "task", "seed", "name", "outside_image", "long_sampler", "duplicate_pending"]
)
def test_online_request_is_exact_current_image_and_task(tmp_path, server, fault):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    policy.bind(server.case_order()[0])
    request = current_request(server, policy)
    if fault == "future":
        request["future_state"] = [1]
    elif fault == "nested_future":
        request["extra"]["action128"]["future_action"] = [1]
    elif fault == "task":
        request["prompt"] = "another task"
    elif fault == "seed":
        request["seed"] = 43
    elif fault == "name":
        request["name"] = "wrong request number"
    elif fault == "outside_image":
        request["vision_path"] = str(tmp_path / "other.png")
    elif fault == "long_sampler":
        request["num_steps"] = 70
    else:
        policy.reserve(request)
    with pytest.raises(ValueError):
        policy.reserve(request)


def test_first_raw_w2_current_pixels_must_pair_even_with_valid_request_paths(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    cases = server.case_order()
    policy.bind(cases[0])
    for _ in range(75):
        complete(policy, current_request(server, policy))
    policy.bind(cases[1])
    with pytest.raises(ValueError, match="PNG"):
        policy.reserve(current_request(server, policy, pixels=b"changed first current image"))


def test_abort_is_explicit_and_preserves_case_identity(tmp_path, server):
    policy = server.DevelopmentPolicy(prepared(tmp_path))
    policy.bind(server.case_order()[0])
    with pytest.raises(ValueError):
        policy.abort(dict(case_id="other", reason="worker failed"))
    body = dict(case_id=policy.current["case_id"], reason="owned CPU worker exited7")
    assert policy.abort(body) == body and policy.poisoned and policy.stopped and not policy.finish_requested
