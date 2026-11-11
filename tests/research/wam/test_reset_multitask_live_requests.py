"""A multi-window model service sees only each actual current image and task."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from s2a.infrastructure.repository_paths import repository_root
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger
from scripts.wam.tracking.preview import reset_multitask_live_requests as multi
from tests.research.wam.test_reset_live_request import Counter, Overrides, Torch


class Channel:
    def __init__(self, requests):
        self.requests, self.sent = list(requests), []

    def receive(self):
        return self.requests.pop(0)

    def send(self, value):
        self.sent.append(value)


class Service:
    def __init__(self, task, counter, evidence):
        self.task, self.counter, self.evidence = task, counter, evidence
        self.requests = []

    def prepare(self, request, _directory):
        self.requests.append(request)
        return {}, {"ai_caption": [json.dumps(dict(task_title=self.task, command=self.task, task_plan=[]))]}

    def execute(self, _sample, _batch):
        self.counter.calls.extend({"completed": True} for _ in range(35))
        self.evidence.append(dict(stage="initial_noise", inactive_rms_before=0.1, inactive_rms_after=0.0))
        values = [[0.0] * 128 for _ in range(40)]
        return SimpleNamespace(
            model_dump=lambda mode: {
                "status": "success",
                "outputs": [{"content": {"action": values, "action128": {"normalized_action": values}}}],
            }
        )


def setup(tmp_path, task):
    root = repository_root()
    image_root, directory = tmp_path / "captures", tmp_path / "model"
    image_root.mkdir()
    directory.mkdir()
    registration = dict(
        session="multi-test",
        initial=dict(step=8, sim_time_s=0.65, state_sha256="a" * 64),
        image_root=str(image_root.relative_to(root)),
        task=task,
        seed=42,
        model_sha256="b" * 64,
        max_requests=5,
        denoise_steps=35,
    )
    ledger = CausalStreamLedger(**registration)
    requests = []
    for window in range(5):
        data = f"current image {window}".encode()
        name = f"frame{window}.png"
        (image_root / name).write_bytes(data)
        current = ledger.current
        requests.append(
            ledger.issue(
                before_render=current,
                after_render=current,
                image=dict(path=name, sha256=hashlib.sha256(data).hexdigest()),
            )
        )
        from tests.research.native_runtime.test_streaming_continuation import response_for

        ledger.accept(response_for(requests[-1]), after_wait=current)
        for _ in range(8):
            before = ledger.current
            after = dict(step=before["step"] + 1, sim_time_s=before["sim_time_s"] + 0.05, state_sha256="c" * 64)
            ledger.complete_period(before=before, after=after, inner_samples=25)
    counter, evidence, result = Counter(), [], {}
    service = Service(task, counter, evidence)
    return registration, image_root, directory, requests, counter, evidence, result, service


def call(items, requests):
    registration, image_root, directory, _requests, counter, evidence, result, service = items
    channel = Channel(requests)
    multi.serve_live_requests(
        registration=registration,
        channel=channel,
        image_root=image_root,
        directory=directory,
        template={},
        checkpoint={},
        service=service,
        torch=Torch,
        counter=counter,
        overrides=Overrides,
        domain_reader=lambda *_args: SimpleNamespace(is_libero=True),
        mask_evidence=evidence,
        result=result,
        request_builder=lambda _template, image, task: {
            "name": f"synthetic_{image['step']}",
            "vision_path": image["path"],
            "task": task,
        },
        response_validator=lambda *_args, **_kwargs: None,
    )
    return channel


@pytest.mark.parametrize("task", multi.TASKS)
def test_five_windows_each_bind_their_own_image_and_task(tmp_path, task):
    items = setup(tmp_path, task)
    channel = call(items, items[3])
    result, service = items[6], items[7]
    assert result["model_sequence_complete"] is True and result["responses_sent"] == 5
    assert result["denoise_calls_attempted"] == result["denoise_calls_completed"] == 175
    assert [response["step"] for response in channel.sent] == [8, 16, 24, 32, 40]
    assert all(request["task"] == task for request in service.requests)
    assert all(row["reference_sent"] for row in result["responses"])


def test_early_stop_after_two_completed_requests_does_not_fabricate_remaining_work(tmp_path):
    items = setup(tmp_path, multi.TASKS[0])
    channel = call(items, [*items[3][:2], make_stop(session="multi-test", next_step=24)])
    assert len(channel.sent) == items[6]["responses_sent"] == 2
    assert items[6]["model_sequence_complete"] is False and items[6]["model_sequence_stop"]["next_step"] == 24
    assert items[6]["denoise_calls_completed"] == 70


def test_wrong_initial_state_is_rejected_before_model_use(tmp_path):
    items = setup(tmp_path, multi.TASKS[1])
    request = copy.deepcopy(items[3][0])
    request["state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="actual registered endpoint"):
        call(items, [request])
    assert not items[7].requests and items[6]["model_requests_attempted"] == 0


def test_real_task_request_builder_keeps_camera_and_sampling_contract(tmp_path):
    root = repository_root()
    template = {
        "name": "s2a_task0_init0_s42_0",
        "model_mode": "wam",
        "prompt": multi.TASKS[0],
        "vision_path": str(tmp_path / "old.png"),
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
                "task_title": multi.TASKS[0],
                "selected_roles": ["auxiliary_0", "left_wrist"],
                "camera_layout": [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                "return_video": False,
            }
        },
    }
    before = copy.deepcopy(template)
    image = dict(step=16, path=str((tmp_path / "current.png").relative_to(root)), sha256="a" * 64)
    for task in multi.TASKS:
        request = multi.make_task_request(template, image, task)
        assert request["prompt"] == request["extra"]["action128"]["task_title"] == task
        assert request["vision_path"] == str(root / image["path"])
        assert request["seed"] == 42 and request["num_steps"] == 35
    assert template == before
    bad = copy.deepcopy(template)
    bad["extra"]["action128"]["selected_roles"].reverse()
    with pytest.raises(ValueError, match="camera"):
        multi.make_task_request(bad, image, multi.TASKS[1])
    bad = copy.deepcopy(template)
    bad["seed"] = 43
    with pytest.raises(ValueError, match="fixed development"):
        multi.make_task_request(bad, image, multi.TASKS[1])
