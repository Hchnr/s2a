"""Check offline replay identity and prevent accidental action dispatch."""

import copy
import shutil

import pytest

from scripts.simulation.runtime import adaptive_multiseed_real43_admission as admission
from scripts.simulation.runtime import adaptive_multiseed_real43_replay as replay
from scripts.wam.tracking.preview import gpu_resume_repro_20261111 as repro
from tests.research.native_runtime.test_streaming_continuation import response_for


@pytest.fixture(scope="module")
def saved():
    return admission.load_inputs()


def registration(saved, directory):
    result = admission.model_registration(saved, admission.OUTPUT / "producer/simulation")
    result["image_root"] = str(directory.relative_to(admission.ROOT))
    return result


class Channel:
    def __init__(self, transform=None):
        self.sent = []
        self.transform = transform or (lambda response: response)

    def send(self, request):
        self.sent.append(copy.deepcopy(request))

    def receive(self):
        return self.transform(response_for(self.sent[-1]))


def copied_image(saved, tmp_path):
    shutil.copyfile(saved["refs"]["historical_image"]["path"], tmp_path / "current.png")
    return tmp_path


def test_saved_frame_exchange_uses_recorded_identity_and_no_actions(saved, tmp_path):
    directory = copied_image(saved, tmp_path)
    arrays = repro.response_arrays(saved["previous_response"])

    def respond(response):
        response["action"] = arrays["action"].tolist()
        response["normalized_action"] = arrays["normalized_action"].tolist()
        return response

    channel = Channel(respond)
    result = replay.exchange_saved_frame(
        saved, registration=registration(saved, directory), image_root=directory, channel=channel
    )
    assert len(channel.sent) == result["accepted_requests"] == 1
    assert result["request"]["seed"] == result["response"]["seed"] == 43
    assert result["request"]["state_sha256"] == saved["initial"]["state_sha256"]
    assert result["snapshot_scope"] == "historical_recorded_endpoint_not_live"
    assert (
        result["completed_periods"]
        == result["physics_steps"]
        == result["native_models"]
        == result["action_dispatches"]
        == 0
    )
    assert all(row["elementwise_equal"] for row in result["compared_to_saved_seed42"].values())
    assert "native_success" not in channel.sent[0] and "current_source_pose" not in channel.sent[0]


@pytest.mark.parametrize("change", ["seed", "image", "denoise", "mask", "inactive", "nonfinite"])
def test_bad_response_is_rejected_before_any_action(saved, tmp_path, change):
    directory = copied_image(saved, tmp_path)

    def tamper(response):
        if change == "seed":
            response["seed"] = 42
        elif change == "image":
            response["image_sha256"] = "0" * 64
        elif change == "denoise":
            response["denoise_calls"] = 34
        elif change == "mask":
            response["action_dim_mask"][10] = True
        elif change == "inactive":
            response["action"][0][10] = 1.0
        else:
            response["action"][0][0] = float("nan")
        return response

    channel = Channel(tamper)
    with pytest.raises(ValueError):
        replay.exchange_saved_frame(
            saved, registration=registration(saved, directory), image_root=directory, channel=channel
        )
    assert len(channel.sent) == 1


def test_changed_image_is_rejected_before_model_request(saved, tmp_path):
    directory = copied_image(saved, tmp_path)
    (directory / "current.png").write_bytes(b"not the recorded image")
    channel = Channel()
    with pytest.raises(ValueError, match="historical image"):
        replay.exchange_saved_frame(
            saved, registration=registration(saved, directory), image_root=directory, channel=channel
        )
    assert channel.sent == []


def test_changed_image_during_exchange_is_rejected(saved, tmp_path):
    directory = copied_image(saved, tmp_path)

    def change(response):
        (directory / "current.png").write_bytes(b"changed during inference")
        return response

    with pytest.raises(ValueError, match="changed during"):
        replay.exchange_saved_frame(
            saved, registration=registration(saved, directory), image_root=directory, channel=Channel(change)
        )


def test_wrong_registered_seed_is_rejected_before_model_request(saved, tmp_path):
    directory = copied_image(saved, tmp_path)
    reg = registration(saved, directory)
    reg["seed"] = 42
    channel = Channel()
    with pytest.raises(ValueError, match="registered saved-frame"):
        replay.exchange_saved_frame(saved, registration=reg, image_root=directory, channel=channel)
    assert channel.sent == []
