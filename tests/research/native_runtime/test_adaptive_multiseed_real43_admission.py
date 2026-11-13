"""Verify the saved-image seed-43 replay admits its real historical inputs."""

import copy

import pytest

from scripts.simulation.runtime import adaptive_multiseed_real43_admission as admission


@pytest.fixture(scope="module")
def saved():
    return admission.load_inputs()


def test_saved_input_is_bound_without_exposing_labels_or_future_state(saved):
    receipt = admission.input_receipt(saved)
    assert receipt["evidence_level"] == "saved_development_image_model_replay"
    assert receipt["recorded_initial"] == {
        "step": 0,
        "sim_time_s": 0.25000000000000017,
        "state_sha256": admission.FIXED["initial_state_sha256"],
    }
    assert receipt["source_files"] == 1004 and receipt["weight_files"] == 163
    assert all(
        item["elementwise_equal"] and item["different_elements"] == 0
        for item in receipt["seed42_reproduction"].values()
    )
    assert "native_success" not in str(receipt) and "current_source_pose" not in str(receipt)
    registration = admission.model_registration(saved, admission.OUTPUT / "producer/simulation")
    assert registration["initial"] == receipt["recorded_initial"]
    assert registration["seed"] == 43 and registration["max_requests"] == registration["max_periods"] == 1
    assert registration["denoise_steps"] == 35 and registration["model_sha256"] == receipt["model_sha256"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seed", 42),
        ("seed", True),
        ("model_requests", 2),
        ("denoise_calls", 36),
        ("physics_steps", 1),
        ("fresh_cases", 1),
        ("automatic_retry", True),
        ("gpu_uuid", "GPU-unregistered"),
        ("historical_case_sha256", "0" * 64),
        ("initial_sim_time_s", 0.25),
    ],
)
def test_registration_changes_cannot_expand_or_relabel_the_run(key, value):
    config = dict(admission.FIXED, **{key: value})
    with pytest.raises(ValueError, match="exact budget"):
        admission.validate_config(config)


def test_other_capture_root_is_rejected(saved, tmp_path):
    with pytest.raises(ValueError, match="replay image root"):
        admission.model_registration(saved, tmp_path)


def test_changed_recorded_render_snapshot_is_rejected(monkeypatch):
    original = admission.prefix.read

    def changed(ref, *args, **kwargs):
        value = original(ref, *args, **kwargs)
        if ref["sha256"] == admission.FIXED["historical_case_sha256"]:
            value = copy.deepcopy(value)
            value["images"][0]["render_after"]["sim_state_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(admission.prefix, "read", changed)
    with pytest.raises(ValueError, match="saved development image"):
        admission.load_inputs()


def test_changed_seed42_reproduction_is_rejected(monkeypatch):
    monkeypatch.setattr(
        admission.repro,
        "compare_responses",
        lambda *_args: {"action": {"elementwise_equal": False, "different_elements": 1}},
    )
    with pytest.raises(ValueError, match="seed-42 reproduction"):
        admission.load_inputs()
