"""025's real preparation risks: whole sources, exact clocks and prior admission."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparer():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_wam_readout.py"
    spec = importlib.util.spec_from_file_location("readout_preparation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uniform_clock_covers_full_h40_without_moving_an_invalid_anchor(preparer):
    assert preparer.uniform_anchors(89) == [0, 16, 32, 48]
    assert preparer.uniform_anchors(44) == [0, 1, 2, 3]
    for length in (43, 41, 89.0, True):
        with pytest.raises(ValueError):
            preparer.uniform_anchors(length)
    with pytest.raises(ValueError, match="H40"):
        preparer.uniform_anchors(89, horizon=16)


def test_preparation_rejects_gpu_or_missing_environment_before_reading_inputs(preparer, monkeypatch):
    monkeypatch.chdir(preparer.REPO)
    for cuda, ld in (("0", ""), ("6", ""), ("", "/tmp"), (None, "")):
        if cuda is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
        monkeypatch.setenv("LD_LIBRARY_PATH", ld)
        with pytest.raises(ValueError, match="explicitly empty CUDA/LD"):
            preparer.prepare()


def sample(episode, split, source=None):
    return dict(
        episode=episode,
        source_episode_id=source or f"source:{episode}",
        task="task",
        task_index=0,
        arrays_path=f"episode{episode}.npz",
        arrays_sha256=f"digest{episode}",
        split=split,
    )


def test_whole_source_splits_and_phase_aliases_cannot_leak(preparer):
    config = dict(train_episodes=[1], validation_episodes=[2])
    a, b = sample(1, "diagnostic"), sample(2, "study_heldout")
    aliases = [dict(a, phase="approach"), dict(a, phase="closing"), b]
    result = preparer.split_episodes(config, aliases)
    assert [(x["sample"]["episode"], x["split"]) for x in result] == [(1, "train"), (2, "validation")]
    with pytest.raises(ValueError, match="aliases disagree"):
        preparer.split_episodes(config, [a, dict(a, arrays_sha256="changed"), b])
    with pytest.raises(ValueError, match="whole-source episode leaks"):
        preparer.split_episodes(config, [a, dict(b, source_episode_id=a["source_episode_id"])])
    with pytest.raises(ValueError, match="overlap"):
        preparer.split_episodes(dict(train_episodes=[1], validation_episodes=[1]), [a])
    with pytest.raises(ValueError, match="split changed"):
        preparer.split_episodes(config, [a, dict(b, split="diagnostic")])


def arrays(module):
    n = 44
    return dict(
        states=np.zeros((n, 128)),
        state_mask=np.broadcast_to(module.ACTIVE, (n, 128)).copy(),
        actions=np.zeros((n, 128)),
        action_mask=np.broadcast_to(module.ACTIVE, (n, 128)).copy(),
        action_valid=np.ones(n, dtype=bool),
        auxiliary_0=np.broadcast_to(np.zeros((1, 1, 1, 3), np.uint8), (n, 256, 256, 3)),
        left_wrist=np.broadcast_to(np.zeros((1, 1, 1, 3), np.uint8), (n, 256, 256, 3)),
    )


def test_source_window_uses_next40_valid_transitions_and_exact41_cameras(preparer):
    data = arrays(preparer)
    data["action_valid"][-1] = False
    observed = preparer.validate_window(data, 3)
    assert observed["valid_future_steps"] == 40
    assert len(observed["measured_state128"]) == len(observed["state_mask128"]) == 128
    with pytest.raises(ValueError, match="complete41"):
        preparer.validate_window(data, 4)
    data["action_valid"][42] = False
    with pytest.raises(ValueError, match="invalid future action"):
        preparer.validate_window(data, 3)
    data["action_valid"][42] = True
    data["left_wrist"] = data["left_wrist"][:-1]
    with pytest.raises(ValueError, match="camera41"):
        preparer.validate_window(data, 3)


def test_state_and_action_masks_are_independent_and_missing_slots_reject(preparer):
    data = arrays(preparer)
    data["state_mask"][3, 9] = False
    with pytest.raises(ValueError, match="state mask"):
        preparer.validate_window(data, 3)
    data["state_mask"][3, 9] = True
    data["action_mask"][4, 10] = True
    with pytest.raises(ValueError, match="action mask"):
        preparer.validate_window(data, 3)
    data["action_mask"][4, 10] = False
    data["actions"][4, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite future action"):
        preparer.validate_window(data, 3)


def test_normalization_and_shared_projection_ids_are_not_interchangeable(preparer):
    config = dict(
        hidden_size=1280,
        domain=dict(name=preparer.DOMAIN, normalization_id=31, projection_id=0, active_slots=10),
    )
    names = [f"other{i}" for i in range(67)]
    names[31] = preparer.DOMAIN
    spec = dict(registry=dict(domains=names, domain_projection_ids=[0] * 67))
    checked = preparer.validate_domain(config, spec)
    assert checked["normalization_id"] == 31 and checked["projection_id"] == 0
    assert checked["normalization_mask_is_not_active_mask"] is True
    wrong = copy.deepcopy(spec)
    wrong["registry"]["domains"][0], wrong["registry"]["domains"][31] = names[31], names[0]
    with pytest.raises(ValueError, match="normalization domain"):
        preparer.validate_domain(config, wrong)
    wrong = copy.deepcopy(spec)
    wrong["registry"]["domain_projection_ids"][31] = 8
    with pytest.raises(ValueError, match="sharedslot0"):
        preparer.validate_domain(config, wrong)


def test_manifest_preserves_complete_episode_train_split_and_paired_sampling(preparer):
    anchors = [
        dict(name=f"ep{ep}_a{i}", split="train" if ep < 6 else "validation", request=dict(seed=42))
        for ep in range(12)
        for i in range(4)
    ]
    teacher, sampling = preparer.build_call_manifests(anchors, dict(sigma=[0.2, 0.5, 0.8], sampling_seeds=[42, 43]))
    assert len(teacher) == 144 and len(sampling) == 96
    assert sum(x["split"] == "validation" for x in teacher) == 72
    assert {x["anchor_name"] for x in sampling} == {x["name"] for x in anchors if x["split"] == "validation"}
    for raw, adapted in zip(sampling[::2], sampling[1::2], strict=True):
        assert raw["variant"] == "raw" and adapted["variant"] == "adapt"
        assert raw["pair_id"] == adapted["pair_id"]
        assert raw["request"] == adapted["request"]
        assert raw["request"] is not adapted["request"]
    with pytest.raises(ValueError, match="manifest differs"):
        preparer.build_call_manifests(anchors[:-1], dict(sigma=[0.2, 0.5, 0.8], sampling_seeds=[42, 43]))


def test_historical_exclusion_keeps_suspected_rotation_and_unsynced_status(preparer):
    s = sample(
        850,
        "diagnostic",
        json.dumps(dict(source_id="/share/mixed/projset/eai_pwm/datasets/action_processed_r1/libero/v3_eef_droid:384")),
    )
    record = dict(
        r1_episode=384, task_index=0, demo="demo_27", initial_sync_certified=False, basis="suspected; rotation unproven"
    )
    fresh = dict(offline_source_exclusion_records=[record], excluded_demos_by_task=[["demo_27"]])
    result = preparer.source_exclusion(s, fresh)
    assert result["basis"] == record["basis"] and result["initial_sync_certified"] is False
    fresh["excluded_demos_by_task"][0] = []
    with pytest.raises(ValueError, match="was not excluded"):
        preparer.source_exclusion(s, fresh)


def test_prior_admission_requires_complete_actual_calls_independent_result_and_exited_births(
    preparer, tmp_path, monkeypatch
):
    target = tmp_path / "current/run/result.json"
    target.parent.mkdir(parents=True)
    verification_path = tmp_path / "verified.json"
    orchestration = target.parent.parent / "orchestration"
    (orchestration / "model_ownership").mkdir(parents=True)
    owner, model = dict(pid=123, start_ticks=100), dict(pid=124, start_ticks=101)
    external = dict(head="external", tracked_source_clean=True)

    def write(path, value):
        path.write_text(json.dumps(value))

    result = dict(
        complete=True,
        passed=True,
        physical_gpu=6,
        sampling_calls=0,
        physics_steps=0,
        denoise_calls=[
            dict(label=f"{t}:{s}:{b}", completed=True)
            for t in range(3)
            for s in (0.2, 0.5, 0.8)
            for b in ("A", "B", "C")
        ],
    )
    write(target, result)
    verification = dict(
        passed=True,
        complete=True,
        prepare_only=False,
        calls_verified=27,
        independent_losses_verified=18,
        input_sha256={str(target.resolve()): preparer.sha256(target)},
    )
    write(verification_path, verification)
    write(
        orchestration / "launch.json",
        dict(owner=owner, model=model, external_cosmos=external, environment=dict(CUDA_VISIBLE_DEVICES="6")),
    )
    write(orchestration / "exit.json", dict(actual_wait=True, automatic_retry=False, returncode=0))
    write(
        orchestration / "model_ownership/release.json",
        dict(**model, returncode=0, reaped=True, original_error=None, only_owned_child_signalled=True),
    )
    write(orchestration / "external_cosmos_after.json", external)
    config = dict(required_numerics_result=str(target), required_numerics_verification=str(verification_path))
    monkeypatch.setattr(preparer, "absent_owned_birth", lambda _: True)
    assert preparer.numerics_admission(config, preparer.Evidence())["owner_and_model_births_terminal"] is True
    monkeypatch.setattr(preparer, "absent_owned_birth", lambda _: False)
    with pytest.raises(ValueError, match="still running"):
        preparer.numerics_admission(config, preparer.Evidence())
    monkeypatch.setattr(preparer, "absent_owned_birth", lambda _: True)
    verification["prepare_only"] = True
    write(verification_path, verification)
    with pytest.raises(ValueError, match="independent27"):
        preparer.numerics_admission(config, preparer.Evidence())
    verification["prepare_only"] = False
    write(verification_path, verification)
    result["denoise_calls"].pop()
    write(target, result)
    with pytest.raises(ValueError, match="not bound"):
        preparer.numerics_admission(config, preparer.Evidence())
    verification["input_sha256"][str(target.resolve())] = preparer.sha256(target)
    write(verification_path, verification)
    with pytest.raises(ValueError, match="budget/order"):
        preparer.numerics_admission(config, preparer.Evidence())
