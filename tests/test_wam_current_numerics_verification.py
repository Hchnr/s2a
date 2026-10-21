"""Independent024 equations and reject boundaries, without torch or real forwards."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_current_numerics.py"
    spec = importlib.util.spec_from_file_location("current_numerics_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bfloat16_round_to_nearest_even_positive_negative_ties_and_signed_zero(checker):
    values = np.asarray([1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8, 0.0, -0.0], dtype=np.float32)
    result = checker.cast_model_precision(values, "torch.bfloat16")
    expected = np.asarray([1, 1 + 2**-6, -1, -1 - 2**-6, 0.0, -0.0], dtype=np.float32)
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    assert not np.shares_memory(result, values)
    with pytest.raises(ValueError, match="unsupported"):
        checker.cast_model_precision(values, "float8")


def test_flow_conditioning_and_invalid_action_zero_differ_from_epsilon_zero(checker):
    x0 = np.asarray([[0.25, 0.0], [0.5, 0.0]], dtype=np.float32)
    noise = np.asarray([[0.75, 9.0], [-0.5, -8.0]], dtype=np.float32)
    valid = np.asarray([[True, False], [True, False]])
    result = checker.flow_equations(x0, noise, np.zeros((2, 1)), valid, 0.5, modality="action", model_dtype="float32")
    np.testing.assert_array_equal(result["xt"], [[0.5, 0], [0, 0]])
    np.testing.assert_array_equal(result["target"], [[0.5, 0], [-1, 0]])
    assert np.any(noise[:, 1] != 0)
    vision = checker.flow_equations(
        x0.T[:, :, None, None],
        noise.T[:, :, None, None],
        np.asarray([[[1]], [[0]]]),
        None,
        0.5,
        modality="vision",
        model_dtype="float32",
    )
    np.testing.assert_array_equal(vision["xt"][:, 0, 0, 0], x0[0])
    # Conditioned-frame targets remain epsilon-x0; only the loss masks them.
    np.testing.assert_array_equal(vision["target"][:, 0, 0, 0], noise[0] - x0[0])


def test_explicit_valid_loss_broadcasts_channels_uses_active_denominator_and_uniform_weights(checker):
    prediction = np.full((2, 3, 1, 2), 2, dtype=np.float32)
    target = np.zeros_like(prediction)
    condition = np.asarray([[[1]], [[0]], [[0]]], dtype=np.float32)
    valid = np.asarray([[[True, True]], [[True, False]], [[True, True]]])
    result = checker.explicit_mask_loss(
        prediction, target, condition, valid, [[200]], modality="vision", time_weight_method="uniform"
    )
    assert result["denominator"] == result["active_elements"] == 6
    assert result["loss"] == result["per_instance_unweighted"] == 4
    np.testing.assert_array_equal(result["time_weight"], [1])
    empty = checker.explicit_mask_loss(
        prediction, target, condition, np.zeros_like(valid), [[200]], modality="vision", time_weight_method="uniform"
    )
    assert empty["denominator"] == 1 and empty["active_elements"] == 0 and empty["loss"] == 0
    with pytest.raises(ValueError, match="unregistered time"):
        checker.explicit_mask_loss(
            prediction, target, condition, valid, [[200]], modality="vision", time_weight_method="snr"
        )


def test_exact_contract_rejects_tiny_drift_and_grid_rejects_missing_duplicate_or_wrong_call(checker):
    checker.exact(np.asarray([0.1], dtype=np.float32), np.asarray([0.1], dtype=np.float32), "equal")
    with pytest.raises(ValueError, match="exact mismatch"):
        checker.exact(np.asarray([0.1]), np.asarray([0.1 + 1e-16]), "drift")
    records = [
        dict(task_index=task, sigma=sigma, branch=branch)
        for task in range(3)
        for sigma in (0.2, 0.5, 0.8)
        for branch in ("A", "B", "C")
    ]
    checker.registered_call_grid(records)
    for wrong in (records[:-1], records + [records[0]], records[:-1] + [dict(task_index=2, sigma=0.8, branch="B")]):
        with pytest.raises(ValueError, match="27-call grid"):
            checker.registered_call_grid(wrong)


def test_real_model_singleton_vision_batch_preserves_time_axis_for_flow_and_loss(checker):
    x0 = np.zeros((1, 2, 3, 1, 2), np.float32)
    noise = np.full_like(x0, 2)
    condition = np.asarray([[[1]], [[0]], [[0]]], np.float32)
    flow = checker.flow_equations(x0, noise, condition, None, 0.5, modality="vision", model_dtype="bfloat16")
    np.testing.assert_array_equal(flow["xt"][:, :, 0], 0)
    np.testing.assert_array_equal(flow["xt"][:, :, 1:], 1)
    result = checker.explicit_mask_loss(
        flow["target"],
        np.zeros_like(x0),
        condition,
        np.ones((3, 1, 2)),
        [[500]],
        modality="vision",
        time_weight_method="uniform",
    )
    assert result["denominator"] == 8 and result["loss"] == 4


def test_independent_normalization_respects_transform_mask_clip_and_invalid_slots(checker):
    physical = np.zeros((2, 128), np.float64)
    physical[:, :3] = [[0.2, 0.5, -3], [2, -0.5, 0.6]]
    transform = dict(
        normalization_mask=[True, False] + [True] * 126, scale=[2] * 128, bias=[0.1] * 128, clip_min=-1, clip_max=1
    )
    valid = np.broadcast_to([True] * 10 + [False] * 118, physical.shape).copy()
    valid[1, 0] = False
    result = checker.normalized_actions(physical, transform, valid)
    np.testing.assert_array_equal(result[:, :3], np.asarray([[0.5, 0.5, -1], [0, -0.5, 1]], np.float32))
    assert not result[:, 10:].any() and result.dtype == np.float32
    assert not np.shares_memory(result, physical)


def loss_fixture():
    vision = np.full((1, 2, 3, 1, 1), 2, np.float32)
    action = np.zeros((2, 128), np.float32)
    action[:, :10] = 3
    target_action = np.zeros_like(action)
    target_action[:, :10] = 1
    condition_v = np.asarray([[[1]], [[0]], [[0]]], np.float32)
    condition_a = np.zeros((2, 1), np.float32)
    packed = dict(
        vision=dict(condition_mask=[condition_v], loss_valid_mask=[np.ones((3, 1, 1), bool)]),
        action=dict(
            condition_mask=[condition_a], loss_valid_mask=[np.broadcast_to([True] * 10 + [False] * 118, action.shape)]
        ),
    )
    arrays = dict(
        pred_vision=vision,
        pred_action=action,
        timestep=np.asarray([[200]], np.float32),
        time_weight_vision=np.ones(1),
        time_weight_action=np.ones(1),
    )
    noised = dict(vt_target_vision=[np.zeros_like(vision)], vt_target_action=[target_action])
    metadata = dict(
        losses=dict(
            total=10.0,
            cross_entropy_loss_text=0.0,
            flow_matching_loss_vision=4.0,
            flow_matching_loss_vision_per_instance=[4.0],
            flow_matching_loss_action=4.0,
            flow_matching_loss_sound=0.0,
        ),
        loss_config=dict(
            time_weight_vision="uniform",
            time_weight_action="uniform",
            sample_level_loss_averaging=False,
            predict_text_tokens=False,
            sound_gen=False,
            vision_gen=True,
            action_gen=True,
            vision_weight=0.5,
            action_weight=2.0,
        ),
    )
    return metadata, arrays, noised, packed


def test_full_objective_independent_active_denominators_and_weights_reject_all_slot_or_omitted_loss(checker):
    metadata, arrays, noised, packed = loss_fixture()
    comparison = dict(loss_rtol=1e-5, loss_atol=1e-6)
    result = checker.verify_recorded_loss(metadata, arrays, noised, packed, comparison)
    assert result["total"] == 10 and result["action"]["denominator"] == 20 and result["vision"]["denominator"] == 4
    metadata["losses"]["flow_matching_loss_action"] = 4 * 10 / 128
    with pytest.raises(ValueError, match="action loss"):
        checker.verify_recorded_loss(metadata, arrays, noised, packed, comparison)
    metadata["losses"]["flow_matching_loss_action"] = 4
    metadata["losses"]["total"] = 8
    with pytest.raises(ValueError, match="weighted total"):
        checker.verify_recorded_loss(metadata, arrays, noised, packed, comparison)
    metadata["losses"]["total"] = 10
    metadata["losses"]["aux_loss_gen"] = 0.001
    with pytest.raises(ValueError, match="unaccounted"):
        checker.verify_recorded_loss(metadata, arrays, noised, packed, comparison)


def test_forward_metadata_compares_positions_exactly_and_only_excludes_supervision(checker):
    packed = dict(
        position_ids=np.asarray([[0, 1]]),
        label_ids=None,
        vision=dict(tokens=[np.ones((1, 1, 2, 1, 1))], loss_valid_mask=None),
        action=dict(tokens=[np.ones((2, 128))], loss_valid_mask=None),
    )
    other = dict(packed, vision=dict(packed["vision"], loss_valid_mask=[np.ones((2, 1, 1))]))
    checker.same_tree(checker.forward_metadata(packed), checker.forward_metadata(other), "forward")
    other["position_ids"] = np.asarray([[0, 2]])
    with pytest.raises(ValueError, match="position_ids"):
        checker.same_tree(checker.forward_metadata(packed), checker.forward_metadata(other), "forward")


def test_hash_bound_bundle_rejects_stale_arrays_false_index_and_changed_input(checker, tmp_path):
    archive = tmp_path / "recorded.npz"
    np.savez(archive, actual=np.asarray([1, 2], np.float32))
    metadata = dict(
        arrays=dict(path=str(archive), sha256=checker.digest(archive)),
        array_index=dict(actual=dict(shape=[2], dtype="float32")),
    )
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(metadata))
    reference = dict(path=str(path), sha256=checker.digest(path))
    evidence = checker.Evidence()
    _, arrays = checker.load_bundle(reference, evidence)
    np.testing.assert_array_equal(arrays["actual"], [1, 2])
    np.savez(archive, actual=np.asarray([1, 3], np.float32))
    with pytest.raises(ValueError, match="source SHA"):
        evidence.recheck()
    with pytest.raises(ValueError, match="source SHA"):
        checker.load_bundle(reference, checker.Evidence())
    metadata["arrays"]["sha256"] = checker.digest(archive)
    metadata["array_index"]["actual"]["shape"] = [1, 2]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="bundle index differs"):
        checker.load_bundle(dict(path=str(path), sha256=checker.digest(path)), checker.Evidence())


def test_partial_actual_result_rejected_before_any_full_success_claim(checker, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version="s2a.current-numerics-result.v1",
                complete=False,
                passed=False,
                denoise_calls=[dict(label="0:0.2:A", completed=True)],
            )
        )
    )
    with pytest.raises(ValueError, match="partial/rejected"):
        checker.verify_result({}, path, checker.Evidence())
