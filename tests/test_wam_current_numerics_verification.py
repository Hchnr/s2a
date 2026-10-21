"""Independent024 equations and reject boundaries, without torch or real forwards."""

import importlib.util
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
