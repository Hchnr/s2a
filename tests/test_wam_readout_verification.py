"""025 independent NumPy math and rejection boundaries; no Torch/model/data."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_readout.py"
    spec = importlib.util.spec_from_file_location("readout_independent_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_fp64_bfloat16_rounding_avoids_double_rounding_and_preserves_signed_zero(checker):
    values = np.asarray([1 + 2**-8, 1 + 2**-8 + 2**-30, 1 + 2**-8 - 2**-30, -1 - 2**-8 - 2**-30, 0.0, -0.0])
    actual = checker.quantize_bfloat16(values)
    expected = np.asarray([1, 1 + 2**-7, 1, -1 - 2**-7, 0.0, -0.0], np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    tiny = np.asarray([2**-134, 3 * 2**-134, -(2**-134), -3 * 2**-134])
    np.testing.assert_array_equal(
        checker.quantize_bfloat16(tiny).view(np.uint32),
        np.asarray([0, 2**-132, -0.0, -(2**-132)], np.float32).view(np.uint32),
    )
    with pytest.raises(ValueError, match="overflow"):
        checker.quantize_bfloat16([float(2**128)])


def test_independent_augmented_solve_matches_dual_and_regularizes_bias(checker):
    rng = np.random.default_rng(42)
    hidden, residual = rng.normal(size=(9, 4)), rng.normal(size=(9, 10))
    weights = np.arange(1, 10, dtype=np.float64)
    weights /= weights.sum()
    models = checker.augmented_fit(hidden, residual, weights)
    design = np.column_stack((hidden / 2, np.ones(9)))
    x, y = np.sqrt(weights)[:, None] * design, np.sqrt(weights)[:, None] * residual
    expected = x.T @ np.linalg.solve(x @ x.T + 0.1 * np.eye(9), y)
    np.testing.assert_allclose(models["adapt"], expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(models["bias"][:-1], 0)
    np.testing.assert_allclose(models["bias"][-1], (weights @ residual) / 1.1)
    with pytest.raises(ValueError, match="positive and sum"):
        checker.augmented_fit(hidden, residual, weights * 2)


def test_readout_reconstruction_quantizes_once_and_preserves_all_inactive_slots(checker):
    hidden = np.zeros((2, 3), np.float32)
    base = np.ones((2, 128), np.float32)
    base[:, 10:] = np.arange(118)
    before = base.copy()
    weights = np.zeros((4, 10))
    weights[-1] = 2**-8 + 2**-30
    adapted = checker.apply_quantized(hidden, base, weights)
    np.testing.assert_array_equal(adapted[:, :10], np.full((2, 10), 1 + 2**-7))
    np.testing.assert_array_equal(adapted[:, 10:], before[:, 10:])
    np.testing.assert_array_equal(base, before)
    np.testing.assert_array_equal(checker.apply_quantized(hidden, base, np.zeros_like(weights)), base)


def test_hierarchy_equalizes_episodes_then_anchors_then_sigmas_then_endpoints_and_refuses_validation(checker):
    rows = [dict(split="train", episode=1, anchor_name="a", sigma=0.2, endpoint=i) for i in range(2)]
    rows += [dict(split="train", episode=2, anchor_name="b", sigma=s, endpoint=0) for s in (0.2, 0.5)]
    rows += [dict(split="train", episode=2, anchor_name="c", sigma=0.2, endpoint=0)]
    np.testing.assert_array_equal(checker.hierarchical_weights(rows), [0.25, 0.25, 0.125, 0.125, 0.25])
    with pytest.raises(ValueError, match="validation"):
        checker.hierarchical_weights(rows + [dict(rows[0], split="validation")])
    with pytest.raises(ValueError, match="duplicate endpoint"):
        checker.hierarchical_weights(rows + [rows[0]])


def test_teacher_gate_has_zero_post_result_slack_and_requires_constant_bias_and_each_sigma(checker):
    def metrics(adapt=0.95, bias=0.951, sigma=1.05):
        return dict(
            validation=dict(
                pooled=dict(mse=dict(raw=1.0, adapt=adapt, bias=bias)),
                by_sigma={str(s): dict(mse=dict(raw=1.0, adapt=sigma, bias=2.0)) for s in (0.2, 0.5, 0.8)},
            )
        )

    assert checker.independent_teacher_gate(metrics())["passed"]
    assert not checker.independent_teacher_gate(metrics(adapt=np.nextafter(0.95, 1.0)))["passed"]
    assert not checker.independent_teacher_gate(metrics(bias=0.95))["passed"]
    assert not checker.independent_teacher_gate(metrics(sigma=np.nextafter(1.05, 2.0)))["passed"]


def feature_fixture(checker):
    x0 = np.zeros((40, 128), np.float32)
    x0[:, :10] = 0.25
    epsilon = np.full_like(x0, 0.75)
    target = epsilon - x0
    target[:, 10:] = 0
    xt = checker.quantize_bfloat16(epsilon * np.float32(0.2) + x0 * np.float32(0.8))
    xt[:, 10:] = 0
    arrays = dict(
        hidden=np.zeros((40, 1280), np.float32),
        head_base=np.ones((40, 128), np.float32),
        base=np.broadcast_to([1.0] * 10 + [0.0] * 118, (40, 128)).astype(np.float32),
        target=target,
        xt=xt,
        x0_normalized=x0,
        epsilon=epsilon,
        valid=np.broadcast_to([True] * 10 + [False] * 118, (40, 128)),
        projection_ids=np.zeros(40, np.int64),
        condition_action=np.zeros((40, 1), np.float32),
        sigma_action=np.full((40, 1), np.float32(0.2)),
    )
    metadata = dict(original_dtypes=dict(hidden="torch.bfloat16", base="torch.bfloat16"))
    return arrays, metadata, x0


def test_actual_hidden_head_mask_label_and_flow_boundaries_reject_misrouting_or_changed_xt(checker):
    arrays, metadata, x0 = feature_fixture(checker)
    checker.check_feature_arrays(arrays, metadata, x0, 0.2)
    arrays["projection_ids"][0] = 1
    with pytest.raises(ValueError, match="projection IDs"):
        checker.check_feature_arrays(arrays, metadata, x0, 0.2)
    arrays["projection_ids"][0] = 0
    arrays["xt"][0, 0] += 0.001
    with pytest.raises(ValueError, match="xt flow"):
        checker.check_feature_arrays(arrays, metadata, x0, 0.2)
    arrays["xt"][0, 0] -= 0.001
    arrays["hidden"][0, 0] = 1 + 2**-10
    with pytest.raises(ValueError, match="BF16 export"):
        checker.check_feature_arrays(arrays, metadata, x0, 0.2)


def test_four_uniform_source_anchors_and_full_domain_context_are_not_interchangeable(checker):
    assert checker.source_anchors(89) == [0, 16, 32, 48]
    with pytest.raises(ValueError, match="four-anchor"):
        checker.source_anchors(43)
    context = dict(domain=checker.DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=checker.ACTIVE.tolist())
    checker.context_check(context)
    for wrong in (
        dict(context, domain="another/shared/slot0"),
        dict(context, normalization_id=0),
        dict(context, action_dim_mask=[True] * 128),
    ):
        with pytest.raises(ValueError, match="full-domain"):
            checker.context_check(wrong)


def test_bundle_source_sha_and_tensor_index_cannot_hide_changed_features(checker, tmp_path):
    archive, meta = tmp_path / "features.npz", tmp_path / "features.json"
    np.savez(archive, hidden=np.ones((2, 3), np.float32))
    metadata = dict(
        arrays=dict(path=str(archive), sha256=checker.digest(archive)),
        array_index=dict(hidden=dict(shape=[2, 3], dtype="float32")),
    )
    meta.write_text(json.dumps(metadata))
    reference = dict(path=str(meta), sha256=checker.digest(meta))
    checker.bundle(reference, checker.Evidence())
    np.savez(archive, hidden=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match="SHA changed"):
        checker.bundle(reference, checker.Evidence())


def test_partial_features_never_admit_fit_and_component_metric_reconstruction_covers_all_slots(checker, tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(dict(schema_version="s2a.frozen-readout-features.v1", complete=False, passed=False)))
    with pytest.raises(ValueError, match="partial/rejected"):
        checker.verify_features({}, {}, {}, path, checker.Evidence())
    target = np.zeros((2, 128))
    pred = target.copy()
    pred[:, 9] = 2
    metrics = checker.metric_group(dict(raw=pred, adapt=target, bias=pred), target, [0, 1])
    assert metrics["mse"]["raw"] == 0.4 and metrics["per_slot_mse"]["raw"] == [0.0] * 9 + [4.0]
    wrong = json.loads(json.dumps(metrics))
    wrong["per_slot_mse"]["raw"][9] = 0
    with pytest.raises(ValueError, match="reconstruction differs"):
        checker.compare_metrics(wrong, metrics)


def state_gate_fixture():
    def group():
        result = {}
        for horizon in (8, 16, 40):
            raw = dict(position_rmse_m=0.01, rotation_rmse_rad=0.1, opening_mae=0.1, joint_score=10.0)
            result[str(horizon)] = dict(
                raw=raw,
                adapt=dict(raw, joint_score=9.0),
                hold=dict(raw, joint_score=11.0),
                constant_velocity=dict(raw, joint_score=12.0),
            )
        return result

    return dict(
        pooled=group(), by_episode={str(e): group() for e in range(6)}, by_seed={str(s): group() for s in (42, 43)}
    )


def test_sampling_gate_requires_nine_component_additive_limits_and_all_episode_seed_protections(checker):
    groups = state_gate_fixture()
    assert checker.independent_state_gate(groups)["passed"]
    groups["pooled"]["8"]["adapt"]["joint_score"] = np.nextafter(9.0, 10.0)
    assert not checker.independent_state_gate(groups)["passed"]
    groups = state_gate_fixture()
    groups["pooled"]["40"]["raw"]["position_rmse_m"] = 0.1
    groups["pooled"]["40"]["adapt"]["position_rmse_m"] = 0.103
    assert not checker.independent_state_gate(groups)["checks"]["all_components_protected"]
    groups = state_gate_fixture()
    for episode in ("0", "1"):
        groups["by_episode"][episode]["8"]["adapt"]["joint_score"] = 10.6
    assert not checker.independent_state_gate(groups)["checks"]["at_least_five_episodes"]
    groups = state_gate_fixture()
    groups["by_episode"]["0"]["16"]["adapt"]["joint_score"] = 12.50001
    assert not checker.independent_state_gate(groups)["checks"]["no_episode_short_outlier"]
    groups = state_gate_fixture()
    groups["by_seed"]["43"]["16"]["adapt"]["joint_score"] = 10.50001
    assert not checker.independent_state_gate(groups)["checks"]["both_seeds_protected"]


def test_state_pool_scores_endpoint_errors_with_correct_units_and_shared_baseline_counts(checker):
    rows = []
    for variant, position in (("raw", 0.01), ("adapt", 0.005)):
        errors = dict(position=[position] * 40, rotation=[0.05] * 40, opening=[-0.1] * 40)
        rows.append(dict(variant=variant, errors=dict(wam=errors, hold=errors, constant_velocity=errors)))
    groups = checker.pool_state_rows(rows)
    assert groups["8"]["raw"]["valid_endpoints"] == groups["8"]["hold"]["valid_endpoints"] == 8
    assert groups["40"]["raw"]["joint_score"] == 3
    assert groups["40"]["adapt"]["joint_score"] == 2.25
    assert groups["8"]["raw"]["opening_mae"] == 0.1


def test_sampling_initial_mask_and_inactive_noise_checked_before_pair_comparison(checker):
    shape = (1, 48, 11, 60, 40)
    offset = int(np.prod(shape))
    arrays = dict(
        initial_noise=np.zeros(offset + 5120, np.float32),
        condition_mask=np.zeros(offset + 5120, np.float32),
        first_condition_latent=np.ones((1, 48, 1, 60, 40), np.float32),
    )
    arrays["initial_noise"][:offset].reshape(shape)[:, :, :1] = 1
    arrays["condition_mask"][:offset].reshape(shape)[:, :, :1] = 1
    meta = dict(caption="actual current task", text_tokens=[[1, 2, 3]])
    checker.initial_signature(meta, arrays)
    arrays["initial_noise"][offset + 10] = 1
    with pytest.raises(ValueError, match="initial inactive"):
        checker.initial_signature(meta, arrays)
    arrays["initial_noise"][offset + 10] = 0
    arrays["condition_mask"][-1] = 1
    with pytest.raises(ValueError, match="first-frame-only"):
        checker.initial_signature(meta, arrays)


def test_frozen_independent_integrator_uses_current_anchor_local_delta_and_past_only_baseline(checker):
    integrator = checker.frozen_integrator(checker.Evidence())
    states = np.zeros((45, 128))
    states[:, 0] = np.arange(45) * 0.01
    states[:, 3] = states[:, 7] = 1
    actions = np.zeros((40, 128))
    actions[:, 0] = 0.01
    actions[:, 3] = actions[:, 7] = 1
    count, errors, _ = integrator(states, dict(anchor_frame=4, valid_future_steps=40), actions)
    assert count == 40
    np.testing.assert_allclose(errors["wam"]["position"], 0, atol=1e-15)
    np.testing.assert_allclose(errors["constant_velocity"]["position"], 0, atol=1e-15)
    assert errors["hold"]["position"][-1] == pytest.approx(0.4)


def test_explicit_cli_phases_require_exact_evidence_roster_and_never_ignore_extra_stage(checker):
    fields = ("features", "fit", "fit_verification", "sampling")
    for phase, present in (("prepare", 0), ("features", 1), ("fit", 2), ("sampling", 4)):
        values = {name: Path("actual.json") if i < present else None for i, name in enumerate(fields)}
        checker.phase_arguments(SimpleNamespace(phase=phase, **values))
        changed = dict(values)
        changed[fields[0]] = None if values[fields[0]] is not None else Path("extra.json")
        with pytest.raises(ValueError, match="explicit audit phase"):
            checker.phase_arguments(SimpleNamespace(phase=phase, **changed))
