"""026 independent numerical and admission checks using synthetic CPU inputs."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_flow_readout.py"
    spec = importlib.util.spec_from_file_location("independent_flow_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_fp32_sigmas_expand_losslessly_and_reject_zero_before_zero_candidate(checker):
    sigma = np.asarray([0.2, 0.5, 0.8], np.float32)
    hidden = np.ones((3, 4))
    design = checker.flow_design(hidden, sigma[:, None])
    expected = -np.column_stack((hidden / 2, np.ones(3))) / sigma.astype(np.float64)[:, None]
    np.testing.assert_array_equal(design, expected)
    assert design[0, -1] != -1 / 0.2
    for invalid in (0, -0.2, np.nan, np.inf, [1, 2], np.ones((3, 2))):
        with pytest.raises(ValueError, match="actual sigma"):
            checker.quantized_velocity(hidden, np.zeros((3, 128)), invalid, np.zeros((5, 10)))


def test_independent_augmented_velocity_solve_matches_dual_and_sigma_aware_bias(checker):
    rng = np.random.default_rng(42)
    hidden, residual = rng.normal(size=(12, 4)), rng.normal(size=(12, 10))
    sigma = np.resize(np.asarray([0.2, 0.5, 0.8], np.float32), 12)
    weights = np.arange(1, 13, dtype=np.float64)
    weights /= weights.sum()
    actual = checker.augmented_fit(hidden, sigma, residual, weights)
    design = -np.column_stack((hidden / 2, np.ones(12))) / sigma.astype(np.float64)[:, None]
    x, y = np.sqrt(weights)[:, None] * design, np.sqrt(weights)[:, None] * residual
    expected = x.T @ np.linalg.solve(x @ x.T + 0.1 * np.eye(12), y)
    np.testing.assert_allclose(actual["adapt"], expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(actual["bias"][:-1], 0)
    bias_column = -1 / sigma.astype(np.float64)
    np.testing.assert_allclose(
        actual["bias"][-1], (weights * bias_column) @ residual / (np.dot(weights, bias_column**2) + 0.1), rtol=1e-15
    )
    assert not np.allclose(actual["bias"][-1], weights @ residual / 1.1)
    changed_weights = weights * sigma.astype(np.float64) ** 2
    changed_weights /= changed_weights.sum()
    wrong_loss = checker.augmented_fit(hidden, sigma, residual, changed_weights)
    assert np.linalg.norm(actual["adapt"] - wrong_loss["adapt"]) > 0.01
    with pytest.raises(ValueError, match="positive and sum"):
        checker.augmented_fit(hidden, sigma, residual, weights * 2)


def test_flow_deployment_sign_sigma_direct_rounding_zero_identity_and_inactive_bytes(checker):
    hidden = np.zeros((3, 4), np.float32)
    base = np.ones((3, 128), np.float32)
    base[:, 10:] = np.arange(118)
    base[:, 10] = -0.0
    before = base.copy()
    weights = np.zeros((5, 10), np.float64)
    assert checker.quantized_velocity(hidden, base, 0.5, weights) is base
    weights[-1] = -0.5 * (2**-8 + 2**-30)
    result = checker.quantized_velocity(hidden, base, 0.5, weights)
    np.testing.assert_array_equal(result[:, :10], np.full((3, 10), 1 + 2**-7))
    np.testing.assert_array_equal(result[:, 10:].view(np.uint32), before[:, 10:].view(np.uint32))
    np.testing.assert_array_equal(base.view(np.uint32), before.view(np.uint32))
    result_double_sigma = checker.quantized_velocity(hidden, base, 1.0, weights)
    np.testing.assert_array_equal(result_double_sigma[:, :10], np.ones((3, 10)))


def test_cached_target_remains_original_velocity_and_does_not_fit_literal_bf16_xt_error(checker):
    x0 = np.zeros((40, 128), np.float32)
    x0[:, :10] = np.float32(0.1234)
    epsilon = np.zeros_like(x0)
    epsilon[:, :10] = np.float32(0.8333)
    sigma = np.float32(0.2)
    target = epsilon - x0
    xt = checker.OLD.quantize_bfloat16(sigma * epsilon + (np.float32(1) - sigma) * x0)
    base = np.zeros_like(x0)
    arrays = dict(
        hidden=np.zeros((40, 1280), np.float32),
        head_base=base.copy(),
        base=base,
        target=target,
        xt=xt,
        epsilon=epsilon,
        x0_normalized=x0,
        sigma_action=np.full((40, 1), sigma),
        condition_action=np.zeros((40, 1), np.float32),
        projection_ids=np.zeros(40, np.int64),
        valid=np.broadcast_to(checker.OLD.ACTIVE, (40, 128)),
    )
    metadata = dict(original_dtypes=dict(hidden="torch.bfloat16", base="torch.bfloat16"))
    checker.cached_contract(arrays, metadata, dict(sigma=0.2))
    desired = target[:, :10].astype(np.float64) - base[:, :10]
    literal_clean_target = -(
        x0[:, :10].astype(np.float64) - (xt[:, :10].astype(np.float64) - float(sigma) * base[:, :10])
    ) / float(sigma)
    assert np.max(np.abs(desired - literal_clean_target)) > 0.001
    arrays["sigma_action"] = np.full((40, 1), 0.2, np.float64)
    with pytest.raises(ValueError, match="stored FP32"):
        checker.cached_contract(arrays, metadata, dict(sigma=0.2))


def trace_fixture(checker):
    sigmas, times = checker.solver_schedule()
    return dict(
        complete=True,
        name="raw_s42",
        seed=42,
        sampler_calls=1,
        solver_sigma_dtype="torch.float32",
        terminal_sigma=0.0,
        terminal_neural_calls=0,
        sigma_values=sigmas.astype(np.float64).tolist(),
        scheduler_timesteps=times.tolist(),
        parameters_before=dict(count=1, metadata_sha256="a" * 64),
        parameters_after=dict(count=1, metadata_sha256="a" * 64),
        actual_sampler_arguments=dict(num_steps=35, shift=5.0, seed=[42]),
        source_sha256={"original_forward.py": "b" * 64},
        steps=[
            dict(
                step_index=i,
                solver_step_index_before=None if i == 0 else i,
                solver_sigma=float(s),
                solver_timestep=int(t),
                velocity_delegations=1,
                head_calls=1,
            )
            for i, (s, t) in enumerate(zip(sigmas[:-1], times, strict=True))
        ],
    )


def test_solver_trace_requires_actual_fp32_schedule_not_integer_time_division_and_no_terminal_nn(checker):
    trace = trace_fixture(checker)
    prepared = dict(source_sha256=trace["source_sha256"])
    checker.sigma_trace(trace, "raw_s42", 42, prepared)
    assert trace["steps"][-1]["solver_sigma"] == pytest.approx(0.1280900687)
    assert trace["steps"][-1]["solver_sigma"] != trace["steps"][-1]["solver_timestep"] / 1000
    for change in ("sigma", "step", "twice", "terminal", "weights", "source"):
        broken = copy.deepcopy(trace)
        if change == "sigma":
            broken["steps"][0]["solver_sigma"] = broken["steps"][0]["solver_timestep"] / 1000
        elif change == "step":
            broken["steps"][0]["solver_step_index_before"] = 0
        elif change == "twice":
            broken["steps"][0]["velocity_delegations"] = 2
        elif change == "terminal":
            broken["terminal_neural_calls"] = 1
        elif change == "weights":
            broken["parameters_after"]["metadata_sha256"] = "c" * 64
        else:
            broken["source_sha256"]["original_forward.py"] = "c" * 64
        with pytest.raises(ValueError):
            checker.sigma_trace(broken, "raw_s42", 42, prepared)


def test_hierarchy_excludes_validation_and_teacher_gate_retains_zero_slack(checker):
    rows = [dict(split="train", episode=1, anchor_name="a", sigma=0.2, endpoint=i) for i in range(2)]
    np.testing.assert_array_equal(checker.OLD.hierarchical_weights(rows), [0.5, 0.5])
    with pytest.raises(ValueError, match="validation"):
        checker.OLD.hierarchical_weights(rows + [dict(rows[0], split="validation")])
    metrics = dict(
        validation=dict(
            pooled=dict(mse=dict(raw=1.0, adapt=0.95, bias=0.951)),
            by_sigma={str(s): dict(mse=dict(raw=1.0, adapt=1.05, bias=2.0)) for s in (0.2, 0.5, 0.8)},
        )
    )
    assert checker.OLD.independent_teacher_gate(metrics)["passed"]
    metrics["validation"]["by_sigma"]["0.8"]["mse"]["adapt"] = np.nextafter(1.05, 2.0)
    assert not checker.OLD.independent_teacher_gate(metrics)["passed"]


def test_sampling_roster_counts_all_three_zero_controls_before96_generations(checker):
    model = dict(path="adapt.npz", sha256="a" * 64)
    prepared, fit = dict(checkpoint="8400/model"), dict(models=dict(adapt=model))
    labels = [f"zero:{i}" for i in range(3)] + [f"sampling:{i}" for i in range(96)]
    actual = dict(
        schema_version="s2a.flow-readout-sampling.v1",
        complete=True,
        passed=True,
        all_original_parameters_frozen=True,
        head_call_counts=dict(zero=105, raw=1680, adapt=1680),
        physics_steps=0,
        fresh_cases=0,
        new_teacher_calls=0,
        denoise_calls=3465,
        zero_identity=[{}] * 3,
        generations=[{}] * 96,
        adapted_model_ref=model,
        runtime=dict(checkpoint_path="8400/model", ema_weights=False),
        denoise_ledger=[
            dict(index=i * 35 + j, label=label, completed=True) for i, label in enumerate(labels) for j in range(35)
        ],
    )
    checker.sampling_execution_contract(actual, prepared, fit)
    for key, value in (
        ("complete", False),
        ("denoise_calls", 3360),
        ("physics_steps", 1),
        ("head_call_counts", dict(zero=0, raw=1680, adapt=1680)),
        ("denoise_ledger", actual["denoise_ledger"][105:]),
    ):
        with pytest.raises(ValueError):
            checker.sampling_execution_contract(dict(actual, **{key: value}), prepared, fit)


def test_current_noise_text_mask_pairing_is_exact_before_future_state_reads(checker):
    signature = dict(
        caption="current task", text_tokens=[[42]], arrays=dict(initial_noise=np.ones(3), condition_mask=np.zeros(3))
    )
    checker.require_same_initial(copy.deepcopy(signature), signature)
    for key in ("initial_noise", "condition_mask"):
        changed = copy.deepcopy(signature)
        changed["arrays"][key][0] += np.finfo(float).eps
        with pytest.raises(ValueError, match="actual initial"):
            checker.require_same_initial(changed, signature)
    changed = copy.deepcopy(signature)
    changed["text_tokens"] = [[43]]
    with pytest.raises(ValueError, match="paired text"):
        checker.require_same_initial(changed, signature)


def test_exact_cli_evidence_roster_and_partial_fits_fail_before_cache_or_solve(checker, tmp_path):
    for phase, present in (("prepare", 0), ("fit", 1), ("sampling", 3)):
        fields = {
            key: Path("record.json") if i < present else None
            for i, key in enumerate(("fit", "fit_verification", "sampling"))
        }
        checker.phase_arguments(SimpleNamespace(phase=phase, **fields))
        fields["sampling"] = Path("extra.json") if fields["sampling"] is None else None
        with pytest.raises(ValueError, match="phase/evidence"):
            checker.phase_arguments(SimpleNamespace(phase=phase, **fields))
    rejected = tmp_path / "partial.json"
    rejected.write_text(json.dumps(dict(schema_version="s2a.flow-readout-fit.v1", complete=False, passed=False)))
    with pytest.raises(ValueError, match="partial/rejected"):
        checker.verify_fit({}, {}, rejected, checker.OLD.Evidence())


def test_preparation_rejects_raw_episode_or_png_reads_before_hashing_their_bytes(checker):
    config = json.loads(checker.CONFIG.read_text())
    prepared = dict(
        schema_version="s2a.flow-readout-preparation.v1",
        config=config,
        config_ref=dict(path=str(checker.CONFIG), sha256=checker.CONFIG_SHA),
        registration=dict(path=str(checker.PLAN), sha256=checker.PLAN_SHA),
        output_root=str(checker.REPO / config["output_dir"]),
        budget=config["budget"],
        anchors=[dict(arrays_path="/unopened_episode.npz", image_path="/unopened_current.png")],
        source_sha256={"/unopened_episode.npz": "a" * 64},
    )
    checked = []

    class Evidence:
        def json(self, path, expected=None):
            return config if path == checker.CONFIG else prepared

        def check(self, path, expected=None):
            checked.append(str(path))
            assert str(path) not in ("/unopened_episode.npz", "/unopened_current.png")

    with pytest.raises(ValueError, match="reopened inherited"):
        checker.verify_preparation(Path("prepared.json"), Evidence())
    assert checked == [str(checker.PLAN), str(checker.HELPER)]


def precision_fixture(checker):
    half, eps, spacing = 2**-8, 2**-30, 2**-133
    positive = [
        half - eps,
        half,
        half + eps,
        -half + eps,
        -half,
        -half - eps,
        3 * half - eps,
        3 * half,
        3 * half + eps,
        0.0,
    ]
    tiny = [
        spacing / 2 - 2**-160,
        spacing / 2,
        spacing / 2 + 2**-160,
        -spacing / 2 + 2**-160,
        -spacing / 2,
        -spacing / 2 - 2**-160,
        1.5 * spacing,
        2.5 * spacing,
        2**-126 - spacing / 2,
        -0.0,
    ]
    arrays, cases = {}, []
    for index, sigma in enumerate(
        np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64)
    ):
        for name, scalar, residual in (
            ("positive", 1.0, positive),
            ("negative", -1.0, -np.asarray(positive)),
            ("subnormal", 0.0, tiny),
        ):
            prefix = f"sigma{index}_{name}"
            hidden = np.zeros((2, 1280), np.float32)
            hidden[1, :2] = [0.5, -0.25]
            base = np.full((2, 128), scalar, np.float32)
            base[:, 10:] = np.arange(118)
            base[:, 10] = -0.0
            weights = np.zeros((1281, 10))
            weights[-1] = -np.asarray(residual) * sigma
            weights[0, 0], weights[1, 1] = 2**-20, -(2**-21)
            actual_sigma = np.full((2, 1), sigma)
            output = checker.quantized_velocity(hidden, base, actual_sigma, weights)
            bits = (output.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)
            for key, value in dict(
                hidden=hidden,
                base=base,
                weights=weights,
                sigma_actual=actual_sigma,
                cpu_bits=bits,
                cuda_bits=bits.copy(),
            ).items():
                arrays[f"{prefix}_{key}"] = value
            cases.append(
                dict(
                    name=prefix,
                    cpu_cuda_bitwise_equal=True,
                    actual_hook_matches_apply=True,
                    zero_original_object=True,
                    other_domain_original_object=True,
                    other118_preserved=True,
                    input_tensors_unchanged=True,
                )
            )
    primitive = np.asarray(
        [
            0.0,
            -0.0,
            2**-134 - 2**-160,
            2**-134,
            2**-134 + 2**-160,
            -(2**-134) + 2**-160,
            -(2**-134),
            -(2**-134) - 2**-160,
            1 + half + eps,
            -1 - half - eps,
        ]
    )
    arrays.update(
        primitive_values=primitive,
        primitive_cpu_bits=checker.bf16_bits(primitive),
        primitive_cuda_bits=checker.bf16_bits(primitive),
    )
    return arrays, cases


def test_new_precision_audit_checks_each_saved_input_sigma_and_all128_hardware_bits(checker):
    arrays, cases = precision_fixture(checker)
    checker.precision_arrays(arrays, cases)
    for key, slot in (
        ("sigma0_positive_cpu_bits", (0, 0)),
        ("sigma4_subnormal_cuda_bits", (1, 127)),
        ("primitive_cuda_bits", 1),
    ):
        original = arrays[key][slot].copy()
        arrays[key][slot] ^= np.int16(1)
        with pytest.raises(ValueError, match="bits|RNE"):
            checker.precision_arrays(arrays, cases)
        arrays[key][slot] = original
    arrays["sigma0_positive_sigma_actual"][:] = 0.2
    with pytest.raises(ValueError, match="sigma_actual"):
        checker.precision_arrays(arrays, cases)
