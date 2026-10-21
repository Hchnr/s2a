"""027 independent synthetic CPU math, causality and evidence rejection."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_scaled_readout.py"
    spec = importlib.util.spec_from_file_location("independent_scaled_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_train_scale_has_unit_total_energy_without_column_whitening_or_extra_sqrt(checker):
    hidden = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 2.0], [6.0, 2.0]])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    center, scale = checker.preprocessing(hidden, weights)
    np.testing.assert_allclose(center, [4.0, 1.4], rtol=0, atol=1e-15)
    assert scale.shape == () and scale == pytest.approx(np.sqrt(4.84))
    z = (hidden - center) / scale
    energy = np.sum(weights[:, None] * z**2, axis=0)
    assert energy.sum() == pytest.approx(1.0) and not np.allclose(energy, 1.0)
    design = checker.scaled_design(hidden, np.float32(0.2), center, scale)
    np.testing.assert_array_equal(design[:, :-1], -z / float(np.float32(0.2)))
    np.testing.assert_array_equal(design[:, -1], np.full(4, -1 / float(np.float32(0.2))))


def test_augmented_candidate_solve_uses_original_velocity_objective_and_regularizes_bias(checker, monkeypatch):
    rng = np.random.default_rng(42)
    hidden, target = rng.normal(size=(11, 4)), rng.normal(size=(11, 10))
    sigma = np.resize(np.asarray([0.2, 0.5, 0.8], np.float32), 11)
    weights = np.arange(1, 12, dtype=np.float64)
    weights /= weights.sum()
    center, scale = checker.preprocessing(hidden, weights)
    design = -np.column_stack(((hidden - center) / scale, np.ones(11))) / sigma.astype(np.float64)[:, None]
    x, y = np.sqrt(weights)[:, None] * design, np.sqrt(weights)[:, None] * target
    expected = x.T @ np.linalg.solve(x @ x.T + 0.1 * np.eye(11), y)
    original, calls = np.linalg.lstsq, []

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counted)
    actual = checker.augmented_fit(hidden, sigma, target, weights)
    assert calls == [(16, 5)]
    np.testing.assert_allclose(actual["weights"], expected, atol=2e-14, rtol=2e-14)
    wrong_weights = weights * sigma.astype(np.float64) ** 2
    wrong_weights /= wrong_weights.sum()
    wrong = checker.augmented_fit(hidden, sigma, target, wrong_weights)
    assert np.linalg.norm(actual["weights"] - wrong["weights"]) > 0.01


def test_centering_does_not_claim_time_weighted_hidden_bias_orthogonality(checker):
    hidden = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 2.0], [6.0, 2.0]])
    weights = np.full(4, 0.25)
    center, scale = checker.preprocessing(hidden, weights)
    sigma = np.asarray([0.2, 0.2, 0.8, 0.8], np.float32)
    design = checker.scaled_design(hidden, sigma, center, scale)
    gram = design.T @ (weights[:, None] * design)
    assert np.linalg.norm(gram[:-1, -1]) > 1.0


def test_whole_episode_selection_keeps_validation_out_of_center_scale_and_fit(checker):
    rows = [
        dict(split=split, episode=episode, anchor_name=str(episode), sigma=0.2, endpoint=i)
        for split, episode in (("train", 1), ("validation", 2))
        for i in range(3)
    ]
    selected, weights = checker.training_rows(rows, [1], [2])
    hidden = np.asarray([[0.0, 1.0], [2.0, 2.0], [4.0, 7.0], [11.0, 13.0], [15.0, 17.0], [19.0, 23.0]])
    target = np.arange(60).reshape(6, 10) / 100
    model = checker.augmented_fit(hidden[selected], 0.5, target[selected], weights)
    hidden[~selected], target[~selected] = 1e100, -1e100
    unchanged = checker.augmented_fit(hidden[selected], 0.5, target[selected], weights)
    for key in model:
        np.testing.assert_array_equal(unchanged[key], model[key])
    for invalid in ([dict(rows[0], split="validation")], [dict(rows[-1], split="train")]):
        with pytest.raises(ValueError, match="wrong whole-episode"):
            checker.training_rows(invalid, [1], [2])
    with pytest.raises(ValueError, match="overlap"):
        checker.training_rows(rows, [1, 2], [2])


def test_invalid_global_scale_and_nonpositive_actual_sigmas_are_rejected_without_epsilon(checker):
    with pytest.raises(ValueError, match="finite and positive"):
        checker.preprocessing(np.ones((4, 3)), np.full(4, 0.25))
    with pytest.raises(ValueError, match="positive and sum"):
        checker.preprocessing(np.arange(12).reshape(4, 3), np.full(4, 0.5))
    for scale in (0.0, -1.0, [1.0, 2.0], np.nan):
        with pytest.raises(ValueError):
            checker.scaled_design(np.ones((2, 2)), 0.5, np.zeros(2), scale)
    for sigma in (0.0, -0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="actual sigma"):
            checker.quantized_velocity(np.ones((2, 2)), np.zeros((2, 128)), sigma, np.zeros((3, 10)), np.zeros(2), 1.0)


def test_saved_actual_sigma_direct_bf16_rounding_zero_object_and_all_inactive_bits(checker):
    hidden = np.zeros((2, 4), np.float32)
    center = np.full(4, 3.0)
    base = np.ones((2, 128), np.float32)
    base[:, 10:] = np.arange(118)
    base[:, 10] = -0.0
    before = base.copy()
    weights = np.zeros((5, 10))
    assert checker.quantized_velocity(hidden, base, 0.5, weights, center, 2.0) is base
    weights[-1] = -0.5 * (2**-8 + 2**-30)
    actual = checker.quantized_velocity(hidden, base, 0.5, weights, center, 2.0)
    np.testing.assert_array_equal(actual[:, :10], np.full((2, 10), 1 + 2**-7))
    np.testing.assert_array_equal(actual[:, 10:].view(np.uint32), before[:, 10:].view(np.uint32))
    np.testing.assert_array_equal(base.view(np.uint32), before.view(np.uint32))
    first = checker.scaled_design(hidden, np.float32(0.2), center, 2.0)
    assert first[0, -1] != -1 / 0.2


def test_reused_time_aware_bias_has_exact_original_predictions_under_new_preprocessing(checker):
    rng = np.random.default_rng(42)
    hidden = checker.OLD.quantize_bfloat16(rng.normal(size=(12, 1280)))
    base = checker.OLD.quantize_bfloat16(rng.normal(size=(12, 128)))
    base[:, 10:] = 0
    sigma = np.resize(np.asarray([0.2, 0.5, 0.8], np.float32), 12).astype(np.float64)
    bias = np.zeros((1281, 10))
    bias[-1] = np.linspace(-0.1, 0.1, 10)
    center, scale = checker.preprocessing(hidden[:6], np.full(6, 1 / 6))
    original = checker.FLOW.quantized_velocity(hidden, base, sigma, bias)
    actual = checker.quantized_velocity(hidden, base, sigma, bias, center, scale)
    np.testing.assert_array_equal(actual.view(np.uint32), original.view(np.uint32))


def test_teacher_gate_uses_original_bias_and_each_sigma_with_no_result_slack(checker):
    metrics = dict(
        validation=dict(
            pooled=dict(mse=dict(raw=1.0, adapt=0.95, bias=0.951)),
            by_sigma={str(s): dict(mse=dict(raw=1.0, adapt=1.05, bias=2.0)) for s in (0.2, 0.5, 0.8)},
        )
    )
    assert checker.OLD.independent_teacher_gate(metrics)["passed"]
    metrics["validation"]["pooled"]["mse"]["bias"] = 0.95
    assert not checker.OLD.independent_teacher_gate(metrics)["passed"]
    metrics["validation"]["pooled"]["mse"]["bias"] = 0.951
    metrics["validation"]["by_sigma"]["0.8"]["mse"]["adapt"] = np.nextafter(1.05, 2.0)
    assert not checker.OLD.independent_teacher_gate(metrics)["passed"]


def test_prepare_rejects_an_episode_or_png_reference_before_opening_it(checker):
    config = json.loads(checker.CONFIG.read_text())
    prepared = dict(
        schema_version="s2a.scaled-readout-preparation.v1",
        config=config,
        config_ref=checker.reference(checker.CONFIG),
        registration=checker.reference(checker.PLAN),
        output_root=str(checker.REPO / config["output_dir"]),
        budget=config["budget"],
        anchors=[dict(arrays_path="/unopened_ep.npz", image_path="/unopened_current.png")],
        zero_identity_cases=[dict(sample=dict(arrays_path="/unopened_zero.npz", image_path="/unopened_zero.png"))],
        source_sha256={"/unopened_ep.npz": "a" * 64},
    )
    checked = []

    class Evidence:
        def json(self, path, expected=None):
            return config if path == checker.CONFIG else prepared

        def check(self, path, expected=None):
            checked.append(str(path))
            assert str(path) != "/unopened_ep.npz"

    with pytest.raises(ValueError, match="reopened inherited"):
        checker.verify_preparation(Path("prepared.json"), Evidence())
    assert checked == [str(checker.PLAN), str(checker.HELPER)]


def test_exact84_source_and_snapshot_contract_includes_actual_checker_owner_and_fit(checker, tmp_path):
    old = {str(checker.HELPER): checker.HELPER_SHA}
    old.update({str(tmp_path / f"old_{i}.py"): f"{i:064x}" for i in range(71)})
    old = json.loads(json.dumps(old, sort_keys=True))
    sources = dict(old)
    sources.update({str(checker.REPO / p): "a" * 64 for p in checker.NEW_SOURCES})
    sources.update({str(checker.CONFIG): checker.CONFIG_SHA, str(checker.PLAN): checker.PLAN_SHA})
    path = tmp_path / "preparation/prepared.json"
    snapshots = {
        str(path.parent / "source_snapshot" / f"{i:03d}_{Path(p).name}"): digest
        for i, (p, digest) in enumerate(sources.items())
    }
    prepared, flow = dict(source_sha256=sources, snapshot_sha256=snapshots), dict(source_sha256=old)
    prepared = json.loads(json.dumps(prepared, sort_keys=True))
    checker.source_closure(prepared, flow, path)
    for source in (
        "scripts/verify_wam_scaled_readout.py",
        "scripts/run_wam_scaled_readout.py",
        "scripts/fit_wam_scaled_readout.py",
    ):
        incomplete = dict(sources)
        incomplete.pop(str(checker.REPO / source))
        with pytest.raises(ValueError, match="required84 source"):
            checker.source_closure(dict(prepared, source_sha256=incomplete), flow, path)
    changed = dict(snapshots)
    key = next(iter(changed))
    changed[key] = "f" * 64
    with pytest.raises(ValueError, match="snapshot paths/digests"):
        checker.source_closure(dict(prepared, snapshot_sha256=changed), flow, path)


def test_zero_episode_and_every_png_suffix_are_forbidden_before_read(checker):
    forbidden = checker.forbidden_observations(
        dict(
            anchors=[dict(arrays_path="/train.npz", image_path="/current.png")],
            zero_identity_cases=[dict(sample=dict(arrays_path="/zero.npz", image_path="/zero.png"))],
        )
    )

    class NoRead:
        def check(self, *args):
            raise AssertionError("raw observation bytes must not be read")

    for path in ("/zero.npz", "/zero.png", "/unregistered.PNG"):
        with pytest.raises(ValueError, match="reopened inherited"):
            checker.cache_only_hashes({path: "a" * 64}, forbidden, NoRead())


def test_partial_fits_and_wrong_cli_phase_fail_before_reading_cache_or_solving(checker, tmp_path):
    for phase, fit in (("prepare", None), ("fit", Path("fit.json"))):
        checker.phase_arguments(SimpleNamespace(phase=phase, fit=fit))
        with pytest.raises(ValueError, match="phase requires"):
            checker.phase_arguments(SimpleNamespace(phase=phase, fit=Path("extra.json") if fit is None else None))
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(dict(schema_version="s2a.scaled-readout-fit.v1", complete=False, passed=False)))
    with pytest.raises(ValueError, match="partial/rejected"):
        checker.verify_fit({}, {}, {}, path, checker.OLD.Evidence())
