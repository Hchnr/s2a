"""025 fit-stage isolation and quantized teacher gates on synthetic CPU inputs."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fit_stage():
    path = Path(__file__).resolve().parents[1] / "scripts/fit_wam_readout.py"
    spec = importlib.util.spec_from_file_location("wam_readout_fit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preparation():
    train, validation = list(range(6)), list(range(6, 12))
    anchors, episodes, teacher = [], [], []
    for episode in train + validation:
        split = "train" if episode in train else "validation"
        source = f"synthetic-source-{episode}"
        names = []
        for index in range(4):
            name = f"episode{episode}_anchor{index}"
            names.append(name)
            anchors.append(
                dict(
                    name=name,
                    episode=episode,
                    source_episode_id=source,
                    split=split,
                    anchor_index=index,
                    anchor_frame=index * 40,
                    arrays_path=f"synthetic{episode}.npz",
                    task_index=episode % 3,
                )
            )
            for sigma in (0.2, 0.5, 0.8):
                teacher.append(dict(call_index=len(teacher), anchor_name=name, split=split, sigma=sigma))
        episodes.append(dict(episode=episode, source_episode_id=source, split=split, anchors=names))
    return dict(
        config=dict(train_episodes=train, validation_episodes=validation),
        anchors=anchors,
        episodes=episodes,
        teacher_calls=teacher,
    )


def test_whole_episode_roster_rejects_validation_leak_and_source_alias(fit_stage):
    registered = preparation()
    assert len(fit_stage.validate_anchors(registered)) == 48
    crossed = copy.deepcopy(registered)
    crossed["anchors"][-1]["split"] = "train"
    with pytest.raises(ValueError, match="whole-episode split"):
        fit_stage.validate_anchors(crossed)
    aliased = copy.deepcopy(registered)
    for anchor in aliased["anchors"][-4:]:
        anchor["source_episode_id"] = registered["anchors"][0]["source_episode_id"]
    with pytest.raises(ValueError, match="aliases leak"):
        fit_stage.validate_anchors(aliased)


def test_hierarchical_weights_are_train_only_equal_per_source_and_complete(fit_stage):
    registered = preparation()
    rows = [
        dict(episode=a["episode"], anchor_name=a["name"], sigma=s, endpoint=i, split=a["split"])
        for a in registered["anchors"]
        if a["split"] == "train"
        for s in (0.2, 0.5, 0.8)
        for i in range(40)
    ]
    weights = fit_stage.hierarchical_weights(rows)
    assert len(weights) == 2880
    np.testing.assert_array_equal(weights, np.full(2880, 1 / 2880))
    np.testing.assert_allclose(
        [weights[np.asarray([r["episode"] == e for r in rows])].sum() for e in range(6)],
        np.full(6, 1 / 6),
        rtol=0,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="training rows only"):
        fit_stage.hierarchical_weights(rows + [dict(rows[0], split="validation")])
    with pytest.raises(ValueError, match="40 distinct"):
        fit_stage.hierarchical_weights(rows[:-1])


def test_feature_export_requires_exact_bfloat16_real_torch_mask_and_no_future_invalid_input(fit_stage):
    torch = pytest.importorskip("torch")
    arrays = dict(
        hidden=np.zeros((40, 1280), dtype=np.float32),
        base=np.zeros((40, 128), dtype=np.float32),
        target=np.zeros((40, 128), dtype=np.float32),
        xt=np.zeros((40, 128), dtype=np.float32),
        valid=np.broadcast_to([True] * 10 + [False] * 118, (40, 128)),
    )
    meta = dict(original_dtypes=dict(hidden="torch.bfloat16", base="torch.bfloat16"))
    fit_stage.validate_feature_array(arrays, meta, torch)
    arrays["base"][0, 0] = np.float32(1.001)
    with pytest.raises(ValueError, match="exact BF16 expansion"):
        fit_stage.validate_feature_array(arrays, meta, torch)
    arrays["base"][0, 0] = 0
    arrays["xt"][0, 10] = 1
    with pytest.raises(ValueError, match="inactive teacher"):
        fit_stage.validate_feature_array(arrays, meta, torch)


def test_feature_call_order_is_checked_before_any_array_or_fit_read(fit_stage):
    registered = preparation()
    calls = copy.deepcopy(registered["teacher_calls"])
    calls[0]["split"] = "validation"
    features = dict(teacher_calls=calls, denoise_calls=249, zero_identity=[{}, {}, {}])

    class Evidence:
        def json_ref(self, ref):
            return {}

    with pytest.raises(ValueError, match="registered order/split/sigma"):
        fit_stage.load_rows(registered, features, Evidence(), None)


def test_teacher_metric_reports_actual_quantized_prediction_and_strict_gate(fit_stage):
    rows = [
        dict(episode=e, task_index=e % 3, split=s, sigma=t)
        for s in ("train", "validation")
        for e in range(6)
        for t in (0.2, 0.5, 0.8)
    ]
    target = np.zeros((len(rows), 128), dtype=np.float32)
    predictions = {name: np.full_like(target, value) for name, value in (("raw", 1), ("adapt", 0.5), ("bias", 0.75))}
    metrics = fit_stage.teacher_metrics(predictions, target, rows)
    assert metrics["validation"]["pooled"]["mse"] == dict(raw=1, adapt=0.25, bias=0.5625)
    assert metrics["validation"]["pooled"]["per_slot_mse"]["adapt"] == [0.25] * 10
    assert fit_stage.teacher_gate(metrics)["passed"]
    tied = copy.deepcopy(metrics)
    tied["validation"]["pooled"]["mse"]["bias"] = 0.25
    assert not fit_stage.teacher_gate(tied)["passed"]
    regressed = copy.deepcopy(metrics)
    regressed["validation"]["by_sigma"]["0.2"]["mse"]["adapt"] = np.nextafter(1.05, np.inf)
    assert not fit_stage.teacher_gate(regressed)["passed"]


def test_cpu_environment_rejection_happens_before_loading_real_features(fit_stage, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(ValueError, match="empty CUDA/LD"):
        fit_stage.execute(SimpleNamespace(seed=42))


def test_evidence_cannot_change_and_output_cannot_be_overwritten(fit_stage, tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"source":1}')
    evidence = fit_stage.Evidence()
    evidence.json(path)
    path.write_text('{"source":2}')
    with pytest.raises(ValueError, match="source SHA changed"):
        evidence.recheck()
    out = tmp_path / "result.json"
    fit_stage.write_new(out, dict(complete=True))
    with pytest.raises(FileExistsError):
        fit_stage.write_new(out, dict(complete=False))
