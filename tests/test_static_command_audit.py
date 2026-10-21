"""022 measures only actual command changes and applies all registered state gates."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "static_command_audit_test", root / "scripts/audit_wam_static_command.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def caption(command):
    return dict(
        task_title="short task",
        command=command,
        task_plan=[],
        embodiment="same robot",
        duration="2s",
        action_frequency_hz=20,
        camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
    )


def test_actual_caption_pair_preserves_empty_renderer_plan_and_every_other_field(audit):
    left, right = caption("short task"), caption("one static long command")

    def check(value):
        return audit.caption_pair(
            [json.dumps(left)], [json.dumps(value)], task="short task", command="one static long command"
        )

    assert check(right)["renderer_empty_task_plan_preserved"]
    alternatives = []
    removed = copy.deepcopy(right)
    del removed["task_plan"]
    alternatives.append(removed)
    alternatives.extend(
        dict(right, **{key: value})
        for key, value in (
            ("current_subtask", "future label"),
            ("task_plan", ["demonstration plan"]),
            ("embodiment", "other robot"),
            ("command", "another command"),
            ("task_title", "phase label"),
        )
    )
    for changed in alternatives:
        with pytest.raises(AssertionError):
            check(changed)


def response():
    raw = np.zeros((40, 128))
    raw[:, 3] = 1
    raw[:, 7] = 1
    raw[:, 9] = 0.5
    return dict(
        status="success",
        outputs=[
            dict(
                content=dict(
                    action=raw.tolist(),
                    action128=dict(
                        normalized_action=raw.tolist(),
                        checkpoint_path="/ckpt",
                        seed=42,
                        action_domain="domain",
                        action_units="raw_physical_or_source_units",
                        action_fps=20,
                        action_dim_mask=[True] * 10 + [False] * 118,
                        action_timestamps_seconds=(np.arange(40) / 20).tolist(),
                    ),
                )
            )
        ],
    )


def test_exact_reproduction_cannot_ignore_the_last_endpoint_or_inactive_slots(audit):
    original = response()
    assert audit.exact_reproduction(original, copy.deepcopy(original))
    for field in ("raw", "normalized"):
        changed = copy.deepcopy(original)
        content = changed["outputs"][0]["content"]
        values = content["action"] if field == "raw" else content["action128"]["normalized_action"]
        values[39][127] = 1e-15
        with pytest.raises(AssertionError):
            audit.exact_reproduction(changed, original)


def test_full128_physical_affine_mask_and_seed_are_authenticated(audit):
    value = response()
    transform = dict(normalization_mask=[True] * 10 + [False] * 118, scale=[2.0] * 128, bias=[1.0] * 128)
    raw = np.asarray(value["outputs"][0]["content"]["action"])
    norm = np.zeros_like(raw)
    norm[:, :10] = raw[:, :10] * 2 + 1
    value["outputs"][0]["content"]["action128"]["normalized_action"] = norm.tolist()
    result, _ = audit.full_actions(value, transform, checkpoint="/ckpt", seed=42, domain="domain")
    np.testing.assert_array_equal(raw, result)
    for key, changed in (("seed", True), ("action_fps", 30), ("action_dim_mask", [1] * 10 + [0] * 118)):
        bad = copy.deepcopy(value)
        bad["outputs"][0]["content"]["action128"][key] = changed
        with pytest.raises(AssertionError):
            audit.full_actions(bad, transform, checkpoint="/ckpt", seed=42, domain="domain")
    bad = copy.deepcopy(value)
    bad["outputs"][0]["content"]["action128"]["normalized_action"][39][127] = 1e-9
    with pytest.raises(AssertionError):
        audit.full_actions(bad, transform, checkpoint="/ckpt", seed=42, domain="domain")


def test_actual_receipt_pair_is_bound_to_the_response_caption_hash(audit):
    value = response()
    text = json.dumps(caption("short task"))
    value["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    request = dict(name="case", seed=42)
    sample = dict(image_sha256="i" * 64)
    prepared = dict(
        stage="actual_prepare_return_before_predict",
        request=request,
        name="case",
        seed=42,
        image_sha256="i" * 64,
        active_slot_mask=True,
        actual_action_dim_mask=[True] * 10 + [False] * 118,
        caption=[text],
    )
    predicted = dict(prepared, stage="actual_predict_entry_same_prepared_and_batch")
    assert audit.receipt_pair(prepared, predicted, request, sample, value) == [text]
    for key, changed in (("caption", ["different"]), ("request", dict(name="other", seed=42)), ("seed", 43)):
        bad = dict(predicted, **{key: changed})
        with pytest.raises(AssertionError):
            audit.receipt_pair(prepared, bad, request, sample, value)
    changed = copy.deepcopy(value)
    changed["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        audit.receipt_pair(prepared, predicted, request, sample, changed)


def rows(position=0.01, rotation=0.05, opening=0.1):
    result = []
    for episode in range(6):
        errors = dict(position=[position] * 40, rotation=[rotation] * 40, opening=[opening] * 40)
        result.append(
            dict(
                sample=dict(task=f"t{episode // 2}", episode=episode, phase="one"),
                errors={method: copy.deepcopy(errors) for method in ("wam", "hold", "constant_velocity")},
            )
        )
    return result


def evaluate(audit, short, expanded):
    return audit.eligibility(
        audit.summarize(short), audit.summarize(expanded), audit.group_summaries(short), audit.group_summaries(expanded)
    )


def test_normalized_scores_use_endpoint_si_errors_not_average_metric_ratios(audit):
    scores = audit.normalized_scores(rows())
    assert scores["8"] == dict(valid_endpoints=48, mean_squared_sum=3.0)
    assert scores["16"] == dict(valid_endpoints=96, mean_squared_sum=3.0)
    assert scores["40"] == dict(valid_endpoints=240, mean_squared_sum=3.0)


def test_global_gain_does_not_replace_task_episode_or_h40_gates(audit):
    short = rows(0.1, 0.001, 0.001)
    expanded = rows(0.05, 0.001, 0.001)
    assert evaluate(audit, short, expanded)["eligible_for_separately_registered_development_physics"]
    for index in range(2):
        expanded[index * 2]["errors"]["wam"]["position"] = [0.101] * 40
        result = evaluate(audit, short, expanded)
        assert result["eligible_for_separately_registered_development_physics"] == (index == 0)
    assert result["passing_episodes"] == 4
    assert all(c["passed"] for c in result["global_metric_checks"] + result["per_task_nonregression"])
    baseline = audit.summarize(short)
    candidate = copy.deepcopy(baseline)
    for h in ("8", "16"):
        candidate["metrics"][h]["wam"]["position_rmse_mm"] *= 0.5
    candidate["metrics"]["40"]["wam"]["rotation_rmse_rad"] *= 1.02000001
    result = audit.eligibility(baseline, candidate, audit.group_summaries(short), audit.group_summaries(short))
    assert not result["eligible_for_separately_registered_development_physics"]
    assert not next(
        c for c in result["global_metric_checks"] if c["horizon"] == 40 and c["metric"] == "rotation_rmse_rad"
    )["passed"]
