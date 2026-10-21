"""Independent022 caption-pair and raw-state audit; never requests a model/physics.

SciPy state reconstruction is the unchanged, independently implemented011
helper. Registration, actual-caption evidence, groups and022 eligibility are
separate from the inference coordinator and its pooled metrics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from audit_wam_sampler_steps import Evidence, check_saved_metrics, integrate_and_score, pool

from s2a.artifacts import REPO

STUDY = REPO / "outputs/wam_stability_20261021/static_command_state"
BASELINE = REPO / "outputs/wam_stability_20261021/earlier_screening/8400_g10"
PLAN = "plans/v0.0.1/022_static_command_state_20261021.md"
HELPER_SHA = "12329768297939fe6b15b4127aa6b65e144578fd72b0e896dd306f9e88b10069"
PREFIXES = (8, 16, 40)
METRICS = ("position_rmse_mm", "rotation_rmse_rad", "opening_mae")
MODES = ("reproduction", "expanded")


def caption_pair(short, expanded, *, task, command):
    """Compare actual batch captions, including empty renderer schema fields."""
    assert len(short) == len(expanded) == 1 and isinstance(short[0], str) and isinstance(expanded[0], str)
    left, right = json.loads(short[0]), json.loads(expanded[0])
    assert isinstance(left, dict) and isinstance(right, dict)
    assert left["task_title"] == right["task_title"] == task and left["command"] == task
    assert right["command"] == command
    assert {k: v for k, v in left.items() if k != "command"} == {k: v for k, v in right.items() if k != "command"}
    assert left.get("task_plan", []) == right.get("task_plan", []) == []
    assert "current_subtask" not in left and "current_subtask" not in right
    assert left["action_frequency_hz"] == 20 and left["duration"] == "2s"
    assert left["camera_layout"] == [["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]]
    return dict(
        only_command_differs=True,
        task=task,
        command=command,
        renderer_empty_task_plan_preserved=("task_plan" in left),
        short_sha256=hashlib.sha256(short[0].encode()).hexdigest(),
        expanded_sha256=hashlib.sha256(expanded[0].encode()).hexdigest(),
    )


def full_actions(response, transform, *, checkpoint, seed, domain):
    """Check every raw/normalized slot and independently invert the saved affine."""
    assert response["status"] == "success" and len(response["outputs"]) == 1
    content = response["outputs"][0]["content"]
    meta = content["action128"]
    raw, normalized = np.asarray(content["action"], float), np.asarray(meta["normalized_action"], float)
    assert all(a.shape == (40, 128) and np.isfinite(a).all() and not np.any(a[:, 10:]) for a in (raw, normalized))
    assert meta["checkpoint_path"] == checkpoint and type(meta["seed"]) is int and meta["seed"] == seed
    assert meta["action_domain"] == domain and meta["action_units"] == "raw_physical_or_source_units"
    assert meta["action_fps"] == 20 and meta["action_dim_mask"] == [True] * 10 + [False] * 118
    assert all(type(v) is bool for v in meta["action_dim_mask"])
    np.testing.assert_array_equal(meta["action_timestamps_seconds"], np.arange(40) / 20)
    decoded = np.asarray(normalized, np.float32).copy()
    mask = np.asarray(transform["normalization_mask"], bool)
    scale, bias = np.asarray(transform["scale"], np.float32), np.asarray(transform["bias"], np.float32)
    assert mask.shape == scale.shape == bias.shape == (128,) and np.isfinite(scale).all() and np.isfinite(bias).all()
    assert np.all(scale[mask] != 0)
    decoded[:, mask] = (decoded[:, mask] - bias[mask]) / scale[mask]
    np.testing.assert_allclose(decoded, raw, atol=2e-7, rtol=0)
    return raw, normalized


def exact_reproduction(current, baseline):
    """No tolerance, active-slot slicing, horizon cropping or float32 recasting."""
    for field in ("action", "normalized_action"):

        def values(response):
            content = response["outputs"][0]["content"]
            return np.asarray(content[field] if field == "action" else content["action128"][field])

        left, right = values(current), values(baseline)
        assert left.shape == right.shape == (40, 128) and np.array_equal(left, right), field
    return True


def receipt_pair(prepared, predicted, request, sample, response):
    """Require the persisted actual prepare/predict handoff, not just request text."""
    assert prepared["stage"] == "actual_prepare_return_before_predict"
    assert predicted == dict(prepared, stage="actual_predict_entry_same_prepared_and_batch")
    assert prepared["request"] == request and prepared["name"] == request["name"]
    assert type(prepared["seed"]) is int and prepared["seed"] == request["seed"]
    assert prepared["image_sha256"] == sample["image_sha256"] and prepared["active_slot_mask"] is True
    mask = prepared["actual_action_dim_mask"]
    assert mask == [True] * 10 + [False] * 118 and all(type(v) is bool for v in mask)
    caption = prepared["caption"]
    assert len(caption) == 1 and isinstance(caption[0], str)
    assert (
        response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"]
        == hashlib.sha256(caption[0].encode()).hexdigest()
    )
    return caption


def read_case(item, sample, config, transform, domain, evidence):
    request = item["request"]
    mode, name, seed = item["mode"], request["name"], request["seed"]
    case = REPO / config["output_dir"] / "state_queries/8400_masked" / name
    result, raw = evidence.json(case / "result.json"), evidence.json(case / "response.json")
    assert evidence.json(case / "request.json") == request
    assert result["response_sha256"] == evidence.hashes[str(case / "response.json")]
    for key, value in dict(
        name=name,
        checkpoint="8400_masked",
        sample=sample["name"],
        episode=sample["episode"],
        task=sample["task"],
        phase=sample["phase"],
        split="diagnostic",
        anchor_frame=sample["anchor_frame"],
        sampling_seed=seed,
        horizon=40,
        prompt_mode="task_only" if mode == "reproduction" else "published_caption",
    ).items():
        assert result[key] == value, key
    checkpoint = str(Path(config["checkpoints"][0]["run"]) / "checkpoints/iter_000008400/model")
    actions, _ = full_actions(raw, transform, checkpoint=checkpoint, seed=seed, domain=domain)
    args = raw["args"]
    for key in (
        "name",
        "seed",
        "prompt",
        "vision_path",
        "extra",
        "fps",
        "num_frames",
        "num_steps",
        "guidance",
        "shift",
        "action_chunk_size",
        "model_mode",
        "image_size",
        "domain_name",
    ):
        assert args[key] == request[key], key
    assert args["condition_frame_indexes_vision"] == [0] and args["action_path"] is None
    assert not args["native_prompt_upsampling"] and not args["negative_prompt"]
    output_dir = Path(args["output_dir"])
    assert output_dir.parent == STUDY / "service" and output_dir.name.startswith("generate_")
    assert len(output_dir.name) == len("generate_") + 32
    int(output_dir.name.removeprefix("generate_"), 16)
    assert evidence.json(output_dir / "sample_args.json") == args
    assert evidence.json(output_dir / "sample_outputs.json") == raw
    if mode == "reproduction":
        assert request == evidence.json(item["original_request"])
        exact_reproduction(raw, evidence.json(item["original_response"]))
    receipt = evidence.json(STUDY / "text_input_receipts" / f"{name}.prepare.json")
    caption = receipt_pair(
        receipt, evidence.json(STUDY / "text_input_receipts" / f"{name}.predict.json"), request, sample, raw
    )
    evidence.check(sample["arrays_path"], sample["arrays_sha256"])
    evidence.check(sample["image_path"], sample["image_sha256"])
    with np.load(sample["arrays_path"]) as data:
        count, errors, generated = integrate_and_score(np.asarray(data["states"], float), sample, actions)
    assert count == result["valid_future_steps"]
    check_saved_metrics(errors["wam"], result["metrics"])
    for method in ("hold", "constant_velocity"):
        check_saved_metrics(errors[method], result["baselines"][method])
    return dict(
        name=name,
        mode=mode,
        sample=sample,
        seed=seed,
        valid_future_steps=count,
        errors=errors,
        generated=generated,
        actual_caption=caption,
        service_request_dir=str(output_dir),
        response_sha256=result["response_sha256"],
        original_full128_reproduction_exact=mode == "reproduction",
    )


def verify_input_source(source):
    evidence = Evidence()
    evidence.check(REPO / "scripts/audit_wam_sampler_steps.py", HELPER_SHA)
    return evidence.json(source), evidence


def normalized_scores(rows):
    """Mean of the registered per-endpoint sum in m, radians, normalized opening."""
    result = {}
    for horizon in PREFIXES:
        values = []
        for row in rows:
            errors = row["errors"]["wam"]
            p, r, g = (np.asarray(errors[key][:horizon], float) for key in ("position", "rotation", "opening"))
            assert p.shape == r.shape == g.shape and len(p) and all(np.isfinite(v).all() for v in (p, r, g))
            values.extend(((p / 0.01) ** 2 + (r / 0.05) ** 2 + (g / 0.1) ** 2).tolist())
        result[str(horizon)] = (
            None if not values else dict(valid_endpoints=len(values), mean_squared_sum=float(np.mean(values)))
        )
    return result


def summarize(rows):
    return dict(metrics=pool(rows), normalized_scores=normalized_scores(rows))


def group_summaries(rows):
    return {
        field: {
            str(value): summarize([r for r in rows if r["sample"][field] == value])
            for value in sorted({r["sample"][field] for r in rows}, key=str)
        }
        for field in ("task", "episode", "phase")
    }


def eligibility(short, expanded, short_groups, expanded_groups):
    """022 fixed gates; no added numerical slack, no automatic physical promotion."""
    comparisons, short_improvement = [], []
    for h in PREFIXES:
        reference, candidate = (v["metrics"][str(h)]["wam"] for v in (short, expanded))
        for metric in METRICS:
            factor = 1.02 if h == 40 else 1.0
            comparisons.append(
                dict(
                    horizon=h,
                    metric=metric,
                    reference=reference[metric],
                    candidate=candidate[metric],
                    maximum_ratio=factor,
                    passed=candidate[metric] <= factor * reference[metric],
                )
            )
        if h != 40:
            short_improvement.append(
                dict(
                    horizon=h,
                    passed=any(reference[key] > 0 and candidate[key] <= 0.9 * reference[key] for key in METRICS),
                )
            )
    tasks, episodes = [], []
    assert set(short_groups["task"]) == set(expanded_groups["task"]) and len(short_groups["task"]) == 3
    assert set(short_groups["episode"]) == set(expanded_groups["episode"]) and len(short_groups["episode"]) == 6
    for field, factor, destination in (("task", 1.0, tasks), ("episode", 1.02, episodes)):
        for name in sorted(short_groups[field]):
            checks = []
            for h in (8, 16):
                left, right = (v[field][name]["normalized_scores"][str(h)] for v in (short_groups, expanded_groups))
                assert left["valid_endpoints"] == right["valid_endpoints"]
                checks.append(
                    dict(
                        horizon=h,
                        reference=left["mean_squared_sum"],
                        candidate=right["mean_squared_sum"],
                        passed=right["mean_squared_sum"] <= factor * left["mean_squared_sum"],
                    )
                )
            destination.append(dict(name=name, checks=checks, passed=all(c["passed"] for c in checks)))
    passed = (
        all(c["passed"] for c in comparisons + short_improvement + tasks) and sum(c["passed"] for c in episodes) >= 5
    )
    return dict(
        eligible_for_separately_registered_development_physics=passed,
        global_metric_checks=comparisons,
        each_short_horizon_has_ten_percent_improvement=short_improvement,
        per_task_nonregression=tasks,
        episodes_with_both_short_horizons_within102_percent=episodes,
        passing_episodes=sum(c["passed"] for c in episodes),
        required_passing_episodes=5,
        added_numerical_tolerance=0,
        new_physics_authorized=False,
        fresh_evaluation_authorized=False,
        interpretation="State-only eligibility is not task success or authorization to run physics.",
    )
