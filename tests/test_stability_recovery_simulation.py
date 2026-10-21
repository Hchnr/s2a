"""Causal reference clocks and strict response/protocol identity; no physics."""

import copy
import importlib.util
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from s2a.action_reference import decode_action_targets
from s2a.artifacts import REPO, sha256
from s2a.geometry import matrix_to_rot6d
from s2a.gripper_intent_hold import GripperIntentHoldConfig
from s2a.reference_governor import ReferenceGovernorConfig
from s2a.stability_recovery_simulation import recovery_variant, sample_model_reference, validate_wam_response
from s2a.wam_regularization import attenuate_wam_rotation


def response():
    actions = np.zeros((4, 128))
    actions[:, :3] = [0.01, 0, 0]
    actions[:, 3:9] = matrix_to_rot6d(np.eye(3))
    actions[:, 9] = 0.4
    return dict(
        status="success",
        args=dict(seed=42, action_chunk_size=4),
        outputs=[
            dict(
                content=dict(
                    action=actions.tolist(),
                    action128=dict(
                        checkpoint_path="outputs/test_checkpoint/model",
                        seed=42,
                        action_domain="test/eef",
                        action_dim_mask=[True] * 10 + [False] * 118,
                        action_fps=20,
                        action_timestamps_seconds=[0, 0.05, 0.1, 0.15],
                        action_units="raw_physical_or_source_units",
                    ),
                )
            )
        ],
    )


def validate(value):
    return validate_wam_response(
        value, checkpoint_path="outputs/test_checkpoint/model", seed=42, domain="test/eef", fps=20, horizon=4
    )


def test_every_response_authenticates_identity_not_just_array_shape():
    value = response()
    actions, identity = validate(value)
    assert actions.shape == (4, 128) and identity["seed"] == 42
    for key, wrong in [
        ("checkpoint_path", "outputs/other/model"),
        ("seed", 43),
        ("action_domain", "wrong"),
        ("action_fps", 30),
        ("action_units", "normalized"),
        ("action_timestamps_seconds", [0, 0.05, 0.1, 0.2]),
        ("action_dim_mask", [1] * 10 + [0] * 118),
    ]:
        changed = copy.deepcopy(value)
        changed["outputs"][0]["content"]["action128"][key] = wrong
        with pytest.raises(ValueError):
            validate(changed)
    for value in (float("nan"), 0.01):
        changed = response()
        changed["outputs"][0]["content"]["action"][0][100] = value
        with pytest.raises(ValueError):
            validate(changed)


def test_raw_effective_model_clock_is_distinct_from_an_admitted_bridge():
    actions, _ = validate(response())
    raw = actions.copy()
    raw[:, 3:9] *= 0.99
    np.testing.assert_array_equal(attenuate_wam_rotation(raw, alpha=1), raw)
    poses, openings = decode_action_targets(np.eye(4), actions, horizon=4)
    for clock, expected_x, expected_opening in [
        (0, 0, 0.8),
        (0.025, 0.005, 0.6),
        (0.05, 0.01, 0.4),
        (0.2, 0.04, 0.4),
        (1, 0.04, 0.4),
    ]:
        pose, opening = sample_model_reference(
            np.eye(4), poses, openings, model_time_s=clock, dt=0.05, anchor_opening=0.8
        )
        assert pose[0, 3] == pytest.approx(expected_x)
        assert opening == pytest.approx(expected_opening)
    # Model clock0 denotes the new measured anchor even if admitted bridge
    # tracking is still starting from the preceding chunk's desired target.
    prior_admitted = np.eye(4)
    prior_admitted[0, 3] = 0.2
    pose, _ = sample_model_reference(np.eye(4), poses, openings, model_time_s=0, dt=0.05, anchor_opening=0.8)
    assert not np.array_equal(pose, prior_admitted)


def config():
    return dict(
        gripper_intent_hold=asdict(GripperIntentHoldConfig()),
        reference_governor=asdict(ReferenceGovernorConfig()),
        variants=[
            dict(
                name="candidate",
                rotation="response",
                gripper="intent_hold",
                governor="recovering",
                model_rotation_alpha=1,
            )
        ],
    )


def test_variant_rejects_old_unqualified_paths_and_oracle_reference_changes():
    c = config()
    assert recovery_variant(c, "candidate", source="oracle")["model_rotation_alpha"] == 1
    for key, wrong in [
        ("rotation", "bounded"),
        ("gripper", "tracking"),
        ("governor", True),
        ("model_rotation_alpha", True),
        ("model_rotation_alpha", 0.9),
    ]:
        changed = copy.deepcopy(c)
        changed["variants"][0][key] = wrong
        with pytest.raises(ValueError):
            recovery_variant(changed, "candidate", source="wam")
    c["variants"][0]["model_rotation_alpha"] = 0
    assert recovery_variant(c, "candidate", source="wam")["model_rotation_alpha"] == 0
    with pytest.raises(ValueError):
        recovery_variant(c, "candidate", source="oracle")
    c = config()
    c["gripper_intent_hold"]["release_margin"] = 0.08
    with pytest.raises(ValueError):
        recovery_variant(c, "candidate", source="wam")


def test_protocol_dry_run_has_no_physics_model_or_output_side_effects(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(dict(records=[dict(name="scene")])) + "\n")
    c = dict(
        config(),
        seed=42,
        records=["scene"],
        sampling_seeds=[42],
        control_fps=20,
        online_max_steps=600,
        online_chunk_size=40,
        online_execution_horizon=40,
        max_observation_interval_steps=80,
        stable_success_last_steps=40,
        fixture=str(fixture),
        fixture_sha256=sha256(fixture),
        output_dir=str(tmp_path / "not_created"),
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(c) + "\n")
    script = REPO / "scripts/run_wam_recovery_matrix.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONPATH=str(REPO / "src"))
    result = subprocess.run(
        [
            str(Path(".venv/bin/python").absolute()),
            str(script),
            "--config",
            str(path),
            "--seed",
            "42",
            "--source",
            "oracle",
            "--variant",
            "candidate",
            "--dry-run",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["physics_started"] is False
    assert not Path(c["output_dir"]).exists()
    spec = importlib.util.spec_from_file_location("recovery_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c["checkpoints"] = [dict(name="ck", gpu=0, port=9001)]
    path.write_text(json.dumps(c) + "\n")
    with pytest.raises(ValueError, match="physical GPU"):
        module.protocol(path, source="wam", variant_name="candidate", checkpoint="ck")
