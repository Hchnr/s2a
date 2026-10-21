import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from s2a.artifacts import sha256, write_json
from s2a.libero_simulation import sha256_bytes
from scripts.verify_wam_acceptance import verify_case


@pytest.fixture
def saved_wam_case(tmp_path):
    """Complete two-step arm evidence with object progress and a saved request/image."""
    directory = tmp_path / "simulation/8400_masked/task0_init0_s42"
    directory.mkdir(parents=True)
    path = directory / "result.json"
    record = dict(name="task0_init0", task="put the bowl on the plate")
    c = dict(
        sampling_steps=35,
        guidance=1.5,
        shift=5.0,
        control_fps=20,
        inverse_weight=0.5,
        max_command_rate=5.0,
        opening_hysteresis=0.02,
        online_max_steps=2,
        online_chunk_size=4,
        online_execution_horizon=2,
    )
    variants = {
        name: dict(run=str(tmp_path / name), iteration="iter_000000001") for name in ("8400_masked", "8800_masked")
    }
    checkpoint = str(tmp_path / "8400_masked/checkpoints/iter_000000001/model")
    pose = np.eye(4).tolist()
    tool = tmp_path / "tool.json"
    write_json(tool, dict(source_from_tcp=pose))
    write_json(
        directory / "manifest.json",
        dict(
            config=dict(
                c,
                seed=42,
                task_text=record["task"],
                tool_contract=str(tool),
                action_domain="test_domain",
                return_video=False,
            ),
            record=record,
            source="wam",
            tool_sha256=sha256(tool),
            server_info=dict(
                checkpoint_path=checkpoint,
                ema_weights=False,
                acceptance_intervention=dict(active_slot_mask=True),
            ),
        ),
    )
    write_json(
        path,
        dict(
            record=record["name"],
            source="wam",
            checkpoint="8400_masked",
            sampling_seed=42,
            error=None,
            steps=2,
            expected_steps=2,
            success=False,
            terminal_success=False,
            chunks=1,
            object_progress={
                "akita_black_bowl_1": dict(grasp_contact_steps=1, max_height_gain_m=-0.001, max_displacement_m=0.002),
                "wine_bottle_1": dict(grasp_contact_steps=0, max_height_gain_m=0.0005, max_displacement_m=0.0005),
            },
        ),
    )
    steps = [
        dict(
            step=i,
            actual_dt=0.05,
            success=False,
            measured_source=pose,
            desired_source=pose,
            opening_target=1.0,
            objects={
                "akita_black_bowl_1": dict(
                    position=[0.0, 0.0, 1 - (i + 1) * 0.001],
                    grasp_contact=i == 0,
                    height_gain_m=-(i + 1) * 0.001,
                    displacement_m=(i + 1) * 0.001,
                ),
                "wine_bottle_1": dict(
                    position=[0.2, 0.3, 0.4005],
                    grasp_contact=False,
                    height_gain_m=0.0005,
                    displacement_m=0.0005,
                ),
            },
        )
        for i in range(2)
    ]
    (directory / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps))
    write_json(directory / "controller.json", {})
    pixels = np.zeros((960, 640, 3), dtype=np.uint8)
    image_path = directory / "input_0000.png"
    Image.fromarray(pixels).save(image_path)
    actions = np.zeros((4, 128))
    actions[:, [3, 7, 9]] = 1
    write_json(
        directory / "chunks.json",
        [
            dict(
                step=0,
                physics_unchanged=True,
                anchor_tcp=pose,
                actions=actions,
                targets_tcp=[pose, pose],
                image_sha256=sha256_bytes(pixels.tobytes()),
            )
        ],
    )
    write_json(
        directory / "request_0000.json",
        dict(
            name="s2a_task0_init0_s42_0",
            model_mode="wam",
            prompt=record["task"],
            vision_path=str(image_path),
            domain_name="franka_panda__parallel_gripper",
            image_size=960,
            action_chunk_size=4,
            fps=20,
            num_frames=5,
            seed=42,
            num_steps=35,
            guidance=1.5,
            shift=5.0,
            extra={
                "action128": dict(
                    action_domain="test_domain",
                    action_dim_mask=[True] * 10 + [False] * 118,
                    action_fps=20,
                    task_title=record["task"],
                    selected_roles=["auxiliary_0", "left_wrist"],
                    camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                    return_video=False,
                )
            },
        ),
    )
    write_json(
        directory / "response_0000.json",
        dict(
            status="success",
            outputs=[dict(content=dict(action=actions, action128=dict(checkpoint_path=checkpoint, seed=42)))],
        ),
    )
    return path, c, {record["name"]: record}, variants


def test_complete_wam_evidence_is_accepted(saved_wam_case):
    checked = verify_case(*saved_wam_case)
    assert checked["source"] == "wam" and checked["checkpoint"] == "8400_masked" and checked["steps"] == 2


@pytest.mark.parametrize("source", ["native", "oracle", "probe"])
def test_valid_object_progress_is_accepted_for_each_control_source(saved_wam_case, source):
    path, *arguments = saved_wam_case
    label = "probe_test" if source == "probe" else source
    directory = path.parent.parent.with_name(label) / path.parent.name
    directory.parent.mkdir(parents=True, exist_ok=True)
    path.parent.rename(directory)
    path = directory / path.name
    result = json.loads(path.read_text())
    result.update(source=source, checkpoint=None)
    write_json(path, result)
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest.update(source=source, probe=dict(name="test") if source == "probe" else None)
    write_json(manifest_path, manifest)
    assert verify_case(path, *arguments)["source"] == source


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [("grasp_contact_steps", 0), ("max_height_gain_m", 0.0), ("max_displacement_m", 0.003)],
)
def test_object_progress_summary_drift_is_rejected(saved_wam_case, field, wrong_value):
    path, *arguments = saved_wam_case
    result = json.loads(path.read_text())
    result["object_progress"]["akita_black_bowl_1"][field] = wrong_value
    write_json(path, result)
    with pytest.raises(ValueError, match="object progress summary differs from endpoint logs"):
        verify_case(path, *arguments)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("grasp_contact_steps", True),
        ("grasp_contact_steps", 1.0),
        ("max_height_gain_m", False),
        ("max_height_gain_m", float("nan")),
        ("max_displacement_m", float("inf")),
    ],
)
def test_object_progress_summary_requires_integer_count_and_finite_metrics(saved_wam_case, field, wrong_value):
    path, *arguments = saved_wam_case
    result = json.loads(path.read_text())
    result["object_progress"]["akita_black_bowl_1"][field] = wrong_value
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="object progress summary schema"):
        verify_case(path, *arguments)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("grasp_contact", 1),
        ("grasp_contact", "false"),
        ("height_gain_m", float("nan")),
        ("height_gain_m", True),
        ("displacement_m", float("inf")),
        ("displacement_m", -0.001),
        ("position", [0.0, 0.0]),
        ("position", [0.0, False, 1.0]),
        ("position", [0.0, 0.0, float("inf")]),
    ],
)
def test_object_progress_rejects_invalid_endpoint_values(saved_wam_case, field, wrong_value):
    path, *arguments = saved_wam_case
    trace = path.with_name("steps.jsonl")
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[0]["objects"]["akita_black_bowl_1"][field] = wrong_value
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="object progress endpoint"):
        verify_case(path, *arguments)


@pytest.mark.parametrize("location", ["summary", "first_endpoint", "second_endpoint"])
def test_object_progress_requires_the_same_object_set_for_every_endpoint(saved_wam_case, location):
    path, *arguments = saved_wam_case
    if location == "summary":
        result = json.loads(path.read_text())
        del result["object_progress"]["wine_bottle_1"]
        write_json(path, result)
    else:
        trace = path.with_name("steps.jsonl")
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        del rows[0 if location == "first_endpoint" else 1]["objects"]["wine_bottle_1"]
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="object progress object set"):
        verify_case(path, *arguments)


def test_another_checkpoint_directory_cannot_relabel_valid_evidence(saved_wam_case):
    path, *arguments = saved_wam_case
    old_label = path.parent.parent
    new_label = old_label.with_name("8800_masked")
    old_label.rename(new_label)
    with pytest.raises(ValueError, match="directory source/checkpoint"):
        verify_case(new_label / path.parent.name / path.name, *arguments)


@pytest.mark.parametrize("source", ["native", "oracle"])
def test_control_evidence_cannot_count_as_a_wam_checkpoint(saved_wam_case, source):
    path, *arguments = saved_wam_case
    result = json.loads(path.read_text())
    result.update(source=source, checkpoint=None)
    write_json(path, result)
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = source
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="directory source/checkpoint"):
        verify_case(path, *arguments)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("vision_path", "outputs/unrelated_future_image.png"),
        ("model_mode", "video"),
        ("domain_name", "another_robot"),
        ("image_size", 640),
        ("action_chunk_size", 40),
        ("fps", 999),
        ("num_frames", 41),
        ("num_steps", 1),
        ("guidance", 99),
        ("shift", 99),
        ("negative_prompt", "a different conditioning input"),
        ("extra.action128.action_domain", "another_domain"),
        ("extra.action128.action_dim_mask", [True] * 128),
        ("extra.action128.action_fps", 999),
        ("extra.action128.selected_roles", ["left_wrist", "auxiliary_0"]),
        ("extra.action128.camera_layout", [["left_wrist", "top", 640, 480], ["auxiliary_0", "bottom", 640, 480]]),
        ("extra.action128.return_video", True),
    ],
)
def test_request_conditioning_must_match_frozen_protocol_and_current_image(saved_wam_case, field, wrong_value):
    path, *arguments = saved_wam_case
    request_path = path.with_name("request_0000.json")
    request = json.loads(request_path.read_text())
    target = request
    keys = field.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = wrong_value
    write_json(request_path, request)
    with pytest.raises(ValueError, match="online (conditioning|input)"):
        verify_case(path, *arguments)


def test_summary_uses_the_eight_executed_endpoints_of_the_last_h16_chunk(saved_wam_case, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts import summarize_wam_independent as summary

    path, c, fixtures, variants = saved_wam_case
    root = path.parents[3]
    c.update(
        seed=42,
        tasks=[fixtures["task0_init0"]["task"]],
        checkpoints=[dict(name="8400_masked", **variants["8400_masked"])],
        sampling_seeds=[42],
        online_initial_states_per_task=1,
        online_max_steps=600,
        online_chunk_size=40,
        online_execution_horizon=16,
        acceptance=dict(stable_success_last_steps=40, reliable_overall_fraction=0.8, reliable_each_task_fraction=0.5),
        qualification_position_tolerance_m=0.01,
        qualification_rotation_tolerance_rad=0.05,
        fixture_source="prior_fixture.json",
    )
    fixtures["task0_init0"].update(task_index=0, initial_index=0)
    write_json(root / "simulation/fixtures.json", dict(config=c, exclusions={}, records=list(fixtures.values())))
    weights_path = root / "weights.json"
    checkpoint = variants["8400_masked"]
    write_json(
        weights_path,
        dict(
            checkpoints=[
                dict(
                    name="8400_masked",
                    checkpoint=str(Path(checkpoint["run"]) / "checkpoints" / checkpoint["iteration"] / "model"),
                    files={},
                )
            ]
        ),
    )
    manifest = json.loads(path.with_name("manifest.json").read_text())
    manifest["config"].update(action_chunk_size=40, execution_horizon=16)
    pose = np.eye(4).tolist()
    steps = [dict(success=False, desired_source=pose, after_source=pose) for _ in range(600)]
    for label in ("native", "oracle", "8400_masked"):
        case = root / "simulation" / label / path.parent.name
        case.mkdir(parents=True, exist_ok=True)
        write_json(case / "manifest.json", manifest)
        write_json(
            case / "result.json",
            dict(
                record="task0_init0",
                sampling_seed=42,
                source="wam" if label == "8400_masked" else label,
                object_progress={},
            ),
        )
        (case / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps))
    chunks = []
    for start in range(0, 600, 16):
        actions = np.ones((40, 128))
        if start == 592:
            actions[8:16, 9] = 0
        chunks.append(dict(step=start, actions=actions))
    write_json(path.with_name("chunks.json"), chunks)
    monkeypatch.setattr(summary, "load_config", lambda _: (c, root))
    # The artifact verifier has separate end-to-end tests above. Exercise the
    # summary's budget-to-prefix wiring here without constructing 38 PNGs.
    monkeypatch.setattr(summary, "verify_case", lambda *_: {})
    monkeypatch.setattr(
        sys, "argv", ["summarize_wam_independent.py", "--seed", "42", "--fingerprints", str(weights_path)]
    )
    summary.main()
    cases = json.loads((root / "summary.json").read_text())["cases"]
    opening = next(row for row in cases if row["label"] == "8400_masked")["opening_diagnostics"]
    assert opening["executed_prefix_min"] == 1
    assert opening["deferred_closure_chunks"] == 1
