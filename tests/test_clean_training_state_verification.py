"""Pure CPU synthetic state/receipt QA; no real validation archive is decoded."""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO
from s2a.wam_clean_capture import CleanCapturePoint
from s2a.wam_clean_compact import CompactCaptureSink, CompactRequest


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_wam_clean_training_state")


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "tests"))
    return importlib.import_module("test_wam_clean_compact")


def reference(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def columns(matrix):
    return np.r_[matrix[:, 0], matrix[:, 1]]


def states_and_actions(frame=4):
    states = np.zeros((frame + 41, 128))
    actions = np.zeros((40, 128))
    attitude = Rotation.from_euler("z", np.pi / 2).as_matrix()
    increment = Rotation.from_euler("x", 0.01).as_matrix()
    for i in range(frame + 1):
        states[i, :3] = [0.001 * i, 0, 0]
        states[i, 3:9] = columns(attitude)
        states[i, 9] = 0.5
    for i in range(40):
        actions[i, :3] = [0.001, 0, 0]
        actions[i, 3:9] = columns(increment)
        actions[i, 9] = 0.5
        states[frame + i + 1, :3] = states[frame + i, :3] + attitude @ actions[i, :3]
        attitude = attitude @ increment
        states[frame + i + 1, 3:9] = columns(attitude)
        states[frame + i + 1, 9] = 0.5
    return states, dict(anchor_frame=frame, valid_future_steps=40), actions


def test_right_local_translation_rotation_and_absolute_opening(checker):
    states, anchor, actions = states_and_actions()
    errors = checker.integrate(states, anchor, actions)
    assert max(errors["wam"]["position"]) < 1e-14
    assert max(errors["wam"]["rotation"]) < 1e-14
    assert not np.any(errors["wam"]["opening"])
    assert errors["hold"]["position"][0] == pytest.approx(0.001)
    assert errors["hold"]["rotation"][0] == pytest.approx(0.01)
    assert errors["constant_velocity"]["position"][0] == pytest.approx(np.sqrt(2) * 0.001)
    actions[:, 9] = 2
    errors = checker.integrate(states, anchor, actions)
    assert np.array_equal(errors["wam"]["opening"], np.full(40, 0.5))


@pytest.mark.parametrize("frame", [0, 4, 17])
def test_independent_integration_matches_frozen011_scoring(checker, frame):
    legacy = importlib.import_module("audit_wam_sampler_steps")
    states, anchor, actions = states_and_actions(frame)
    errors = checker.integrate(states, anchor, actions)
    count, original, _ = legacy.integrate_and_score(states, anchor, actions)
    assert count == 40
    for method in errors:
        for component in errors[method]:
            np.testing.assert_allclose(errors[method][component], original[method][component], rtol=0, atol=2e-12)


@pytest.mark.parametrize("fault", ["incomplete", "mask", "nan", "degenerate"])
def test_scoring_refuses_incomplete_or_invalid_reference_geometry(checker, fault):
    states, anchor, actions = states_and_actions()
    if fault == "incomplete":
        states = states[:-1]
    elif fault == "mask":
        actions[0, 10] = 1
    elif fault == "nan":
        states[5, 2] = np.nan
    else:
        actions[0, 3:9] = 0
    with pytest.raises(ValueError):
        checker.integrate(states, anchor, actions)


def metric_rows(checker):
    rows = []
    for episode in checker.EPISODES:
        for anchor in range(4):
            for seed in (42, 43):
                for variant, error in (("raw", 0.01), ("w2", 0.008)):
                    errors = {
                        method: {k: [e] * 40 for k in ("position", "rotation", "opening")}
                        for method, e in (("wam", error), ("hold", 0.02), ("constant_velocity", 0.03))
                    }
                    rows.append(
                        dict(
                            name=f"{episode}_{anchor}_{seed}_{variant}",
                            episode=episode,
                            task_index=checker.EPISODES.index(episode) // 2,
                            sampling_seed=seed,
                            variant=variant,
                            errors=errors,
                        )
                    )
    return rows


def test_full96_pooled_metrics_keep_horizons_tasks_episodes_and_seeds(checker):
    groups, gate = checker.diagnosis(metric_rows(checker))
    assert gate["passed"] is True
    assert groups["pooled"]["8"]["raw"]["valid_endpoints"] == 48 * 8
    assert groups["pooled"]["40"]["w2"]["valid_endpoints"] == 48 * 40
    assert groups["pooled"]["16"]["raw"]["joint_score"] == pytest.approx(1 + 0.04 + 0.01)
    assert set(groups["by_task"]) == {"0", "1", "2"}
    assert set(groups["by_seed"]) == {"42", "43"}


@pytest.mark.parametrize("fault", [None, "joint", "baseline_tie", "component", "two_episodes", "outlier", "seed"])
def test_original028_gate_thresholds_match_independent_legacy_gate(checker, fault):
    groups, _ = checker.diagnosis(metric_rows(checker))
    if fault == "joint":
        groups["pooled"]["8"]["w2"]["joint_score"] = 0.900001 * groups["pooled"]["8"]["raw"]["joint_score"]
    elif fault == "baseline_tie":
        groups["pooled"]["8"]["hold"]["joint_score"] = groups["pooled"]["8"]["w2"]["joint_score"]
    elif fault == "component":
        groups["pooled"]["40"]["w2"]["position_rmse_m"] = 0.013
    elif fault == "two_episodes":
        for value in list(groups["by_episode"].values())[:2]:
            value["8"]["w2"]["joint_score"] = 1.06 * value["8"]["raw"]["joint_score"]
    elif fault == "outlier":
        value = groups["by_episode"][str(checker.EPISODES[0])]
        value["16"]["w2"]["joint_score"] = 1.250001 * value["16"]["raw"]["joint_score"]
    elif fault == "seed":
        value = groups["by_seed"]["43"]
        value["8"]["w2"]["joint_score"] = 1.050001 * value["8"]["raw"]["joint_score"]
    actual = checker.state_gate(groups)
    old = copy.deepcopy(groups)
    for value in [old["pooled"], *old["by_episode"].values(), *old["by_seed"].values()]:
        for horizon in value.values():
            horizon["adapt"] = horizon.pop("w2")
    expected = importlib.import_module("verify_wam_readout").independent_state_gate(old)
    assert actual == expected
    assert actual["passed"] is (fault is None)


def validation_fixture(tmp_path, checker, *, length=50):
    anchors = []
    for episode in checker.EPISODES:
        states = np.zeros((length, 128))
        states[:, 3] = states[:, 7] = 1
        states[:, 9] = 0.5
        mask = np.broadcast_to(checker.CAPTURE.MASK, states.shape).copy()
        path = tmp_path / f"episode{episode}.npz"
        np.savez(path, states=states, state_mask=mask, future_rgb=np.ones(1))
        for index in range(4):
            frame = index * (length - 41) // 3
            anchors.append(
                dict(
                    name=f"ep{episode}_a{index}",
                    episode=episode,
                    anchor_index=index,
                    anchor_frame=frame,
                    arrays_ref=reference(path),
                    initial_sync_certified=False,
                    measured_state128=states[frame].tolist(),
                    state_mask128=checker.CAPTURE.MASK,
                    task_index=checker.EPISODES.index(episode) // 2,
                )
            )
    return dict(validation_anchors=anchors)


def test_validation_decoder_is_unreachable_until_numerical_and_exit_gate(tmp_path, checker, monkeypatch):
    prepared = validation_fixture(tmp_path, checker)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("future archive decoded before complete numerical gate")
    )
    with pytest.raises(ValueError, match="complete numerical/exit gate"):
        checker.validation_labels(checker.CAPTURE.Evidence(), prepared, numerical_contract_passed=False)


def test_only_six_registered_validation_state_and_mask_arrays_are_decoded(tmp_path, checker, monkeypatch):
    prepared = validation_fixture(tmp_path, checker)
    original = np.load
    decoded = []

    class SelectiveArchive:
        def __init__(self, path, **kwargs):
            self.archive = original(path, **kwargs)

        def __enter__(self):
            return self

        def __getitem__(self, key):
            assert key in ("states", "state_mask")
            decoded.append(key)
            return self.archive[key]

        def __exit__(self, *args):
            self.archive.close()

    monkeypatch.setattr(np, "load", SelectiveArchive)
    assert len(checker.validation_labels(checker.CAPTURE.Evidence(), prepared, numerical_contract_passed=True)) == 6
    assert decoded == ["states", "state_mask"] * 6


@pytest.mark.parametrize("fault", ["missing_episode", "anchor", "current", "mask", "history", "synchronization"])
def test_validation_source_mask_and_sync_boundary_are_checked(tmp_path, checker, fault):
    prepared = validation_fixture(tmp_path, checker, length=200 if fault == "history" else 50)
    first = prepared["validation_anchors"][0]
    if fault == "missing_episode":
        prepared["validation_anchors"].pop()
    elif fault == "anchor":
        first["anchor_frame"] += 1
    elif fault == "current":
        first["measured_state128"][0] = 1
    elif fault in ("mask", "history"):
        path = tmp_path / f"episode{checker.EPISODES[0]}.npz"
        with np.load(path) as source:
            states, mask = source["states"].copy(), source["state_mask"].copy()
        mask[49 if fault == "history" else 1, 0] = False
        np.savez(path, states=states, state_mask=mask)
        for anchor in prepared["validation_anchors"][:4]:
            anchor["arrays_ref"] = reference(path)
    else:
        first["initial_sync_certified"] = True
    with pytest.raises(ValueError):
        checker.validation_labels(checker.CAPTURE.Evidence(), prepared, numerical_contract_passed=True)


def online(protocol, step=0, adapted=False):
    original = protocol.point(step, adapted=adapted)
    metadata = dict(original.metadata, purpose="evaluation", noise_seed=42)
    arrays = dict(original.arrays)
    arrays["condition_vision"] = np.zeros((protocol.VISION[2], 1, 1), np.float32)
    arrays["condition_vision"][0] = 1
    return CleanCapturePoint(metadata, arrays)


def test_complete35_evaluation_compact_nodes_and037_raw_fullpoint_regression(tmp_path, checker, protocol):
    directory = tmp_path / "000_zero"
    writer = CompactCaptureSink(
        directory / "compact",
        CompactRequest("native-anchor0", 42, "evaluation", "state", "zero", None, "a" * 64),
        expected_vision_shape=protocol.VISION,
        expected_grid=protocol.GRID,
    )
    historical = tmp_path / "037_raw"
    historical.mkdir()
    points, steps = [], []
    for i in range(35):
        actual = online(protocol, i)
        writer(actual)
        array_path = historical / f"point_{i:02d}.npz"
        np.savez(array_path, **actual.arrays)
        meta = dict(
            actual.metadata,
            arrays=reference(array_path),
            array_index={k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in actual.arrays.items()},
        )
        json_path = historical / f"point_{i:02d}.json"
        json_path.write_text(json.dumps(meta))
        points.append(reference(json_path))
        steps.append(
            dict(
                step_index=i,
                solver_sigma=protocol.GRID[i][0],
                solver_timestep=protocol.GRID[i][1],
                returned_original_velocity_object=True,
            )
        )
    ledger_ref = writer.finish()
    evidence = checker.CAPTURE.Evidence()
    ledger = evidence.json(ledger_ref)
    header = evidence.json(ledger["request_ref"])
    first, changed = checker.compact_points(
        evidence,
        ledger,
        header,
        directory,
        dict(steps=steps),
        head=None,
        historical=dict(points=points),
        vision_shape=protocol.VISION,
        time_vision_shape=(2,),
    )
    assert first["metadata"]["step_index"] == 0 and changed == 0
    bad = copy.deepcopy(first)
    bad["source_arrays"]["head_base_full128"]["snapshot_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="zero fullpoint"):
        checker.zero_sources(evidence, bad, points[0], historical, 0)


@pytest.mark.parametrize("fault", [None, "future_purpose", "wrong_w2"])
def test_state_point_uses_independent_fixed_w2_math(tmp_path, checker, protocol, fault):
    directory = tmp_path / "state"
    writer = CompactCaptureSink(
        directory / "compact",
        CompactRequest("native-anchor0", 42, "evaluation", "state", "anchor", "b" * 64, "a" * 64),
        expected_vision_shape=protocol.VISION,
        expected_grid=protocol.GRID,
    )
    saved = writer(online(protocol, adapted=True))
    evidence = checker.CAPTURE.Evidence()
    record = evidence.json(saved)
    arrays = evidence.arrays(record["arrays_ref"], directory / "compact/step00.npz")
    header = evidence.json(record["request_ref"])
    coefficient = np.zeros((1291, 10))
    coefficient[-1] = protocol.GRID[0][0] / 8
    if fault == "future_purpose":
        record["metadata"]["purpose"] = "training"
    elif fault == "wrong_w2":
        coefficient[-1] += 0.01
    step = dict(
        step_index=0,
        solver_sigma=protocol.GRID[0][0],
        solver_timestep=protocol.GRID[0][1],
        returned_original_velocity_object=False,
    )
    if fault is None:
        assert (
            checker.point_math(
                record,
                arrays,
                header,
                step,
                head=(coefficient, np.zeros(1290), 1),
                vision_shape=protocol.VISION,
                time_vision_shape=(2,),
            )
            == 400
        )
    else:
        with pytest.raises(ValueError):
            checker.point_math(
                record,
                arrays,
                header,
                step,
                head=(coefficient, np.zeros(1290), 1),
                vision_shape=protocol.VISION,
                time_vision_shape=(2,),
            )


def test_state_cli_rejects_incomplete_execution_without_reading_gt_or_writing_pass(tmp_path):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps(dict(schema_version="s2a.clean-training-preparation.v1")))
    registration = tmp_path / "registration.json"
    registration.write_text(
        json.dumps(
            dict(
                schema_version="s2a.clean-training-execution-registration.v1",
                prepared_ref=reference(prepared),
                complete=False,
                passed=False,
            )
        )
    )
    command = [sys.executable, str(REPO / "scripts/verify_wam_clean_training_state.py")]
    for name in (
        "prepared",
        "execution-registration",
        "sampling",
        "worker-launch",
        "worker-exit",
        "worker-release",
        "gpu-after",
    ):
        path = registration if name == "execution-registration" else prepared
        command.extend(["--" + name, str(path), "--" + name + "-sha256", reference(path)["sha256"]])
    output = tmp_path / "must_not_exist.json"
    command += ["--output", str(output), "--seed", "42"]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "independent final execution registration required" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("state_passed", [False, True])
def test_registered_development_eligibility_is_separate_from_state_proximity(checker, state_passed):
    verdict = checker.diagnostic_verdict(dict(passed=state_passed), numerical_contract_passed=True)
    assert verdict["state_gate_passed"] is state_passed
    assert verdict["eligible_for_development"] is True
    assert verdict["physics_authorized"] is verdict["fresh_authorized"] is False
    with pytest.raises(ValueError, match="numerical contract"):
        checker.diagnostic_verdict(dict(passed=state_passed), numerical_contract_passed=False)


@pytest.mark.parametrize("fault", [None, "wait", "signal", "birth", "gpu", "uuid", "interpreter", "w2"])
def test_state_lifecycle_requires_real_owned_wait_and_qualified_w2_command(tmp_path, checker, fault):
    prepared_ref = dict(path="/prepared.json", sha256="a" * 64)
    registration_ref = dict(path="/registration.json", sha256="b" * 64)
    actual = dict(
        w2_fit_ref=dict(path="/w2.json", sha256="c" * 64),
        w2_verification_ref=dict(path="/w2_verified.json", sha256="d" * 64),
    )
    command = ["qualified-python", "scripts/sample_wam_clean_training_state.py"]
    for name, ref in (
        ("prepared", prepared_ref),
        ("execution-registration", registration_ref),
        ("w2-fit", actual["w2_fit_ref"]),
        ("w2-verification", actual["w2_verification_ref"]),
    ):
        command.extend(["--" + name, ref["path"], "--" + name + "-sha256", ref["sha256"]])
    command.extend(["--gpu-uuid", checker.GPU, "--seed", "42"])
    launch = dict(
        child=dict(pid=123, start_ticks=456),
        command=command,
        selected_environment=dict(CUDA_VISIBLE_DEVICES=checker.GPU, LD_LIBRARY_PATH=""),
    )
    exit_record = dict(returncode=0, actual_wait=True, automatic_retry=False)
    release = dict(
        pid=123,
        start_ticks=456,
        reaped=True,
        returncode=0,
        original_error=None,
        signal_sent=False,
        only_owned_child_signalled=True,
    )
    gpu = dict(released=True, uuid=checker.GPU, memory_mib=0, compute_apps=[], device_minor=6, visible_gpu=2)
    if fault == "wait":
        exit_record["actual_wait"] = False
    elif fault == "signal":
        release["signal_sent"] = True
    elif fault == "birth":
        release["start_ticks"] += 1
    elif fault == "gpu":
        gpu["memory_mib"] = 1
    elif fault == "uuid":
        launch["selected_environment"]["CUDA_VISIBLE_DEVICES"] = "GPU-forbidden"
    elif fault == "interpreter":
        command[0] = "other-python"
    elif fault == "w2":
        command[command.index("--w2-fit-sha256") + 1] = "e" * 64
    refs = {}
    for key, value in (
        ("worker_launch", launch),
        ("worker_exit", exit_record),
        ("worker_release", release),
        ("gpu_after", gpu),
    ):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        refs[key] = reference(path)
    prepared = dict(root=str(tmp_path), model_config=dict(model_python="qualified-python"))
    if fault is None:
        checker.lifecycle(checker.CAPTURE.Evidence(), refs, prepared_ref, registration_ref, actual, prepared)
    else:
        with pytest.raises(ValueError):
            checker.lifecycle(checker.CAPTURE.Evidence(), refs, prepared_ref, registration_ref, actual, prepared)


@pytest.mark.parametrize(
    "fault", [None, "timestamp", "mask", "normalizer", "caption", "future_condition", "degenerate"]
)
def test_response_physical_units_mask_and_clock_are_independently_bound(checker, fault):
    action = np.zeros((40, 128))
    action[:, 3] = action[:, 7] = 1
    request = dict(seed=42, name="current-only")
    metadata = dict(
        checkpoint_path="/8400/model",
        seed=42,
        action_domain=checker.CAPTURE.DOMAIN,
        action_dim_mask=checker.CAPTURE.MASK,
        action_units="raw_physical_or_source_units",
        action_fps=20,
        conditioning_caption_sha256="a" * 64,
        action_timestamps_seconds=(np.arange(40) / 20).tolist(),
        normalized_action=action.tolist(),
    )
    args = dict(
        request,
        condition_frame_indexes_vision=[0],
        action_path=None,
        native_prompt_upsampling=False,
        negative_prompt="",
    )
    response = dict(
        status="success", outputs=[dict(content=dict(action=action.tolist(), action128=metadata))], args=args
    )
    transform = dict(normalization_mask=[True] * 10 + [False] * 118, scale=[1.0] * 128, bias=[0.0] * 128)
    if fault == "timestamp":
        metadata["action_timestamps_seconds"][1] += 0.01
    elif fault == "mask":
        metadata["normalized_action"][0][10] = 1
    elif fault == "normalizer":
        transform["scale"][3] = 2
    elif fault == "caption":
        metadata["conditioning_caption_sha256"] = "b" * 64
    elif fault == "future_condition":
        args["condition_frame_indexes_vision"] = [0, 1]
    elif fault == "degenerate":
        response["outputs"][0]["content"]["action"][0][3:9] = [0] * 6
    if fault is None:
        actual = checker.response_actions(response, request, "/8400/model", transform, "a" * 64)
        np.testing.assert_array_equal(actual, action)
    else:
        with pytest.raises(ValueError):
            checker.response_actions(response, request, "/8400/model", transform, "a" * 64)


@pytest.mark.parametrize("fault", [None, "future_state", "plan", "duration", "context"])
def test_caption_contains_only_fixed_current_task_and_camera_contract(checker, fault):
    caption = dict(
        task_title="task",
        command="task",
        embodiment="franka_panda__parallel_gripper",
        duration="2s",
        fps=20.0,
        action_frequency_hz=20.0,
        resolution=dict(W=640, H=960),
        aspect_ratio="2:3",
        camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
        task_plan=[],
    )
    if fault == "future_state":
        caption["future_state"] = [1, 2]
    elif fault == "plan":
        caption["task_plan"] = ["privileged future subtask"]
    elif fault == "duration":
        caption["duration"] = "4s"
    text = json.dumps(caption)
    conditioning = dict(
        caption=text,
        caption_sha256=hashlib.sha256(text.encode()).hexdigest(),
        inference_only_current_image=True,
        context=dict(
            domain=checker.CAPTURE.DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=checker.CAPTURE.MASK
        ),
    )
    if fault == "context":
        conditioning["context"]["measured_state"] = [1]
    if fault is None:
        checker.conditioning_check(conditioning, dict(prompt="task"))
    else:
        with pytest.raises(ValueError):
            checker.conditioning_check(conditioning, dict(prompt="task"))


@pytest.mark.parametrize("fault", ["w1", "partial_rows", "unverified"])
def test_fixed_w2_rejects_intermediate_or_incomplete_fit_before_any_array_decoder(
    tmp_path, checker, monkeypatch, fault
):
    prepared = dict(path="/prepared.json", sha256="a" * 64)
    registration = dict(path="/registration.json", sha256="b" * 64)
    fit = dict(
        schema_version="s2a.clean-training-fit.v1",
        complete=True,
        passed=True,
        stage="W2",
        actual_rows=840000,
        prepared_ref=prepared,
        execution_registration_ref=registration,
    )
    verified = dict(fit, schema_version="s2a.clean-training-fit-verification.v1")
    if fault == "w1":
        fit["stage"] = "W1"
    elif fault == "partial_rows":
        fit["actual_rows"] -= 1
    elif fault == "unverified":
        verified["passed"] = False
    fit_path, verified_path = tmp_path / "w2.json", tmp_path / "verified.json"
    fit_path.write_text(json.dumps(fit))
    verified_path.write_text(json.dumps(verified))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("W2 gate opened future labels"))
    with pytest.raises(ValueError, match="W2 qualification"):
        checker.fixed_w2(
            checker.CAPTURE.Evidence(),
            prepared,
            registration,
            dict(w2_fit_ref=reference(fit_path), w2_verification_ref=reference(verified_path)),
        )
