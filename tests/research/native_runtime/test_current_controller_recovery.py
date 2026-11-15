"""The recovery owner must retain training identity and actual endpoint evidence."""

import copy
import json

import numpy as np
import pytest

from s2a.geometry import pose_error
from s2a.references.training_recovery_timeline import training_recovery_timeline
from s2a.research.native_runtime import bootstrap
from s2a.research.native_runtime.training_recovery_capture import TrainingRecoveryCapture
from scripts.simulation.runtime import run_current_controller_recovery as run
from tests.research.native_runtime.test_training_recovery_capture import image, sample


def test_fixed_config_and_training_roster_keep_excluded_initial_states_closed():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for key, value in (
        ("offsets_m", [0.0, 0.03, -0.02]),
        ("max_periods_per_phase", 2308),
        ("automatic_retry", 1),
        ("new_attempts", 2),
        ("max_probe_inner_steps", 1),
    ):
        with pytest.raises(ValueError):
            run.validate_config(dict(config, **{key: value}))
    changed = copy.deepcopy(config)
    changed["training_preparation"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        run.validate_config(changed)
    _config, identity = run.load_inputs()
    assert len(identity["cases"]) == 9
    assert [case["periods"] for case in identity["cases"]] == [220] * 3 + [310] * 3 + [239] * 3
    assert [case["offset_m"] for case in identity["cases"]] == [0.0, 0.02, -0.02] * 3


def test_original_failed_run_rejects_changed_counts_or_worker_lifecycle():
    owner = run.prefix.read(run.preparation.configured_ref(run.REGISTERED_REFS["failed_producer"]))
    lifecycle = run.prefix.read(run.preparation.configured_ref(run.REGISTERED_REFS["failed_lifecycle"]))
    branches = [run.prefix.read(owner["cases"][str(i)]["available_branch_ref"]) for i in range(9)]
    run.validate_previous_failure(owner, lifecycle, branches)
    bad = copy.deepcopy(branches)
    bad[3]["runtime"]["counters"]["mj_step"]["completed"] = 1
    with pytest.raises(ValueError, match="zero actions"):
        run.validate_previous_failure(owner, lifecycle, bad)
    bad_lifecycle = copy.deepcopy(lifecycle)
    bad_lifecycle["workers"][0]["reaped"] = False
    with pytest.raises(ValueError, match="cleanup"):
        run.validate_previous_failure(owner, bad_lifecycle, branches)


def test_probe_gate_rejects_any_action_or_incomplete_owner(monkeypatch):
    counters = {
        key: dict(attempted=value, completed=value)
        for key, value in (("mj_step", 0), ("mj_step2", 0), ("model_construct", 3), ("context", 3))
    }
    runtime = dict(
        complete=True,
        passed=True,
        reset_observation_verified=True,
        cleanup_complete=True,
        continuation=dict(segments=[], stop=None, probe_no_actions=True),
        control_steps_attempted=0,
        control_steps_completed=0,
        newly_selected_commands=0,
        world_model_calls=0,
        counters=counters,
    )
    branch = dict(
        complete=True,
        passed=True,
        runtime=runtime,
        requested_phase="probe",
        candidate_modules_unloaded=True,
        fresh_cases=0,
    )
    checked = dict(endpoint_count=1, labels_reconstructed=False, images=[{}], anchor_windows=[dict(full_h40=False)])
    monkeypatch.setattr(run, "verify_capture", lambda *_args: (checked, dict(sent_commands=np.empty((0, 7)))))
    run.verify_probe_branch(branch, dict(case=dict(tool=np.eye(4))), type("Timeline", (), dict(periods=1))())
    bad = copy.deepcopy(branch)
    bad["runtime"]["control_steps_attempted"] = 1
    with pytest.raises(ValueError, match="zero physical actions"):
        run.verify_probe_branch(bad, dict(case=dict(tool=np.eye(4))), type("Timeline", (), dict(periods=1))())
    with pytest.raises(ValueError, match="source-matched"):
        run.verify_probe_admission(
            dict(complete=True, passed=False, owner_closed=True, sources_unchanged=True), {}, [], {}, {}
        )


def test_reset_capture_writes_within_runtime_root_after_runtime_creation(tmp_path):
    runtime_dir = tmp_path / "worker" / "runtime"
    runtime_dir.mkdir(parents=True)
    initial, receipt = sample(0)

    def guarded_image(env, frame, current, endpoint_receipt, directory):
        destination = directory / f"anchor{frame:04d}"
        bootstrap._safe_path(destination, (runtime_dir,), exists=False)
        destination.mkdir()
        return image(env, frame, current, endpoint_receipt, directory)

    with pytest.raises(ValueError, match="outside approved roots"):
        bootstrap._safe_path(runtime_dir.parent / "capture" / "anchor0000", (runtime_dir,), exists=False)
    capture = run.capture_reset(
        runtime_dir, np.eye(4), None, dict(sample=initial, endpoint_receipt=receipt), guarded_image
    )
    assert capture.directory == runtime_dir / "capture"
    assert (capture.directory / "anchor0000").is_dir()
    assert capture.receipts == [receipt]
    with pytest.raises(FileExistsError):
        run.capture_reset(runtime_dir, np.eye(4), None, dict(sample=initial, endpoint_receipt=receipt), guarded_image)


def test_independent_capture_check_rejects_changed_state_and_motor(tmp_path, monkeypatch):
    source = np.repeat(np.eye(4)[None], 41, axis=0)
    timeline, _metadata = training_recovery_timeline(source, np.full(41, 0.8), np.eye(4), offset_m=0.0)
    capture = TrainingRecoveryCapture(np.eye(4), tmp_path, image)
    rows = []
    for index in range(4):
        actual, receipt = sample(index)
        capture.append(None, actual, receipt)
        if index:
            target, opening, _ = timeline.window(index - 1)
            rows.append(
                dict(
                    step=index - 1,
                    command=np.zeros(7),
                    actual_state=actual["sim_states"],
                    endpoint_receipt=receipt,
                    task_success=False,
                    tcp_error=pose_error(target[0], actual["source_pose"]),
                    completed_feedback=dict(score=dict(opening=float(actual["opening"]), opening_reference=opening[0])),
                )
            )
    path = tmp_path / "actual.npz"
    arrays = capture.arrays()
    arrays["sent_commands"] = np.zeros((3, 7))
    np.savez(path, **arrays)
    branch = dict(
        runtime=dict(continuation=dict(segments=[dict(continuation=dict(periods=rows))])),
        arrays_ref=dict(path=str(path)),
        endpoint_receipts=capture.receipts,
        images=capture.images,
        anchor_windows=capture.anchor_windows(),
    )
    monkeypatch.setattr(run.prefix, "checked", lambda ref: path)
    monkeypatch.setattr(run, "verify_canvas", lambda item, _data, _receipt: dict(frame_index=item["frame_index"]))
    checked, _data = run.verify_capture(branch, timeline.periods, np.eye(4), timeline)
    assert checked["endpoint_count"] == 4 and checked["labels_reconstructed"] is True
    assert not any(row["full_h40"] for row in checked["anchor_windows"])
    corrupted = copy.deepcopy(branch)
    corrupted["endpoint_receipts"][1]["gripper_current_action_sha256"] = "changed"
    with pytest.raises(ValueError, match="endpoint state"):
        run.verify_capture(corrupted, timeline.periods, np.eye(4), timeline)
    corrupted = copy.deepcopy(branch)
    corrupted["runtime"]["continuation"]["segments"][0]["continuation"]["periods"][1]["actual_state"][1] += 1
    with pytest.raises(AssertionError):
        run.verify_capture(corrupted, timeline.periods, np.eye(4), timeline)
