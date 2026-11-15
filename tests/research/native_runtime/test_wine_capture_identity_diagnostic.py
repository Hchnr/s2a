"""The wine diagnostic preserves failed evidence and stops before future references."""

import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from scripts.simulation.runtime import diagnose_wine_capture_identity as runner


def test_config_rejects_retries_and_changed_physical_budget():
    config = json.loads(runner.CONFIG.read_text())
    runner.validate_config(config)
    for key, value in (("capture_step", 105), ("max_control_steps_per_phase", 143), ("automatic_retry", True)):
        with pytest.raises(ValueError):
            runner.validate_config(dict(config, **{key: value}))


def test_target_metadata_is_saved_before_original_guard(monkeypatch):
    binding = dict(sim_time=5.0, sim_state_sha256="state")
    expected = dict(
        frame_index=104,
        current_endpoint=dict(endpoint_step=104, **binding),
        render_before=dict(binding),
        render_after=dict(binding),
        future_rgb_frames=0,
        camera_keys=["left", "right"],
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
        canvas_rgb_sha256="rgb",
        canvas_png=dict(sha256="png"),
        current_source_pose=np.eye(4),
        current_opening=0.5,
    )
    actual = copy.deepcopy(expected)
    actual["current_endpoint"]["sim_state_sha256"] = "different"
    events = []

    def publish(path, payload):
        events.append(("saved", path, payload))
        return dict(path=str(path), sha256="receipt")

    def original_guard(_expected, _actual):
        assert events and events[0][0] == "saved"
        raise ValueError("original capture guard failed")

    monkeypatch.setattr(runner, "publish_result", publish)
    monkeypatch.setattr(runner.captures, "compare_capture", original_guard)
    result = runner.persist_target_capture("runtime", expected, actual)
    assert result["original_guard_passed"] is False
    assert result["fields"]["differing_paths"] == ["current_endpoint.sim_state_sha256"]
    assert result["original_guard_error"]["message"] == "original capture guard failed"
    assert events[0][2]["actual"]["current_endpoint"]["sim_state_sha256"] == "different"
    assert str(events[0][1]) == "runtime/capture104.json"
    with pytest.raises(ValueError, match="controlled runtime"):
        runner.persist_target_capture("worker", expected, actual)


def test_capture_persists_inside_runtime_with_write_audit_hook():
    script = r"""
import os, sys
from pathlib import Path
from scripts.simulation.runtime import diagnose_wine_capture_identity as runner

runtime = Path(sys.argv[1]).resolve()
runtime.mkdir()
binding = dict(sim_time=5.0, sim_state_sha256='state')
capture = dict(frame_index=104, current_endpoint=dict(endpoint_step=104, **binding),
    render_before=dict(binding), render_after=dict(binding), future_rgb_frames=0,
    camera_keys=['left', 'right'], canvas_shape=[960, 640, 3], canvas_dtype='uint8',
    canvas_rgb_sha256='rgb', canvas_png=dict(sha256='png'),
    current_source_pose=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
    current_opening=0.5)

def guard(event, args):
    path = None
    if event == 'os.mkdir':
        path = args[0]
    elif event == 'open':
        value, mode, flags = args
        if not isinstance(value, int) and ((mode and any(x in mode for x in 'wax+')) or
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
            path = value
    if path is not None and not Path(os.fsdecode(path)).resolve().is_relative_to(runtime):
        raise ValueError('write outside controlled runtime')

sys.addaudithook(guard)
receipt = runner.persist_target_capture(runtime, capture, capture)
assert receipt['original_guard_passed'] is True
assert Path(receipt['capture_ref']['path']) == runtime / 'capture104.json'
assert (runtime / 'capture104.json').is_file()
"""
    with tempfile.TemporaryDirectory(prefix="capture-audit-", dir=runner.ROOT / "outputs") as temporary:
        command = [sys.executable, "-c", script, str(os.path.join(temporary, "runtime"))]
        result = subprocess.run(
            command,
            cwd=runner.ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_prepare_exposes_no_post_capture_reference_to_runtime(monkeypatch):
    config = json.loads(runner.CONFIG.read_text())
    failed = dict(
        complete=False,
        passed=False,
        error=dict(type="PrefixReplayFailure", message="The independent current image or endpoint differs"),
        runtime=dict(
            control_steps_attempted=104,
            control_steps_completed=104,
            counters=dict(mj_step=dict(completed=2600)),
            cleanup_complete=True,
        ),
    )
    steps = list(range(8, 104, 8))
    windows = [
        dict(
            start_step=step,
            boundary="window_end",
            signals=[],
            continuation=dict(periods=[dict(step=x) for x in range(step, step + 8)]),
        )
        for step in steps
    ]
    references = [dict(frame_index=step) for step in steps + [104]]
    exchanges = [dict(step=step, request_ref="request", response_ref="response") for step in steps]
    responses = [dict(step=step, actual_response_ref="actual") for step in steps]
    producer = dict(
        runtime=dict(
            continuation=dict(
                first_window=dict(periods=[dict(step=x) for x in range(8)]),
                adaptive=dict(segments=windows + [None]),
            ),
            adaptive_captures=references + [None],
            adaptive_model_exchanges=exchanges + [None],
        )
    )
    model = dict(responses=responses + [None])
    inputs = dict(case=dict(case=dict(sampling_seed=42)), baseline_periods=[])
    row = dict(
        case_index=4, producer_ref="producer", model_ref="model", checker_ref="checker", original_owner_ref="owner"
    )
    registration = dict(source_refs=[])

    monkeypatch.setattr(runner.prefix, "reference", lambda path: dict(path=str(path), sha256="hash"))
    monkeypatch.setattr(runner.prefix, "read", lambda ref: failed if "case04/worker" in ref["path"] else config)
    monkeypatch.setattr(runner.prefix, "checked", lambda _ref: None)
    monkeypatch.setattr(
        runner.admission, "admit", lambda _index: (inputs, producer, model, None, row, registration, [])
    )
    monkeypatch.setattr(runner.admission.admission, "input_receipt", lambda _inputs: {})
    monkeypatch.setattr(runner.comparison, "compare_periods", lambda *_args: {})
    result = runner.prepare()
    assert result["identity"]["consumed_reference_steps"] == steps
    assert [item["frame_index"] for item in result["captures"]] == steps + [104]
    assert [item["step"] for item in result["responses"]] == steps
    assert len(result["windows"]) == 12
