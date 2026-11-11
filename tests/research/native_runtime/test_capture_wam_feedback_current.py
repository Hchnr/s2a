import subprocess
import sys

import pytest

from scripts.simulation.runtime import capture_wam_feedback_current as capture


def test_capture_is_bound_to_both_completed_and_checked_wam_feedback():
    inputs = capture.load_inputs()
    assert [period["step"] for period in inputs["periods"]] == list(range(80, 88))
    assert inputs["branch"]["runtime"]["continuation"]["stop"] is None
    assert capture.prefix.read(inputs["refs"]["producer"])["branch_refs"] == [inputs["refs"]["producer_branch"]]
    assert capture.prefix.read(inputs["refs"]["checker"])["counts"]["continuation_steps"] == 8


def test_capture_rejects_more_than_registered_physical_replay():
    config = capture.read_config()[1]
    runtime = {
        "control_steps_completed": 89,
        "counters": {"mj_step": {"completed": 2200}},
        "prefix_inner_periods": [{"samples": [None] * 25} for _ in range(80)],
        "continuation": {"steps_completed": 8},
        "captures": [{"frame_index": 80}, {"frame_index": 88}],
        "newly_selected_commands": 0,
        "cleanup_complete": True,
    }
    branch = {"runtime": runtime, "complete": True, "passed": True, "candidate_modules_unloaded": True}
    with pytest.raises(ValueError, match="Complete bounded replay"):
        capture.summarize_phase([branch], "producer", config)


def test_capture_replay_does_not_import_candidate_policies():
    source = (
        "from scripts.simulation.runtime import capture_wam_feedback_current as c; "
        "c.load_inputs(); c.study.prior.checker_isolation(); print('isolated')"
    )
    child = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True, timeout=90)
    assert "isolated" in child.stdout


def test_step88_callback_passes_observer_and_writes_inside_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    endpoint = {"sim_state_sha256": "state", "sim_time": 4.65}

    class Observer:
        @staticmethod
        def endpoint(env, step):
            assert env == "live" and step == 88
            return "sample", endpoint

        @staticmethod
        def render_anchor(env, step, sample, receipt, destination):
            assert env == "live" and step == 88 and sample == "sample" and receipt == endpoint
            path = destination / "anchor0088"
            path.mkdir()
            return {"frame_index": step, "path": path}

    result = capture.render_completed_endpoint(Observer, "live", {"endpoint_receipt": endpoint}, runtime)
    assert result["path"] == runtime / "anchor0088" and result["path"].is_dir()
    with pytest.raises(ValueError, match="actual recorded WAM"):
        capture.render_completed_endpoint(
            Observer, "live", {"endpoint_receipt": {**endpoint, "sim_time": 4.6}}, runtime
        )
