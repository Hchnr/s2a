import subprocess
import sys

import pytest

from scripts.simulation.runtime import capture_ordered_feedback_observations as capture


def test_capture_inputs_are_bound_to_completed_and_independently_replayed_actions():
    inputs = capture.load_inputs()
    assert inputs["config"]["capture_steps"] == [80, 88]
    assert [item["step"] for item in inputs["periods"]] == list(range(80, 88))
    assert inputs["producer"]["runtime"]["continuation"]["steps_completed"] == 16
    assert inputs["refs"]["producer"] == capture.prefix.read(inputs["refs"]["producer_summary"])["branch_refs"][0]


def test_capture_process_does_not_import_candidate_policies():
    source = (
        "from scripts.simulation.runtime import capture_ordered_feedback_observations as c; "
        "c.load_inputs(); c.prior.checker_isolation(); print('isolated')"
    )
    child = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True, timeout=90)
    assert "isolated" in child.stdout


def test_capture_budget_rejects_incomplete_or_extra_replay():
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
    with pytest.raises(ValueError, match="Bounded complete replay"):
        capture.summarize_phase([branch], "producer", config)


def test_capture_renderer_writes_inside_the_runtime_root(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    class Observer:
        @staticmethod
        def render_anchor(env, step, sample, endpoint, destination):
            assert env == "live" and sample == "sample" and endpoint == "endpoint"
            (destination / f"anchor{step:04d}").mkdir()
            return destination / f"anchor{step:04d}"

    result = capture.render_registered_endpoint(Observer, "live", 80, "sample", "endpoint", runtime)
    assert result == runtime / "anchor0080" and result.is_dir()
    with pytest.raises(ValueError, match="registered capture steps"):
        capture.render_registered_endpoint(Observer, "live", 81, "sample", "endpoint", runtime)
