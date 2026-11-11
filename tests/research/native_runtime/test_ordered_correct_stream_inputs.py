import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_ordered_correct_stream as binding
from scripts.simulation.runtime import run_ordered_correct_stream as runner


def test_complete_prefix_and_overlapping_correct_reference_are_bound():
    inputs = binding.load_inputs()
    assert len(inputs["rows"]) == 80 and inputs["rows"][-1]["step"] == 79
    assert len(inputs["prefix_chunks"]) == 10
    np.testing.assert_array_equal([row["command"] for row in inputs["rows"]], inputs["previous"]["commands"])
    np.testing.assert_allclose(inputs["targets"][0][8:], inputs["targets"][1][:32], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(inputs["openings"][0][8:], inputs["openings"][1][:32])
    assert len(binding.input_receipt(inputs)["source_sha256"]) > 0


def test_checker_import_does_not_load_candidate_policies():
    source = (
        "from scripts.simulation.runtime import run_ordered_correct_stream as r; "
        "r.binding.load_inputs(); r.checker_isolation(); print('isolated')"
    )
    child = subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True, timeout=90)
    assert "isolated" in child.stdout


def test_experiment_summary_rejects_unregistered_extra_physics():
    config = binding.read_config()[1]
    runtime = {
        "control_steps_completed": 97,
        "counters": {"mj_step": {"completed": 2400}},
        "newly_selected_commands": 16,
        "continuation": {"periods": [{}] * 16},
        "prefix_inner_periods": [{"samples": [None] * 25} for _ in range(80)],
        "checked_endpoints": 81,
        "cleanup_complete": True,
        "segment_receipts": [],
    }
    branch = {"complete": True, "passed": True, "runtime": runtime}
    with pytest.raises(ValueError, match="Bounded producer"):
        runner.summarize_phase([branch], "producer", config)
