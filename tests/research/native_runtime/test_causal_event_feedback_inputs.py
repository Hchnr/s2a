"""Paired feedback references must be causal, aligned and isolated."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.simulation.runtime import prepare_causal_event_feedback as binding


def test_real_cases_bind_reproduced_prefixes_and_paired_absolute_endpoints():
    inputs = binding.load_inputs()
    assert [case["spec"]["case_index"] for case in inputs["cases"]] == [0, 5]
    assert [case["references"]["remembered"]["cursor"] for case in inputs["cases"]] == [8, 16]
    for case in inputs["cases"]:
        old = binding.selected_case(case, "remembered")
        new = binding.selected_case(case, "latest")
        assert old["fixed_targets"].shape == new["fixed_targets"].shape == (8, 4, 4)
        assert old["fixed_openings"].shape == new["fixed_openings"].shape == (8,)
        assert "references" not in old and "references" not in new
        assert np.any(old["fixed_openings"] != new["fixed_openings"])
        old["targets"][0, 0, 3] += 1
        assert not np.array_equal(old["targets"], case["references"]["remembered"]["poses"])
        long_old = binding.selected_case(case, "remembered", horizon=16)
        assert long_old["fixed_targets"].shape == (16, 4, 4) and len(long_old["fixed_openings"]) == 16
        with pytest.raises(ValueError, match="preview horizon"):
            binding.selected_case(case, "remembered", horizon=24)


def test_reference_binding_rejects_wrong_time_or_detached_opening():
    case = binding.load_inputs()["cases"][0]
    spec, refs, shadow = case["spec"], case["references"], case["opportunity"]["shadow"]
    old = {
        "step": spec["old_step"],
        "effective_targets_tcp": refs["remembered"]["poses"],
        "openings": refs["remembered"]["openings"],
        "anchor_tcp": refs["remembered"]["anchor"],
    }
    latest = {
        "step": spec["prefix_steps"],
        "effective_targets_tcp": refs["latest"]["poses"],
        "openings": refs["latest"]["openings"],
        "anchor_tcp": refs["latest"]["anchor"],
    }
    measured = case["saved"]["source_pose"][-1] @ case["tool"]
    altered = copy.deepcopy(shadow)
    altered["absolute_endpoints"][0] += 1
    with pytest.raises(ValueError):
        binding.validate_references(spec, old, latest, altered, measured)
    altered = copy.deepcopy(shadow)
    altered["openings"][0] += 0.1
    with pytest.raises(AssertionError):
        binding.validate_references(spec, old, latest, altered, measured)
    with pytest.raises(ValueError):
        binding.selected_case(case, "future")


def test_checker_binding_does_not_import_candidate_policies():
    program = (
        "import sys; from scripts.simulation.runtime import prepare_causal_event_feedback as b; "
        "b.load_inputs(); "
        "assert 's2a.research.joint_control.preview.refined_feedback' not in sys.modules; "
        "assert 's2a.research.joint_control.preview.quality_feedback_refiner' not in sys.modules; "
        "assert 's2a.research.joint_control.tracking.loaded_velocity_control' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
