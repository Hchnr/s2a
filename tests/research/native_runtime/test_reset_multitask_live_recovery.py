"""Recovery must preserve the failed run and account for its spent budget."""

import copy
import json
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime import reset_multitask_live_admission_v2 as admission


def test_recovery_config_rejects_ignored_spending_and_extra_retries():
    config = json.loads(admission.CONFIG.read_text())
    admission.validate_config(config)
    for field, value in (
        ("previously_completed_physical_periods", 0),
        ("previously_completed_denoise", 0),
        ("previously_attempted_model_requests", 3),
        ("automatic_retry", True),
        ("max_model_requests_per_case", 6),
    ):
        changed = copy.deepcopy(config)
        changed[field] = value
        with pytest.raises(ValueError, match="cumulative budgets"):
            admission.validate_config(changed)


def test_recovery_loads_audited_failure_and_the_three_original_cases():
    inputs = admission.load_inputs(verify_assets=False)
    receipt = admission.input_receipt(inputs)
    assert receipt["incident"] == admission.INCIDENT
    assert inputs["config"]["previously_completed_inner_steps"] == 1200
    assert [row["spec"]["case_index"] for row in receipt["cases"]] == [0, 4, 8]
    assert receipt["prior_config_ref"]["path"].endswith("reset_multitask_live_48_20261111.json")
    for spec in admission.CASES:
        index = spec["case_index"]
        image_root = admission.OUTPUT / f"case_{index:02d}/runtime"
        registration = admission.model_registration(inputs, index, image_root)
        assert registration["initial"] == inputs["selected"][index]["initial"]
        assert registration["max_requests"] == 5 and registration["denoise_steps"] == 35


def test_recovery_binds_pinned_supervisor_and_distinct_outputs_in_new_process():
    code = (
        "from scripts.simulation.runtime import run_reset_multitask_live_48_v2 as s; "
        "x=s.bind_recovery(); "
        "assert x.admission is s.admission and x.workers is s.workers; "
        "assert x.OUTPUT == s.admission.OUTPUT and x.MODULE.endswith('_v2'); "
        "assert x.workers.admission is s.admission"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
