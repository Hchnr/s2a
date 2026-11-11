"""The multi-task pilot must bind its original cases and exact budgets."""

import copy
import json

import pytest

from scripts.simulation.runtime import reset_multitask_live_admission as admission


def test_config_rejects_case_device_budget_and_type_changes():
    registered = json.loads(admission.CONFIG.read_text())
    admission.validate_config(registered)
    changes = []
    for mutate in (
        lambda c: c["cases"][0].update(gpu_uuid=c["cases"][1]["gpu_uuid"]),
        lambda c: c["cases"][1].update(task=c["cases"][0]["task"]),
        lambda c: c["cases"].reverse(),
        lambda c: c.update(max_native_steps_per_case_per_phase=1201),
        lambda c: c.update(max_model_requests_per_case=6),
        lambda c: c.update(fresh_cases=False),
        lambda c: c.update(automatic_retry=True),
    ):
        changed = copy.deepcopy(registered)
        mutate(changed)
        changes.append(changed)
    for changed in changes:
        with pytest.raises(ValueError, match="exact three-case"):
            admission.validate_config(changed)


def synthetic_case(spec):
    identity = dict(
        case_index=spec["case_index"],
        case_id=spec["case_id"],
        task_index=spec["case_index"] // 4,
        sampling_seed=42,
        variant="raw",
    )
    case = dict(
        case=identity,
        record=dict(task=spec["task"], task_index=identity["task_index"], initial_index=0),
        target_object=spec["target"],
    )
    periods = [dict(step=i) for i in range(8)]
    periods[-1]["endpoint_receipt"] = dict(endpoint_step=8, sim_time=0.65, sim_state_sha256="a" * 64)
    branch = dict(
        complete=True,
        passed=True,
        case=copy.deepcopy(identity),
        runtime=dict(continuation=dict(periods=periods, stop=None)),
    )
    return case, branch


def test_each_case_binds_task_target_and_completed_original_endpoint():
    for spec in admission.CASES:
        case, branch = synthetic_case(spec)
        assert admission.validate_case(spec, case, branch) == dict(step=8, sim_time_s=0.65, state_sha256="a" * 64)
        for field, value in (("task", "wrong task"), ("initial_index", 1)):
            changed = copy.deepcopy(case)
            changed["record"][field] = value
            with pytest.raises(ValueError, match="matching completed raw reset"):
                admission.validate_case(spec, changed, branch)
        changed = copy.deepcopy(branch)
        changed["runtime"]["continuation"]["periods"].pop()
        with pytest.raises(ValueError, match="matching completed raw reset"):
            admission.validate_case(spec, case, changed)


def test_actual_registered_development_cases_and_model_bindings():
    inputs = admission.load_inputs(verify_assets=False)
    assert list(inputs["selected"]) == [0, 4, 8]
    assert len(inputs["weight_hashes"]) == 163 and len(inputs["model_source_hashes"]) >= 1000
    for spec in inputs["config"]["cases"]:
        index = spec["case_index"]
        registration = admission.model_registration(inputs, index, admission.OUTPUT / f"case_{index:02d}/runtime")
        assert registration["initial"] == inputs["selected"][index]["initial"]
        assert registration["task"] == spec["task"]
        assert registration["max_requests"] == 5 and registration["denoise_steps"] == 35
    with pytest.raises(ValueError, match="registered task"):
        admission.model_registration(inputs, 1, admission.OUTPUT / "other/runtime")
