"""All three command branches and original rejection/optimality gates are retained."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from velocity_plan_compiled_audit import audit_plan as compiled
from velocity_plan_independent_audit import audit_plan as scalar

from s2a.preview_result_publication import result_json
from s2a.velocity_model_planner import VelocityModelPlanner


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_full_scalar_enumeration_matches_reference_on_frozen_parameters(horizon):
    fixture = json.loads((Path(__file__).parent / "fixtures/velocity_scalar_audit_parameters.json").read_text())
    parameters, envelope = fixture["parameters"], fixture["envelope"]
    reference = np.linspace(0.2, 0.6, horizon).tolist()
    opening, motor, velocity = 0.4, np.array([-0.25, 0.25]), 0.01
    plan = json.loads(
        result_json(
            asdict(
                VelocityModelPlanner(parameters, velocity_envelope=envelope).plan(opening, motor, velocity, reference)
            )
        )
    )
    expected = scalar(parameters, envelope, opening, motor, velocity, reference, plan)
    actual = compiled(parameters, envelope, opening, motor, velocity, reference, plan)
    assert actual["valid_sequences"] == expected["valid_sequences"]
    assert abs(actual["minimum_cost"] - expected["minimum_cost"]) <= 1e-12
    assert abs(actual["chosen_cost"] - expected["chosen_cost"]) <= 1e-12
    wrong = dict(plan, cost=plan["cost"] + 0.01)
    with pytest.raises(AssertionError):
        compiled(parameters, envelope, opening, motor, velocity, reference, wrong)
