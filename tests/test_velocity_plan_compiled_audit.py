"""All three command branches and original rejection/optimality gates are retained."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from s2a.infrastructure.publication import result_json
from s2a.research.trajectory_search.velocity_model_planner import VelocityModelPlanner
from scripts.verification.joint_control.planning import velocity_plan_compiled_audit as compiled_module
from scripts.verification.joint_control.planning.velocity_plan_compiled_audit import audit_plan as compiled
from scripts.verification.joint_control.planning.velocity_plan_independent_audit import audit_plan as scalar


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


def test_original_compiled_build_declaration_and_current_source_storage_are_separate(tmp_path, monkeypatch):
    build_path = Path(compiled_module.BUILD_REF["path"])
    build_bytes = build_path.read_bytes()
    original_ref = dict(compiled_module._build["source_ref"])
    actual_storage = compiled_module.current_source_path(compiled_module.CXX_SOURCE_IDENTITY)
    current_storage = tmp_path / "current" / "velocity_scalar_audit.cpp"
    current_storage.parent.mkdir()
    current_storage.write_bytes(actual_storage.read_bytes())

    def current_path(identity):
        assert identity == "scripts/velocity_scalar_audit.cpp"
        return current_storage

    monkeypatch.setattr(compiled_module, "current_source_path", current_path)
    storage_ref = compiled_module.compiled_source_storage(original_ref)
    assert storage_ref["path"] == str(current_storage.resolve())
    assert storage_ref["sha256"] == original_ref["sha256"]
    assert original_ref["path"] != storage_ref["path"]
    with pytest.raises(ValueError, match=r"Original compiled C\+\+ source declaration"):
        compiled_module.compiled_source_storage(dict(original_ref, path=str(current_storage)))
    with pytest.raises(ValueError, match=r"Original compiled C\+\+ source declaration"):
        compiled_module.compiled_source_storage(dict(original_ref, sha256="0" * 64))
    current_storage.write_bytes(current_storage.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"Current compiled C\+\+ source bytes"):
        compiled_module.compiled_source_storage(original_ref)
    assert build_path.read_bytes() == build_bytes
    assert compiled_module._build["source_ref"] == original_ref
