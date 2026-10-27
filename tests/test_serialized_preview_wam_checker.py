"""Independent identity audit and unchanged physical replay equations."""

import ast
import json
from pathlib import Path

import pytest
from serialized_preview_wam_acceptance_checker import verify_development_identity

ROOT = Path(__file__).resolve().parents[1]


def values():
    entries = json.loads((ROOT / "tests/fixtures/coupled_preview_wam_responses.json").read_text())["cases"]
    value = entries[1]
    config = dict(
        wam_development=dict(
            case=dict(
                case_index=1,
                case_id="task0_init0_s42_terminal058",
                task_index=0,
                record_name="task0_init0",
                sampling_seed=42,
                variant="terminal058",
                expected_requests=75,
            ),
            readout_ref=value["response"]["clean_training"]["readout_ref"],
        )
    )
    return config, value["request"], value["response"]


def test_independent_ledger_matches_archived_response():
    config, request, response = values()
    assert verify_development_identity(config, 0, request, response) == response["clean_training"]


@pytest.mark.parametrize(
    "field,value", [("variant", "raw"), ("request_index", 1), ("readout_ref", None), ("case_index", True)]
)
def test_independent_ledger_rejects_cross_variant_and_stale_response(field, value):
    config, request, response = values()
    response["clean_training"][field] = value
    with pytest.raises(AssertionError):
        verify_development_identity(config, 0, request, response)


def test_independent_replay_retains_every_per_step_physical_check():
    bodies = []
    for name in ("coupled_preview_acceptance_checker_registered.py", "serialized_preview_wam_acceptance_checker.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text())
        loop = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [v.id for v in n.target.elts if isinstance(v, ast.Name)] == ["step", "row"]
        ]
        assert len(loop) == 1
        bodies.append(ast.dump(loop[0], include_attributes=False))
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize("nested", [False, True])
def test_future_labels_rejected_even_when_response_echoes_them(nested):
    config, request, response = values()
    if nested:
        request["extra"]["future_state"] = [1]
        response["args"]["extra"]["future_state"] = [1]
    else:
        request["future_state"] = response["args"]["future_state"] = [1]
    with pytest.raises(AssertionError):
        verify_development_identity(config, 0, request, response)
