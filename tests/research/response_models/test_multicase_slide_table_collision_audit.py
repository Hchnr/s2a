"""Verify that the independent audit rejects mismatched data and witnesses."""

import copy
import json

import numpy as np
import pytest

from s2a.research.wam.training.native.native_training import state_sha256
from scripts.tracking.response_models.multicase_slide_table_collision import CONFIG
from scripts.verification.response_models.audit_multicase_slide_table_collision import audit_queries


def fixture():
    config = json.loads(CONFIG.read_text())
    q = -0.02
    vertices = [[0.0, 0.0, 0.0]] * 4
    pose = np.eye(4).tolist()
    states = np.zeros((601, 3))
    chunks, queries = [], []
    for i in range(75):
        chunks.append({"step": i * 8, "effective_targets_tcp": [pose] * 40})
        fixed = {
            "certified": False,
            "joint_target_interval_m": [q, q],
            "measured_joint_m": q,
            "numerical_guard_m": 1e-9,
        }
        full = {
            "certified": False,
            "joint_target_interval_m": [-0.04, 0.0],
            "measured_joint_m": q,
            "numerical_guard_m": 1e-9,
        }
        queries.append(
            {
                "query": i,
                "step": i * 8,
                "state_sha256": state_sha256(states[i * 8]),
                "measured_joint_m": q,
                "measured_joint_in_range": True,
                "vertices_tcp": vertices,
                "slide_axis_tcp": [1.0, 0.0, 0.0],
                "targets": [
                    {
                        "cursor": j,
                        "target_tcp": pose,
                        "fixed_current": copy.deepcopy(fixed),
                        "full_joint_range": copy.deepcopy(full),
                    }
                    for j in range(8)
                ],
            }
        )
    analysis = {
        "case": {"case_id": "synthetic"},
        "queries": queries,
        "finger_joint_range_m": [-0.04, 0.0],
        "table_pose": np.eye(4).tolist(),
        "table_half_size_m": [0.5, 0.5, 0.5],
    }
    return analysis, chunks, states, config


def test_original_target_and_state_pairing_are_mandatory():
    analysis, chunks, states, config = fixture()
    summary, checks = audit_queries(analysis, chunks, states, config)
    assert summary["queries"] == 75 and summary["targets"] == 600 and checks == []
    changed = copy.deepcopy(analysis)
    changed["queries"][2]["targets"][4]["target_tcp"] = np.diag([1, 1, 1, 2]).tolist()
    with pytest.raises(ValueError, match="original query"):
        audit_queries(changed, chunks, states, config)
    changed = copy.deepcopy(analysis)
    changed["queries"][3]["state_sha256"] = "wrong"
    with pytest.raises(ValueError, match="original state"):
        audit_queries(changed, chunks, states, config)


def test_positive_witness_requires_independent_strict_interior():
    analysis, chunks, states, config = fixture()
    witness = analysis["queries"][0]["targets"][0]["fixed_current"]
    witness.update(certified=True, vertex_index=0, vertex_tcp=[0.0, 0.0, 0.0])
    summary, checks = audit_queries(analysis, chunks, states, config)
    assert summary["fixed_current_certified"] == 1 and len(checks) == 1
    analysis["table_half_size_m"] = [0.005, 0.005, 0.005]
    with pytest.raises(ValueError, match="Independent bound"):
        audit_queries(analysis, chunks, states, config)
