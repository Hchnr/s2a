"""Compare C/F/strided inputs with the SHA-bound external replay algebra."""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from s2a.research.response_models.forward.native_osc_torque import replay_order_uncoupled_torque
from s2a.research.response_models.forward.native_replay_joint_response import rollout_replay_order_joint_response


@pytest.fixture(scope="module")
def external_functions():
    root = Path(__file__).resolve().parents[3]
    path = root / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/control_utils.py"
    if not path.is_file():
        pytest.skip("Pinned external NumPy reference not installed; no native arithmetic qualification")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "1d0f1a063894b1e8289339e223665e14e8aec492c31e9e00f42edea8dc735f43"
    names = {"opspace_matrices", "orientation_error", "nullspace_torques"}
    nodes = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 3
    for node in nodes:
        node.decorator_list = []
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("rank_deficient", [False, True])
def test_layout_matches_external_replay_operation_tree(external_functions, layout, rank_deficient):
    rng = np.random.default_rng(42)
    for _ in range(16):
        a = rng.normal(size=(7, 7))
        mass = a.T @ a + np.eye(7)
        j = rng.normal(size=(6, 7))
        if rank_deficient:
            j[5] = j[4]
        if layout == "strided":
            storage = np.zeros((12, 14))
            storage[::2, ::2] = j
            j = storage[::2, ::2]
        else:
            j = np.array(j, order=layout)
        pose, goal = np.eye(4), np.eye(4)
        pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        goal[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        goal[:3, 3] = rng.normal(size=3)
        q, v, initial = rng.normal(size=(3, 7))
        kp, kd = rng.uniform(1, 100, size=(2, 6))
        _, lp, lr, null = external_functions["opspace_matrices"](mass, j, j[:3], j[3:])
        error = external_functions["orientation_error"](goal[:3, :3], pose[:3, :3])
        force = kp[:3] * (goal[:3, 3] - pose[:3, 3]) - kd[:3] * (j[:3] @ v)
        moment = kp[3:] * error - kd[3:] * (j[3:] @ v)
        expected = j.T @ np.r_[lp @ force, lr @ moment]
        expected += external_functions["nullspace_torques"](mass, null, initial, q, v, joint_kp=10)
        actual = replay_order_uncoupled_torque(mass, j, pose, goal, q, v, initial, kp, kd)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("layout", ["C", "F"])
def test_response_sampling_preserves_external_first_torque(external_functions, layout):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 7))
    mass = a.T @ a + np.eye(7)
    j = np.array(rng.normal(size=(6, 7)), order=layout)
    q, v, initial = rng.normal(size=(3, 7))
    kp, kd = rng.uniform(1, 100, size=(2, 6))
    goal = np.eye(4)
    goal[:3, 3] = [0.125, -0.25, 0.5]
    _, lp, lr, null = external_functions["opspace_matrices"](mass, j, j[:3], j[3:])
    force = kp[:3] * goal[:3, 3] - kd[:3] * (j[:3] @ v)
    moment = -kd[3:] * (j[3:] @ v)
    expected = j.T @ np.r_[lp @ force, lr @ moment]
    expected += external_functions["nullspace_torques"](mass, null, initial, q, v, joint_kp=10)

    def query(q, velocity):
        return dict(mass=mass, jacobian=j, pose=np.eye(4), bias=np.zeros(7), damping=np.ones(7))

    result = rollout_replay_order_joint_response(
        q,
        v,
        np.r_[goal[:3, 3], np.zeros(3)],
        initial,
        kp,
        kd,
        query,
        -np.ones(7) * 1000,
        np.ones(7) * 1000,
        dt=0.002,
        substeps=1,
        update_matrices=True,
        latched_rotation_goal=np.eye(3),
    )
    assert result["requested_torque"][0].tobytes() == expected.tobytes()
