"""Native-rounded goal regression, without treating NumPy2 as the native runtime.

Component tests run with the ordinary repository pytest. To independently
check the actual archived source and its enabled Numba arithmetic, run from
the repository (CPU only, no simulation or robosuite import):

CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src outputs/libero_env311/bin/python tests/test_native_goal_joint_response.py

That explicit native-1.26 check prints source/input SHA256 and runtime versions,
and compares all nine archived increments plus seed42 random1000 and three
edge cases against numba.njit(cache=False)(archived_quat2mat). Its scope is this
runtime and these cases, not a cross-BLAS/backend bitwise guarantee.
"""

import numpy as np
from scipy.spatial.transform import Rotation

import s2a.native_goal_joint_response as native_response
from s2a.nonlinear_joint_response import rollout_nonlinear_joint_response

# Recorded from archived transform_utils.py SHA256 below, using native
# NumPy1.26.4/Numba0.68.0 JIT. This exposes dot/normalization and scalar-promotion
# differences that a small pure-z example would conceal.
NATIVE_SOURCE_SHA256 = "1c2317895733449883b56af0decefb3529649b3a7779dc2ef3ed00816bb25bb0"
DELTA = np.array([-1.6365676692913387, 0.3275087220950086, -2.6170964633749483])
NATIVE_JIT_MATRIX = np.array(
    [
        [-0.4435156136751175, -0.07953272759914398, 0.8927310109138489],
        [-0.14291104674339294, -0.9770359992980957, -0.15804265439510345],
        [0.8847997784614563, -0.19767548143863678, 0.4219643324613571],
    ]
)


def _dynamics(q, velocity):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
    pose[:3, 3] = q[:3]
    return dict(
        mass=np.diag(1.0 + q * q),
        jacobian=np.c_[np.diag(1.0 + q[:6] * q[:6]), np.zeros(6)],
        pose=pose,
        bias=0.2 * q,
        damping=0.1 + q * q,
        passive_extra=0.03 * q,
    )


def _arguments():
    q = np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.15, 0.4])
    return dict(
        q=q,
        velocity=np.array([0.02, 0.0, -0.01, 0.0, 0.0, -0.02, 0.01]),
        physical_delta=np.r_[0.01, -0.02, 0.03, DELTA],
        initial_joint=q + 0.02,
        kp=np.full(6, 20.0),
        kd=np.full(6, 4.0),
        dynamics=_dynamics,
        torque_lower=np.full(7, -0.3),
        torque_upper=np.full(7, 0.3),
        dt=0.002,
        substeps=4,
    )


def test_explicit_rounding_matches_native_jit_fixture_without_so3_projection():
    actual = native_response.native_goal_rotation(DELTA)
    np.testing.assert_array_equal(actual, NATIVE_JIT_MATRIX)
    assert actual.dtype == np.float64
    assert np.max(abs(actual - Rotation.from_rotvec(DELTA).as_matrix())) > 1e-8
    assert np.linalg.norm(actual.T @ actual - np.eye(3)) > 1e-8


def test_native_goal_is_a_world_increment_anchored_once_and_inputs_remain_owned():
    for update in (False, True):
        arguments = _arguments()
        initial_pose = _dynamics(arguments["q"], arguments["velocity"])["pose"]
        initial_q, initial_v = arguments["q"].copy(), arguments["velocity"].copy()
        seen = []

        def dynamics(q, velocity):
            seen.append((q.copy(), velocity.copy()))
            state = _dynamics(q, velocity)
            q[:] = 1000.0
            velocity[:] = -1000.0
            return state

        arguments["dynamics"] = dynamics
        actual = native_response.rollout_native_goal_joint_response(**arguments, update_matrices=update)
        np.testing.assert_array_equal(actual["goal"][:3, :3], NATIVE_JIT_MATRIX @ initial_pose[:3, :3])
        np.testing.assert_array_equal(actual["goal"][:3, 3], initial_pose[:3, 3] + arguments["physical_delta"][:3])
        np.testing.assert_array_equal(arguments["q"], initial_q)
        np.testing.assert_array_equal(arguments["velocity"], initial_v)
        np.testing.assert_array_equal(
            np.asarray([value[0] for value in seen]), np.vstack((initial_q, actual["joint_position"]))
        )
        np.testing.assert_array_equal(
            np.asarray([value[1] for value in seen]), np.vstack((initial_v, actual["joint_velocity"]))
        )
        assert actual["dynamics_queries"] == 5
        assert not np.array_equal(actual["endpoint_pose"][:3, 3], initial_pose[:3, 3])


def test_only_goal_construction_changes_the_frozen_numerical_recurrence(monkeypatch):
    # Restoring the original double-precision goal constructor must make every
    # output identical, including clipping, fixed/updated M/J and passive terms.
    monkeypatch.setattr(native_response, "native_goal_rotation", lambda delta: Rotation.from_rotvec(delta).as_matrix())
    for update in (False, True):
        arguments = _arguments()
        actual = native_response.rollout_native_goal_joint_response(**arguments, update_matrices=update)
        original = rollout_nonlinear_joint_response(**arguments, update_matrices=update)
        assert set(actual) == set(original)
        for name in actual:
            np.testing.assert_array_equal(actual[name], original[name], err_msg=name)
        assert np.any(actual["clipped"])


def test_zero_rotation_still_requires_missing_latch_state_before_any_query():
    np.testing.assert_array_equal(native_response.native_goal_rotation(np.zeros(3)), np.eye(3))

    def forbidden(q, velocity):
        raise AssertionError("No dynamics query is valid without the previous orientation goal")

    arguments = _arguments()
    arguments["physical_delta"][3:] = 0.0
    arguments["dynamics"] = forbidden
    with np.testing.assert_raises_regex(ValueError, "nonzero rotation"):
        native_response.rollout_native_goal_joint_response(**arguments, update_matrices=True)


def test_explicit_zero_rotation_keeps_historical_goal_and_resets_position():
    history = Rotation.from_rotvec([0.0, 0.0, 0.7]).as_matrix()
    original_history = history.copy()
    for update in (False, True):
        arguments = _arguments()
        arguments["physical_delta"][3:] = 0.0
        current = _dynamics(arguments["q"], arguments["velocity"])["pose"]
        actual = native_response.rollout_native_goal_joint_response(
            **arguments, update_matrices=update, latched_rotation_goal=history
        )
        np.testing.assert_array_equal(actual["goal"][:3, :3], original_history)
        np.testing.assert_array_equal(actual["goal"][:3, 3], current[:3, 3] + arguments["physical_delta"][:3])
        np.testing.assert_array_equal(history, original_history)
        assert not np.allclose(actual["goal"][:3, :3], current[:3, :3])
        # q_z=0.15 and the retained goal is 0.7: the positive sine-angle
        # spring must act. Holding the current orientation would not saturate.
        assert actual["requested_torque"][0, 5] > 1.0
        assert actual["applied_torque"][0, 5] == 0.3


def test_nonzero_rotation_ignores_supplied_latch_in_both_matrix_modes():
    for update in (False, True):
        arguments = _arguments()
        reset = native_response.rollout_native_goal_joint_response(**arguments, update_matrices=update)
        supplied = native_response.rollout_native_goal_joint_response(
            **arguments,
            update_matrices=update,
            latched_rotation_goal=Rotation.from_euler("xyz", [0.3, -0.5, 0.2]).as_matrix(),
        )
        for name in reset:
            np.testing.assert_array_equal(supplied[name], reset[name], err_msg=name)


def test_zero_rotation_latch_must_be_finite_three_by_three_before_any_query():
    def forbidden(q, velocity):
        raise AssertionError("Invalid historical state must be rejected before querying dynamics")

    for history in (np.zeros(3), np.full((3, 3), np.nan), np.full((3, 3), np.inf)):
        arguments = _arguments()
        arguments["physical_delta"][3:] = 0.0
        arguments["dynamics"] = forbidden
        with np.testing.assert_raises(ValueError):
            native_response.rollout_native_goal_joint_response(
                **arguments, update_matrices=True, latched_rotation_goal=history
            )


def test_goal_rotation_rejects_nonfinite_or_nonvector_physical_input():
    for value in ([np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.1, 0.2], [[0.1, 0.2, 0.3]]):
        with np.testing.assert_raises(ValueError):
            native_response.native_goal_rotation(value)


def check_native_jit_fidelity():
    """Explicit independent functional check; not collected as a NumPy2 test."""
    import ast
    import hashlib
    import json
    import math
    import os
    from pathlib import Path

    import numba
    import scipy

    from s2a.artifacts import REPO

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    assert Path.cwd().resolve() == REPO
    assert np.__version__ == "1.26.4" and numba.__version__ == "0.68.0"
    source = REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/transform_utils.py"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == NATIVE_SOURCE_SHA256
    tree = ast.parse(source.read_text())
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"axisangle2quat", "quat2mat"}
    ]
    assert len(functions) == 2
    for node in functions:
        node.decorator_list = []
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "EPS"
    ]
    assert len(constants) == 1
    namespace = {"np": np, "math": math}
    module = ast.fix_missing_locations(ast.Module(body=constants + functions, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    native_quat2mat = numba.njit(cache=False)(namespace["quat2mat"])
    inputs = REPO / "outputs/tracking_v5_state_updated_response_20261028/verified.json"
    input_bytes = inputs.read_bytes()
    recorded = json.loads(input_bytes)
    assert recorded["complete"] is recorded["passed"] is True
    assert [entry["step"] for entry in recorded["entries"]] == list(range(368, 377))
    cases = [np.asarray(entry["delta"])[3:] for entry in recorded["entries"]]
    cases += list(np.random.default_rng(42).uniform(-3.0, 3.0, (1000, 3)))
    cases += [np.zeros(3), np.array([1e-12, -2e-12, 3e-12]), np.array([np.pi, 0.0, 0.0])]
    max_difference, max_python_jit_difference = 0.0, 0.0
    for delta in cases:
        quaternion = namespace["axisangle2quat"](delta)
        expected = native_quat2mat(quaternion)
        actual = native_response.native_goal_rotation(delta)
        max_difference = max(max_difference, float(np.max(abs(actual - expected))))
        max_python_jit_difference = max(
            max_python_jit_difference, float(np.max(abs(namespace["quat2mat"](quaternion) - expected)))
        )
        np.testing.assert_array_equal(actual, expected)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert inputs.read_bytes() == input_bytes
    return dict(
        passed=True,
        seed=42,
        cases=len(cases),
        max_native_jit_difference=max_difference,
        max_python_only_vs_jit_difference=max_python_jit_difference,
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        native_source=dict(path=str(source), sha256=source_sha),
        inputs=dict(path=str(inputs), sha256=hashlib.sha256(input_bytes).hexdigest()),
        physics_steps=0,
        world_model_calls=0,
        fresh_cases=0,
        control_qualified=False,
        scope="These goal conversions on this pinned native runtime; no general backend or response qualification",
    )


if __name__ == "__main__":
    import json

    print(json.dumps(check_native_jit_fidelity(), indent=2))
