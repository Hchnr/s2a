import numpy as np
from coupled_pose_preview_independent import verify_preview
from test_coupled_pose_preview import problem

from s2a.coupled_pose_preview_executable import plan_pose
from s2a.current_coupled_state import rotation_goal_from_command
from s2a.osc import OSCInterface


def test_zero_tracking_error_resets_stale_orientation_goal_instead_of_latching():
    p = problem()
    p["twist"] = np.zeros(6)
    p["previous_command"] = np.zeros(6)
    p["innovation"] = np.zeros(12)
    p["references"] = np.repeat(p["measured"][None], 4, axis=0)
    p["reference_twists"] = np.zeros((4, 6))
    plan = plan_pose(**p)
    assert plan["reset_representative_adjusted"] and plan["goal_mode"] == "reset"
    assert plan["command_projection_max_abs"] <= 1e-9
    interface = OSCInterface(*(p["contract"][k] for k in ("input_min", "input_max", "output_min", "output_max")))
    assert np.any(interface.decode(plan["command"])[3:] != 0)
    from scipy.spatial.transform import Rotation

    current = p["measured"][:3, :3]
    stale = Rotation.from_rotvec([0.2, 0.1, -0.3]).as_matrix() @ current
    goal, _ = rotation_goal_from_command(
        plan["command"], interface, current, stale, matrix_decoder=lambda v: Rotation.from_rotvec(v).as_matrix()
    )
    np.testing.assert_allclose(goal, current, atol=1e-9, rtol=0)
    assert np.max(np.abs(goal - stale)) > 0.1
    verify_preview(plan, **p)


def test_dynamic_plan_remains_feasible_and_independently_stationary():
    p = problem()
    plan = plan_pose(**p)
    verify_preview(plan, **p)
    assert np.all(plan["planned_commands"] <= np.asarray(p["contract"]["input_max"]))
    assert np.all(plan["planned_commands"] >= np.asarray(p["contract"]["input_min"]))
