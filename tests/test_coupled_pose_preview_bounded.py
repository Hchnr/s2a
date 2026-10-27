import json
from pathlib import Path

import numpy as np
from coupled_pose_preview_independent import verify_preview

from s2a.coupled_pose_preview import plan_pose as original
from s2a.coupled_pose_preview_bounded import plan_pose
from s2a.coupled_response_history import CoupledResponseHistory
from s2a.current_coupled_response import build_response


def test_real_strict_commit_failure_is_bounded_and_recertified():
    p = json.loads((Path(__file__).parent / "fixtures/coupled_preview_case07_step88.json").read_text())["parameters"]
    raw = original(**p)
    # The historical JSON is the platform-independent reproducer of the strict failure.
    historical = np.array(
        [
            0.13938946819447343,
            -0.16635418282469722,
            1.0000000000000007,
            0.0713121175054731,
            -0.17106951847817872,
            0.01127915275874232,
        ]
    )
    assert historical[2] > p["contract"]["input_max"][2]
    fixed = plan_pose(**p)
    np.testing.assert_allclose(fixed["planned_commands"], raw["planned_commands"], atol=1e-9, rtol=0)
    assert np.all(fixed["planned_commands"] <= np.asarray(p["contract"]["input_max"]))
    assert np.all(fixed["planned_commands"] >= np.asarray(p["contract"]["input_min"]))
    verify_preview(fixed, **p)
    model = build_response(p["mass"], p["jacobian"], p["contract"]["kp"], p["contract"]["kd"], np.eye(3))
    history = CoupledResponseHistory()
    history.commit_executed(p["measured"], p["twist"], model, fixed["command"][:3], p["contract"])
    assert history.pending is not None
