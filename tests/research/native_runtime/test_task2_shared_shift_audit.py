"""First-pose eligibility must not imply two-point or safety qualification."""

import numpy as np
import pytest

from scripts.tracking.trajectory_search import audit_task2_shared_shift as runner
from scripts.tracking.trajectory_search import audit_task2_shifted_seed as previous


def test_first_pose_candidate_requires_safety_and_does_not_imply_second_point():
    errors = np.zeros((8, 6))
    errors[0, 0] = 0.009
    errors[1, 0] = 0.012
    result = runner.qualify(errors, True)
    assert result["model_first_pose_candidate"] is True
    assert result["model_prefix_candidate"] is False and not bool(result["point_pose_caps_satisfied"][1])
    assert runner.qualify(errors, False)["model_first_pose_candidate"] is False
    errors[0, 3] = 0.051
    assert runner.qualify(errors, True)["model_first_pose_candidate"] is False


def test_invalid_error_shape_and_ambiguous_safety_are_rejected():
    with pytest.raises(ValueError):
        runner.qualify(np.zeros((7, 6)), True)
    with pytest.raises(ValueError, match="Explicit independent safety"):
        runner.qualify(np.zeros((8, 6)), 1)


def test_isolated_worker_preserves_prior_namespace():
    worker = runner.isolated_worker()
    assert worker.__code__ is previous.worker.__code__ and worker.__globals__ is not previous.worker.__globals__
    assert worker.__globals__["OUTPUT"] == runner.OUTPUT and previous.OUTPUT != runner.OUTPUT
    assert worker.__globals__["analyze"] is runner.analyze
