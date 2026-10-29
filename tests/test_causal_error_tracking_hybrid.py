"""The execution adapter must retain tightened constraints or report no action."""

import numpy as np
import pytest
from test_configurable_tracking_terminal_plan import inputs

from s2a.causal_error_tracking_hybrid import NoCausalWitness, refine_long_plan


def test_tightened_witness_is_preserved_by_execution_adapter():
    result = refine_long_plan(*inputs(), past_position_error_rad=np.full(7, 0.003))
    assert result["selection"] == "interior_witness"
    assert result["minimum_tightened_inner_slack_rad"] >= -1e-10
    assert result["soft_response_evaluations"] == 0
    np.testing.assert_array_equal(result["past_position_error_rad"], np.full(7, 0.003))


def test_infeasible_error_budget_has_no_relaxed_fallback_action():
    with pytest.raises(NoCausalWitness) as error:
        refine_long_plan(*inputs(), past_position_error_rad=np.full(7, 0.1))
    assert error.value.diagnostic["witness_found"] is False
    assert "command" not in error.value.diagnostic
