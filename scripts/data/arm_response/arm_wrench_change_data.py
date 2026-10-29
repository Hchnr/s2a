"""112 qualified current wrench followed by complete-episode causal differences."""

from s2a.research.response_models.wrench.causal_wrench_change_forward import attach_wrench_change
from scripts.data.arm_response.arm_wrench_data import attach_current_wrench as previous_attach


def attach_current_wrench(data, entry, config, e):
    return attach_wrench_change(previous_attach(data, entry, config, e))
