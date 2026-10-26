"""112 qualified current wrench followed by complete-episode causal differences."""

from arm_wrench_data import attach_current_wrench as previous_attach

from s2a.causal_wrench_change_forward import attach_wrench_change


def attach_current_wrench(data, entry, config, e):
    return attach_wrench_change(previous_attach(data, entry, config, e))
