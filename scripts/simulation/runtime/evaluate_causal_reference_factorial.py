"""Score selected mixed components and the three original references."""

from s2a.geometry import pose_error
from scripts.simulation.runtime import evaluate_causal_event_bridge as bridge_scoring


def score_completed(env, case, selected, bridge, offset, own_error, *, expected=None):
    """Read only the completed physical endpoint and label each reference."""
    result = bridge_scoring.score_completed(env, case, selected, offset, own_error)
    for key in ("reference_errors", "opening_targets", "opening_errors"):
        result[key]["selected"] = result[key].pop("bridge")
    result["reference_errors"]["bridge"] = pose_error(bridge["poses"][offset], result["actual_tcp"])
    target = float(bridge["openings"][offset])
    result["opening_targets"]["bridge"] = target
    result["opening_errors"]["bridge"] = result["gripper"]["opening"] - target
    if expected is not None:
        bridge_scoring.compare_scores(result, expected, names=("latest", "remembered", "bridge", "selected"))
    return result
