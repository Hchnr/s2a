"""177 causal loading-route ablation of learned translation corrections."""

from s2a.support_aware_inverse import support_aware_translation


class LoadingPhaseSupport:
    def __init__(self, support, route):
        if route not in ("native_close_load", "model_tracking"):
            raise ValueError("Known causal gripper route required")
        self.support = support
        self.route = route

    def query(self, point):
        result = dict(self.support.query(point))
        result["learned_weight"] *= 0 if self.route == "native_close_load" else 0.5
        return result


def loading_phase_translation(*args, support, route, **kwargs):
    result = support_aware_translation(*args, support=LoadingPhaseSupport(support, route), **kwargs)
    result["arm_loading_route"] = route
    return result
