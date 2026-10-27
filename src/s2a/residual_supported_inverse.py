"""175 fixed half-weight learned branch for a causal residual physical candidate."""

from s2a.support_aware_inverse import support_aware_translation


class HalfLearningSupport:
    def __init__(self, support):
        self.support = support

    def query(self, point):
        result = dict(self.support.query(point))
        result["learned_weight"] *= 0.5
        return result


def residual_supported_translation(*args, support, **kwargs):
    """The caller supplies173 causal candidate and commits the final actual command."""
    return support_aware_translation(*args, support=HalfLearningSupport(support), **kwargs)
