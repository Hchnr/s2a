"""Build the step-107 model source manifest without inherited binding state."""

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_replan_step107_model as model


def causal_model_source_refs():
    """Replace only the legacy owner's dynamic config and plan with step-107 refs.

    The historical adapter's source list reads two mutable module globals.
    Its previous callers bind those globals during input loading, but a fresh
    model worker must be able to reproduce the same list before model loading.
    Every other historical source reference is preserved in its original order.
    """
    inherited = model.source_refs()
    legacy_owner = model.previous.previous
    prefix.require(
        len(inherited) >= 2
        and inherited[:2] == [prefix.reference(legacy_owner.CONFIG), prefix.reference(legacy_owner.PLAN)],
        "The inherited model source list must begin with its active config and plan",
    )
    return [prefix.reference(model.CONFIG), prefix.reference(model.PLAN), *inherited[2:]]
