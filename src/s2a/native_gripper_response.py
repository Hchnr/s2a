"""Fixed training-only command intervention schedule; no observation-dependent policy."""

import numpy as np


def intervention_commands(sent_commands, anchor, override):
    sent = np.asarray(sent_commands)
    if sent.ndim != 2 or sent.shape[1] != 7 or not np.isfinite(sent).all():
        raise ValueError("Finite native7 command sequence required")
    if type(anchor) is not int or anchor < 0 or anchor + 24 > len(sent):
        raise ValueError("Registered anchor must have24 existing command periods")
    if override is not None and (isinstance(override, bool) or override not in (-1.0, 0.0, 1.0)):
        raise ValueError("Override must be native-1/0/+1 or explicit baselineNone")
    selected = sent[: anchor + 24].copy()
    if override is not None:
        selected[anchor : anchor + 8, -1] = override
    return selected
