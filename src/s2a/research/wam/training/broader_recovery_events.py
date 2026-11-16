"""Offline event summaries and current-image candidates for recorded trajectories."""

import numpy as np

from s2a.research.wam.training.recovery_h40_contract import window_indices


def event_candidates(openings, successes, *, anchors=(0, 16, 56), threshold=0.35, lead=8, horizon=40):
    """Select first measured low-opening and task-success events, never online inputs."""
    opening, success = np.asarray(openings, dtype=float), np.asarray(successes)
    if (
        opening.ndim != 1
        or len(opening) < 41
        or not np.isfinite(opening).all()
        or np.any((opening < 0) | (opening > 1))
        or success.shape != opening.shape
        or success.dtype != np.bool_
        or not np.isfinite(threshold)
        or not 0 < threshold < 1
        or type(lead) is not int
        or not 0 < lead <= horizon
    ):
        raise ValueError("Measured opening, boolean native outcomes and finite event limits are required")
    periods = len(opening) - 1
    covered = sorted({step for row in window_indices(anchors, horizon, periods)["windows"] for step in row})
    kinds = {"low_opening": opening[1:] < threshold, "native_success": success[1:]}
    candidates, summaries = {}, {}
    for kind, mask in kinds.items():
        indices = np.flatnonzero(mask)
        first = None if not len(indices) else int(indices[0])
        summaries[kind] = dict(
            total=int(len(indices)),
            covered=int(np.count_nonzero(mask[covered])),
            first_action_index=first,
            last_action_index=None if not len(indices) else int(indices[-1]),
        )
        if first is not None:
            anchor = min(max(0, first - lead + 1), periods - horizon)
            candidate = candidates.setdefault(
                anchor,
                dict(
                    anchor_step=anchor,
                    events=[],
                    current_image_available=anchor in anchors,
                    full_h40=True,
                    selection_uses_future_training_labels=True,
                ),
            )
            candidate["events"].append(
                dict(kind=kind, action_index=first, endpoint_index=first + 1, actual_lead_periods=first + 1 - anchor)
            )
    return dict(
        periods=periods,
        reset_success=bool(success[0]),
        covered_case_transitions=len(covered),
        low_opening=summaries["low_opening"],
        native_success=summaries["native_success"],
        candidates=[candidates[key] for key in sorted(candidates)],
    )
