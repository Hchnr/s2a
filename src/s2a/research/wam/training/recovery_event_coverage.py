"""Training-only event coverage and source-level validation separation."""

import numpy as np

from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from s2a.research.wam.training.recovery_h40_contract import window_indices


def validation_identity_check(groups, training_prepared, preflight, validation_prepared, verification):
    """Compare six later synchronized validation identities without future labels."""
    exclusion = training_prepared["exclusion"]
    training = {(row["task_index"], row["demo"], row["initial_state_sha256"]) for row in groups}
    fresh = {(row["task_index"], row["demo"], row["initial_state_sha256"]) for row in exclusion["fresh_sources"]}

    def identities(records):
        return {(row["task_index"], row["demo"], row["initial_state_sha256"]) for row in records}

    records = preflight["records"]
    validation = identities(records)
    prepared_validation = identities(validation_prepared["records"])
    training_keys, validation_keys, fresh_keys = (
        {(task, demo) for task, demo, _ in group} for group in (training, validation, fresh)
    )
    training_hashes, validation_hashes, fresh_hashes = (
        {sha for _, _, sha in group} for group in (training, validation, fresh)
    )
    expected_names = {f"task{task}_validation_{demo}" for task, demo in validation_keys}
    if not (
        len(groups) == len(training) == len(training_keys) == len(training_hashes) == 3
        and len(exclusion["fresh_sources"]) == len(fresh) == len(fresh_keys) == len(fresh_hashes) == 12
        and len(records)
        == len(validation_prepared["records"])
        == len(validation)
        == len(validation_keys)
        == len(validation_hashes)
        == 6
        and validation_keys == VALIDATION_DEMO_KEYS
        and validation == prepared_validation
        and not training_keys & validation_keys
        and not training_keys & fresh_keys
        and not validation_keys & fresh_keys
        and not training_hashes & validation_hashes
        and not training_hashes & fresh_hashes
        and not validation_hashes & fresh_hashes
        and preflight["complete"] is preflight["passed"] is preflight["training_fresh_identity_disjoint"] is True
        and verification["complete"] is verification["passed"] is verification["eligible_for_model_validation"] is True
        and verification["fresh_authorized"] is False
        and verification["native_stable_cases"] == 6
        and verification["physics_endpoints_verified"] == 1218
        and verification["current_images_verified"] == 24
        and len(verification["cases"]) == 6
        and {row["name"] for row in verification["cases"]} == expected_names
        and all(row["native_stable_last40"] is True for row in verification["cases"])
    ):
        raise ValueError("Synchronized validation and training/fresh demo or initial identities differ")
    return dict(
        training_groups=len(training),
        synchronized_validation_sources=len(validation),
        fresh_metadata_sources=len(fresh),
        identity_sets_disjoint=True,
        unknown_near_duplicates_excluded=False,
        validation_future_labels_read=False,
        fresh_future_labels_read=False,
    )


def event_coverage(opening, native_success, anchors=(0, 16, 56), *, threshold=0.35, lead=8, horizon=40):
    """Count actual next-endpoint events and preselect one offline training anchor."""
    opening, native_success = np.asarray(opening), np.asarray(native_success)
    if (
        opening.ndim != 1
        or len(opening) < 2
        or not np.isfinite(opening).all()
        or np.any((opening < 0) | (opening > 1))
        or native_success.shape != opening.shape
        or native_success.dtype != np.bool_
        or not 0 < threshold < 1
        or type(lead) is not int
        or not 0 < lead <= horizon
    ):
        raise ValueError("Finite measured opening and boolean endpoint outcomes are required")
    periods = len(opening) - 1
    indices = window_indices(anchors, horizon, periods)
    covered = sorted({step for window in indices["windows"] for step in window})
    low = np.flatnonzero(opening[1:] < threshold)
    success = np.flatnonzero(native_success[1:])
    covered_array = np.asarray(covered, dtype=np.int64)
    candidate = None
    if len(low) and periods >= horizon:
        first = int(low[0])
        anchor = min(max(0, first - lead + 1), periods - horizon)
        candidate = dict(
            anchor_step=anchor,
            event_action_index=first,
            event_endpoint_index=first + 1,
            event_opening=float(opening[first + 1]),
            event_lead_periods=first - anchor + 1,
            complete_horizon=True,
            selection_uses_future_training_labels=True,
            current_image_available=anchor in anchors,
        )
    return dict(
        periods=periods,
        covered_case_transitions=len(covered),
        low_opening_total=int(len(low)),
        low_opening_covered=int(np.count_nonzero(opening[1:][covered_array] < threshold)),
        first_low_action_index=None if not len(low) else int(low[0]),
        last_low_action_index=None if not len(low) else int(low[-1]),
        successful_endpoints_total=int(len(success)),
        successful_endpoints_covered=int(np.count_nonzero(native_success[1:][covered_array])),
        first_success_action_index=None if not len(success) else int(success[0]),
        selected_anchor=candidate,
    )
