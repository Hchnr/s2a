"""Synthetic checks of what saved finite differences can, and cannot, establish."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.saved_trajectory_differences import diagnose_saved_differences

STEP = 1e-4
PREVIOUS = np.zeros(6)
LOWER = np.vstack((np.full(6, -0.25), np.full(6, -1.0)))
UPPER = -LOWER


def affine_scores(commands):
    """Independent affine model of the four normalized squared pose errors."""
    coefficients = np.arange(1, 49, dtype=float).reshape(4, 12) / 100.0
    return (np.array([0.36, 0.49, 1.44, 1.21]) + coefficients @ commands.ravel()).reshape(2, 2)


def error_from_scores(scores):
    errors = np.zeros((2, 6))
    errors[:, 0] = 0.01 * np.sqrt(scores[:, 0])
    errors[:, 3] = 0.05 * np.sqrt(scores[:, 1])
    return errors


def trace_entry(commands, *, scores=affine_scores, safety=lambda _: 0.1, errors=None):
    commands = np.array(commands, dtype=float, copy=True)
    errors = error_from_scores(scores(commands)) if errors is None else np.array(errors, copy=True)
    normalized = np.column_stack(
        (np.linalg.norm(errors[:, :3], axis=1) ** 2 / 0.01**2, np.linalg.norm(errors[:, 3:], axis=1) ** 2 / 0.05**2)
    )
    failures = np.flatnonzero(np.any(normalized > 1.0, axis=1))
    prefix = int(failures[0]) if len(failures) else 2
    safety_min = float(safety(commands))
    rate = np.diff(np.vstack((PREVIOUS, commands)), axis=0)
    joint_feasible = bool(
        np.all(commands >= -1.0 - 1e-12)
        and np.all(commands <= 1.0 + 1e-12)
        and np.all(0.25 - np.abs(rate) >= -1e-12)
        and safety_min >= -1e-10
    )
    return dict(
        commands=commands,
        tcp_errors=errors,
        safety_min=safety_min,
        joint_feasible=joint_feasible,
        prefix_length=prefix,
        feasible=joint_feasible and prefix == 2,
    )


def answer_for(commands, *, phases=None, scores=affine_scores, safety=lambda _: 0.1):
    entries = [trace_entry(command, scores=scores, safety=safety) for command in commands]
    return dict(
        horizon=2,
        finite_difference_step=STEP,
        lower_command=LOWER.copy(),
        upper_command=UPPER.copy(),
        evaluated_candidates=entries,
        phases=phases
        if phases is not None
        else [dict(prefix_before=1, target_cursor=1, evaluations_before=1, evaluations_after=len(entries))],
        selected=dict(commands=entries[-1]["commands"].copy()),
    )


def probes_for(base):
    probes, deltas = [], []
    for axis in range(12):
        delta = STEP if base.ravel()[axis] + STEP <= UPPER.ravel()[axis] else -STEP
        changed = base.copy()
        changed.ravel()[axis] += delta
        probes.append(changed)
        deltas.append(delta)
    return probes, np.array(deltas)


def complete_case(*, base=None, trial=None, safety=lambda _: 0.1):
    base = np.zeros((2, 6)) if base is None else np.array(base, copy=True)
    trial = base + np.linspace(-0.03, -0.01, 12).reshape(2, 6) if trial is None else trial
    probes, deltas = probes_for(base)
    return answer_for([base, *probes, trial], safety=safety), deltas


def diagnose(answer):
    return diagnose_saved_differences(answer, PREVIOUS, target_index=1)


def test_full_axis_group_predicts_affine_squared_errors_and_preserves_inputs():
    answer, deltas = complete_case()
    answer["phases"][0]["last_callback_commands"] = answer["evaluated_candidates"][-1]["commands"].copy()
    before = deepcopy(answer)
    result = diagnose(answer)

    assert len(result["complete_groups"]) == len(result["pairs"]) == 1
    group = result["complete_groups"][0]
    assert group["base_index"] == 0
    assert group["probe_indices"] == list(range(1, 13))
    np.testing.assert_array_equal(group["nominal_deltas"], deltas)
    pair = result["pairs"][0]
    assert pair["trial_index"] == 13
    assert pair["main_subset"]
    expected_base = affine_scores(answer["evaluated_candidates"][0]["commands"])
    expected_trial = affine_scores(answer["evaluated_candidates"][13]["commands"])
    np.testing.assert_allclose(pair["qbase"], expected_base, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pair["qpred"], expected_trial, rtol=0, atol=2e-11)
    np.testing.assert_allclose(pair["qactual"], expected_trial, rtol=0, atol=1e-14)
    expected_change = float(np.max(expected_trial[1]) - np.max(expected_base[1]))
    assert pair["target"]["predicted_change"] == pytest.approx(expected_change, abs=2e-11)
    assert pair["target"]["actual_change"] == pytest.approx(expected_change, abs=1e-14)
    assert pair["target"]["prediction_residual"] == pytest.approx(0.0, abs=2e-11)
    assert pair["target"]["descent_ratio"] == pytest.approx(1.0, abs=2e-9)
    displacement = answer["evaluated_candidates"][13]["commands"].ravel()
    assert pair["step"]["linf"] == pytest.approx(np.max(np.abs(displacement)))
    assert pair["step"]["rms"] == pytest.approx(np.sqrt(np.mean(displacement**2)))
    assert pair["step"]["ratio_to_h"] == pytest.approx(300.0)
    summary = result["summaries"]["main_subset"]
    assert summary["count"] == 1
    assert summary["target_prediction_residual"]["maximum_absolute"] < 2e-11
    assert [row["count"] for row in summary["step_bins"]] == [0, 0, 0, 1, 0]
    assert result["links"]["callbacks"][0]["trace_index"] == 13
    assert result["links"]["callbacks"][0]["exact_match"]
    for actual, expected in zip(answer["evaluated_candidates"], before["evaluated_candidates"], strict=True):
        np.testing.assert_array_equal(actual["commands"], expected["commands"])
        np.testing.assert_array_equal(actual["tcp_errors"], expected["tcp_errors"])
    np.testing.assert_array_equal(answer["lower_command"], before["lower_command"])
    np.testing.assert_array_equal(answer["upper_command"], before["upper_command"])


def test_upper_bound_uses_exact_backward_probe_and_nominal_denominator():
    base = np.full((2, 6), 0.2)
    base[0, 0] = 0.25
    answer, deltas = complete_case(base=base, trial=base - 0.2)
    result = diagnose(answer)
    group = result["complete_groups"][0]

    assert deltas[0] == -STEP
    assert np.all(deltas[1:] == STEP)
    np.testing.assert_array_equal(group["nominal_deltas"], deltas)
    actual_delta = answer["evaluated_candidates"][1]["commands"][0, 0] - base[0, 0]
    assert actual_delta != -STEP
    assert group["nominal_deltas"][0] == -STEP
    pair = result["pairs"][0]
    # The archived TCP errors have their own rounding, so derive secants from
    # those errors below, while the affine truth is checked independently.
    archived = []
    for row in answer["evaluated_candidates"][:13]:
        errors = row["tcp_errors"]
        archived.append(
            np.column_stack(
                (np.sum(errors[:, :3] ** 2, axis=1) / 0.01**2, np.sum(errors[:, 3:] ** 2, axis=1) / 0.05**2)
            )
        )
    secants = (np.array(archived[1:]) - archived[0]) / deltas[:, None, None]
    expected = archived[0] + np.tensordot(
        answer["evaluated_candidates"][-1]["commands"].ravel() - base.ravel(), secants, axes=1
    )
    np.testing.assert_allclose(pair["qpred"], expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        pair["qpred"], affine_scores(answer["evaluated_candidates"][-1]["commands"]), rtol=0, atol=2e-11
    )


def test_cached_base_can_start_later_phase_but_all_probes_belong_to_that_phase():
    base = np.zeros((2, 6))
    probes, _ = probes_for(base)
    trial = np.full((2, 6), -0.02)
    answer = answer_for(
        [base, np.full((2, 6), -0.05), *probes, trial],
        phases=[
            dict(prefix_before=1, target_cursor=1, evaluations_before=1, evaluations_after=2),
            dict(prefix_before=1, target_cursor=1, evaluations_before=2, evaluations_after=15),
        ],
    )
    result = diagnose(answer)
    group = next(group for group in result["complete_groups"] if group["base_index"] == 0)
    assert group["phase_index"] == 1
    assert group["base_cached_before_phase"]
    assert group["probe_indices"] == list(range(2, 14))
    assert result["pairs"][0]["trial_index"] == 14


@pytest.mark.parametrize(
    "defect", ["missing", "cached_axis", "out_of_order", "noncontiguous", "partial", "cross_phase"]
)
def test_incomplete_or_misattributed_difference_groups_cannot_support_predictions(defect):
    base = np.zeros((2, 6))
    probes, _ = probes_for(base)
    trial = np.full((2, 6), -0.02)
    commands = [base, *probes, trial]
    base_index = 0
    phases = None
    if defect == "missing":
        del commands[5]
    elif defect == "cached_axis":
        commands = [probes[0], base, *probes[1:], trial]
        base_index = 1
    elif defect == "out_of_order":
        commands[2], commands[3] = commands[3], commands[2]
    elif defect == "noncontiguous":
        commands.insert(6, np.full((2, 6), -0.05))
    elif defect == "partial":
        commands = commands[:7]
    elif defect == "cross_phase":
        phases = [
            dict(prefix_before=1, target_cursor=1, evaluations_before=1, evaluations_after=7),
            dict(prefix_before=1, target_cursor=1, evaluations_before=7, evaluations_after=14),
        ]
    result = diagnose(answer_for(commands, phases=phases))
    assert all(group["base_index"] != base_index for group in result["complete_groups"])
    assert result["pairs"] == []
    reason = {
        "missing": "missing_probe",
        "cached_axis": "cached_or_self_probe",
        "out_of_order": "out_of_order_probe",
        "noncontiguous": "noncontiguous_probe",
        "partial": "partial_group",
        "cross_phase": "cross_phase",
    }[defect]
    assert result["exclusions"]["group_counts"][reason] > 0


def test_adjacent_trial_that_is_a_probe_of_any_earlier_base_is_excluded_without_skipping():
    base = np.zeros((2, 6))
    probes, _ = probes_for(base)
    nested_probe = probes[0].copy()
    nested_probe[0, 1] += STEP
    # This row is not one of base 0's own probes. It is an exact axis probe
    # of row 1, which has no complete derivative group of its own.
    answer = answer_for([base, *probes, nested_probe, np.full((2, 6), -0.02)])
    result = diagnose(answer)

    assert len(result["complete_groups"]) == 1
    assert result["pairs"] == []
    assert result["exclusions"]["adjacent_trial_counts"]["earlier_base_fit_probe"] == 1


@pytest.mark.parametrize("different_phase", [False, True])
def test_complete_group_needs_an_adjacent_trial_in_its_own_phase(different_phase):
    answer, _ = complete_case()
    if different_phase:
        answer["phases"] = [
            dict(prefix_before=1, target_cursor=1, evaluations_before=1, evaluations_after=13),
            dict(prefix_before=1, target_cursor=1, evaluations_before=13, evaluations_after=14),
        ]
    else:
        answer["evaluated_candidates"].pop()
        answer["selected"]["commands"] = answer["evaluated_candidates"][-1]["commands"].copy()
        answer["phases"][0]["evaluations_after"] = 13
    result = diagnose(answer)
    assert len(result["complete_groups"]) == 1
    assert result["pairs"] == []
    reason = "different_phase" if different_phase else "no_next_row"
    assert result["exclusions"]["adjacent_trial_counts"][reason] == 1


def test_selected_probe_is_not_validation_and_callback_link_requires_exact_bytes():
    answer, _ = complete_case()
    answer["selected"]["commands"] = answer["evaluated_candidates"][4]["commands"].copy()
    callback = answer["evaluated_candidates"][-1]["commands"].copy()
    callback[0, 0] = np.nextafter(callback[0, 0], np.inf)
    answer["phases"][0]["last_callback_commands"] = callback
    result = diagnose(answer)

    selected = result["links"]["selected"]
    assert selected["exact_match"]
    assert selected["trace_index"] == 4
    assert selected["fit_probe_of"]
    assert all(pair["trial_index"] != 4 for pair in result["pairs"])
    linked_callback = result["links"]["callbacks"][0]
    assert linked_callback["trace_index"] is None
    assert not linked_callback["exact_match"]


def test_one_ulp_perturbation_does_not_match_an_expected_probe():
    answer, _ = complete_case()
    changed = answer["evaluated_candidates"][6]["commands"].copy()
    changed[0, 5] = np.nextafter(changed[0, 5], np.inf)
    answer["evaluated_candidates"][6] = trace_entry(changed)

    result = diagnose(answer)

    assert result["complete_groups"] == []
    assert result["pairs"] == []


def test_scalar_minimum_secants_cannot_certify_all_safety_constraints():
    base = np.zeros((2, 6))
    base[0, :2] = 0.05
    trial = base.copy()
    trial[0, 0] = -0.01
    answer, _ = complete_case(base=base, trial=trial, safety=lambda x: min(x[0, 0], x[0, 1]))
    pair = diagnose(answer)["pairs"][0]

    # Both +h coordinate probes leave min(x,y) unchanged at x=y. A linear
    # model of each underlying constraint would instead predict [-.01,.05].
    expected_full_slack = np.array([0.05, 0.05]) + (trial - base)[0, :2]
    np.testing.assert_allclose(expected_full_slack, [-0.01, 0.05], rtol=0, atol=1e-16)
    assert pair["scalar_safety"]["predicted"] == pytest.approx(0.05, abs=1e-15)
    assert pair["scalar_safety"]["actual"] == pytest.approx(-0.01, abs=1e-15)
    assert pair["scalar_safety"]["prediction_residual"] == pytest.approx(-0.06, abs=1e-15)
    assert not pair["scalar_safety"]["full_safety_jacobian_available"]
    assert pair["main_subset"]  # Failed trials remain evidence of a bad prediction.
    assert not answer["evaluated_candidates"][-1]["joint_feasible"]


def test_protected_pose_cap_rejects_a_single_ulp_excess():
    answer, _ = complete_case()
    trial = answer["evaluated_candidates"][-1]["commands"]
    errors = error_from_scores(affine_scores(trial))
    errors[0, :3] = [np.nextafter(0.01, np.inf), 0.0, 0.0]
    answer["evaluated_candidates"][-1] = trace_entry(trial, errors=errors)
    pair = diagnose(answer)["pairs"][0]

    assert pair["prefix"]["predicted_passed"]
    assert pair["prefix"]["actual_max_cap_excess"] > 0.0
    assert not pair["prefix"]["actual_passed"]


@pytest.mark.parametrize("violation", [0.5e-12, 2e-12])
@pytest.mark.parametrize("period", [0, 1])
def test_original_rate_tolerance_is_applied_to_first_and_between_period_commands(violation, period):
    trial = np.full((2, 6), -0.02)
    trial[period, 0] = (0.25 if period == 0 else trial[0, 0] + 0.25) + violation
    if period == 0:
        trial[1, 0] = trial[0, 0]
    answer, _ = complete_case(trial=trial)
    if period == 0 and violation > 1e-12:
        # The solver rejects this before evaluating a response: its first-row
        # bounds already contain the previous-command rate intersection.
        with pytest.raises(ValueError):
            diagnose(answer)
        return
    result = diagnose(answer)
    assert len(result["pairs"]) == 1
    assert answer["evaluated_candidates"][-1]["joint_feasible"] is (violation < 1e-12)
    audit = result["pairs"][0]["trial_command_audit"]
    assert audit["box_feasible"]
    assert audit["rate_feasible"] is (violation < 1e-12)
    assert audit["rate_minimum"] == pytest.approx(-violation, abs=3e-17)
    # A trace that falsely claims the opposite side of the original rate
    # qualification must be rejected, even though the discrepancy is tiny.
    answer["evaluated_candidates"][-1]["joint_feasible"] = violation >= 1e-12
    with pytest.raises(ValueError):
        diagnose(answer)


@pytest.mark.parametrize("displacement", [-0.1, 0.1])
def test_extrapolation_keeps_negative_predictions_and_non_descent_ratio_is_absent(displacement):
    def scores(commands):
        return np.array([[0.36, 0.49], [1.44, 1.21]]) * np.array([[1.0], [np.exp(20.0 * commands[0, 0])]])

    base = np.zeros((2, 6))
    probes, _ = probes_for(base)
    trial = base.copy()
    trial[0, 0] = displacement
    result = diagnose(answer_for([base, *probes, trial], scores=scores))
    pair = result["pairs"][0]
    expected_prediction = scores(base) + (scores(probes[0]) - scores(base)) / STEP * displacement

    np.testing.assert_allclose(pair["qpred"], expected_prediction, rtol=0, atol=1e-12)
    assert np.all(np.asarray(pair["qactual"]) > 0.0)
    if displacement < 0:
        assert np.all(np.asarray(pair["qpred"])[1] < 0.0)
    else:
        assert pair["target"]["predicted_change"] > 0.0
        assert pair["target"]["actual_change"] > 0.0
        assert pair["target"]["descent_ratio"] is None


@pytest.mark.parametrize("safety_min", [0.5e-8, 1e-8])
def test_main_subset_buffer_qualifies_only_the_base_without_hiding_trial_errors(safety_min):
    answer, _ = complete_case(safety=lambda _: safety_min)
    result = diagnose(answer)

    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["main_subset"] is (safety_min >= 1e-8)
    assert result["summaries"]["all_pairs"]["count"] == 1
    assert result["summaries"]["main_subset"]["count"] == int(safety_min >= 1e-8)


@pytest.mark.parametrize(
    "corruption", ["duplicate", "command_nan", "error_inf", "safety_nan", "bounds_inf", "previous_nan"]
)
def test_duplicate_or_nonfinite_saved_data_is_rejected(corruption):
    answer, _ = complete_case()
    previous = PREVIOUS.copy()
    if corruption == "duplicate":
        answer["evaluated_candidates"][-1] = deepcopy(answer["evaluated_candidates"][0])
    elif corruption == "command_nan":
        answer["evaluated_candidates"][2]["commands"][0, 2] = np.nan
    elif corruption == "error_inf":
        answer["evaluated_candidates"][2]["tcp_errors"][0, 2] = np.inf
    elif corruption == "safety_nan":
        answer["evaluated_candidates"][2]["safety_min"] = np.nan
    elif corruption == "bounds_inf":
        answer["upper_command"][1, 2] = np.inf
    elif corruption == "previous_nan":
        previous[0] = np.nan
    with pytest.raises(ValueError):
        diagnose_saved_differences(answer, previous, target_index=1)
