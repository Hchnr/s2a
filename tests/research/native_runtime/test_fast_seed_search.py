"""Fixed-state comparison preserves candidate identity and resource accounting."""

import copy
import json

import numpy as np
import pytest

from scripts.simulation.runtime import compare_fast_seed_search as run
from tests.research.trajectory_search.test_full_horizon_quality import fixture


def sample_fixture():
    args, source = fixture()
    _model, velocity, seed, previous, interface, site, tool, targets, twists, lower, upper = args
    mass, jacobian, kp, kd, q, _velocity = source
    return dict(
        step=352,
        mass=mass,
        jacobian=jacobian,
        kp=kp,
        kd=kd,
        q=q,
        velocity=velocity,
        initial_joint=q.copy(),
        seed=seed,
        previous=previous,
        site=site,
        site_from_tcp=tool,
        targets=targets,
        twists=twists,
        lower=lower,
        upper=upper,
        past_inner_error_rad=np.zeros((25, 7)),
        contract={
            key: getattr(interface, key).tolist() for key in ("input_min", "input_max", "output_min", "output_max")
        },
    )


def ledger():
    return dict(
        solver_attempted=0,
        solver_completed=0,
        responses_attempted=0,
        responses_completed=0,
        audits_attempted=0,
        audits_completed=0,
        calls=[],
    )


def test_fixed_sample_runs_paired_solvers_and_original_audits():
    counts = ledger()
    result = run.run_sample(sample_fixture(), counts)
    assert [row["order"] for row in result["pairs"]] == [["baseline", "fast"], ["fast", "baseline"]]
    assert all(row["early_selected"] and row["max_selected_rollout_difference"] == 0 for row in result["pairs"])
    assert counts["solver_attempted"] == counts["solver_completed"] == 4
    assert counts["audits_attempted"] == counts["audits_completed"] == 4
    assert counts["responses_attempted"] == counts["responses_completed"]
    assert all(row["audit_passed"] is True for row in counts["calls"])


def test_changed_selected_command_cannot_pass_paired_comparison():
    def altered(*args, **kwargs):
        result = run.solve_fast_preferred_seed(*args, **kwargs)
        result["best_joint_safe"]["planned_commands"][0, 0] += 0.001
        return result

    def accept_for_comparison(witness, *_args, **_kwargs):
        return dict(
            passed=True,
            prefix_pose_caps_satisfied=True,
            braking_constraint_satisfied=True,
            tcp_errors=witness["tcp_errors"],
        )

    with pytest.raises(ValueError, match="selected plan"):
        run.run_sample(sample_fixture(), ledger(), fast=altered, auditor=accept_for_comparison)


def test_frozen_config_rejects_changed_budget_or_source():
    config = json.loads(run.CONFIG.read_text())
    run.validate_config(config)
    for change in (dict(max_responses_per_call=321), dict(physical_qualification=True), dict(gpu_calls=1)):
        with pytest.raises(ValueError, match="fixed three-state"):
            run.validate_config(dict(config, **change))
    changed = copy.deepcopy(config)
    changed["source_profile"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixed three-state"):
        run.validate_config(changed)


def test_actual_profile_and_comparison_config_identities_do_not_alias():
    config, identity, inputs = run.load_inputs()
    assert config["steps"] == [352, 355, 359]
    assert identity["comparison_config_ref"] != identity["profile_inputs"]["config_ref"]
    assert identity["profile_inputs"]["sample_fingerprints"] == run.binding.input_receipt(inputs)["sample_fingerprints"]
