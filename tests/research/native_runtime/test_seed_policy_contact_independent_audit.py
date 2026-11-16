"""Independent placement auditing rejects inconsistent labels and missing futures."""

import copy
import json

import pytest

from scripts.verification.geometry import audit_seed_policy_contacts as audit


def wine_row():
    return dict(
        support_pairs=[dict(target_geom="bottle", other_geom="rack", distance_m=-0.001)],
        conditions=dict(
            support_contact=True,
            spatial_satisfied=True,
            reconstructed_success=True,
            site_local_m=[0.01, 0.01, 0.02],
            xy_clearance_m=[0.09, 0.01],
            lower_height_clearance_m=0.022,
            upper_height_clearance_m=0.083,
        ),
    )


def test_independent_wine_boundaries_detect_tampered_clearance_or_label():
    row = wine_row()
    assert audit.scalar_conditions(9, row, [0.1, 0.02, 0.003], 1e-10) is True
    changed = copy.deepcopy(row)
    changed["conditions"]["xy_clearance_m"][1] += 0.001
    with pytest.raises(ValueError, match="clearances"):
        audit.scalar_conditions(9, changed, [0.1, 0.02, 0.003], 1e-10)
    changed = copy.deepcopy(row)
    changed["conditions"]["reconstructed_success"] = False
    with pytest.raises(ValueError, match="predicate"):
        audit.scalar_conditions(9, changed, [0.1, 0.02, 0.003], 1e-10)


def test_a_partial_trajectory_tail_is_never_called_final():
    row = wine_row()
    row.update(native_success=True, native_grasp=False)
    partial = audit.tail_summary([row] * 40, 100, 9)
    assert partial["available"] is True and partial["full_trajectory"] is False
    assert partial["native_success"] == 40
    short = audit.tail_summary([row] * 3, 100, 9)
    assert short["available"] is False and short["full_trajectory"] is False


def test_sources_and_zero_new_simulation_budget_are_fixed():
    config = json.loads(audit.CONFIG.read_text())
    audit.validate_config(config)
    for change in (dict(forwards=1), dict(gpu_calls=1), dict(endpoints=9457)):
        with pytest.raises(ValueError):
            audit.validate_config(dict(config, **change))


def test_actual_wine_case_recomputes_all_saved_labels_and_tail_boundaries():
    config, identity, reconstruction, _audits, _first, mapping = audit.inputs()
    result = audit.audit_case(12, config, identity, reconstruction, mapping)
    fast, bounded = result["tails"]["fast"], result["tails"]["bounded"]
    assert result["checked_endpoints"] == 456
    assert fast["full_trajectory"] is bounded["full_trajectory"] is True
    assert fast["support_contact"] == bounded["support_contact"] == 40
    assert fast["native_success"] == 0 and bounded["native_success"] == 40
    assert fast["axis_1_clearance_m"]["median"] < 0 < bounded["axis_1_clearance_m"]["median"]
