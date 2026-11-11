"""Independent scalar audit catches a changed target or a concealed refusal."""

from itertools import product

import numpy as np
import pytest

from s2a.geometry import transform
from s2a.research.joint_control.tracking.tabletop_reference_projection import project_tabletop_reference
from scripts.verification.reference_ports import audit_tabletop_reference_projection as audit


def example(limit=0.05):
    vertices = np.array(list(product((-0.01, 0.01), repeat=3)))
    target = np.eye(4)
    table = transform(np.array([0.0, 0.0, -0.05]), np.eye(3))
    size = np.array([0.2, 0.2, 0.05])
    config = {"clearance_m": 0.001, "numerical_margin_m": 1e-9, "max_translation_m": limit}
    variant = {"position_uncertainty_m": 0.0, "rotation_uncertainty_rad": 0.0, "shape_uncertainty_m": 0.0}
    return vertices, target, table, size, config, variant


def test_scalar_audit_uses_clearance_and_cap():
    vertices, target, table, size, config, variant = example()
    result = audit.scalar_projection(vertices, target, table, size, config, variant)
    assert result["overlap"] and result["accepted"] and result["status"] == "projected"
    assert result["original_lower"] == pytest.approx(-0.01, abs=1e-12)
    assert result["required"] == pytest.approx(0.011000001, abs=1e-12)
    config["max_translation_m"] = 0.005
    refused = audit.scalar_projection(vertices, target, table, size, config, variant)
    assert not refused["accepted"] and refused["status"] == "correction_exceeds_limit"
    target[0, 3] = 0.3
    clear = audit.scalar_projection(vertices, target, table, size, config, variant)
    assert not clear["overlap"] and clear["status"] == "unchanged" and clear["required"] == 0


def test_audit_rejects_tampered_projected_reference():
    vertices, target, table, size, config, variant = example()
    expected = audit.scalar_projection(vertices, target, table, size, config, variant)
    result = project_tabletop_reference(
        target,
        vertices,
        table,
        size,
        position_uncertainty=0,
        rotation_uncertainty=0,
        shape_uncertainty=0,
        clearance=config["clearance_m"],
        numerical_margin=config["numerical_margin_m"],
        max_translation=config["max_translation_m"],
    )
    result["projected_proxy_finger_table_contact_distances_m"] = []
    result["current_tcp_error_to_projected"] = np.r_[result["projected_tcp"][:3, 3], [0.0, 0.0, 0.0]]
    audit.check_result(result, expected, target, table, np.eye(4))
    result["projected_tcp"][0, 3] += 0.001
    with pytest.raises(AssertionError):
        audit.check_result(result, expected, target, table, np.eye(4))
