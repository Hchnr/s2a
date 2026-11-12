import copy

import pytest

from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.trajectory_search import audit_replayed_reference_pose_bounds_v2 as recovery


def test_software_library_binding_checks_the_path_and_previously_consumed_digest(monkeypatch):
    library = prefix.reference(prefix.REPO / "AGENTS.md")
    monkeypatch.setattr(recovery.static, "OSMESA", library)
    inputs = {
        "case": {"config": {"osmesa_library": library["path"]}},
        "source": {"source_sha256": {}},
        "refs": {},
    }
    recovery.bind_software_gl(inputs)
    assert inputs["source"]["source_sha256"][library["path"]] == library["sha256"]
    assert inputs["refs"]["software_gl"] == library
    wrong_path = copy.deepcopy(inputs)
    wrong_path["case"]["config"]["osmesa_library"] += ".other"
    with pytest.raises(ValueError):
        recovery.bind_software_gl(wrong_path)
    conflicting = copy.deepcopy(inputs)
    conflicting["source"]["source_sha256"][library["path"]] = "0" * 64
    with pytest.raises(ValueError):
        recovery.bind_software_gl(conflicting)
