"""Independent fixed-input profiles before executable reference binding."""

import copy
import json
from pathlib import Path

import pytest
import verify_serialized_preview_wam_metadata as audit

from s2a import serialized_preview_wam_control as control


@pytest.fixture
def profile(monkeypatch):
    base = json.loads(Path(control.BASE_REF["path"]).read_text())
    qualified = json.loads((control.REPO / "configs/coupled_preview_control_registered_20261026.json").read_text())
    ref = dict(path="/synthetic/control.json", sha256="a" * 64)
    p = dict(
        root=str(control.ROOT),
        tool_relocation_ref=control.RELOCATION_REF,
        preview_control_closure_ref=dict(path="/synthetic/closed.json", sha256="b" * 64),
        model_config=dict(
            cosmos_source_root="/readonly/cosmos", checkpoint=dict(run="/readonly/run", iteration="iter_000008400")
        ),
    )
    monkeypatch.setattr(control, "read", lambda r: copy.deepcopy(base if r == control.BASE_REF else qualified))
    monkeypatch.setattr(control, "safe_template", lambda r: dict(path="/synthetic/template.json"))
    return base, qualified, ref, p


def test_all12_profiles_have_no_circular_execution_references(profile):
    base, qualified, ref, p = profile
    for case in audit.order():
        c = control.make_case_profile(p, ref, case=case)
        assert "wam_development" not in c and "profiles" not in c and "oracle_reference_root" not in c
        audit.check_profile(base, c, case, p, "/synthetic/template.json", qualified)


@pytest.mark.parametrize("fault", ["gain", "uuid", "budget", "expert", "early_registration"])
def test_independent_profile_rejects_control_or_input_changes(profile, fault):
    base, qualified, ref, p = profile
    case = audit.order()[0]
    c = control.make_case_profile(p, ref, case=case)
    if fault == "gain":
        c["inverse_weight"] = 0.1
    elif fault == "uuid":
        c["checkpoints"][0]["gpu_uuid"] = "GPU-other"
    elif fault == "budget":
        c["online_max_steps"] = 599
    elif fault == "expert":
        c["oracle_reference_root"] = "/expert"
    else:
        c["wam_development"] = {}
    with pytest.raises(ValueError, match="whole fixed control"):
        audit.check_profile(base, c, case, p, "/synthetic/template.json", qualified)
