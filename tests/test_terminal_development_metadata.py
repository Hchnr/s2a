import copy
import importlib
import json

import pytest

from s2a.artifacts import REPO


@pytest.fixture
def metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_wam_terminal_development_metadata")


def profile(m):
    base = json.loads((REPO / "configs/wam_short_replan_8_alpha1_20261021.json").read_text())
    p = dict(
        model_config=dict(
            cosmos_source_root="/active/cosmos", checkpoint=dict(run="/active/run", iteration="iter_000008400")
        )
    )
    case = m.order()[0]
    c = copy.deepcopy(base)
    c.update(
        output_dir=str((m.ROOT / "development/cases" / case["case_id"]).relative_to(REPO)),
        records=[case["record_name"]],
        sampling_seeds=[42],
        simulation_python=str(REPO / base["simulation_python"]),
        fixture=str(REPO / base["fixture"]),
        policy_template="/safe/template.json",
        cosmos_repo="/active/cosmos",
    )
    c["checkpoints"] = [dict(p["model_config"]["checkpoint"], name="8400_masked", gpu=4, port=31363)]
    return base, c, case, p


def test_independent_profile_accepts_only_fixed12_matrix(metadata):
    base, c, case, p = profile(metadata)
    metadata.check_profile(base, c, case, p, "/safe/template.json")
    assert len(metadata.order()) == 12 and sum(c["expected_requests"] for c in metadata.order()) == 900


@pytest.mark.parametrize("fault", ["steps", "tail", "controller", "port", "template"])
def test_independent_profile_rejects_control_or_protocol_changes(metadata, fault):
    base, c, case, p = profile(metadata)
    if fault == "steps":
        c["online_max_steps"] = 599
    elif fault == "tail":
        c["stable_success_last_steps"] = 39
    elif fault == "controller":
        c["inverse_weight"] = 0.4
    elif fault == "port":
        c["checkpoints"][0]["port"] = 31364
    else:
        c["policy_template"] = "/other/template.json"
    with pytest.raises(ValueError):
        metadata.check_profile(base, c, case, p, "/safe/template.json")


def test_metadata_wrong_root_fails_before_any_array_or_model_read(metadata, monkeypatch):
    class Evidence:
        def json(self, ref):
            return dict(schema_version="s2a.terminal-development-preparation.v1", root="/unregistered")

    monkeypatch.setattr(metadata, "Evidence", Evidence)
    monkeypatch.setattr(metadata.np, "load", lambda *a, **k: pytest.fail("metadata decoded arrays"))
    with pytest.raises(ValueError):
        metadata.verify({})
