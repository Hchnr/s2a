import copy
import json
from pathlib import Path

import pytest

import s2a.terminal_development_control as control


@pytest.fixture
def profile(tmp_path, monkeypatch):
    base = json.loads(Path(control.BASE_REF["path"]).read_text())

    def ref(name):
        return dict(path=str(tmp_path / name), sha256="a" * 64)

    old_ref, new_ref, safe_ref = ref("old.json"), ref("new.json"), ref("contract.json")
    old = dict(tool_contract="/historical/path", gain=0.5)
    new = dict(old, tool_contract=safe_ref["path"])
    relocation = dict(
        passed=True,
        complete=True,
        external_writes=0,
        only_template_field_changed="tool_contract",
        old_template_ref=old_ref,
        new_template_ref=new_ref,
        safe_copy_ref=safe_ref,
        historical_sha256="a" * 64,
    )
    docs = {
        control.BASE_REF["path"]: base,
        control.RELOCATION_REF["path"]: relocation,
        old_ref["path"]: old,
        new_ref["path"]: new,
    }
    monkeypatch.setattr(control, "read", lambda r: copy.deepcopy(docs[r["path"]]))
    monkeypatch.setattr(control, "checked", lambda r: None)
    prepared = dict(
        root=str(control.ROOT),
        tool_relocation_ref=control.RELOCATION_REF,
        model_config=dict(
            cosmos_source_root=str(tmp_path / "cosmos"),
            checkpoint=dict(run="/active/checkpoint", iteration="iter_000008400"),
        ),
    )
    return base, prepared, new


def test_all12_preserve_entire_control_profile_except_explicit_metadata(profile):
    base, p, _ = profile
    changes = {
        "output_dir",
        "records",
        "sampling_seeds",
        "simulation_python",
        "fixture",
        "policy_template",
        "cosmos_repo",
        "notes",
        "checkpoints",
    }
    for case in control.case_order():
        c = control.make_case_config(p, case)
        assert set(c) == set(base)
        assert {k: v for k, v in c.items() if k not in changes} == {k: v for k, v in base.items() if k not in changes}
        assert c["records"] == [case["record_name"]] and c["sampling_seeds"] == [case["sampling_seed"]]
        assert (
            c["checkpoints"][0]["port"] == control.PORT and c["cosmos_repo"] == p["model_config"]["cosmos_source_root"]
        )
        assert (control.REPO / c["output_dir"]).is_relative_to(control.ROOT)


def test_template_relocation_cannot_change_dynamics(profile):
    _, p, new = profile
    new["gain"] = 0.6
    with pytest.raises(ValueError, match="Only tool_contract"):
        control.make_case_config(p, control.case_order()[0])


def test_unregistered_case_or_template_rejected(profile):
    _, p, _ = profile
    wrong = dict(control.case_order()[0], sampling_seed=43)
    with pytest.raises(ValueError):
        control.make_case_config(p, wrong)
    p["tool_relocation_ref"] = dict(path="/other", sha256="b" * 64)
    with pytest.raises(ValueError):
        control.make_case_config(p, control.case_order()[0])
