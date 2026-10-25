"""Preserve the fixed raw8 controller while relocating metadata paths for061."""

import copy
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import GPU, checked, read, require
from s2a.terminal_development_execution import PORT, ROOT
from s2a.terminal_development_policy import case_order

CHECKPOINT = "8400_masked"
FIXTURE_SHA = "f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e"
BASE_REF = dict(
    path=str(REPO / "configs/wam_short_replan_8_alpha1_20261021.json"),
    sha256="51f64aa81a53f89cd91f8cf9925efe3e3c9a3a8de13a1856b054a37693dc979b",
)
RELOCATION_REF = dict(
    path=str(REPO / "outputs/clean_training_development_dependency_repair_20261024/path_relocation.json"),
    sha256="35f43c12d90c098aa2625bae5c994bf336d7488ce996a829c642f39a80ddf8a7",
)


def safe_template(relocation_ref):
    relocation = read(relocation_ref)
    require(
        relocation["passed"] is relocation["complete"] is True
        and relocation["external_writes"] == 0
        and relocation["only_template_field_changed"] == "tool_contract",
        "Qualified051 contract relocation required",
    )
    old, new = read(relocation["old_template_ref"]), read(relocation["new_template_ref"])
    expected = copy.deepcopy(old)
    expected["tool_contract"] = relocation["safe_copy_ref"]["path"]
    require(
        new == expected and relocation["safe_copy_ref"]["sha256"] == relocation["historical_sha256"],
        "Only tool_contract path may differ from the original control template",
    )
    checked(relocation["safe_copy_ref"])
    return relocation["new_template_ref"]


def make_case_config(prepared, case):
    require(
        Path(prepared["root"]) == ROOT
        and type(case["case_index"]) is int
        and 0 <= case["case_index"] < 12
        and case == case_order()[case["case_index"]],
        "Fixed061 case/root required",
    )
    base = read(BASE_REF)
    require(
        base["fixture_sha256"] == FIXTURE_SHA
        and base["variants"]
        == [
            dict(
                name="position_hold",
                rotation="response",
                gripper="position_hold",
                governor=False,
                model_rotation_alpha=1.0,
            )
        ],
        "Frozen raw8 controller differs",
    )
    require(prepared["tool_relocation_ref"] == RELOCATION_REF, "Registered051 contract relocation identity differs")
    template = safe_template(prepared["tool_relocation_ref"])
    value = copy.deepcopy(base)
    root = ROOT / "development/cases" / case["case_id"]
    value.update(
        output_dir=str(root.relative_to(REPO)),
        records=[case["record_name"]],
        sampling_seeds=[case["sampling_seed"]],
        simulation_python=str(REPO / base["simulation_python"]),
        fixture=str(REPO / base["fixture"]),
        policy_template=template["path"],
        cosmos_repo=prepared["model_config"]["cosmos_source_root"],
        notes=["061 fixed controller; legacy gpu label4 is CPU executor metadata only; actual model UUID=" + GPU],
    )
    value["checkpoints"] = [dict(prepared["model_config"]["checkpoint"], name=CHECKPOINT, gpu=4, port=PORT)]
    return value
