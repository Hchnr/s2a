"""Actual Python3.11 all12 input preflight; no simulator/model initialization."""

import os
import sys
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require

from s2a.clean_training_runtime import publish, reference
from s2a.coupled_preview_control_gate import admit_preview_control
from s2a.dependency_safe_wam_admission import admit_tracking_control
from s2a.terminal_development_gate import ROOT, admit_state
from s2a.terminal_development_policy import case_order
from s2a.tracking_preview_wam_control_v4 import BASE_REF, RELOCATION_REF, make_case_profile
from s2a.tracking_preview_wam_execution_v4 import ROOT as DEVELOPMENT_ROOT
from s2a.tracking_preview_wam_simulation_v4 import execution_inputs


def tracking_evidence():
    from s2a.artifacts import REPO

    return {
        key: reference(REPO / path)
        for key, path in dict(
            qualification_ref="outputs/dependency_safe_matrix_closed_20261027/verified.json",
            closure_owner_ref="outputs/dependency_safe_closure_execution_20261027/result.json",
            transaction_owner_ref="outputs/tracking_transaction_feedback_execution_20261027/result.json",
        ).items()
    }


def main():
    require(sys.version_info[:2] == (3, 11), "Actual sim Python3.11 required")
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    e = Evidence()
    _, admission = admit_state(reference(ROOT / "orchestration/result.json"), read_json=e.json)
    state = e.json(admission["state_prepared_ref"])
    policy = e.json(reference(Path("outputs/terminal_development_preparation_20261025/policy_metadata.json")))
    closure_ref = reference(Path("outputs/coupled_preview_control_20261026/closure_verified.json"))
    physical = admit_preview_control(closure_ref)
    prepared = dict(policy["policy_manifest"], model_config=state["model_config"], tool_relocation_ref=RELOCATION_REF)
    tracking = tracking_evidence()
    admit_tracking_control(tracking)
    prepared.update(root=str(DEVELOPMENT_ROOT), preview_control_closure_ref=closure_ref, tracking_admission=tracking)
    e.path(BASE_REF)
    relocation = e.json(RELOCATION_REF)
    for key in ("old_template_ref", "new_template_ref", "safe_copy_ref"):
        e.path(relocation[key])
    root = Path("outputs/tracking_preview_wam_preparation_v4_20261027/input_preflight_py311")
    require(not root.exists(), "New preflight output required")
    root.mkdir(parents=True)
    rows = []
    for case in case_order():
        config = make_case_profile(prepared, physical["control_config_ref"], case=case)
        config_ref = publish(root / "configs" / f"{case['case_id']}.json", config)
        inputs = execution_inputs(config, config_ref["path"])
        rows.append(dict(case=case, config_ref=config_ref, input_sha256=inputs))
    e.finish()
    publish(
        root / "result.json",
        dict(
            passed=True,
            complete=True,
            case_count=len(rows),
            records=rows,
            admission=admission,
            physical_control_admission=physical,
            input_sha256=e.inputs,
            source_refs=[reference(Path(__file__)), reference(Path("src/s2a/tracking_preview_wam_control_v4.py"))],
            interpreter=sys.executable,
            actual_model_calls=0,
            actual_physics_steps=0,
            array_decodes=0,
            execution_authorized=False,
            fresh_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
