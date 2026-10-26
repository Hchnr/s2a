"""Exercise156 actual-state binding on152 observed native prefix; reject corruption."""

from pathlib import Path

import numpy as np
from arm_joint_wrench_independent import INPUTS
from arm_wrench_data import attach_current_wrench
from audit_wam_position import Evidence
from collect_wam_native_training import write_once
from qualified_input_state_audit import verify_inputs

from s2a.joint_wrench_condition_forward import with_history


def main():
    e = Evidence()
    for p in (
        __file__,
        "scripts/qualified_input_state_audit.py",
        "plans/v0.0.1/156_qualified_control_log_checker_20261025.md",
    ):
        e.read(p)
    root = Path("outputs/runtime_prefix_parity_corrected_20261025/case00")
    collection = e.json(root / "collection_result.json")
    assert collection["runtime_input_parity"]
    ref = collection["arrays_ref"]
    e.check(ref["path"], ref["sha256"])
    xml = e.read(root / "model.xml")
    q = e.json("outputs/left_table_forward_qualification_20261025/qualification.json")
    ref = q["parameter_ref"]
    e.check(ref["path"], ref["sha256"])
    parameters = e.json(ref["path"])
    c = e.json("configs/left_table_forward_20261025.json")
    ref = c["data_result_ref"]
    e.check(ref["path"], ref["sha256"])
    entry = next(r for r in e.json(ref["path"])["trajectories"] if r["index"] == 23)
    ref = entry["cache_ref"]
    e.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as a:
        expected = attach_current_wrench(with_history({k: a[k].copy() for k in a.files}, parameters), entry, c, e)
    ref = c["wrench_closure_ref"]
    closed = e.json(ref["path"])
    ref = closed["summary_ref"]
    e.check(ref["path"], ref["sha256"])
    record = next(r for r in e.json(ref["path"])["trajectories"] if r["index"] == 23)
    ref = record["wrench_ref"]
    e.check(ref["path"], ref["sha256"])
    wrench = e.json(ref["path"])["samples"]
    rows = []
    with np.load(collection["arrays_ref"]["path"], allow_pickle=False) as data:
        for k in range(48):
            rows.append(
                dict(
                    qualified_forward_input={key: expected[key][k : k + 1].copy() for key in INPUTS},
                    current_wrench=wrench[k],
                    controller_goal_position_before=data["controller_goal_pos"][k],
                    controller_goal_before=data["controller_goal_ori"][k],
                    controller_goal_position_after=data["controller_goal_pos"][k + 1],
                    controller_goal_after=data["controller_goal_ori"][k + 1],
                )
            )
        verify_inputs(xml, data, rows)
        rows[17]["qualified_forward_input"]["x_joint_velocity"][0, 3] += 0.01
        try:
            verify_inputs(xml, data, rows)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted current joint velocity was accepted")
    e.recheck()
    out = Path("outputs/qualified_control_checker_admission_20261025")
    out.mkdir(exist_ok=False)
    write_once(
        out / "state_binding_verified.json",
        dict(
            complete=True,
            actual_prefix_cycles=48,
            corruption_rejected=True,
            source_sha256=e.hashes,
            physics_steps=0,
            control_qualified=False,
        ),
    )
    print("156 actual48-cycle current-state binding passed;corrupted input rejected;physics0")


if __name__ == "__main__":
    main()
