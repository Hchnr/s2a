"""169 paired first-case evidence only after both independent audits and waits."""

import json

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.owned_receipt_contract import normal_receipts


def main():
    root = REPO / "outputs/supported_first_case_comparison_20261025"
    if root.exists():
        raise FileExistsError(root)
    e = Evidence()
    e.read(__file__)
    registrations = [e.json(f"configs/qualified_inverse_{name}_20261025.json") for name in ("enumerated", "supported")]
    assert registrations[0]["profiles"] == registrations[1]["profiles"]
    assert registrations[0]["arm_forward_parameter_ref"] == registrations[1]["arm_forward_parameter_ref"]
    cases = []
    logs = []
    for name in ("enumerated", "supported"):
        base = REPO / f"outputs/qualified_inverse_{name}_20261025"
        for parent in ("orchestration", "verification"):
            directory = base / parent / "case00"
            normal_receipts(
                e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
            )
        case = base / "case00"
        checked = e.json(case / "independent_verified.json")
        assert checked["complete"] and checked["mechanical_labels_images_passed"]
        control = checked["control"]
        qualified = (
            control["outcome"]["stable"]
            and control["tracking"]["admitted"]["position_rmse_m"] <= 0.01
            and control["tracking"]["admitted"]["so3_rmse_rad"] <= 0.05
        )
        assert bool(qualified) == checked["performance_qualified"]
        path = next(case.rglob("steps.jsonl"))
        rows = [json.loads(line) for line in e.read(path).decode().splitlines()]
        assert len(rows) == 164
        logs.append(rows)
        command = np.asarray([r["command"] for r in rows])
        rate = np.diff(np.vstack([np.zeros((1, 7)), command]), axis=0)
        assert np.max(np.abs(rate[:, :3])) <= 0.25 + 1e-12
        cases.append(
            dict(
                name=name,
                periods=len(rows),
                performance_qualified=bool(qualified),
                outcome=control["outcome"],
                tracking=control["tracking"]["admitted"],
                max_translation_command_change=float(np.max(np.abs(rate[:, :3]))),
            )
        )
    rows = logs[1]
    decisions = [r["qualified_forward_inverse"]["support_decision"] for r in rows]
    weights = np.asarray([d["learned_weight"] for d in decisions])
    transitions = [
        dict(step=r["step"], distance=d["distance"], learned_weight=d["learned_weight"])
        for r, d in zip(rows, decisions, strict=True)
    ]
    e.recheck()
    publish(
        root / "verified.json",
        dict(
            complete=True,
            cases=cases,
            full_learning_periods=int(np.sum(weights == 1)),
            full_physical_periods=int(np.sum(weights == 0)),
            mixed_periods=int(np.sum((weights > 0) & (weights < 1))),
            decisions=transitions,
            source_sha256=e.hashes,
            new_physics_steps=0,
            new_fits=0,
            full18_control_qualified=False,
            limitation="Same registered first profile only; no full18 or WAM/fresh generalization claim",
        ),
    )
    print(json.dumps(cases))


if __name__ == "__main__":
    main()
