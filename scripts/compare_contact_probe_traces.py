"""094 compare observed branch contact sequences and saturation with baseline."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

ROOT = Path("outputs/contact_probe_trace_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    sources = [(Path("outputs/native_controller_trace_20261025"), [4]), (ROOT, list(range(6)))]
    rows = []
    for root, indices in sources:
        closure = e.json(root / "closure_verified.json")
        assert closure["complete"] and closure["owner_normal_wait_verified"] and closure["source_roster_unchanged"]
        for group in indices:
            folder = root / f"group{group:02d}"
            result = e.json(folder / "collection_result.json")
            verified = e.json(folder / "independent_verified.json")
            assert result["complete"] and verified["prefix_matches"]
            for path, signature in verified["source_sha256"].items():
                e.check(path, signature)
            ref = result["trace_ref"]
            e.check(ref["path"], ref["sha256"])
            records = e.json(ref["path"])["records"]
            assert len(records) == 25
            sequences = []
            for r in records:
                pairs = []
                for c in r["contacts"]:
                    bodies = c["bodies"]
                    if any((b or "").startswith(("gripper0_", "robot0_")) for b in bodies):
                        pairs.append(tuple(sorted(bodies)))
                sequences.append(sorted(set(pairs)))
            table = [any("table" in pair for pair in pairs) for pairs in sequences]
            raw = np.array([r["raw_torque"] for r in records])
            low, high = np.array(records[0]["low"]), np.array(records[0]["high"])
            row = dict(
                case=28 if root != ROOT else 29 + group,
                branch="baseline" if root != ROOT else ["x-", "x+", "y-", "y+", "z-", "z+"][group],
                inner_calls=25,
                saturated_calls=verified["saturated_inner_calls"],
                minimum_limit_margin_nm=float(np.minimum(raw - low, high - raw).min()),
                table_contact_calls=sum(table),
                table_contact_sequence=table,
                robot_contact_body_pairs=sequences,
            )
            rows.append(row)
    baseline = rows[0]["robot_contact_body_pairs"]
    for row in rows:
        row["same_body_pair_set_as_baseline_calls"] = sum(
            a == b for a, b in zip(row["robot_contact_body_pairs"], baseline, strict=True)
        )
    e.recheck()
    result = dict(
        complete=True,
        rows=rows,
        source_sha256=e.hashes,
        scope="observed integration-before contact body pairs, not contact force or unique causal attribution",
        physics_steps=0,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
        control_qualification=False,
    )
    with (ROOT / "branch_comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for row in rows:
        print({k: v for k, v in row.items() if k not in ("table_contact_sequence", "robot_contact_body_pairs")})


if __name__ == "__main__":
    main()
