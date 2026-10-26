"""093 independently verify replay endpoints and actual raw/clipped/ctrl trace."""

import argparse
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_controller_trace_20261025")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", type=int, choices=range(6), required=True)
    args = p.parse_args()
    e = Evidence()
    e.read(__file__)
    folder = ROOT / f"group{args.group:02d}"
    receipts = ROOT / "orchestration" / f"group{args.group:02d}"
    normal_receipts(e, receipts / "launch.json", receipts / "exit.json", receipts / "release.json", format="durable")
    c = e.json(folder / "collection_result.json")
    assert c["complete"] and c["group"] == args.group and c["inner_records"] == 25
    for path, signature in c["source_sha256"].items():
        e.check(path, signature)
    for key in ("arrays_ref", "trace_ref", "baseline_ref"):
        ref = c[key]
        e.check(ref["path"], ref["sha256"])
    with (
        np.load(c["arrays_ref"]["path"], allow_pickle=False) as actual,
        np.load(c["baseline_ref"]["path"], allow_pickle=False) as old,
    ):
        for key in actual.files:
            if key == "sent_commands":
                np.testing.assert_array_equal(actual[key], old[key])
            else:
                np.testing.assert_allclose(actual[key], old[key], rtol=0, atol=1e-9)
        assert len(actual["sent_commands"]) == c["physics_steps"]
        nqnv = actual["sim_states"].copy()
    rows = e.json(c["trace_ref"]["path"])["records"]
    assert len(rows) == 25
    np.testing.assert_allclose(np.diff([r["time"] for r in rows]), 0.002, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rows[0]["time"], nqnv[-2, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(rows[-1]["time"] + 0.002, nqnv[-1, 0], rtol=0, atol=1e-10)
    saturated = 0
    for row in rows:
        raw, clipped, ctrl, lo, hi = (
            np.asarray(row[k]) for k in ("raw_torque", "clipped_torque", "ctrl", "low", "high")
        )
        assert raw.shape == clipped.shape == ctrl.shape == lo.shape == hi.shape == (7,)
        assert all(np.isfinite(v).all() for v in (raw, clipped, ctrl, lo, hi))
        np.testing.assert_array_equal(clipped, np.minimum(hi, np.maximum(lo, raw)))
        np.testing.assert_array_equal(ctrl, clipped)
        saturated += int(np.any((raw < lo) | (raw > hi)))
    flags = sorted(set(r["uncoupling"] for r in rows))
    assert len(flags) == 1
    e.recheck()
    write_once(
        folder / "independent_verified.json",
        dict(
            complete=True,
            group=args.group,
            prefix_matches=True,
            actual_uncoupling=flags[0],
            saturated_inner_calls=saturated,
            inner_calls=25,
            physics_steps=c["physics_steps"],
            source_sha256=e.hashes,
            model_calls=0,
            new_fits=0,
            fresh_cases=0,
            control_qualification=False,
        ),
    )
    print(dict(group=args.group, uncoupling=flags[0], saturated=saturated, inner_calls=25), flush=True)


if __name__ == "__main__":
    main()
