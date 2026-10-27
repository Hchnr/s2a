"""Measure changes to predictions for identical absolute physical endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference
from scipy.spatial.transform import Rotation


def analyze(path):
    chunks = json.loads(path.read_text())
    pairs = []
    for previous, current in zip(chunks[:-1], chunks[1:], strict=True):
        shift = current["step"] - previous["step"]
        if shift != 8 or len(previous["openings"]) != 40 or len(current["openings"]) != 40:
            raise ValueError("Expected the registered H40/replan8 development protocol")
        # Index j represents endpoint step+j+1. Compare old indices 8:16 to new 0:8.
        old = np.asarray(previous["effective_targets_tcp"])[shift : shift + 8]
        new = np.asarray(current["effective_targets_tcp"])[:8]
        translation = np.linalg.norm(new[:, :3, 3] - old[:, :3, 3], axis=1)
        relative_rotation = new[:, :3, :3] @ old[:, :3, :3].transpose(0, 2, 1)
        angle = Rotation.from_matrix(relative_rotation).magnitude()
        old_open = np.asarray(previous["openings"])[shift : shift + 8]
        new_open = np.asarray(current["openings"])[:8]
        delta = new_open - old_open
        old_edge = (
            next((j for j, x in enumerate(previous["openings"]) if x >= 0.9), None)
            if previous["anchor_opening"] < 0.9
            else None
        )
        new_edge = (
            next((j for j, x in enumerate(current["openings"]) if x >= 0.9), None)
            if current["anchor_opening"] < 0.9
            else None
        )
        pairs.append(
            {
                "new_chunk_step": current["step"],
                "absolute_endpoint_steps": list(range(current["step"] + 1, current["step"] + 9)),
                "position_revision_m": translation.tolist(),
                "rotation_revision_rad": angle.tolist(),
                "opening_revision": delta.tolist(),
                "old_opening": old_open.tolist(),
                "new_opening": new_open.tolist(),
                "old_first_open_absolute_step": None if old_edge is None else previous["step"] + old_edge + 1,
                "new_first_open_absolute_step": None if new_edge is None else current["step"] + new_edge + 1,
            }
        )
    summary = {}
    for label, rows in [("all", pairs), ("last5_revisions", pairs[-5:])]:
        p = np.concatenate([r["position_revision_m"] for r in rows])
        r = np.concatenate([r["rotation_revision_rad"] for r in rows])
        o = np.concatenate([r["opening_revision"] for r in rows])
        edges = [
            v["new_first_open_absolute_step"] - v["old_first_open_absolute_step"]
            for v in rows
            if v["new_first_open_absolute_step"] is not None and v["old_first_open_absolute_step"] is not None
        ]
        summary[label] = {
            "paired_endpoints": len(p),
            "position_revision_rmse_m": float(np.sqrt(np.mean(p**2))),
            "rotation_revision_rmse_rad": float(np.sqrt(np.mean(r**2))),
            "opening_revision_bias": float(o.mean()),
            "opening_revision_rmse": float(np.sqrt(np.mean(o**2))),
            "old_ge_0_9_revised_below": sum(
                sum(a >= 0.9 > b for a, b in zip(v["old_opening"], v["new_opening"], strict=True)) for v in rows
            ),
            "opening_edge_absolute_delay_steps_when_both_exist": edges,
        }
    return {"case": path.parents[3].name, "chunks_ref": reference(path), "summary": summary, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.cases.glob("*/simulation/*/*/chunks.json"))
    if len(paths) != 12:
        raise ValueError("Expected all 12 fixed development cases")
    result = {
        "scope": "Offline prediction revision, not execution error or counterfactual task success; 0.9 is descriptive only",
        "script_ref": reference(Path(__file__)),
        "helper_ref": reference(Path(__file__).with_name("analyze_wam_release_horizon.py")),
        "cases": [analyze(path) for path in paths],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for case in result["cases"]:
        print(case["case"], case["summary"]["last5_revisions"])


if __name__ == "__main__":
    main()
