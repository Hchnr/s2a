"""Describe commanded gripper phases and fixed training-anchor coverage offline."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_wam_release_horizon import reference


def native_coverage(prepared_path):
    prepared = json.loads(prepared_path.read_text())
    rows = []
    for anchor in prepared["training_anchors"]:
        path = Path(anchor["trajectory_ref"]["path"])
        if reference(path) != anchor["trajectory_ref"]:
            raise ValueError("Training trajectory fingerprint changed")
        frame = anchor["anchor_frame"]
        with np.load(path, allow_pickle=False) as z:
            commands = z["sent_commands"][:, -1]
            opening = z["opening"]
            window = commands[frame : frame + 40]
            future_opening = opening[frame + 1 : frame + len(window) + 1]
            if len(window) != 40:
                raise ValueError("Incomplete registered training future")
            events = [i for i in range(frame, frame + 40) if commands[i] < 0 and (i == 0 or commands[i - 1] >= 0)]
            rows.append(
                {
                    "anchor": anchor["name"],
                    "task_index": anchor["source_identity"]["task_index"],
                    "frame": frame,
                    "trajectory_ref": anchor["trajectory_ref"],
                    "opening_at_anchor": float(opening[frame]),
                    "future40_opening_min": float(future_opening.min()),
                    "future40_opening_max": float(future_opening.max()),
                    "future40_close_commands": int(sum(window > 0)),
                    "future40_open_commands": int(sum(window < 0)),
                    "future40_release_edges": events,
                }
            )
    if len(rows) != 150:
        raise ValueError("Expected all150 registered training anchors")
    summary = {}
    for task in range(3):
        rr = [r for r in rows if r["task_index"] == task]
        summary[str(task)] = {
            "anchors": len(rr),
            "windows_with_close_commands": sum(r["future40_close_commands"] > 0 for r in rr),
            "windows_with_open_commands": sum(r["future40_open_commands"] > 0 for r in rr),
            "windows_with_release_edges": sum(bool(r["future40_release_edges"]) for r in rr),
            "windows_with_observed_opening_ge_0_9": sum(r["future40_opening_max"] >= 0.9 for r in rr),
        }
    return {"prepared_ref": reference(prepared_path), "summary": summary, "anchors": rows}


def development(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = "wine_bottle_1"
    close = [r for r in rows if r["gripper"]["command"] > 0]
    contact = [r for r in rows if r["objects"][target]["grasp_contact"]]
    return {
        "case": path.parents[3].name,
        "steps_ref": reference(path),
        "close_command_steps": len(close),
        "first_close_command_step": close[0]["step"] if close else None,
        "first_contact_step": contact[0]["step"] if contact else None,
        "contact_steps": len(contact),
        "max_logged_height_gain_m": max(r["objects"][target]["height_gain_m"] for r in rows),
        "reference_opening_range": [min(r["opening_target"] for r in rows), max(r["opening_target"] for r in rows)],
        "actual_opening_range": [min(r["opening_after"] for r in rows), max(r["opening_after"] for r in rows)],
        "closed_motor_hold_steps": sum(r["gripper"]["state"] == "closed_motor_hold" for r in rows),
        "success_steps": sum(r["success"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = sorted(args.cases.glob("task1_*/simulation/*/*/steps.jsonl"))
    if len(cases) != 4:
        raise ValueError("Expected the fixed four wine development cases")
    result = {
        "scope": "Offline native-label coverage and logged development behavior; neither contact nor command sign is task success",
        "script_ref": reference(Path(__file__)),
        "helper_ref": reference(Path(__file__).with_name("analyze_wam_release_horizon.py")),
        "training": native_coverage(args.prepared),
        "development": [development(path) for path in cases],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["training"]["summary"])
    for case in result["development"]:
        print({k: v for k, v in case.items() if k != "steps_ref"})


if __name__ == "__main__":
    main()
