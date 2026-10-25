"""Collect one fixed063 physical training recovery trial on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
from run_matched_oracle_case import CONTROL, CONTROL_SHA

from s2a.artifacts import sha256
from s2a.native_environment import restore_cv2_environment, strict_native_environment

CONFIG = Path("configs/motor_state_pilot_20261025.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(5), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    strict_native_environment()
    if sha256(CONTROL) != CONTROL_SHA:
        raise ValueError("Import guard changed")
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    readiness = restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from collect_wam_native_training import endpoint, render_anchor, save_arrays

    from s2a.motor_state_simulation import run_motor_state_case
    from s2a.native_training import pack_gripper_caches, synchronous_labels

    strict_native_environment()
    registration = json.loads(CONFIG.read_text())
    entry = registration["profiles"][args.case_index]
    preview_steps = entry["preview_steps"]
    offset = 0
    record = entry["record"]
    trajectory = entry["trajectory_ref"]
    if sha256(trajectory["path"]) != trajectory["sha256"]:
        raise ValueError("Registered training trajectory changed")
    with np.load(trajectory["path"], allow_pickle=False) as data:
        if entry["format"] == "native_endpoints":
            targets = data["source_pose"][1:].copy()
            openings = data["opening"][1:].copy()
        elif entry["format"] == "world_reference":
            targets = data["poses"].copy()
            openings = data["openings"].copy()
        else:
            raise ValueError("Unknown registered reference format")
    if len(targets) != entry["physics_steps"] or len(openings) != len(targets):
        raise ValueError("Registered physical budget differs")
    base = json.loads(Path("configs/matched_oracle_requalification_20261025.json").read_text())
    root = Path("outputs/motor_state_pilot_20261025").resolve() / f"case{args.case_index:02d}"
    base.update(
        output_dir=str(root),
        records=[record["name"]],
        online_max_steps=len(targets),
        gripper_preview_steps=preview_steps,
    )
    template = json.loads(Path(base["policy_template"]).read_text())
    h = np.asarray(json.loads(Path(template["tool_contract"]).read_text())["source_from_tcp"])
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    case=args.case_index,
                    steps=len(targets),
                    readiness=readiness["passed"],
                    model_calls=0,
                    new_physics_steps=0,
                )
            )
        )
        return
    root.mkdir(parents=True, exist_ok=False)
    config_path = root / "execution_config.json"
    config_path.write_text(json.dumps(base, indent=2) + "\n")
    samples, receipts, images = [], [], []

    def observe(env, index):
        sample, receipt = endpoint(env, index)
        samples.append(sample)
        receipts.append(receipt)
        with (root / "endpoints.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt, allow_nan=False) + "\n")
        if index in (0, 16, 56):
            images.append(render_anchor(env, index, sample, receipt, root))

    signatures = {str(Path(trajectory["path"]).resolve()): trajectory["sha256"]}
    physical = run_motor_state_case(
        base,
        root,
        record,
        source="oracle",
        variant_name="position_hold",
        sampling_seed=42,
        config_path=str(config_path),
        oracle_reference=(targets, openings, signatures),
        endpoint_observer=observe,
    )
    if not physical["complete"] or len(samples) != len(targets) + 1:
        raise ValueError("Physical collection incomplete; original logs retained")
    arrays = {key: np.stack([s[key] for s in samples]) for key in samples[0] if key != "gripper_current_action"}
    arrays.update(pack_gripper_caches([s["gripper_current_action"] for s in samples]))
    arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], h, arrays["sim_time"]))
    array_ref = save_arrays(root / "actual_trajectory.npz", arrays)
    result = dict(
        complete=True,
        physical_result=physical,
        arrays_ref=array_ref,
        images=images,
        readiness=readiness,
        case_index=args.case_index,
        offset_m=offset,
        label_source="actual physical endpoints only; not target references",
        new_model_calls=0,
        fresh_cases=0,
    )
    payload = json.dumps(result, indent=2, default=lambda value: value.tolist(), allow_nan=False)
    with (root / "collection_result.json").open("x") as stream:
        stream.write(payload + "\n")
    print(json.dumps(dict(case=args.case_index, complete=True, physics_steps=len(targets))))


if __name__ == "__main__":
    main()
