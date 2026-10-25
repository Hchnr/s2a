"""062 CPU-only fixed oracle worker with explicit native import admission."""

import argparse
import hashlib
import json
from pathlib import Path

from s2a.native_environment import restore_cv2_environment, strict_native_environment

CONFIG = Path("configs/matched_oracle_requalification_20261025.json")
CONTROL = Path("outputs/clean_training_development_preflight_20261023/dependency_addendum.json")
CONTROL_SHA = "106f45543f8bc0aa0e336c0ad9b5c01e0bce252a5e5dad0ec3ebbba349e5fda0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, choices=["task0_init0", "task1_init0", "task2_init0"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    strict_native_environment()
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA:
        raise ValueError("Qualified import guard changed")
    guard = json.loads(CONTROL.read_text())["cv2_guard_registration"]
    readiness = restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    from run_wam_position_matrix import protocol

    from s2a.stability_position_simulation import execution_inputs, run_position_case
    from s2a.stability_recovery_simulation import _read_oracle

    strict_native_environment()
    config, root, records, _ = protocol(CONFIG, source="oracle", variant_name="position_hold", checkpoint=None)
    if config["sampling_seeds"] != [42] or config["records"] != ["task0_init0", "task1_init0", "task2_init0"]:
        raise ValueError("Fixed062 cohort differs")
    record = next(r for r in records if r["name"] == args.record)
    inputs = execution_inputs(config, CONFIG)
    _, _, reference_inputs = _read_oracle(config, record, 42)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    record=args.record,
                    readiness=readiness["passed"],
                    inputs=len(inputs),
                    oracle_inputs=len(reference_inputs),
                    model_calls=0,
                    physics_steps=0,
                )
            )
        )
        return
    receipt = root / "readiness" / f"{args.record}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as stream:
        json.dump(dict(readiness=readiness, source_sha256=inputs, oracle_sha256=reference_inputs), stream, indent=2)
    result = run_position_case(
        config,
        root,
        record,
        source="oracle",
        variant_name="position_hold",
        sampling_seed=42,
        config_path=str(CONFIG),
        frozen_inputs=inputs,
    )
    print(json.dumps(dict(record=args.record, result=result)))


if __name__ == "__main__":
    main()
