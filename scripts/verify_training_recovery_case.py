"""Full saved063 control replay against independently constructed targets."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from training_recovery_acceptance_checker import verify_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, required=True, choices=range(9))
    args = parser.parse_args()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.read(Path(__file__).with_name("training_recovery_acceptance_checker.py"))
    registration = evidence.json("configs/training_state_recovery_qualification_20261025.json")
    entry = registration["records"][args.case_index // 3]
    offset = registration["base_x_offsets_m"][args.case_index % 3]
    ref = entry["trajectory_ref"]
    evidence.check(ref["path"], ref["sha256"])
    with np.load(ref["path"], allow_pickle=False) as data:
        original = data["source_pose"]
        opening = data["opening"]
        prefix = np.repeat(original[:1], 56, axis=0)
        for index in range(16):
            prefix[index, 0, 3] += offset * (index + 1) / 16
        for index in range(40):
            prefix[16 + index, 0, 3] += offset * (39 - index) / 40
        poses = np.concatenate([prefix, original[1:]])
        openings = np.r_[np.repeat(opening[0], 56), opening[1:]]
    root = Path(
        "outputs/training_state_recovery_qualification_20261025"
        if args.case_index == 0
        else "outputs/training_state_recovery_continuation_20261025"
    )
    directory = root / f"case{args.case_index:02d}"
    config = evidence.json(directory / "execution_config.json")
    case = next(directory.glob("simulation/*/*"))
    manifest = evidence.json(case / "manifest.json")
    result = verify_case(
        config,
        manifest["record"],
        config["variants"][0],
        case,
        evidence,
        source="oracle",
        checkpoint=None,
        full=True,
        sampling_seed=42,
        first_query_validator=None,
        oracle_reference=(poses, openings),
    )
    evidence.recheck()
    output = dict(
        passed=True,
        full_cohort_qualification=False,
        scope="Full saved mechanical replay only; source adaptation, labels, images and lifecycle have separate witnesses",
        case=result,
        input_sha256=evidence.hashes,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
    )
    with (directory / "mechanical_replay_witness.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(passed=True, steps=result["steps"], outcome=result["outcome"])))


if __name__ == "__main__":
    main()
