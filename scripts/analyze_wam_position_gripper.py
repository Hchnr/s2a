"""Compare existing003/v2 and005/v3 gripper logs; no new control or inference.

The shared first-query interval ends before either case's next observation.
Full600 comparisons are descriptive because subsequent observations diverge.
The requested .8/.2 counters are diagnostic, never policy or acceptance gates.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if expected is not None:
            assert actual == expected, str(path)
        if str(path) in self.hashes:
            assert self.hashes[str(path)] == actual, str(path)
        self.hashes[str(path)] = actual
        return data

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def recheck(self):
        for path, expected in list(self.hashes.items()):
            self.read(path, expected)


def example(row):
    return dict(
        step=row["step"],
        reference=row["opening_target"],
        observed_before=row["opening_before"],
        observed_after=row["opening_after"],
        motor_before=row["motor_opening_before"],
        motor_after=row["motor_opening_after"],
        command=row["command"][-1],
        gripper=row["gripper"],
    )


def counts(rows):
    high = [r for r in rows if r["opening_target"] > 0.8]
    mismatches = [r for r in high if r["opening_after"] < 0.2]
    reasons = {}
    for row in mismatches:
        name = row["gripper"]["reason"]
        reasons[name] = reasons.get(name, 0) + 1
    latch_onsets, previous = [], False
    for row in rows:
        latched = row["gripper"]["closing_episode"]
        if latched and not previous:
            latch_onsets.append(example(row))
        previous = latched
    return dict(
        endpoint_count=len(rows),
        high_reference_endpoints=len(high),
        high_reference_low_observed_before=sum(r["opening_before"] < 0.2 for r in high),
        high_reference_low_observed_after=len(mismatches),
        high_reference_low_motor_after=sum(r["motor_opening_after"] < 0.2 for r in high),
        high_reference_full_closed_motor_after=sum(r["motor_opening_after"] <= 0.005 for r in high),
        high_reference_close_commands=sum(r["command"][-1] == 1 for r in high),
        high_reference_close_latch_onsets=sum(r["reference"] > 0.8 for r in latch_onsets),
        all_close_latch_onsets=latch_onsets,
        mismatch_gripper_reason_counts=reasons,
        first_mismatch=None if not mismatches else example(mismatches[0]),
        last_mismatch=None if not mismatches else example(mismatches[-1]),
    )


def first_difference(previous, current, key, *, component=None):
    for old, new in zip(previous, current, strict=True):
        a, b = old[key], new[key]
        if component is not None:
            a, b = a[component], b[component]
        if not np.array_equal(a, b):
            return dict(step=new["step"], old=a, new=b, max_abs=float(np.max(np.abs(np.asarray(a) - b))))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != REPO
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("CPU-only existing-log analysis requires repository cwd and empty CUDA/LD")
    target = Path(args.output).resolve()
    assert target.is_relative_to(REPO / "outputs") and not target.exists()
    inputs = Inputs()
    inputs.read(__file__)
    audit = inputs.json(args.audit)
    assert audit["schema_version"] == "s2a.position-readonly-audit.v1" and audit["scope"] == "full"
    for path, expected in audit["source_before_after_hashes"].items():
        inputs.read(path, expected)
    prior_path = REPO / "outputs/wam_stability_20261021/recovery_full_complete_20261021.json"
    prior = inputs.json(prior_path, "b5ad882e7e696ec7d57a05987585c7ca5eea2ffc4c957c48ac00459c99a30eef")
    assert prior["whole_registered_protocol_complete"]
    results = []
    for case in audit["cases"]:
        if case["source"] != "wam" or case["status"] != "verified":
            continue
        assert all(v is True for k, v in case["first_query003_regression"].items() if k.endswith("exact"))
        new, old = Path(case["case"]), Path(case["first_query003_regression"]["source_case"])
        old_rows_path, old_chunks_path = old / "steps.jsonl", old / "chunks.json"
        old_rows = [
            json.loads(line)
            for line in inputs.read(old_rows_path, prior["source_before_after_hashes"][str(old_rows_path)]).splitlines()
        ]
        old_chunks = inputs.json(old_chunks_path, prior["source_before_after_hashes"][str(old_chunks_path)])
        new_rows = [json.loads(line) for line in inputs.read(new / "steps.jsonl").splitlines()]
        new_chunks = inputs.json(new / "chunks.json")
        assert len(old_rows) == len(new_rows) == 600 and old_chunks[0]["step"] == new_chunks[0]["step"] == 0
        np.testing.assert_array_equal(old_chunks[0]["raw128"], new_chunks[0]["raw128"])
        assert inputs.read(old / "input_0000.png") == inputs.read(new / "input_0000.png")
        end = min(
            old_chunks[1]["step"] if len(old_chunks) > 1 else 600, new_chunks[1]["step"] if len(new_chunks) > 1 else 600
        )
        before, after = old_rows[:end], new_rows[:end]
        changes = {
            "any_command": first_difference(before, after, "command"),
            "gripper_command": first_difference(before, after, "command", component=-1),
            "arm_command": first_difference(before, after, "command", component=slice(0, 6)),
            "observed_opening": first_difference(before, after, "opening_after"),
            "motor_opening": first_difference(before, after, "motor_opening_after"),
            "arm_source_pose": first_difference(before, after, "after_source"),
            "admitted_reference_pose": first_difference(before, after, "desired_tcp"),
            "admitted_reference_opening": first_difference(before, after, "opening_target"),
            "model_clock": first_difference(before, after, "model_time_s"),
        }
        first_grip = changes["gripper_command"]
        if first_grip is not None:
            index = first_grip["step"]
            first_grip["old_detail"], first_grip["new_detail"] = example(before[index]), example(after[index])
        results.append(
            dict(
                checkpoint=case["checkpoint"],
                variant=case["variant"],
                record=case["record"],
                old_case=str(old),
                new_case=str(new),
                same_first_query_png_and_full_raw128=True,
                paired_first_query_physical_steps=end,
                paired_first_query=dict(
                    old_v2=counts(before), new_v3=counts(after), first_exact_numeric_differences=changes
                ),
                full600_different_closed_loop_inputs=dict(old_v2=counts(old_rows), new_v3=counts(new_rows)),
            )
        )
    inputs.recheck()
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        schema_version="s2a.position-gripper-paired-diagnostic.v1",
        complete_registered_wam_matrix=audit["whole_registered_protocol_complete"],
        verified_wam_pairs=len(results),
        registered_wam_pairs=24,
        cases=results,
        thresholds=dict(
            high_reference_strictly_above=0.8, low_observed_strictly_below=0.2, descriptive_not_acceptance=True
        ),
        input_sha256=inputs.hashes,
        new_model_calls=0,
        new_physics_steps=0,
        limits=[
            "Shared first-query physical interval ends before either case's second query; raw generated reference and initial image are exact, but causal governor clocks/admitted targets may diverge after different gripper actions.",
            "Exact numeric first differences are logged without a fitted significance threshold; tiny floating residuals alone need not be practically meaningful.",
            "High-reference/low-observed mismatch can follow an earlier valid low-reference closing latch and release confirmation; the counter alone does not prove an incorrect mapping.",
            "Full600 counters compare trajectories with different later observations/model generations and are descriptive, not a fixed-reference causal intervention.",
            "Counters do not establish task success, grasp force, checkpoint correctness or generalization. Partial pairs do not establish a winning setting.",
        ],
    )
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                output=str(target),
                sha256=digest(target),
                verified_pairs=len(results),
                complete=value["complete_registered_wam_matrix"],
            )
        )
    )


if __name__ == "__main__":
    main()
