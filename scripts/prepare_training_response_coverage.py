"""097 deterministic all-remaining-fit roster; no simulation or model fitting."""

from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def main():
    e = Evidence()
    e.read(__file__)
    old = e.json("configs/paired_arm_response_20261025.json")
    source_path = Path("outputs/native_arm_response_data_20261025/reconstruction_result.json")
    closure = e.json(source_path.parent / "closure_verified.json")
    assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
    source = e.json(source_path)
    excluded = {p["trajectory_ref"]["sha256"] for p in old["profiles"]}
    profiles, trials = [], []
    for row in source["trajectories"]:
        if row["split"] != "fit" or row["source"]["sha256"] in excluded:
            continue
        ref = row["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        meta = Path(row["source"]["path"]).parent / "result.json"
        e.check(meta, closure["source_sha256"][str(meta.resolve())])
        m = e.json(meta)
        assert m["complete"] and m["passed"] and m["native_quality_passed"]
        pref = m["prepared_ref"]
        e.check(pref["path"], pref["sha256"])
        with np.load(ref["path"], allow_pickle=False) as cache:
            commands = cache["action_command"]
            anchor = int(np.flatnonzero(commands[:, -1] > 0)[0])
        with np.load(row["source"]["path"], allow_pickle=False) as original:
            np.testing.assert_array_equal(commands, original["sent_commands"])
        group = len(profiles)
        profiles.append(
            dict(
                task=row["task"],
                record=m["record"],
                trajectory_ref=row["source"],
                case_ref=dict(path=str(meta.resolve()), sha256=sha256(meta)),
                native_data_index=row["index"],
            )
        )
        for branch in range(7):
            axis = None if branch == 0 else (branch - 1) // 2
            offset = 0 if branch == 0 else (-0.1 if branch % 2 else 0.1)
            trials.append(
                dict(
                    profile_index=group,
                    anchor_group=group,
                    anchor_cycle=anchor,
                    physics_steps=anchor + 1,
                    phase="first_close",
                    translation_axis=axis,
                    normalized_command_offset=offset,
                    branch="baseline" if branch == 0 else str(axis) + ("_minus" if offset < 0 else "_plus"),
                )
            )
    assert len(profiles) == 21 and len(trials) == 147
    assert [sum(p["task"] == i for p in profiles) for i in range(3)] == [7, 7, 7]
    assert sum(t["physics_steps"] for t in trials) == 8281
    e.recheck()
    write_once(
        "configs/training_response_coverage_20261025.json",
        dict(
            seed=42,
            profiles=profiles,
            trials=trials,
            physics_steps=8281,
            prior_closure_path=old["prior_closure_path"],
            native_data_result_ref=dict(path=str(source_path.resolve()), sha256=sha256(source_path)),
            excluded_086_trajectory_sha256=sorted(excluded),
            all_new_data_split="fit",
            model_calls=0,
            new_fits=0,
            fresh_cases=0,
            source_sha256=e.hashes,
        ),
    )
    print("097 registered21fit anchors/147trials/8281control periods; physics0")


if __name__ == "__main__":
    main()
