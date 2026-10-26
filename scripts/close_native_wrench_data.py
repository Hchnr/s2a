"""103 complete native wrench data and lifecycle audit with old-prefix binding."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/native_wrench_data_20261025")


def absent(child):
    path = Path(f"/proc/{child['pid']}/stat")
    assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]


def main():
    e = Evidence()
    e.read(__file__)
    config = e.json("configs/native_wrench_data_20261025.json")
    execution = e.json(ROOT / "execution_result.json")
    supervision = e.json(ROOT / "supervision_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 60
    assert execution["physics_steps"] == config["physics_steps"] == 6180
    assert supervision["owner_actual_wait_recorded"]
    ref = supervision["owner_stage_ref"]
    e.check(ref["path"], ref["sha256"])
    rows = []
    for profile in config["profiles"]:
        index = profile["index"]
        folder = ROOT / f"group{index:02d}"
        for kind in ("orchestration", "verification"):
            receipts = ROOT / kind / f"group{index:02d}"
            launch = normal_receipts(
                e, receipts / "launch.json", receipts / "exit.json", receipts / "release.json", format="durable"
            )
            absent(launch["child"])
            assert (
                launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
                == launch["selected_environment"]["LD_LIBRARY_PATH"]
                == ""
            )
        verified = e.json(folder / "independent_verified.json")
        collection = e.json(folder / "collection_result.json")
        assert verified["complete"] and verified["prefix_matches"] and verified["current_wrench_contract_verified"]
        assert verified["sensor_frame_independent_xml_verified"]
        assert verified["native_index"] == index and verified["split"] == profile["split"]
        assert verified["wrench_endpoints"] == profile["physics_steps"] + 1
        assert (
            collection["baseline_ref"] == profile["trajectory_ref"]
            and collection["native_cache_ref"] == profile["cache_ref"]
        )
        for path, signature in verified["source_sha256"].items():
            known = e.hashes.get(str(Path(path).resolve()))
            if known is None:
                e.check(path, signature)
            else:
                assert known == signature
        for key in ("wrench_ref", "arrays_ref"):
            ref = collection[key]
            e.check(ref["path"], ref["sha256"])
        rows.append(
            dict(
                index=index,
                split=profile["split"],
                task=profile["task"],
                source=profile["trajectory_ref"],
                physics_steps=profile["physics_steps"],
                wrench_ref=collection["wrench_ref"],
                arrays_ref=collection["arrays_ref"],
            )
        )
    assert (
        len(rows) == 30
        and sum(r["split"] == "fit" for r in rows) == 24
        and sum(r["split"] == "holdout" for r in rows) == 6
    )
    assert sum(r["physics_steps"] for r in rows) == 6180
    assert sum(r["physics_steps"] + 1 for r in rows) == 6210
    # Independently bind all old six probe anchors to the same native causal prefix.
    paired = e.json("configs/paired_arm_response_20261025.json")
    sensor_closed = e.json(config["sensor_closure_ref"]["path"])
    assert sensor_closed["complete"] and sensor_closed["current_wrench_contract_verified"]
    for group in range(6):
        native_sha = paired["profiles"][group // 2]["trajectory_ref"]["sha256"]
        row = next(r for r in rows if r["source"]["sha256"] == native_sha)
        old_folder = Path("outputs/current_wrench_sensor_20261025") / f"group{group:02d}"
        old_collection_path = old_folder / "collection_result.json"
        e.check(old_collection_path, sensor_closed["source_sha256"][str(old_collection_path.resolve())])
        old = e.json(old_collection_path)
        old_ref = old["wrench_ref"]
        e.check(old_ref["path"], old_ref["sha256"])
        earlier = e.json(old_ref["path"])["samples"]
        current = e.json(row["wrench_ref"]["path"])["samples"]
        for a, b in zip(earlier, current[: len(earlier)], strict=True):
            for key in ("local", "sensor_pose", "control_site_position", "world_at_control_site", "time"):
                np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-9, err_msg=f"oldgroup{group}:{key}")
    owner = ROOT / "owner_wait"
    absent(
        normal_receipts(e, owner / "launch.json", owner / "exit.json", owner / "release.json", format="durable")[
            "child"
        ]
    )
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    e.recheck()
    summary = dict(
        complete=True,
        data_qualified=True,
        trajectories=rows,
        verified_trajectories=30,
        fit_trajectories=24,
        holdout_trajectories=6,
        physics_steps=6180,
        wrench_endpoints=6210,
        old_six_anchor_prefixes_matched=True,
        environment_initializations=30,
        initialization_internal_steps_not_counted=True,
        model_calls=0,
        new_fits=0,
        fresh_cases=0,
        control_qualification=False,
    )
    with (ROOT / "data_summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    e.read(ROOT / "data_summary.json")
    e.recheck()
    result = dict(
        complete=True,
        data_qualified=True,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        verified_trajectories=30,
        physics_steps=6180,
        wrench_endpoints=6210,
        old_six_anchor_prefixes_matched=True,
        source_sha256=e.hashes,
        summary_ref=dict(path=str((ROOT / "data_summary.json").resolve()), sha256=sha256(ROOT / "data_summary.json")),
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
        control_qualification=False,
    )
    with (ROOT / "closure_verified.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print("103 qualified30 trajectories/6210 current endpoints;60stages+owner actualwait0;no fit")


if __name__ == "__main__":
    main()
