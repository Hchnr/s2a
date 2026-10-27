"""191 complete21-trial data and actual lifecycle qualification; no task claim."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import sha256
from s2a.owned_receipt_contract import normal_receipts

ROOT = Path("outputs/finger_loading_interventions_20261025")


def absent(child):
    path = Path(f"/proc/{child['pid']}/stat")
    assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]


def main():
    e = Evidence()
    e.read(__file__)
    config = e.json("configs/finger_loading_interventions_20261025.json")
    prior = e.json(config["prior_closure_path"])
    assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
    execution = e.json(ROOT / "execution_result.json")
    assert execution["complete"] and execution["passed"] and len(execution["stages"]) == 42
    assert execution["physics_steps"] == config["physics_steps"] == 3003
    assert len(config["trials"]) == 21
    assert len(config["profiles"]) == 3 and config["all_new_data_split"] == "fit"
    assert [sum(p["task"] == t for p in config["profiles"]) for t in range(3)] == [0, 3, 0]
    for path, signature in config["source_sha256"].items():
        e.check(path, signature)
    cases = []
    for i, trial in enumerate(config["trials"]):
        for parent in ("orchestration", "verification"):
            folder = ROOT / parent / f"case{i:02d}"
            launch = normal_receipts(
                e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable"
            )
            absent(launch["child"])
            assert (
                launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
                == launch["selected_environment"]["LD_LIBRARY_PATH"]
                == ""
            )
        folder = ROOT / f"case{i:02d}"
        verified = e.json(folder / "independent_verified.json")
        assert verified["complete"] and verified["mechanical_verified"] and verified["prefix_native_match"]
        assert verified["source_twist_independent_xml_verified"] and verified["current_images_verified"]
        assert verified["baseline_native_match"] == (i % 7 == 0)
        assert verified["task_success_qualification"] is False
        assert verified["physics_steps"] == trial["physics_steps"] and verified["anchor_group"] == i // 7
        collection = e.json(folder / "collection_result.json")
        assert collection["complete"] and collection["physics_steps"] == trial["physics_steps"]
        expected_profile = dict(config["profiles"][trial["profile_index"]], **trial)
        assert collection["profile"] == expected_profile
        for evidence in (verified, collection):
            for path, signature in evidence["source_sha256"].items():
                known = e.hashes.get(str(Path(path).resolve()))
                if known is None:
                    e.check(path, signature)
                else:
                    assert known == signature, path
        ref = collection["arrays_ref"]
        e.check(ref["path"], ref["sha256"])
        assert len(collection["images"]) == 2
        cases.append(dict(case=i, trial=trial, arrays_ref=ref, baseline_native_match=verified["baseline_native_match"]))
    folder = ROOT / "owner_wait"
    owner = normal_receipts(e, folder / "launch.json", folder / "exit.json", folder / "release.json", format="durable")[
        "child"
    ]
    absent(owner)
    supervision = e.json(ROOT / "supervision_result.json")
    assert supervision["owner_actual_wait_recorded"]
    e.check(supervision["owner_stage_ref"]["path"], supervision["owner_stage_ref"]["sha256"])
    for path, signature in e.json(ROOT / "source_sha256.json").items():
        e.check(path, signature)
    assert sum(c["trial"]["physics_steps"] for c in cases) == 3003
    e.recheck()
    result = dict(
        complete=True,
        data_qualified=True,
        task_success_qualification=False,
        verified_cases=21,
        physics_steps=3003,
        baseline_native_matched=3,
        owner_normal_wait_verified=True,
        source_roster_unchanged=True,
        cases=cases,
        environment_initializations=21,
        initialization_internal_steps_not_counted=True,
        explicit_integration_substeps=32725,
        all_new_data_split="fit",
        new_fits=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "data_summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    e.read(ROOT / "data_summary.json")
    e.recheck()
    closure = dict(
        complete=True,
        data_qualified=True,
        verified_cases=21,
        owner_normal_wait_verified=True,
        task_success_qualification=False,
        physics_steps=3003,
        source_sha256=e.hashes,
        summary_ref=dict(path=str((ROOT / "data_summary.json").resolve()), sha256=sha256(ROOT / "data_summary.json")),
    )
    with (ROOT / "closure_verified.json").open("x") as stream:
        json.dump(closure, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("191 closed21trials3003steps;3baselines;42stages+owner normalwait0;data only")


if __name__ == "__main__":
    main()
