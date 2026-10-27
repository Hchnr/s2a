"""053 metadata admission and independent saved-byte physical dataset audit."""

import argparse
import json
from pathlib import Path

import numpy as np
from verify_settled_native_training import (
    array_file,
    bits,
    verify_canvas,
    verify_commands,
    verify_goals,
    verify_labels,
    verify_receipts,
)

from s2a.synchronous_validation import checked, current_model_input, load_prepared, publish, reference


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(prepared, pref):
    import h5py

    expected = [
        (0, "demo_16", 95),
        (0, "demo_32", 92),
        (1, "demo_4", 158),
        (1, "demo_27", 177),
        (2, "demo_14", 106),
        (2, "demo_24", 104),
    ]
    require(len(prepared["records"]) == 6, "Independent six-case denominator")
    for record, identity in zip(prepared["records"], expected, strict=True):
        require((record["task_index"], record["demo"], record["native_steps"]) == identity, "Independent fixed roster")
        n = record["native_steps"]
        require(record["anchor_frames"] == [j * (n - 40) // 3 for j in range(4)], "Independent native windows")
        with h5py.File(checked(record["source_hdf5_ref"]), "r") as archive:
            group = archive["data"][record["demo"]]
            native, initial = np.asarray(group["actions"], np.float64), np.asarray(group["states"][0], np.float64)
            xml = group.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
        bits(np.load(record["expert_path"], allow_pickle=False), native, "Native copy differs from R1 HDF")
        bits(np.asarray(record["initial_state"], np.float64), initial, "Selected R1 initial state changed")
        require(xml == record["model_xml"], "Selected R1 XML changed")
    for filename, digest in prepared["snapshot_sha256"].items():
        require(reference(filename)["sha256"] == digest, "Snapshot changed")
    preflight = json.loads(checked(prepared["preflight_ref"]).read_text())
    old = json.loads(checked(preflight["settled_prepared_ref"]).read_text())["original_prepared"]
    a, b = prepared["policy_config"], old["policy_config"]
    allowed = {"tool_contract", "output_dir"}
    require(
        {k: v for k, v in a.items() if k not in allowed} == {k: v for k, v in b.items() if k not in allowed},
        "Control policy changed beyond explicit paths",
    )
    require(
        prepared["tool_contract_ref"]["sha256"] == old["tool_contract_ref"]["sha256"], "Tool migration changed bytes"
    )
    require(np.array_equal(prepared["tool_source_from_tcp"], old["tool_source_from_tcp"]), "Tool axes changed")
    return {
        "schema_version": "s2a.synchronous-validation-metadata.v1",
        "passed": True,
        "complete": True,
        "prepared_ref": pref,
        "verifier_ref": reference(Path(__file__)),
        "cases_verified": 6,
        "source_entries": len(prepared["source_sha256"]),
        "snapshots_verified": len(prepared["snapshot_sha256"]),
        "execution_authorized": True,
        "model_calls": 0,
        "physics_steps": 0,
        "fresh_cases": 0,
    }


def physical(prepared, pref):
    root = Path(prepared["root"])
    result_ref = reference(root / "collection/result.json")
    collection = json.loads(checked(result_ref).read_text())
    require(
        collection["complete"] and collection["passed"] and collection["prepared_ref"] == pref, "Incomplete collection"
    )
    require(len(collection["cases"]) == 6, "Full collection required")
    summaries, model_inputs, case_refs = [], [], []
    for record, entry in zip(prepared["records"], collection["cases"], strict=True):
        case = json.loads(checked(entry["case"]).read_text())
        require(case["record"] == record and case["prepared_ref"] == pref, "Case identity changed")
        data = array_file(case["arrays"])
        n = record["native_steps"]
        native = np.load(record["expert_path"], allow_pickle=False)
        verify_commands(data, native, n)
        geometry = verify_labels(data, n, prepared["tool_source_from_tcp"], record["initial_state"])
        verify_goals(json.loads(checked(entry["goals"]).read_text()), data, n)
        receipts = verify_receipts(case, data)
        require([a["frame_index"] for a in case["anchors"]] == record["anchor_frames"], "Current anchor roster changed")
        canvases = []
        for anchor in case["anchors"]:
            canvases.append(verify_canvas(anchor, data, receipts[anchor["frame_index"]]))
            model_inputs.append(current_model_input(anchor, data, task=record["task"], case_name=record["name"]))
        stable = bool(data["native_success"][-40:].all())
        require(
            case["native_quality_passed"] is stable and case["future_rgb_frames"] == 0,
            "Quality/future-image claim differs",
        )
        require(case["current_canvas_images"] == 4 and case["physics_steps"] == n + 80, "Case budget changed")
        summaries.append(
            {
                "name": record["name"],
                "native_stable_last40": stable,
                "physics_steps": n + 80,
                "geometry_max_errors": geometry,
                "canvases": canvases,
            }
        )
        case_refs.append({"case_ref": entry["case"], "arrays_ref": case["arrays"], "goals_ref": entry["goals"]})
    load_prepared(pref)
    return {
        "schema_version": "s2a.synchronous-validation-physical-audit.v1",
        "complete": True,
        "passed": True,
        "prepared_ref": pref,
        "collection_ref": result_ref,
        "verifier_ref": reference(Path(__file__)),
        "cases": summaries,
        "case_refs": case_refs,
        "current_model_inputs": model_inputs,
        "physics_endpoints_verified": 1218,
        "physics_steps_verified": 1212,
        "current_images_verified": 24,
        "native_stable_cases": sum(s["native_stable_last40"] for s in summaries),
        "eligible_for_model_validation": all(s["native_stable_last40"] for s in summaries),
        "new_model_calls": 0,
        "fresh_cases": 0,
        "fresh_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    pref = reference(args.prepared)
    prepared = load_prepared(pref)
    root = Path(prepared["root"])
    if args.metadata:
        result = metadata(prepared, pref)
        print(json.dumps(publish(root / "admitted.json", result)), flush=True)
    else:
        result = physical(prepared, pref)
        print(json.dumps(publish(root / "verified.json", result)), flush=True)


if __name__ == "__main__":
    main()
