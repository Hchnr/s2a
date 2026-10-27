"""Audit only the four completed054 requests before its exact dispatch failure."""

import argparse
import json
from pathlib import Path

import verify_wam_clean_training_capture as CAPTURE
from verify_wam_clean_training_state import compact_points, conditioning_check, response_actions, values_equal

from s2a.clean_training_runtime import publish, reference
from s2a.synchronous_state_sampling import admit

require = CAPTURE.require


def verify(pref, regref, output):
    prepared, _, sources = admit(pref, regref, required_sources=[])
    root = Path(prepared["root"])
    evidence = CAPTURE.Evidence()
    actual_ref = reference(root / "state_sampling/result.json")
    actual = evidence.json(actual_ref)
    require(
        actual["passed"] is actual["complete"] is False
        and actual["error"] == {"type": "KeyError", "message": "'probe052'"}
        and actual["actual_requests"] == 4
        and actual["actual_denoise_calls"] == actual["actual_head_calls"] == 140
        and actual["model_initializations"] == 1,
        "Not the exact054 dispatch failure",
    )
    entries = actual["generations"]
    require(
        len(entries) == 5
        and all(g["complete"] for g in entries[:4])
        and entries[4]["index"] == 4
        and entries[4]["variant"] == "probe052"
        and entries[4]["complete"] is False,
        "054 completed/unexecuted boundary differs",
    )
    failed_dir = root / "state_sampling" / f"004_{entries[4]['name']}"
    require(
        sorted(p.name for p in failed_dir.iterdir()) == ["request.json"],
        "First adapted request has unexpected sampling artifacts",
    )
    expected_calls = [
        {"index": i * 35 + j, "label": prepared["sampling_entries"][i]["name"], "completed": True}
        for i in range(4)
        for j in range(35)
    ]
    require(actual["boundary_calls"] == expected_calls, "054 did model work past completed four requests")
    lifecycle_refs = {
        k: reference(root / p)
        for k, p in {
            "launch": "orchestration/model/launch.json",
            "exit": "orchestration/model/exit.json",
            "release": "orchestration/model/ownership/release.json",
            "gpu": "orchestration/gpu_after.json",
        }.items()
    }
    launch, exited, released, gpu = [evidence.json(lifecycle_refs[k]) for k in ["launch", "exit", "release", "gpu"]]
    require(
        exited == {"returncode": 1, "actual_wait": True, "automatic_retry": False}
        and released["reaped"] is True
        and released["returncode"] == 1
        and released["signal_sent"] is False
        and released["only_owned_child_signalled"] is True
        and all(launch["child"][k] == released[k] for k in ["pid", "start_ticks"]),
        "Actual failed exit/reap evidence differs",
    )
    require(
        gpu["released"] is True and gpu["memory_mib"] == 0 and gpu["compute_apps"] == [], "Failed054 GPU not released"
    )
    require(actual["source_sha256"] == sources, "Failed054 source identity differs")
    evidence.mapping(sources)
    evidence.mapping(prepared["input_sha256"])
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    trace0 = evidence.json(qualified["generations"][0]["trace_ref"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in trace0["steps"]]
    normal = evidence.json(actual["normalization_ref"])
    checkpoint = str(
        Path(prepared["model_config"]["checkpoint"]["run"])
        / "checkpoints"
        / prepared["model_config"]["checkpoint"]["iteration"]
        / "model"
    )
    anchors = {a["name"]: a for a in prepared["current_inputs"]}
    retained = []
    for planned, entry in zip(prepared["sampling_entries"][:4], entries[:4], strict=True):
        require(
            all(entry[k] == planned[k] for k in ["index", "name", "variant", "sampling_seed"]),
            "054 prefix identity differs",
        )
        directory = root / "state_sampling" / f"{entry['index']:03d}_{entry['name']}"
        anchor = (
            {"image_ref": planned["image_ref"], "task": planned["request"]["prompt"]}
            if entry["variant"] == "zero"
            else anchors[entry["anchor_name"]]
        )
        CAPTURE.current_request(planned["request"], dict(planned, name=planned["request"]["name"]), anchor)
        request = evidence.json(entry["request_ref"])
        require(request == planned["request"], "054 prefix saved request differs")
        condition = evidence.json(entry["conditioning_ref"])
        conditioning_check(condition, request)
        response = evidence.json(entry["response_ref"])
        response_actions(response, request, checkpoint, normal["transform"], condition["caption_sha256"])
        trace = evidence.json(entry["trace_ref"])
        require(
            trace["parameters_before"] == trace["parameters_after"] == actual["parameters_before"],
            "Prefix per-request parameters changed",
        )
        CAPTURE.verify_trace(trace, planned, grid, sources, prepared["noise_roster"], purpose="evaluation")
        ledger = evidence.json(entry["compact_ledger_ref"])
        header = evidence.json(ledger["request_ref"])
        require(
            header["request"]["noise_seed"] == planned["sampling_seed"]
            and header["request"]["current_image_sha256"] == anchor["image_ref"]["sha256"]
            and header["request"]["readout_sha256"] is None,
            "Prefix current/head compact identity differs",
        )
        historical = qualified["generations"][entry["index"]] if entry["variant"] == "zero" else None
        first, changed = compact_points(evidence, ledger, header, directory, trace, head=None, historical=historical)
        require(changed == 0, "Zero/raw prefix altered raw velocity")
        initial = evidence.json(entry["initial_ref"])
        arrays = evidence.arrays(initial["arrays"], directory / "initial.npz")
        CAPTURE.verify_initial(initial, arrays, planned, condition["caption"], first)
        if historical is not None:
            original = evidence.json(historical["initial_ref"])
            gold = evidence.arrays(original["arrays"], Path(historical["initial_ref"]["path"]).parent / "initial.npz")
            CAPTURE.paired_initial(initial, arrays, original, gold)
            for ref in [entry["expected_response_ref"], entry["qualified_response_ref"]]:
                old = evidence.json(ref)["outputs"][0]["content"]
                now = response["outputs"][0]["content"]
                values_equal(now["action"], old["action"], "Prefix zero action regression")
                values_equal(
                    now["action128"]["normalized_action"],
                    old["action128"]["normalized_action"],
                    "Prefix zero normalized regression",
                )
        retained.append(
            {
                "index": entry["index"],
                "name": entry["name"],
                "variant": entry["variant"],
                "response_ref": entry["response_ref"],
                "initial_ref": entry["initial_ref"],
                "trace_ref": entry["trace_ref"],
                "compact_ledger_ref": entry["compact_ledger_ref"],
            }
        )
    evidence.finish()
    return publish(
        output,
        {
            "schema_version": "s2a.synchronous-state-prefix-audit.v1",
            "complete": True,
            "passed": True,
            "scope": "Four completed requests only; original054 whole stage remains failed",
            "prepared_ref": pref,
            "registration_ref": regref,
            "failed_sampling_ref": actual_ref,
            "retained_requests": retained,
            "verified_nodes": 140,
            "zero_fullpoint_nodes_verified": 105,
            "remaining_original_indices": list(range(4, 99)),
            "remaining_requests": 95,
            "remaining_denoise_calls": 3325,
            "per_request_parameters_unchanged": True,
            "whole_worker_parameters_after_available": False,
            "actual_failed_exit_and_reap_verified": True,
            "lifecycle_refs": lifecycle_refs,
            "input_sha256": evidence.inputs,
            "inputs_unchanged_after": True,
            "validation_future_archive_reads": 0,
            "new_model_calls": 0,
            "fresh_authorized": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(reference(args.prepared), reference(args.registration), args.output)), flush=True)


if __name__ == "__main__":
    main()
