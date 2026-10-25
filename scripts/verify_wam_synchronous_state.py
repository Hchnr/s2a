"""Independent054 complete99/3465 audit; never opens validation future labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import verify_wam_clean_training_capture as CAPTURE
from verify_wam_clean_training_state import (
    compact_points,
    conditioning_check,
    response_actions,
    values_equal,
    weight_stats,
)

from s2a.clean_training_runtime import publish
from s2a.synchronous_state_sampling import GPU, admit

require = CAPTURE.require


def lifecycle(evidence, refs, prepared_ref, registration_ref, actual, prepared):
    launch, exit_record, release, gpu = [
        evidence.json(refs[k]) for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")
    ]
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False), "real normal state worker wait0"
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "real owned state reap/noSignal",
    )
    require(all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")), "owned state PID/starttick")
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""), "state worker UUID/LD"
    )
    command = launch["command"]
    require(
        command[:2] == [prepared["model_config"]["model_python"], "scripts/sample_wam_synchronous_state.py"],
        "actual state worker interpreter/entrypoint",
    )
    for flag, value in (
        ("--prepared", prepared_ref["path"]),
        ("--prepared-sha256", prepared_ref["sha256"]),
        ("--execution-registration", registration_ref["path"]),
        ("--execution-registration-sha256", registration_ref["sha256"]),
        ("--candidate-fit", actual["candidate_fit_ref"]["path"]),
        ("--candidate-fit-sha256", actual["candidate_fit_ref"]["sha256"]),
        ("--candidate-verification", actual["candidate_verification_ref"]["path"]),
        ("--candidate-verification-sha256", actual["candidate_verification_ref"]["sha256"]),
        ("--gpu-uuid", GPU),
        ("--seed", "42"),
    ):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, "state worker admission flags")
    require(
        gpu["released"] is True
        and gpu["uuid"] == GPU
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "actual state GPU release",
    )
    require(
        all(
            Path(refs[k]["path"]).resolve().is_relative_to(Path(prepared["root"]))
            for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")
        ),
        "state receipt escaped registered root",
    )


def verify_generations(evidence, prepared, registration, sources, actual, head):
    root = Path(prepared["root"]) / "state_sampling"
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    qualified_trace = evidence.json(qualified["generations"][0]["trace_ref"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in qualified_trace["steps"]]
    require(len(grid) == 35, "qualified35 grid")
    normalizer = evidence.json(actual["normalization_ref"])
    require(
        normalizer["domain"] == CAPTURE.DOMAIN
        and normalizer["normalization_id"] == 31
        and normalizer["projection_id"] == 0,
        "state normalization31/projection0",
    )
    checkpoint = str(
        Path(prepared["model_config"]["checkpoint"]["run"])
        / "checkpoints"
        / prepared["model_config"]["checkpoint"]["iteration"]
        / "model"
    )
    anchors = {a["name"]: a for a in prepared["current_inputs"]}
    roster = prepared["sampling_entries"]
    require(len(actual["generations"]) == 99, "all99 state outputs required")
    require(
        actual["boundary_calls"]
        == [dict(index=i * 35 + j, label=p["name"], completed=True) for i, p in enumerate(roster) for j in range(35)],
        "3465 denoise boundary calls",
    )
    generated, pairs, changed = [], {}, 0
    for planned, entry in zip(roster, actual["generations"], strict=True):
        identity = (
            ("index", "name", "variant", "sampling_seed", "zero_case_name")
            if planned["variant"] == "zero"
            else ("index", "validation_index", "name", "variant", "sampling_seed", "anchor_name")
        )
        require(
            entry["complete"] is True and all(entry[k] == planned[k] for k in identity), "exact99 state identity/order"
        )
        directory = root / f"{entry['index']:03d}_{entry['name']}"
        historical = None
        if entry["variant"] == "zero":
            case = prepared["zero_cases"][entry["index"]]
            anchor = dict(image_ref=case["image_ref"], task=planned["request"]["prompt"])
            require(
                entry["qualified_generation_ref"] == prepared["prerequisite_refs"]["bridge_qualification"]
                and entry["qualified_generation_index"] == entry["index"],
                "zero037 qualified source identity",
            )
            historical = qualified["generations"][entry["index"]]
            require(
                historical["variant"] == "raw"
                and historical["case_id"] == case["name"]
                and entry["expected_response_ref"] == case["expected_response_ref"]
                and entry["qualified_response_ref"] == historical["response_ref"],
                "zero raw/source response origin",
            )
            gold = evidence.json(entry["zero_source_hashes_ref"])
            require(
                Path(entry["zero_source_hashes_ref"]["path"]).resolve() == directory / "historical_source_hashes.json"
                and gold["qualified_generation_ref"] == entry["qualified_generation_ref"]
                and gold["qualified_generation_index"] == entry["qualified_generation_index"]
                and gold["point_refs"] == historical["points"]
                and len(gold["source_arrays"]) == 35,
                "zero source sidecar exact qualified origin",
            )
            historical = dict(historical, source_summary_witnesses=gold["source_arrays"])
        else:
            anchor = anchors[entry["anchor_name"]]
            require(
                entry["sampling_seed"] in (42, 43) and entry["variant"] in ("raw", "probe052"),
                "evaluation noise/variant",
            )
        CAPTURE.current_request(planned["request"], dict(planned, name=planned["request"]["name"]), anchor)
        evidence.path(anchor["image_ref"])
        request = evidence.json(entry["request_ref"])
        require(
            Path(entry["request_ref"]["path"]).resolve() == directory / "request.json"
            and request == planned["request"],
            "exact saved state request",
        )
        conditioning = evidence.json(entry["conditioning_ref"])
        conditioning_check(conditioning, request)
        response = evidence.json(entry["response_ref"])
        actions = response_actions(
            response, request, checkpoint, normalizer["transform"], conditioning["caption_sha256"]
        )
        trace = evidence.json(entry["trace_ref"])
        require(
            trace["purpose"] == "evaluation" and trace["parameters_before"] == actual["parameters_before"],
            "actual evaluation base parameter identity",
        )
        CAPTURE.verify_trace(trace, planned, grid, sources, prepared["noise_roster"], purpose="evaluation")
        ledger = evidence.json(entry["compact_ledger_ref"])
        require(
            Path(entry["compact_ledger_ref"]["path"]).resolve() == directory / "compact/ledger.json"
            and ledger["schema_version"] == "s2a.clean-compact-ledger.v1"
            and ledger["complete"] is True
            and ledger["future_labels_read"] == 0
            and ledger["omitted_vision_reconstructible"] is False
            and ledger["qualification_claim"] is False,
            "complete label-free state compact ledger",
        )
        header = evidence.json(ledger["request_ref"])
        require(
            Path(ledger["request_ref"]["path"]).resolve() == directory / "compact/request.json"
            and header["schema_version"] == "s2a.clean-compact-request.v1"
            and header["expected_vision_shape"] == list(CAPTURE.VISION)
            and header["expected_nodes"] == 35
            and header["expected_grid"] == [list(x) for x in grid]
            and header["source_hash_scope"] == "C-order NumPy snapshot bytes"
            and header["labels_accepted"] is header["omitted_vision_reconstructible"] is False,
            "state compact header",
        )
        require(
            header["request"]
            == dict(
                request_name=entry["name"],
                noise_seed=entry["sampling_seed"],
                purpose="evaluation",
                round_name="state",
                anchor_id=anchor.get("name", entry.get("zero_case_name")),
                readout_sha256=actual["readout_ref"]["sha256"] if entry["variant"] == "probe052" else None,
                current_image_sha256=anchor["image_ref"]["sha256"],
            ),
            "state compact identity",
        )
        first, mutated = compact_points(
            evidence,
            ledger,
            header,
            directory,
            trace,
            head=head if entry["variant"] == "probe052" else None,
            historical=historical,
        )
        changed += mutated
        initial = evidence.json(entry["initial_ref"])
        arrays = evidence.arrays(initial["arrays"], directory / "initial.npz")
        require(
            initial["variant"] == entry["variant"]
            and initial["array_index"] == {k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in arrays.items()},
            "state actual initial metadata",
        )
        CAPTURE.verify_initial(initial, arrays, planned, conditioning["caption"], first)
        if historical is not None:
            original = evidence.json(historical["initial_ref"])
            original_arrays = evidence.arrays(
                original["arrays"], Path(historical["initial_ref"]["path"]).parent / "initial.npz"
            )
            CAPTURE.paired_initial(initial, arrays, original, original_arrays)
            for old_ref in (entry["expected_response_ref"], entry["qualified_response_ref"]):
                old = evidence.json(old_ref)["outputs"][0]["content"]
                content = response["outputs"][0]["content"]
                values_equal(content["action"], old["action"], "zero full128 raw exact regression")
                values_equal(
                    content["action128"]["normalized_action"],
                    old["action128"]["normalized_action"],
                    "zero full128 normalized exact regression",
                )
        else:
            key = (entry["anchor_name"], entry["sampling_seed"])
            if entry["variant"] == "raw":
                require(key not in pairs, "duplicate raw validation pair")
                pairs[key] = entry["initial_ref"]
            else:
                require(key in pairs, "W2 missing corresponding raw initial state")
                original = evidence.json(pairs[key])
                raw_directory = root / f"{entry['index'] - 1:03d}_{roster[entry['index'] - 1]['name']}"
                raw_arrays = evidence.arrays(original["arrays"], raw_directory / "initial.npz")
                CAPTURE.paired_initial(initial, arrays, original, raw_arrays)
            generated.append((planned, anchor, actions))
    require(len(pairs) == 48 and len(generated) == 96, "full48 identical-noise/current-condition pairs")
    return generated, changed


def verify(refs, output):
    evidence = CAPTURE.Evidence()
    prepared, registered, sources = admit(refs["prepared"], refs["registration"], required_sources=[Path(__file__)])
    actual = evidence.json(refs["sampling"])
    require(
        actual["schema_version"] == "s2a.synchronous-state-sampling.v1"
        and actual["complete"] is actual["passed"] is True,
        "Complete054 sampling required",
    )
    require(
        actual["prepared_ref"] == refs["prepared"] and actual["execution_registration_ref"] == refs["registration"],
        "054 worker registration identity differs",
    )
    require(
        actual["candidate_fit_ref"] == prepared["candidate_fit_ref"]
        and actual["candidate_verification_ref"] == prepared["candidate_verification_ref"]
        and actual["readout_ref"] == prepared["candidate_params_ref"],
        "054 candidate identity differs",
    )
    require(
        actual["actual_requests"] == 99
        and actual["actual_denoise_calls"] == actual["actual_head_calls"] == 3465
        and actual["model_initializations"] == actual["model_initialization_attempts"] == 1,
        "054 actual budget differs",
    )
    require(
        actual["future_label_reads"]
        == actual["producer_fits"]
        == actual["physics_steps"]
        == actual["fresh_cases"]
        == 0,
        "054 model worker consumed unregistered inputs/work",
    )
    require(
        actual["source_sha256"] == sources
        and actual["parameters_before"] == actual["parameters_after"]
        and actual["parameters_before"]["count"] > 0
        and actual["gpu_uuid"] == GPU,
        "054 source/base parameter identity differs",
    )
    evidence.mapping(sources)
    evidence.mapping(prepared["input_sha256"])
    evidence.mapping(prepared["snapshot_sha256"])
    lifecycle(evidence, refs, refs["prepared"], refs["registration"], actual, prepared)
    params = evidence.json(actual["readout_ref"])
    coefficient, center = np.asarray(params["coefficients"], np.float64), np.asarray(params["center"], np.float64)
    require(
        coefficient.shape == (1291, 10)
        and center.shape == (1290,)
        and params["ridge"] == 0.1 / 1290
        and np.isfinite(coefficient).all()
        and np.isfinite(center).all()
        and np.isfinite(params["scale"])
        and params["scale"] > 0,
        "054 head numerical geometry differs",
    )
    head = coefficient, center, params["scale"]
    zero = evidence.json(actual["zero_head_ref"])
    require(
        not np.asarray(zero["coefficients"]).any() and not np.asarray(zero["center"]).any() and zero["scale"] == 1.0,
        "054 explicit zero head differs",
    )
    before, after = actual["cosmos_imported_sources_after_load"], actual["cosmos_imported_sources_final"]
    require(before and all(after.get(k) == v for k, v in before.items()), "054 imported source changed")
    model_root = Path(prepared["model_config"]["cosmos_source_root"]).resolve()
    for item in after.values():
        if item.get("file") is not None:
            path = Path(item["file"]).resolve()
            require(
                path.is_relative_to(model_root) and sources.get(str(path)) == item["sha256"],
                "054 model escaped source closure",
            )
            evidence.path({"path": str(path), "sha256": item["sha256"]})
    generated, changed = verify_generations(evidence, prepared, registered, sources, actual, head)
    evidence.finish()
    weight_stats(prepared)
    result = {
        "schema_version": "s2a.synchronous-state-numerical-audit.v1",
        "complete": True,
        "passed": True,
        "prepared_ref": refs["prepared"],
        "registration_ref": refs["registration"],
        "sampling_ref": refs["sampling"],
        "candidate_params_ref": actual["readout_ref"],
        "actual_requests": 99,
        "actual_nodes": 3465,
        "zero_nodes_verified": 105,
        "paired_initial_conditions_verified": 48,
        "numerical_contract_passed": True,
        "lifecycle_passed": True,
        "worker_launch_ref": refs["worker_launch"],
        "worker_exit_ref": refs["worker_exit"],
        "worker_release_ref": refs["worker_release"],
        "gpu_after_ref": refs["gpu_after"],
        "input_sha256": evidence.inputs,
        "inputs_unchanged_after": True,
        "changed_active_values": changed,
        "validation_future_archive_reads": 0,
        "fresh_cases": 0,
        "generated_refs": [
            {
                "name": p["name"],
                "anchor_name": a["name"],
                "variant": p["variant"],
                "sampling_seed": p["sampling_seed"],
                "response_ref": actual["generations"][p["index"]]["response_ref"],
            }
            for p, a, _ in generated
        ],
    }
    return publish(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "registration", "sampling", "worker-launch", "worker-exit", "worker-release", "gpu-after"]:
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = {
        name: {"path": str(getattr(args, name).resolve()), "sha256": getattr(args, name + "_sha256")}
        for name in [
            "prepared",
            "registration",
            "sampling",
            "worker_launch",
            "worker_exit",
            "worker_release",
            "gpu_after",
        ]
    }
    print(json.dumps(verify(refs, args.output)), flush=True)


if __name__ == "__main__":
    main()
