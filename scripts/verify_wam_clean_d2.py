"""056 independent saved-array D2 audit; no labels, fitting or model imports.

Uses the established NumPy capture checker primitives, not producer prediction.
Omitted video/flat tensors remain hash witnesses, not offline reconstructions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from verify_wam_clean_training_capture import (
    DOMAIN,
    MASK,
    VISION,
    Evidence,
    audit_points,
    current_request,
    paired_initial,
    require,
    training_caption,
    verify_initial,
    verify_trace,
)

from s2a.clean_training_d2_execution import admit
from s2a.clean_training_runtime import GPU


def head_parameters(value):
    require(
        set(value) == {"schema_version", "coefficients", "center", "scale", "ridge"}
        and value["schema_version"] == "s2a.clean-residual-readout.v1",
        "D2 parameter schema",
    )
    coefficient, center = np.asarray(value["coefficients"], np.float64), np.asarray(value["center"], np.float64)
    scale = value["scale"]
    require(
        coefficient.shape == (1291, 10)
        and center.shape == (1290,)
        and np.isfinite(coefficient).all()
        and np.isfinite(center).all(),
        "D2 parameter layout",
    )
    require(
        type(scale) in (int, float) and np.isfinite(scale) and scale > 0 and value["ridge"] == 0.1 / 1290,
        "D2 frozen scale/ridge",
    )
    return coefficient, center, float(scale)


def lifecycle(evidence, refs, prepared_ref, registration_ref, collection, root, *, model_python):
    launch, exit_record, release, gpu = [
        evidence.json(refs[k]) for k in ("worker_launch_ref", "worker_exit_ref", "worker_release_ref", "gpu_after_ref")
    ]
    require(
        exit_record == dict(returncode=0, actual_wait=True, automatic_retry=False), "real normal worker wait0 required"
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["original_error"] is None
        and release["signal_sent"] is False
        and release["only_owned_child_signalled"] is True,
        "real owned reap/noSignal required",
    )
    require(all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")), "owned PID/starttick binding")
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=GPU, LD_LIBRARY_PATH=""), "worker UUID/LD identity"
    )
    command = launch["command"]
    require(
        len(command) > 1 and command[0] == model_python and command[1] == "scripts/collect_wam_clean_d2.py",
        "lifecycle is not the collection worker",
    )
    for flag, value in (
        ("--prepared", prepared_ref["path"]),
        ("--prepared-sha256", prepared_ref["sha256"]),
        ("--execution-registration", registration_ref["path"]),
        ("--execution-registration-sha256", registration_ref["sha256"]),
        ("--gpu-uuid", GPU),
        ("--seed", "42"),
    ):
        require(
            command.count(flag) == 1 and command[command.index(flag) + 1] == value, "worker command admission binding"
        )
    require(
        gpu["released"] is True
        and gpu["uuid"] == GPU
        and gpu["memory_mib"] == 0
        and gpu["compute_apps"] == []
        and gpu["device_minor"] == 6
        and gpu["visible_gpu"] == 2,
        "actual normal GPU release required",
    )
    require(
        all(Path(refs[k]["path"]).resolve().is_relative_to(root) for k in refs),
        "lifecycle receipt escaped registered training root",
    )


def verify(args):
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "056 audit requires CPU environment",
    )
    evidence = Evidence()
    names = (
        "prepared",
        "execution_registration",
        "collection",
        "worker_launch",
        "worker_exit",
        "worker_release",
        "gpu_after",
    )
    refs = {
        name: dict(path=str(Path(getattr(args, name)).resolve()), sha256=getattr(args, name + "_sha256"))
        for name in names
    }
    prepared, registration, sources, _, params_ref, d0 = admit(
        refs["prepared"], refs["execution_registration"], required_sources=[Path(__file__)]
    )
    manifest = evidence.json(prepared["current_manifest_ref"])
    evidence.json(refs["prepared"])
    evidence.json(refs["execution_registration"])
    evidence.json(manifest["d0_collection_ref"])
    collection = evidence.json(refs["collection"])
    root, round_name = Path(prepared["root"]), "D2"
    require(Path(refs["collection"]["path"]) == root / "capture/D2/result.json", "056 exact collection path")
    require(
        collection["schema_version"] == "s2a.clean-d2-capture.v1"
        and collection["passed"] is collection["complete"] is True
        and collection["round_name"] == round_name,
        "056 complete D2 producer required",
    )
    require(
        collection["prepared_ref"] == refs["prepared"]
        and collection["execution_registration_ref"] == refs["execution_registration"]
        and collection["current_manifest_ref"] == prepared["current_manifest_ref"],
        "056 producer bindings differ",
    )
    require(
        collection["actual_requests"] == len(collection["requests"]) == 300
        and collection["actual_denoise_calls"] == collection["actual_head_calls"] == 10500
        and collection["model_initialization_attempts"] == collection["model_initializations"] == 1,
        "056 full300/10500 denominator",
    )
    require(
        all(collection[k] == 0 for k in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases"))
        and collection["source_sha256"] == sources
        and collection["parameters_before"] == collection["parameters_after"]
        and collection["gpu_uuid"] == GPU,
        "056 label-free unchanged model capture",
    )
    require(
        collection["training_seed_plan_ref"] == manifest["training_seed_plan_ref"]
        and collection["readout_ref"] == params_ref,
        "056 seed/policy identity differs",
    )
    lifecycle_refs = {k + "_ref": refs[k] for k in ("worker_launch", "worker_exit", "worker_release", "gpu_after")}
    lifecycle(
        evidence,
        lifecycle_refs,
        refs["prepared"],
        refs["execution_registration"],
        collection,
        root,
        model_python=prepared["model_config"]["model_python"],
    )
    qualified = evidence.json(prepared["prerequisite_refs"]["bridge_qualification"])
    trace = evidence.json(qualified["generations"][0]["trace_ref"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in trace["steps"]]
    require(len(grid) == 35, "056 full sigma grid required")
    normalizer = evidence.json(collection["normalization_ref"])
    require(
        normalizer["domain"] == DOMAIN
        and normalizer["normalization_id"] == 31
        and normalizer["projection_id"] == 0
        and collection["normalization_ref"] == d0["normalization_ref"],
        "056 original normalization31 required",
    )
    before, after = collection["cosmos_imported_sources_after_load"], collection["cosmos_imported_sources_final"]
    require(before and all(after.get(k) == v for k, v in before.items()), "056 imported source closure changed")
    for record in after.values():
        if record.get("file") is not None:
            path = Path(record["file"]).resolve()
            require(
                path.is_relative_to(Path(prepared["model_config"]["cosmos_source_root"]))
                and sources.get(str(path)) == record["sha256"],
                "056 import escaped source closure",
            )
            evidence.path(dict(path=str(path), sha256=record["sha256"]))
    head = head_parameters(evidence.json(params_ref))
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    require(
        len(anchors) == 150
        and len({e["sampling_seed"] for e in prepared["training_requests"]}) == 300
        and prepared["noise_roster"]["evaluation_seeds"] == [42, 43],
        "complete training identity/noise matrix",
    )
    labels = [e["name"] for e in prepared["training_requests"] for _ in range(35)]
    require(
        collection["boundary_calls"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "10500 actual denoise boundary ledger",
    )
    points, changed = 0, 0
    for expected, entry in zip(prepared["training_requests"], collection["requests"], strict=True):
        require(
            entry["complete"] is True
            and all(entry[k] == expected[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
            "registered300 request order/identity",
        )
        require(expected["sampling_seed"] not in (42, 43), "evaluation noise entered training capture")
        directory = root / "capture" / round_name / f"{entry['index']:03d}_{entry['name']}"
        anchor = anchors[entry["anchor_name"]]
        current_request(expected["request"], expected, anchor)
        require(
            evidence.json(entry["request_ref"]) == expected["request"]
            and Path(entry["request_ref"]["path"]).resolve() == directory / "request.json",
            "exact current-only saved request",
        )
        evidence.path(anchor["image_ref"])
        conditioning = evidence.json(entry["conditioning_ref"])
        caption = json.loads(conditioning["caption"])
        training_caption(caption, expected["request"]["prompt"], expected["sampling_seed"])
        require(
            conditioning["inference_only_current_image"] is True
            and hashlib.sha256(conditioning["caption"].encode()).hexdigest() == conditioning["caption_sha256"]
            and caption["command"] == caption["task_title"] == expected["request"]["prompt"],
            "current-image/task-only caption",
        )
        context = conditioning["context"]
        require(
            context["domain"] == DOMAIN
            and context["normalization_id"] == 31
            and context["projection_id"] == 0
            and context["action_dim_mask"] == MASK,
            "actual conditioning domain",
        )
        response = evidence.json(entry["response_ref"])["outputs"][0]["content"]
        for value in (response["action"], response["action128"]["normalized_action"]):
            array = np.asarray(value)
            require(
                array.shape == (40, 128) and np.isfinite(array).all() and not array[:, 10:].any(),
                "generated full128 response/mask",
            )
        trace = evidence.json(entry["trace_ref"])
        verify_trace(trace, expected, grid, sources, prepared["noise_roster"])
        ledger = evidence.json(entry["compact_ledger_ref"])
        require(
            Path(entry["compact_ledger_ref"]["path"]).resolve() == directory / "compact/ledger.json"
            and ledger["schema_version"] == "s2a.clean-compact-ledger.v1"
            and ledger["complete"] is True
            and ledger["nodes"] == len(ledger["points"]) == 35
            and ledger["future_labels_read"] == 0
            and ledger["omitted_vision_reconstructible"] is ledger["qualification_claim"] is False,
            "full label-free compact ledger",
        )
        header = evidence.json(ledger["request_ref"])
        require(
            Path(ledger["request_ref"]["path"]).resolve() == directory / "compact/request.json"
            and header["schema_version"] == "s2a.clean-compact-request.v1"
            and header["expected_vision_shape"] == list(VISION)
            and header["expected_grid"] == [list(x) for x in grid]
            and header["expected_nodes"] == 35
            and header["source_hash_scope"] == "C-order NumPy snapshot bytes"
            and header["labels_accepted"] is header["omitted_vision_reconstructible"] is False,
            "registered compact request/layout",
        )
        require(
            header["request"]
            == dict(
                request_name=entry["name"],
                noise_seed=entry["sampling_seed"],
                purpose="training",
                round_name=round_name,
                anchor_id=entry["anchor_name"],
                readout_sha256=None if head is None else collection["readout_ref"]["sha256"],
                current_image_sha256=anchor["image_ref"]["sha256"],
            ),
            "compact parameter/current-image identity",
        )
        first, count, mutated = audit_points(evidence, ledger, header, directory, trace["steps"], head=head)
        changed += mutated
        points += count
        initial = evidence.json(entry["initial_ref"])
        initial_arrays = evidence.arrays(initial["arrays"], directory / "initial.npz")
        require(
            initial["array_index"]
            == {k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in initial_arrays.items()},
            "initial array index",
        )
        require(initial["variant"] == "probe052", "initial policy binding")
        verify_initial(initial, initial_arrays, expected, conditioning["caption"], first)
        if d0 is not None:
            paired = d0["requests"][entry["index"]]
            require(
                all(paired[k] == entry[k] for k in ("index", "name", "anchor_name", "sampling_seed")),
                "D0/D2 paired request",
            )
            original = evidence.json(paired["initial_ref"])
            original_arrays = evidence.arrays(
                original["arrays"],
                Path(manifest["d0_collection_ref"]["path"]).parent
                / f"{entry['index']:03d}_{entry['name']}"
                / "initial.npz",
            )
            paired_initial(initial, initial_arrays, original, original_arrays)
    require(points == 10500, "056 complete numerical denominator")
    evidence.finish()
    admit(refs["prepared"], refs["execution_registration"], required_sources=[Path(__file__)])
    result = dict(
        schema_version="s2a.clean-d2-capture-audit.v1",
        complete=True,
        passed=True,
        prepared_ref=refs["prepared"],
        execution_registration_ref=refs["execution_registration"],
        collection_ref=refs["collection"],
        readout_ref=params_ref,
        actual_requests=300,
        actual_denoise_calls=points,
        changed_active_values=changed,
        initial_pairs_verified=300,
        input_sha256=evidence.inputs,
        future_label_reads=0,
        new_fits=0,
        physics_steps=0,
        fresh_cases=0,
        evidence_boundary="Saved action math and initial pairs; omitted vision/flat values remain online hash witnesses",
        **lifecycle_refs,
    )
    output = Path(args.output).resolve()
    require(output.is_relative_to(root) and not output.exists(), "056 audit output must be new and inside root")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "prepared",
        "execution-registration",
        "collection",
        "worker-launch",
        "worker-exit",
        "worker-release",
        "gpu-after",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    print(json.dumps(verify(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
