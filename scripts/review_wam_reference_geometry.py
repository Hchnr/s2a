"""Derive closure/contact interpretations without rerunning physics or WAM."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

DIRECTORY = REPO / "outputs/wam_stability_20261021/reference_geometry_diagnosis"


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU read-only review requires empty CUDA and LD paths")
    destination = DIRECTORY / "closure_and_objects_review.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest_path = DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    signatures = dict(manifest["input_sha256"], **manifest["output_sha256"])
    signatures[str(manifest_path)] = sha256(manifest_path)
    signatures[str(Path(__file__).resolve())] = sha256(__file__)
    for path, digest in signatures.items():
        assert sha256(path) == digest, path
    geometry = json.loads((DIRECTORY / "geometry_diagnosis.json").read_text())
    cases = []
    for case in geometry["cases"]:
        if case["executor"] == "old_baseline":
            continue
        rows_path = Path(case["case"]) / "steps.jsonl"
        assert str(rows_path) in signatures
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        events = []
        for event in case["low_reference_opening_onsets"]:
            step = event["step"]
            row = rows[step]
            errors = np.asarray(row["tcp_error"])
            events.append(
                dict(
                    event,
                    canonical_tcp_tracking_error_mm=float(1000 * np.linalg.norm(errors[:3])),
                    canonical_so3_tracking_error_rad=float(np.linalg.norm(errors[3:])),
                    privileged_objects_offline_only=True,
                )
            )
        objects = {}
        for name in rows[0]["objects"]:
            positions = np.asarray([r["objects"][name]["position"] for r in rows])
            contact = np.asarray([r["objects"][name]["grasp_contact"] for r in rows])
            objects[name] = dict(
                native_bilateral_contact_endpoints=int(contact.sum()),
                first_native_contact_step=None if not contact.any() else int(np.flatnonzero(contact)[0]),
                last_native_contact_step=None if not contact.any() else int(np.flatnonzero(contact)[-1]),
                max_body_displacement_from_first_endpoint_mm=float(
                    1000 * np.linalg.norm(positions - positions[0], axis=1).max()
                ),
                max_height_gain_mm=float(1000 * max(r["objects"][name]["height_gain_m"] for r in rows)),
                final_body_displacement_from_first_endpoint_mm=float(
                    1000 * np.linalg.norm(positions[-1] - positions[0])
                ),
            )
        consumption = []
        for chunk in case["chunks"]:
            item = chunk["consumption"]
            first = item["first_generated_low_opening_time_s"]
            first_low_not_consumed = first is not None and first > item["max_model_time_s"] + 1e-10
            consumption.append(
                dict(
                    chunk=chunk["chunk"],
                    query_step=chunk["query_step"],
                    max_model_time_s=item["max_model_time_s"],
                    first_generated_low_opening_time_s=first,
                    first_low_opening_not_consumed=first_low_not_consumed,
                    first_low_opening_lost_to_requery=first_low_not_consumed
                    and item["suffix_status"] == "replaced_before_model_horizon",
                    first_low_opening_unconsumed_at_budget_end=first_low_not_consumed
                    and item["suffix_status"] == "unconsumed_at_physics_budget_end",
                    any_low_opening_suffix_unconsumed=item["unconsumed_low_opening_endpoints"] > 0,
                    consumed_admitted_min_opening=item["consumed_admitted_min_opening"],
                    suffix_status=item["suffix_status"],
                )
            )
        cases.append(
            dict(
                label=case["label"],
                record=case["record"],
                executor=case["executor"],
                case=case["case"],
                task_target=case["target"],
                native_task_success=case["task_ever"],
                objects=objects,
                low_opening_events=events,
                consumption=consumption,
                lost_first_low_opening_to_requery_chunks=[
                    c["chunk"] for c in consumption if c["first_low_opening_lost_to_requery"]
                ],
                first_low_opening_unconsumed_at_budget_end_chunks=[
                    c["chunk"] for c in consumption if c["first_low_opening_unconsumed_at_budget_end"]
                ],
            )
        )
    for path, digest in signatures.items():
        assert sha256(path) == digest, path
    write_new(
        destination,
        dict(
            seed=args.seed,
            input_sha256=signatures,
            observed_cases=8,
            interpretation_limits=[
                "A low-opening suffix can be unconsumed even when its first low-opening event was already consumed. These are distinct counts.",
                "The opening<=.2 threshold is an offline descriptive threshold, not a tuned controller or grasp-feasibility criterion.",
                "Object-body displacement cannot identify pushing versus grasping. Native bilateral-contact predicates remain separate endpoint metrics.",
                "Only eight completed raw-alpha1 physical trajectories are reviewed; alpha.5/0 executions are outside this fixed snapshot.",
            ],
            cases=cases,
        ),
    )
    print(json.dumps(dict(output=str(destination), sha256=sha256(destination), cases=len(cases))))


if __name__ == "__main__":
    main()
