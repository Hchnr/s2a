"""Audit and correct saved plate positions without rerunning MuJoCo."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import probe_bowl_native_predicate as original
from scripts.simulation.runtime import registered_feedback_phase as lifecycle
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

RESULT = {
    "path": str(original.OUTPUT / "result.json"),
    "sha256": "1eb86fc00e6f5d65e727a3aaa24131dad4a3925e0766a0b10dc910a63b562147",
}
WORKER = {
    "path": str(original.OUTPUT / "worker/result.json"),
    "sha256": "0fa42bbbcbfc9ed8c633507d696c61945fddfea525ce350ade06cfefaaefd9f4",
}


def free_joint_positions(xml, *, body_names, expected_nq):
    """Derive top-level free-joint qpos addresses from the recorded MJCF order."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    prefix.require(world is not None and len(set(body_names)) == len(body_names), "Unique object bodies are required")
    wanted = set(body_names)
    top = [body for body in world.findall("body") if body.get("name") in wanted]
    prefix.require(len(top) == len(wanted), "The requested bodies must be top-level world bodies")
    for body in top:
        joints = [node for node in body if node.tag in ("joint", "freejoint")]
        prefix.require(
            len(joints) == 1 and (joints[0].tag == "freejoint" or joints[0].get("type") == "free"),
            "Each requested body must have one top-level free joint",
        )
    addresses, offset = {}, 0

    def visit(body):
        nonlocal offset
        for child in body:
            if child.tag == "body":
                visit(child)
            elif child.tag in ("joint", "freejoint"):
                kind = "free" if child.tag == "freejoint" else child.get("type", "hinge")
                prefix.require(kind in ("free", "ball", "hinge", "slide"), "Unsupported recorded joint type")
                if body.get("name") in wanted:
                    prefix.require(kind == "free" and body.get("name") not in addresses, "Unique free joint")
                    addresses[body.get("name")] = offset
                offset += {"free": 7, "ball": 4, "hinge": 1, "slide": 1}[kind]

    for body in world.findall("body"):
        visit(body)
    prefix.require(offset == expected_nq and set(addresses) == wanted, "The recorded qpos layout must match the model")
    return addresses


def frozen_conditions(bowl, plate, pairs, *, xy_limit):
    """Detach mutable simulator buffers before retaining per-endpoint values."""
    value = original.on_conditions(bowl, plate, pairs, xy_limit=xy_limit)
    value["bowl_world_m"] = value["bowl_world_m"].copy()
    value["plate_world_m"] = value["plate_world_m"].copy()
    return value


def compare_rows(saved, periods, *, bowl_qpos, plate_qpos, nq, nv, xy_limit, bowl_geoms, plate_geoms):
    """Recompute position scalars from saved qpos and retain original contacts."""
    prefix.require(
        len(saved) == len(periods) == 40
        and [row["endpoint_step"] for row in saved] == [row["step"] + 1 for row in periods] == list(range(108, 148)),
        "All 40 corresponding adaptive endpoints are required",
    )
    corrected, errors = [], []
    for row, period in zip(saved, periods, strict=True):
        state = np.asarray(period["actual_state"], dtype=np.float64)
        prefix.require(
            state.shape == (1 + nq + nv,)
            and np.isfinite(state).all()
            and state_sha256(state) == row["state_sha256"] == period["endpoint_receipt"]["sim_state_sha256"],
            "The full saved state must match the recorded endpoint digest",
        )
        bowl = state[1 + bowl_qpos : 1 + bowl_qpos + 3].copy()
        plate = state[1 + plate_qpos : 1 + plate_qpos + 3].copy()
        pairs = row["contact_pairs"]
        prefix.require(
            row["contact_count"] == len(pairs)
            and all(
                np.isfinite(pair["distance_m"])
                and (
                    (pair["geom1"] in bowl_geoms and pair["geom2"] in plate_geoms)
                    or (pair["geom2"] in bowl_geoms and pair["geom1"] in plate_geoms)
                )
                for pair in pairs
            ),
            "The saved contacts must use the registered object geom sets",
        )
        value = frozen_conditions(bowl, plate, pairs, xy_limit=xy_limit)
        prefix.require(
            np.array_equal(value["bowl_world_m"], row["bowl_world_m"])
            and value["xy_distance_m"] == row["xy_distance_m"]
            and value["bowl_above_plate_m"] == row["bowl_above_plate_m"]
            and all(
                value[key] == row[key]
                for key in ("height_condition", "contact_condition", "xy_condition", "reconstructed_on")
            )
            and row["native_success"] is period["task_success"],
            "Saved position scalars, conditions and native labels must match the raw qpos",
        )
        error = float(np.max(np.abs(np.asarray(row["plate_world_m"], dtype=float) - plate)))
        errors.append(error)
        value.update(
            endpoint_step=row["endpoint_step"],
            state_sha256=row["state_sha256"],
            native_success=row["native_success"],
            native_grasp=row["native_grasp"],
            contact_count=row["contact_count"],
            contact_pairs=pairs,
            original_saved_plate_max_error_m=error,
        )
        corrected.append(value)
    prefix.require(
        all(row["plate_world_m"] == saved[-1]["plate_world_m"] for row in saved)
        and errors[-1] == 0
        and max(errors) > 0,
        "The original plate-vector alias must remain observable and the final value must match",
    )
    return dict(
        corrected_rows=corrected,
        original_plate_vectors_all_equal_final=True,
        original_plate_vectors_valid=False,
        original_plate_max_absolute_error_m=max(errors),
        position_scalars_match_raw_states=True,
        all_native_labels_match=all(row["native_success"] == row["reconstructed_on"] for row in corrected),
        native_failed_endpoints=[row["endpoint_step"] for row in corrected if not row["native_success"]],
    )


def audit():
    """Bind the unmodified static result, source identity and original states."""
    prefix.environment()
    result, worker = prefix.read(RESULT), prefix.read(WORKER)
    registered = prefix.read(result["registration_ref"])
    workers = prefix.read(result["workers_ref"])
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["worker_ref"] == result["available_worker_ref"] == WORKER
        and worker["complete"] is worker["passed"] is True
        and result["counters"] == worker["counters"]
        and result["counters"]["models"] == result["counters"]["data"] == 1
        and result["counters"]["forwards_attempted"] == result["counters"]["forwards_completed"] == 40
        and result["counters"]["physics_steps"]
        == result["counters"]["contexts"]
        == result["counters"]["model_requests"]
        == 0,
        "The completed non-stepping query and its original counters are required",
    )
    lifecycle.require_closed_workers(workers, ["worker"])
    prefix.checked(workers["workers"][0]["log_ref"])
    for ref in registered["source_refs"]:
        prefix.checked(ref)
    inputs = original.load_inputs()
    prefix.require(
        registered["inputs"] == original.input_receipt(inputs)
        and registered["source_refs"] == original.source_refs()
        and worker["diagnosis"]["all_native_labels_match"] is True,
        "The source-matched registration and 40 originally matching labels are required",
    )
    config, case = inputs["config"], inputs["case"]
    addresses = free_joint_positions(
        case["record_xml"],
        body_names=(config["bowl_body"], config["plate_body"]),
        expected_nq=config["model_shape"]["nq"],
    )
    details = compare_rows(
        worker["diagnosis"]["rows"],
        inputs["adaptive"],
        bowl_qpos=addresses[config["bowl_body"]],
        plate_qpos=addresses[config["plate_body"]],
        nq=config["model_shape"]["nq"],
        nv=config["model_shape"]["nv"],
        xy_limit=config["xy_limit_m"],
        bowl_geoms=set(inputs["bowl_geoms"]),
        plate_geoms=set(inputs["plate_geoms"]),
    )
    prefix.require(
        details["all_native_labels_match"] is True and details["native_failed_endpoints"] == [129],
        "The retained native labels and reconstructed conjunction must agree",
    )
    return dict(
        complete=True,
        passed=True,
        control_qualified=False,
        evidence_level="saved_static_query_alias_audit",
        original_result_ref=RESULT,
        original_worker_ref=WORKER,
        audit_source_ref=prefix.reference(Path(__file__).resolve()),
        free_joint_qpos_addresses=addresses,
        details=details,
        previous_static_work_bounds=result["static_work_bounds"],
        current_qa_synthetic_models=result["current_qa_synthetic_models"],
        cumulative_diagnostic_static_models=result["cumulative_diagnostic_static_models"],
        cumulative_physical_bounds=result["cumulative_physical_bounds"],
        cumulative_model_bounds=result["cumulative_model_bounds"],
        new_static_models=0,
        new_forwards=0,
        new_physics_steps=0,
        new_model_requests=0,
        automatic_retry=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (prefix.REPO / args.output).resolve()
    prefix.require(output.is_relative_to(prefix.REPO / "outputs"), "The audit output must stay in this repository")
    print(publish(output, audit()), flush=True)


if __name__ == "__main__":
    main()
