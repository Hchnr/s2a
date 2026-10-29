"""Freeze event-selected branch inputs and reconstruct causal gripper history offline."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.loaded_velocity_control import LoadedVelocityControl


def main():
    audit_ref = reference(REPO / "outputs/preview_chunk_events_20261027/verified.json")
    audit = read(audit_ref)
    root = REPO / "outputs/preview_chunk_diagnostic_preparation_20261027"
    selected, omitted = [], []
    for case in audit["cases"]:
        event = next(
            (
                e
                for e in case["events"]
                if e["current_opening"] > 0.8 and e["first_close_index"] is not None and e["first_close_index"] >= 8
            ),
            None,
        )
        if event is None:
            omitted.append(dict(case=case["case"], reason="no event satisfying frozen selection rule"))
            continue
        refs = {Path(r["path"]).name: r for r in case["input_refs"]}
        worker, chunks = read(refs["worker_result.json"]), read(refs["chunks.json"])
        config = read(worker["config_ref"])
        rows = [json.loads(line) for line in Path(refs["steps.jsonl"]["path"]).read_text().splitlines()]
        folder = Path(refs["chunks.json"]["path"]).parent
        manifest_ref = reference(folder / "manifest.json")
        manifest = read(manifest_ref)
        tool_ref = reference(Path(manifest["runtime_config"]["tool_contract"]))
        h = np.asarray(read(tool_ref)["source_from_tcp"])
        parameters = read(config["response_parameters_ref"])
        controller = LoadedVelocityControl(
            parameters, rows[0]["opening_before"], velocity_envelope=config["gripper_velocity_envelope"]
        )
        previous_command = 0.0
        maximum_difference = 0.0
        for row in rows[: event["physical_step"]]:
            chunk = chunks[row["chunk_index"]]
            outcome = controller.update(
                row["opening_before"],
                np.asarray(row["motor_target_before"]),
                row["measured_period_velocity"],
                np.asarray(chunk["effective_targets_tcp"]),
                np.asarray(chunk["openings"]),
                row["chunk_physical_step"],
                np.asarray(row["measured_source"]) @ h,
                previous_command=previous_command,
                dt=0.05,
            )
            difference = abs(outcome.command - row["command"][-1])
            require(
                difference <= 1e-12 and outcome.route == row["gripper"]["route"],
                "Causal gripper reconstruction differs",
            )
            maximum_difference = max(maximum_difference, difference)
            previous_command = outcome.command
        selected.append(
            dict(
                case=case["case"],
                event=event,
                original_refs=list(refs.values()),
                config_ref=worker["config_ref"],
                manifest_ref=manifest_ref,
                tool_ref=tool_ref,
                parameter_ref=config["response_parameters_ref"],
                endpoint_receipts_ref=reference(Path(worker["endpoint_receipts_ref"]["path"])),
                replay_periods=event["physical_step"],
                gripper_reconstruction_max_difference=maximum_difference,
                branch_horizons=[8, 40],
                control_solver="original strict_convex_qp; no hot replacement",
            )
        )
    require(len(selected) == 7 and len(omitted) == 1, "Expected fixed7 selected and1 omitted")
    value = dict(
        complete=True,
        passed=True,
        audit_ref=audit_ref,
        cases=selected,
        omitted=omitted,
        source_ref=reference(Path(__file__)),
        plan_ref=reference(REPO / "plans/v0.0.1/preview_chunk_event_diagnostic_20261027.md"),
        total_branch_physics_budget=sum(2 * c["replay_periods"] + 48 for c in selected),
        prepared_only=True,
        execution_authorized=False,
        fresh_authorized=False,
        new_model_calls=0,
        physics_steps=0,
        gripper_history_reconstructed_periods=sum(c["replay_periods"] for c in selected),
    )
    print(json.dumps(publish(root / "prepared.json", value)))


if __name__ == "__main__":
    main()
