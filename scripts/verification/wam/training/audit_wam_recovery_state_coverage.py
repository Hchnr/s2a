"""Compare recorded WAM request states with actual training image anchors."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.geometry import validate_transform
from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.state_anchor_coverage import nearest_position_anchor, summarize_distances
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/wam_recovery_state_coverage_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/wam_recovery_state_coverage_20261116.md"
OUTPUT = ROOT / "outputs/wam_recovery_state_coverage_20261116/result.json"
CASE_INDICES = (0, 2, 4, 6, 8, 10)
REQUESTS = (46, 5, 19, 9, 75, 59)
TASKS = ("put the bowl on the plate", "put the wine bottle on the rack", "put the cream cheese in the bowl")
REFERENCES = {
    "development_audit": (
        "outputs/reset_multitask_adaptive_complete_audit_20261113/result.json",
        "7fe0f7d17ddea1913ffab2f5a4243c5b8e4b6a5def9633269ebec95efc82f8fe",
    ),
    "training_preparation": (
        "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json",
        "e6fab51219fba6d108bd6b858f2467a9a0e54c5fc22890f8abd7c318742f8e54",
    ),
    "recovery_audit": (
        "outputs/current_controller_recovery_repair_20261116/audit.json",
        "a10dc09801d90efd92933e536d04a1e71ea1da75b724105dc384205c4d61c7f6",
    ),
    "h40_audit": (
        "outputs/recovery_h40_contract_20261116/result.json",
        "0753c4c30f432e6b09e103152238cb4f6f47dffa30ce76e60636e59f2329dcec",
    ),
    "event_image_audit": (
        "outputs/recovery_event_images_20261116/audit.json",
        "a9e7c77b5ddb8ebce4b86cae9d6ea9f153dfcc8e69ad1ecbab9a6600da461b8b",
    ),
    "tool_contract": (
        "outputs/clean_training_development_dependency_repair_20261024/inputs/eef_tool_contract.json",
        "6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.wam-recovery-state-coverage.v1",
        output="outputs/wam_recovery_state_coverage_20261116",
        development_case_indices=list(CASE_INDICES),
        expected_requests_by_case=list(REQUESTS),
        baseline_anchor_count=150,
        recovery_anchor_count=39,
        request_count=213,
        distance_bins_m=[0.01, 0.02, 0.05, 0.1],
        seed=42,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
    )
    prefix.require(
        isinstance(config, dict)
        and set(config) == set(fixed) | set(REFERENCES)
        and all(type(config[key]) is type(value) and config[key] == value for key, value in fixed.items()),
        "The exact registered request, anchor roster and scope are required",
    )
    for key, (path, sha) in REFERENCES.items():
        prefix.require(config[key] == dict(path=path, sha256=sha), "The registered source identity must match")


def load_trajectory(evidence, ref, cache):
    path = evidence.path(ref)
    if str(path) not in cache:
        with np.load(path, allow_pickle=False) as archive:
            cache[str(path)] = {
                key: np.asarray(archive[key]).copy()
                for key in ("source_pose", "opening", "sim_states", "sim_time", "state128", "state_mask")
            }
    return cache[str(path)]


def measured_anchor(
    evidence, cache, *, anchor_id, source_kind, source_identity, step, receipt, image_ref, trajectory_ref, tool
):
    """Read one current training endpoint, binding its recorded image and state."""
    data = load_trajectory(evidence, trajectory_ref, cache)
    evidence.path(image_ref)
    prefix.require(type(step) is int and 0 <= step < len(data["sim_states"]), "The actual anchor must exist")
    state = np.ascontiguousarray(data["sim_states"][step])
    prefix.require(
        receipt["endpoint_step"] == step
        and receipt["sim_state_sha256"] == hashlib.sha256(state.tobytes()).hexdigest()
        and receipt["sim_time"] == float(data["sim_time"][step])
        and np.array_equal(data["state_mask"][step], np.array([True] * 10 + [False] * 118)),
        "The actual current state, time and active mask must match the recorded anchor",
    )
    tcp = validate_transform(data["source_pose"][step] @ tool)
    opening = float(data["opening"][step])
    measured = data["state128"][step]
    np.testing.assert_allclose(measured[:3], tcp[:3, 3], atol=1e-10, rtol=0)
    np.testing.assert_allclose(measured[3:6], tcp[:3, 0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(measured[6:9], tcp[:3, 1], atol=1e-10, rtol=0)
    prefix.require(
        measured[9] == opening and not measured[10:].any(), "The actual opening and inactive slots must match"
    )
    return dict(
        anchor_id=anchor_id,
        source_kind=source_kind,
        source_identity=source_identity,
        task_index=source_identity["task_index"],
        anchor_step=step,
        tcp_pose=tcp.tolist(),
        opening=opening,
        image_ref=image_ref,
        trajectory_ref=trajectory_ref,
        current_endpoint=receipt,
    )


def training_anchors(evidence, prepared, h40, recovery, event_images, tool):
    """Rebuild 150 baseline and 39 recovery positions from actual endpoints."""
    prefix.require(
        len(prepared["training_anchors"]) == 150
        and h40["complete"] is h40["passed"] is event_images["complete"] is event_images["passed"] is True
        and h40["window_count"] == 27
        and event_images["images"] == 12
        and recovery["complete"] is recovery["passed"] is True
        and len(recovery["cases"]) == 9,
        "The complete original and recovery image-anchor rosters are required",
    )
    cache, cases, baseline = {}, {}, []
    for row in prepared["training_anchors"]:
        ref = row["case_ref"]
        if ref["path"] not in cases:
            cases[ref["path"]] = evidence.json(ref)
        native = cases[ref["path"]]
        source, frame = row["source_identity"], row["anchor_frame"]
        prefix.require(
            native["complete"] is native["passed"] is native["native_quality_passed"] is True
            and all(native["record"][key] == source[key] for key in ("task_index", "demo", "initial_state_sha256"))
            and native["arrays"]["sha256"] == row["trajectory_ref"]["sha256"]
            and np.allclose(row["tool_source_from_tcp"], tool, atol=1e-12, rtol=0),
            "The native training case and fixed TCP must match",
        )
        anchor = next(item for item in native["anchors"] if item["frame_index"] == frame)
        prefix.require(
            anchor["current_endpoint"] == row["current_endpoint"]
            and anchor["canvas_png"] == row["image_ref"]
            and anchor["render_before"] == anchor["render_after"]
            and anchor["future_rgb_frames"] == 0,
            "The original current-only image identity must match",
        )
        baseline.append(
            measured_anchor(
                evidence,
                cache,
                anchor_id=row["name"],
                source_kind="baseline",
                source_identity=source,
                step=frame,
                receipt=row["current_endpoint"],
                image_ref=row["image_ref"],
                trajectory_ref=row["trajectory_ref"],
                tool=tool,
            )
        )
    prefix.require(
        len(cases) == 30
        and len({(x["task_index"], x["source_identity"]["demo"]) for x in baseline}) == 30
        and all(sum(x["task_index"] == task for x in baseline) == 50 for task in range(3)),
        "Thirty training demos and 50 actual baseline anchors per task are required",
    )
    added, by_case = [], {}
    for row in h40["windows"]:
        index = row["case_index"]
        original = recovery["cases"][index]
        prefix.require(
            original["case"]["index"] == index and original["qualified"] is original["replayed"] is True,
            "Each recovery anchor must come from its qualified replayed case",
        )
        if index not in by_case:
            producer = evidence.json(original["producer_ref"])
            evidence.path(original["checker_ref"])
            by_case[index] = (producer, row["source_identity"])
        producer, source = by_case[index]
        prefix.require(
            row["source_identity"] == source
            and row["trajectory_ref"] == producer["arrays_ref"]
            and row["current_endpoint"] == producer["endpoint_receipts"][row["anchor_step"]],
            "The recovery window must bind its own original physical endpoint",
        )
        added.append(
            measured_anchor(
                evidence,
                cache,
                anchor_id=f"recovery_case{index:02d}_step{row['anchor_step']}",
                source_kind="recovery",
                source_identity=source,
                step=row["anchor_step"],
                receipt=row["current_endpoint"],
                image_ref=row["image_ref"],
                trajectory_ref=row["trajectory_ref"],
                tool=tool,
            )
        )
    for case in event_images["cases"]:
        index = case["case_index"]
        evidence.path(case["branch_ref"])
        producer, source = by_case[index]
        for image in case["images"]:
            frame = image["anchor_step"]
            prefix.require(
                image["endpoint"] == producer["endpoint_receipts"][frame],
                "The event image must match the original actual endpoint",
            )
            added.append(
                measured_anchor(
                    evidence,
                    cache,
                    anchor_id=f"recovery_case{index:02d}_step{frame}",
                    source_kind="recovery",
                    source_identity=source,
                    step=frame,
                    receipt=image["endpoint"],
                    image_ref=image["image_ref"],
                    trajectory_ref=producer["arrays_ref"],
                    tool=tool,
                )
            )
    prefix.require(
        len(added) == 39 and len(by_case) == 9 and len({item["anchor_id"] for item in baseline + added}) == 189,
        "The 39 distinct recovery image anchors must be present",
    )
    return baseline, added


def development_requests(evidence, audit, tool_ref, tool):
    """Recover the actual current image/state at each of 213 accepted requests."""
    prefix.require(
        audit["complete"] is True
        and audit["producer_periods"] == 1025
        and audit["producer_requests"] == 213
        and audit["new_model_requests"] == audit["new_physics_steps"] == audit["fresh_cases"] == 0
        and [row["case_index"] for row in audit["cases"]] == list(CASE_INDICES),
        "The independently replayed six-case development audit is required",
    )
    queries = []
    for row, expected_count in zip(audit["cases"], REQUESTS, strict=True):
        index = row["case_index"]
        producer = evidence.json(row["producer_ref"])
        evidence.path(row["checker_ref"])
        case, runtime = producer["case"], producer["runtime"]
        tool_inputs = feedback.load_inputs(index, verify_assets=False)["cases"][0]
        prefix.require(
            case == tool_inputs["case"]
            and case["case_id"] == row["case_id"]
            and np.allclose(tool_inputs["tool"], tool, atol=1e-12, rtol=0)
            and tool_inputs["original_source_sha256"].get(tool_ref["path"]) == tool_ref["sha256"]
            and all(value == 0 for value in row["independent_max_absolute_difference"].values())
            and runtime["control_steps_completed"] == row["counts"]["control_steps"],
            "The actual case, fixed TCP and independent mechanical replay must match",
        )
        evidence.path(tool_inputs["original_receipt_ref"])
        captures, exchanges = runtime["adaptive_captures"], runtime["adaptive_model_exchanges"]
        periods = list(runtime["continuation"]["first_window"]["periods"])
        for segment in runtime["continuation"]["adaptive"]["segments"]:
            periods.extend(segment["continuation"]["periods"])
        prefix.require(
            len(captures) == len(exchanges) == expected_count == row["model_exchange"]["requests"]
            and len(periods) == row["counts"]["control_steps"]
            and [period["step"] for period in periods] == list(range(len(periods))),
            "The full accepted request and actual endpoint denominators must match",
        )
        for ordinal, (capture, exchange) in enumerate(zip(captures, exchanges, strict=True)):
            request = evidence.json(exchange["request_ref"])
            frame, receipt = capture["frame_index"], capture["current_endpoint"]
            prefix.require(
                type(frame) is int and 0 < frame <= len(periods), "The request must use an actual prior endpoint"
            )
            period = periods[frame - 1]
            actual = np.ascontiguousarray(np.asarray(period["actual_state"], dtype=np.float64))
            binding = dict(sim_time=receipt["sim_time"], sim_state_sha256=receipt["sim_state_sha256"])
            image = capture["canvas_png"]
            evidence.path(image)
            image_path = (Path(row["producer_ref"]["path"]).parent / request["image"]["path"]).resolve()
            prefix.require(
                frame == exchange["step"] == request["step"] == receipt["endpoint_step"]
                and receipt["sim_state_sha256"] == period["endpoint_receipt"]["sim_state_sha256"]
                and receipt["sim_state_sha256"] == hashlib.sha256(actual.tobytes()).hexdigest()
                and receipt["sim_time"] == period["endpoint_receipt"]["sim_time"] == request["sim_time_s"]
                and request["state_sha256"] == receipt["sim_state_sha256"]
                and capture["render_before"] == capture["render_after"] == binding
                and capture["future_rgb_frames"] == 0
                and request["image"]["sha256"] == image["sha256"]
                and image_path == Path(image["path"])
                and request["seed"] == case["sampling_seed"]
                and request["task"] == TASKS[case["task_index"]]
                and all(
                    exchange[key] is True
                    for key in ("request_sent", "response_received", "response_accepted", "policy_ready")
                )
                and abs(float(capture["current_opening"]) - period["completed_feedback"]["score"]["opening"]) <= 1e-10,
                "Each request must bind the actual measured endpoint, current image and seed",
            )
            tcp = validate_transform(np.asarray(capture["current_source_pose"], dtype=np.float64) @ tool)
            queries.append(
                dict(
                    case_index=index,
                    case_id=case["case_id"],
                    task_index=case["task_index"],
                    seed=case["sampling_seed"],
                    ordinal=ordinal,
                    step=frame,
                    tcp_pose=tcp.tolist(),
                    opening=float(capture["current_opening"]),
                    current_endpoint=receipt,
                    image_ref=image,
                    request_ref=exchange["request_ref"],
                    producer_ref=row["producer_ref"],
                )
            )
    prefix.require(len(queries) == 213, "All 213 original current-image requests are required")
    return queries


def audit():
    prefix.environment()
    feedback.require_no_simulator_import()
    evidence = Evidence()
    config_ref = prefix.reference(CONFIG)
    config = evidence.json(config_ref)
    validate_config(config)
    for path in (
        PLAN,
        Path(__file__).resolve(),
        ROOT / "src/s2a/research/wam/training/state_anchor_coverage.py",
        ROOT / "scripts/verification/wam/training/audit_recovery_h40_contract.py",
        ROOT / "tests/research/wam/training/test_state_anchor_coverage.py",
    ):
        evidence.path(prefix.reference(path))
    for ref in feedback.source_refs():
        evidence.path(ref)
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    development, prepared, recovery, h40, event_images, tool_contract = (evidence.json(refs[key]) for key in REFERENCES)
    tool = validate_transform(np.asarray(tool_contract["source_from_tcp"], dtype=np.float64))
    prefix.require(tool.shape == (4, 4), "One fixed source-to-TCP transform is required")
    baseline, added = training_anchors(evidence, prepared, h40, recovery, event_images, tool)
    queries = development_requests(evidence, development, refs["tool_contract"], tool)
    rows = []
    for query in queries:
        old = nearest_position_anchor(query, baseline)
        combined = nearest_position_anchor(query, baseline + added)
        prefix.require(
            combined["position_m"] <= old["position_m"] + 1e-12, "Adding anchors cannot increase nearest distance"
        )
        rows.append(
            dict(
                query=query, baseline=old, augmented=combined, improvement_m=old["position_m"] - combined["position_m"]
            )
        )
    per_case = []
    for index in CASE_INDICES:
        selected = [row for row in rows if row["query"]["case_index"] == index]
        per_case.append(
            dict(
                case_index=index,
                case_id=selected[0]["query"]["case_id"],
                baseline=summarize_distances([row["baseline"]["position_m"] for row in selected]),
                augmented=summarize_distances([row["augmented"]["position_m"] for row in selected]),
                recovery_selected=sum(row["augmented"]["source_kind"] == "recovery" for row in selected),
                strictly_closer=sum(row["improvement_m"] > 1e-12 for row in selected),
            )
        )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        input_refs=list(evidence.refs.values()),
        baseline_anchor_count=len(baseline),
        recovery_anchor_count=len(added),
        request_count=len(rows),
        baseline=summarize_distances([row["baseline"]["position_m"] for row in rows]),
        augmented=summarize_distances([row["augmented"]["position_m"] for row in rows]),
        recovery_selected=sum(row["augmented"]["source_kind"] == "recovery" for row in rows),
        strictly_closer=sum(row["improvement_m"] > 1e-12 for row in rows),
        maximum_improvement_m=max(row["improvement_m"] for row in rows),
        per_case=per_case,
        anchors=baseline + added,
        requests=rows,
        describes_position_only=True,
        support_qualification=False,
        training_authorized=False,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
    )


def main():
    if OUTPUT.exists():
        raise SystemExit("Audit output already exists; no automatic retry or overwrite")
    try:
        result = audit()
    except Exception as error:
        publish_result(
            OUTPUT, dict(complete=False, passed=False, error=dict(type=type(error).__name__, message=str(error)))
        )
        raise
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
