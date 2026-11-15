"""Compare old WAM current requests with replayed broader training anchors."""

from pathlib import Path

import numpy as np

from s2a.geometry import validate_transform
from s2a.infrastructure.publication import publish_result
from s2a.research.wam.training.broader_state_coverage import first_joint_anchor
from s2a.research.wam.training.state_anchor_coverage import nearest_position_anchor, summarize_distances
from scripts.simulation.runtime import prepare_broader_recovery_pilot as pilot
from scripts.simulation.runtime import prepare_reset_correct_reference_cohort as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_reset_start_feedback as feedback
from scripts.verification.wam.training import audit_wam_recovery_state_coverage as previous
from scripts.verification.wam.training.audit_recovery_h40_contract import Evidence

ROOT = prefix.REPO
CONFIG = ROOT / "configs/wam/training/recovery/broader_recovery_coverage_20261116.json"
PLAN = ROOT / "plans/v0.0.1/wam/training/recovery/broader_recovery_coverage_20261116.md"
OUTPUT = ROOT / "outputs/broader_recovery_coverage_20261116/result.json"
CASE_INDICES = (0, 2, 4, 6, 8, 10)
REQUESTS = (46, 5, 19, 9, 75, 59)
STEPS = (0, 16, 56)
GROUPS = ("previous", "all", "sensitivity")
REFERENCES = {
    "previous_coverage": (
        "outputs/wam_recovery_state_coverage_20261116/result.json",
        "b1fe1a46929d4733677cf79e50cdb13b88e7d713ee24bdaf9aedc30c646f3846",
    ),
    "previous_independent": (
        "outputs/wam_recovery_state_coverage_20261116/independent.json",
        "43a779cc0d68bdd76713c8c6df0eb402ebb9b473ebd399cb32072cf396b4def3",
    ),
    "broader_recovery_audit": (
        "outputs/broader_recovery_pilot_20261116/audit.json",
        "16d8c1250690b540b3d7230df329ec4b00aee23770033cfa3ded0c68fcb1847f",
    ),
    "tool_contract": (
        "outputs/clean_training_development_dependency_repair_20261024/inputs/eef_tool_contract.json",
        "6fcaed3fe499aacb15d03170d002b1c897091481a0022db5f2922c06320a3a0c",
    ),
}


def validate_config(config):
    fixed = dict(
        schema="s2a.broader-recovery-coverage.v1",
        output="outputs/broader_recovery_coverage_20261116",
        development_case_indices=list(CASE_INDICES),
        expected_requests_by_case=list(REQUESTS),
        request_count=213,
        previous_anchor_count=189,
        new_case_count=24,
        anchor_steps=list(STEPS),
        new_anchor_count=72,
        sensitivity_excluded_cases=[13],
        sensitivity_new_anchor_count=69,
        distance_bins_m=[0.01, 0.02, 0.05, 0.1],
        joint_position_limits_m=[0.05, 0.1],
        joint_rotation_limit_rad=0.5,
        joint_opening_limit=0.2,
        seed=42,
        world_model_calls=0,
        gpu_calls=0,
        physical_steps=0,
        fits=0,
        fresh_cases=0,
        training_authorized=False,
        automatic_retry=False,
        **{key: dict(path=path, sha256=sha) for key, (path, sha) in REFERENCES.items()},
    )
    prefix.require(
        pilot._same_typed(config, fixed), "The fixed coverage roster, inputs and no-training scope are required"
    )


def verify_pair(index, row, producer, checker, tool_ref):
    """Bind both physical branches, their real images, and one registered source."""
    case = row["case"]
    source, variant = pilot.SOURCES[index // len(pilot.VARIANTS)], pilot.VARIANTS[index % len(pilot.VARIANTS)]
    summary = row["summary"]
    prefix.require(
        case["index"] == index
        and case["source"] == source
        and case["variant"] == variant
        and case["original_identity"]["refs"]["tool"] == tool_ref
        and row["producer_status"] == row["checker_status"] == "closed"
        and row["qualified"] is row["replayed"] is row["evidence_complete"] is True
        and row["observed_h40_anchors"] == list(STEPS)
        and all(value == 0 for value in row["replay_max_difference"].values())
        and summary["completed_periods"] == summary["expected_periods"] == case["periods"]
        and producer["case"] == checker["case"] == case
        and producer["complete"] is producer["passed"] is checker["complete"] is checker["passed"] is True
        and checker["candidate_modules_unloaded"] is True
        and producer["physical_labels_from_reference"] is checker["physical_labels_from_reference"] is False
        and [image["frame_index"] for image in producer["images"]] == list(STEPS)
        and [image["frame_index"] for image in checker["images"]] == list(STEPS),
        "Each image source must be its fixed, qualified and replayed training case",
    )
    return source


def broader_anchors(evidence, audit, tool_ref, tool):
    """Read 72 measured current anchors from the actual producer/checker branches."""
    prefix.require(
        audit["complete"] is audit["passed"] is audit["sources_unchanged"] is True
        and audit["training_authorized"] is False
        and audit["registered_cases"] == audit["evidence_complete_cases"] == len(audit["cases"]) == 24
        and audit["zero_qualified"] == 6
        and audit["nonzero_qualified"] == 18,
        "The complete independently replayed 24-case control audit is required",
    )
    cache, anchors, non_pointwise = {}, [], []
    for index, row in enumerate(audit["cases"]):
        producer, checker = (evidence.json(row[key]) for key in ("producer_ref", "checker_ref"))
        source = verify_pair(index, row, producer, checker, tool_ref)
        if row["summary"]["pose_caps_periods"] != row["summary"]["expected_periods"]:
            non_pointwise.append(index)
        evidence.path(producer["arrays_ref"])
        evidence.path(checker["arrays_ref"])
        source_identity = dict(
            task_index=source["task_index"],
            demo=source["demo"],
            initial_state_sha256=row["case"]["initial_state_sha256"],
        )
        for step, image, replay_image in zip(STEPS, producer["images"], checker["images"], strict=True):
            receipt = producer["endpoint_receipts"][step]
            prefix.require(
                image["current_endpoint"] == replay_image["current_endpoint"] == receipt
                and receipt == checker["endpoint_receipts"][step]
                and image["canvas_rgb_sha256"] == replay_image["canvas_rgb_sha256"]
                and image["render_before"] == image["render_after"]
                and replay_image["render_before"] == replay_image["render_after"]
                and image["future_rgb_frames"] == replay_image["future_rgb_frames"] == 0,
                "Each image must bind the same current physical endpoint in independent replay",
            )
            for ref in (image["raw_cameras"], replay_image["raw_cameras"], replay_image["canvas_png"]):
                evidence.path(ref)
            anchor = previous.measured_anchor(
                evidence,
                cache,
                anchor_id=f"broader_case{index:02d}_step{step:04d}",
                source_kind="broader_recovery",
                source_identity=source_identity,
                step=step,
                receipt=receipt,
                image_ref=image["canvas_png"],
                trajectory_ref=producer["arrays_ref"],
                tool=tool,
            )
            np.testing.assert_allclose(
                image["current_source_pose"],
                cache[str(evidence.path(producer["arrays_ref"]))]["source_pose"][step],
                atol=1e-12,
                rtol=0,
            )
            prefix.require(
                float(image["current_opening"]) == anchor["opening"], "Image opening must match actual state"
            )
            anchor.update(case_index=index, variant=row["case"]["variant"], pointwise_case_eligible=index != 13)
            anchors.append(anchor)
    prefix.require(non_pointwise == [13] and len(anchors) == 72, "The fixed pointwise sensitivity roster must match")
    return anchors


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    prefix.require(data.ndim == 1 and len(data) > 0 and np.isfinite(data).all(), "Finite descriptive values required")
    return dict(median=float(np.percentile(data, 50)), p90=float(np.percentile(data, 90)), maximum=float(np.max(data)))


def summarize(rows, group):
    selected = [row[group]["nearest"] for row in rows]
    return dict(
        position=summarize_distances([item["position_m"] for item in selected]),
        rotation_rad=distribution([item["rotation_rad"] for item in selected]),
        opening_abs=distribution([item["opening_abs"] for item in selected]),
        joint_counts={
            str(limit): sum(row[group]["joint"][str(limit)] is not None for row in rows) for limit in (0.05, 0.1)
        },
        broader_selected=sum(item["source_kind"] == "broader_recovery" for item in selected),
        strictly_closer=sum(
            row[group]["nearest"]["position_m"] < row["previous"]["nearest"]["position_m"] - 1e-12 for row in rows
        ),
    )


def compare_query(query, anchors, config):
    nearest = nearest_position_anchor(query, anchors)
    joint = {
        str(limit): first_joint_anchor(
            query,
            anchors,
            max_position_m=limit,
            max_rotation_rad=config["joint_rotation_limit_rad"],
            max_opening_abs=config["joint_opening_limit"],
        )
        for limit in config["joint_position_limits_m"]
    }
    return dict(nearest=nearest, joint=joint)


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
        ROOT / "src/s2a/research/wam/training/broader_state_coverage.py",
        ROOT / "tests/research/wam/training/test_broader_state_coverage.py",
        ROOT / "tests/research/wam/training/test_broader_recovery_coverage.py",
    ):
        evidence.path(prefix.reference(path))
    refs = {key: preparation.configured_ref(config[key]) for key in REFERENCES}
    prior, independent, broad, contract = (evidence.json(refs[key]) for key in REFERENCES)
    prefix.require(
        prior["complete"] is prior["passed"] is independent["complete"] is independent["passed"] is True
        and prior["training_authorized"]
        is prior["support_qualification"]
        is independent["training_authorized"]
        is False
        and independent["primary_ref"] == refs["previous_coverage"]
        and len(prior["anchors"]) == config["previous_anchor_count"]
        and len(prior["requests"]) == config["request_count"]
        and independent["anchors_checked"] == 189
        and independent["requests_checked"] == 213,
        "The exact previously independently audited requests and anchors are required",
    )
    for ref in prior["input_refs"] + independent["input_refs"]:
        evidence.path(ref)
    tool = validate_transform(np.asarray(contract["source_from_tcp"], dtype=np.float64))
    prefix.require(tool.shape == (4, 4), "One fixed source-to-TCP transform is required")
    old = prior["anchors"]
    added = broader_anchors(evidence, broad, refs["tool_contract"], tool)
    strict = [anchor for anchor in added if anchor["pointwise_case_eligible"]]
    prefix.require(
        len(added) == config["new_anchor_count"]
        and len(strict) == config["sensitivity_new_anchor_count"]
        and len({row["anchor_id"] for row in old + added}) == 261,
        "The complete old, new and sensitivity anchor identities are required",
    )
    groups = dict(previous=old, all=old + added, sensitivity=old + strict)
    rows = []
    for old_row in prior["requests"]:
        query = old_row["query"]
        compared = {name: compare_query(query, group, config) for name, group in groups.items()}
        nearest, expected = compared["previous"]["nearest"], old_row["augmented"]
        prefix.require(
            nearest["anchor_id"] == expected["anchor_id"]
            and nearest["source_kind"] == expected["source_kind"]
            and all(
                abs(nearest[key] - expected[key]) <= 1e-10 for key in ("position_m", "rotation_rad", "opening_abs")
            ),
            "The previous nearest result must reproduce the frozen baseline",
        )
        for name in ("all", "sensitivity"):
            prefix.require(
                compared[name]["nearest"]["position_m"] <= nearest["position_m"] + 1e-12
                and all(
                    compared["previous"]["joint"][key] is None or compared[name]["joint"][key] is not None
                    for key in compared["previous"]["joint"]
                ),
                "Adding anchors cannot reduce nearest or joint descriptive coverage",
            )
        rows.append(dict(query=query, **compared))
    per_case = []
    for index, count in zip(CASE_INDICES, REQUESTS, strict=True):
        subset = [row for row in rows if row["query"]["case_index"] == index]
        prefix.require(len(subset) == count, "The six development request counts must remain unchanged")
        per_case.append(
            dict(
                case_index=index,
                case_id=subset[0]["query"]["case_id"],
                groups={name: summarize(subset, name) for name in GROUPS},
            )
        )
    evidence.verify()
    feedback.require_no_simulator_import()
    return dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        input_refs=list(evidence.refs.values()),
        request_count=len(rows),
        previous_anchor_count=len(old),
        new_anchor_count=len(added),
        sensitivity_new_anchor_count=len(strict),
        new_unique_state_hashes=len({row["current_endpoint"]["sim_state_sha256"] for row in added}),
        new_unique_image_hashes=len({row["image_ref"]["sha256"] for row in added}),
        groups={name: summarize(rows, name) for name in GROUPS},
        per_case=per_case,
        anchors=old + added,
        requests=rows,
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
