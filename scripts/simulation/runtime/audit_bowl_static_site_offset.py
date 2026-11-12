"""Compare two fixed pad configurations against saved bowl collision boxes."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result
from s2a.research.native_runtime.box_separation import box_separation
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/gripper/bowl_static_site_offset_20261112.json"
PLAN = ROOT / "plans/v0.0.1/tracking/gripper/bowl_static_site_offset_20261112.md"
OUTPUT = ROOT / "outputs/bowl_static_site_offset_20261112/result.json"
QA = ROOT / "outputs/qa/bowl_static_site_offset_20261112/result.json"
SOURCE = {
    "path": str(ROOT / "outputs/bowl_pad_alignment_recovery_audit_20261112/result.json"),
    "sha256": "6fb6ad52ce9a81f0e241215578f151b368235813db480e4a892819c8a9175447",
}
FIXED = dict(
    schema="s2a.bowl-static-site-offset.v1",
    output=str(OUTPUT.relative_to(ROOT)),
    seed=42,
    branches=["fixed", "lead"],
    endpoint_steps=[40, 43, 46, 48],
    site_translation_m=[-0.02, 0.0, 0.0],
    classification_tolerance_m=1e-8,
    source_audit_sha256=SOURCE["sha256"],
    new_models=0,
    new_forwards=0,
    new_physics_steps=0,
    new_contexts=0,
    new_model_requests=0,
    fresh_cases=0,
    automatic_retry=False,
)


def classify(gap, tolerance):
    if not np.isfinite(gap) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A finite gap and positive tolerance are required")
    return "separated" if gap > tolerance else "overlap" if gap < -tolerance else "near"


def analyze_point(point, translation_site, tolerance):
    """Move only the two pad boxes rigidly for the registered static hypothesis."""
    a = point["pad_alignment"]
    boxes = a["box_projections"]
    prefix.require(len(boxes) == 42, "Exactly two pad boxes and 40 target boxes are required")
    rotation = np.asarray(a["site_rotation_world"], dtype=float)
    translation = rotation @ np.asarray(translation_site, dtype=float)
    output = {}
    for scenario, delta in (("recorded", np.zeros(3)), ("offset", translation)):
        pads = {}
        for pad in boxes[:2]:
            pairs = []
            for target in boxes[2:]:
                result = box_separation(
                    np.asarray(pad["center_world_m"]) + delta,
                    pad["rotation_world"],
                    pad["half_size_m"],
                    target["center_world_m"],
                    target["rotation_world"],
                    target["half_size_m"],
                )
                pairs.append(
                    dict(
                        target_geom=target["name"],
                        classification=classify(result["maximum_gap_m"], tolerance),
                        **result,
                    )
                )
            best = min(pairs, key=lambda row: row["maximum_gap_m"])
            pads[pad["name"]] = dict(
                pairs=pairs,
                strictly_overlapping_targets=[
                    row["target_geom"] for row in pairs if row["classification"] == "overlap"
                ],
                near_targets=[row["target_geom"] for row in pairs if row["classification"] == "near"],
                minimum_maximum_gap_m=best["maximum_gap_m"],
                best_target=best["target_geom"],
            )
        output[scenario] = dict(pad_translation_world_m=delta, pads=pads)
    output["original_contacts"] = point["contacts"]
    return output


def main():
    prefix.environment()
    config = prefix.read(prefix.reference(CONFIG))
    prefix.require(config == FIXED, "The exact pre-registered read-only scope and budget are required")
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 32
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"],
        "A complete source-matched 32-worker regression is required",
    )
    before = snapshot(ROOT)["sources"]
    prefix.require(before == prefix.read(qa["source_before_ref"])["sources"], "The QA source snapshot differs")
    audit = prefix.read(SOURCE)
    prefix.require(
        audit["complete"] is audit["passed"] is True
        and audit["control_qualified"] is False
        and audit["audited_points"] == 97
        and audit["audited_box_projections"] == 4074
        and audit["audit_models"] == audit["audit_forwards"] == audit["audit_physics_steps"] == 0,
        "The complete independent saved-geometry audit is required",
    )
    worker = prefix.read(audit["worker_ref"])
    rows = worker["geometry"]["points"]
    results = {
        branch: {
            str(step): analyze_point(
                rows[branch][step - 1], config["site_translation_m"], config["classification_tolerance_m"]
            )
            for step in config["endpoint_steps"]
        }
        for branch in config["branches"]
    }
    prefix.require(snapshot(ROOT)["sources"] == before, "Sources changed during the static analysis")
    prefix.checked(SOURCE)
    result = dict(
        complete=True,
        passed=True,
        control_qualified=False,
        source_audit_ref=SOURCE,
        config_ref=prefix.reference(CONFIG),
        plan_ref=prefix.reference(PLAN),
        qa_ref=prefix.reference(QA),
        source_refs=[
            prefix.reference(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/s2a/research/native_runtime/box_separation.py",
                ROOT / "tests/research/native_runtime/test_box_separation.py",
                ROOT / "tests/research/native_runtime/test_audit_bowl_static_site_offset.py",
            )
        ],
        results=results,
        new_models=0,
        new_forwards=0,
        new_physics_steps=0,
        new_contexts=0,
        new_model_requests=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    print(publish_result(OUTPUT, result), flush=True)


if __name__ == "__main__":
    main()
