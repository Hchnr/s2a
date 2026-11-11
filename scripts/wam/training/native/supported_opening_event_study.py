"""Re-score frozen out-of-fold events with support from generated reference only."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import expit

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.gripper.causal_opening_event_metrics import select_threshold
from s2a.research.gripper.supported_opening_event import future_closure_support, supported_probability
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.training.verify_supported_opening_event import verify_supported_selection
from scripts.verification.wam.training.verify_wam_clean_training_capture import require
from scripts.wam.training.native import causal_opening_event_study as original

REPO = prefix.REPO
CONFIG = REPO / "configs/wam/training/supported_opening_event_20261111.json"
PLAN = REPO / "plans/v0.0.1/wam/training/supported_opening_event_20261111.md"
ROOT = REPO / "outputs/supported_opening_event_20261111"
DRY = REPO / "outputs/supported_opening_event_dry_20261111/result.json"
QA = REPO / "outputs/qa/supported_opening_event_study_20261111/result.json"


def source_refs():
    paths = (
        CONFIG,
        PLAN,
        Path(__file__),
        REPO / "src/s2a/research/gripper/supported_opening_event.py",
        REPO / "scripts/verification/wam/training/verify_supported_opening_event.py",
        REPO / "tests/research/gripper/test_supported_opening_event.py",
        REPO / "tests/research/gripper/test_supported_opening_event_study.py",
    )
    return [prefix.reference(path) for path in paths] + original.source_refs()


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    expected = dict(
        schema="s2a.supported-opening-event.v1",
        source_cv="outputs/causal_opening_event_20261111/cv/result.json",
        source_cv_sha256="e36a7a6c6e49b9cc00ebb6d6445139b2cac2206dc8e7475ccc3d72733a472101",
        source_manifest="outputs/causal_opening_event_20261111/training_manifest.json",
        source_manifest_sha256="3edfbae478a9fb9e2b41a251f35fe62ea38415a51b5ed871fd49e6236f1a2ceb",
        output="outputs/supported_opening_event_20261111",
        support_future_steps=16,
        seed=42,
        training_demos=30,
        training_requests=300,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99],
        new_fold_fits=0,
        max_full_fits=1,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    require(config == expected, "registered supported-event study differs")
    _old_ref, old = original.read_config()
    require(config["thresholds"] == old["thresholds"], "original event thresholds must remain fixed")
    return ref, config, old


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["all_tests_passed"] is True
        and qa["skipped_count"] == 0,
        "complete zero-skip CPU regression required",
    )
    before = snapshot(REPO)["sources"]
    require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU regression source identity")
    return before


def frozen_inputs(config):
    cv_ref = original.configured_ref(config, "source_cv")
    manifest_ref = original.configured_ref(config, "source_manifest")
    cv, manifest = prefix.read(cv_ref), prefix.read(manifest_ref)
    require(
        cv["complete"] is cv["passed"] is True
        and cv["training_gate_passed"] is cv["final_fit_attempted"] is False
        and cv["candidate_ref"] is None
        and cv["validation_future_reads"] == 0
        and len(cv["fold_refs"]) == 5
        and manifest["complete"] is manifest["passed"] is True
        and manifest["training_requests"] == 300
        and manifest["training_demos"] == 30
        and manifest["validation_future_reads"] == 0
        and manifest["source_refs"] == original.source_refs(),
        "frozen failed training-only study and source identity required",
    )
    prefix.checked(manifest["dataset_ref"])
    for ref in cv["fold_refs"]:
        prefix.checked(ref)
    return cv_ref, manifest_ref, cv, manifest


def reconstruct_oof(data, manifest_ref, cv, read_fold):
    """Rebuild every query's probability from its own held-out demo fold."""
    features, assignments = np.asarray(data["features"]), np.asarray(data["fold"])
    require(features.shape == (300, 40, 44) and assignments.shape == (300,), "fixed training array shapes")
    recorded = np.asarray(cv["oof_probability"], dtype=float)
    require(recorded.shape == (300, 40) and np.isfinite(recorded).all(), "fixed OOF probability shape")
    probability = np.empty((300, 40))
    covered = np.zeros(300, dtype=bool)
    maximum = 0.0
    for number, ref in enumerate(cv["fold_refs"]):
        fold = read_fold(ref)
        indices = np.asarray(fold["held_indices"], dtype=int)
        held, coefficients = np.asarray(fold["held_probability"]), np.asarray(fold["coefficients"])
        require(
            fold["complete"] is fold["passed"] is True
            and fold["fold"] == number
            and fold["training_manifest_ref"] == manifest_ref
            and fold["train_queries"] == 240
            and fold["held_queries"] == 60
            and fold["validation_future_reads"] == 0
            and np.array_equal(indices, np.flatnonzero(assignments == number))
            and len(indices) == 60
            and not covered[indices].any()
            and held.shape == (60, 40)
            and coefficients.shape == (44,)
            and np.isfinite(held).all()
            and np.isfinite(coefficients).all()
            and np.all((held >= 0) & (held <= 1)),
            "one complete held-demo prediction per query required",
        )
        maximum = max(maximum, float(np.max(abs(held - expit(features[indices] @ coefficients)))))
        probability[indices] = held
        covered[indices] = True
    require(
        covered.all() and np.array_equal(probability, recorded) and maximum <= 1e-12,
        "independent held-fold probabilities and recorded OOF assembly must agree",
    )
    return probability, maximum


def dry_run():
    before = verify_qa_sources()
    config_ref, config, _old = read_config()
    cv_ref, manifest_ref, cv, manifest = frozen_inputs(config)
    sources = source_refs()
    require(snapshot(REPO)["sources"] == before, "source bytes changed during dry run")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                cv_ref=cv_ref,
                manifest_ref=manifest_ref,
                dataset_ref=manifest["dataset_ref"],
                fold_refs=cv["fold_refs"],
                source_refs=sources,
                training_future_reads=0,
                validation_future_reads=0,
                new_fold_fits=0,
                full_fits=0,
                world_model_calls=0,
                physics_steps=0,
                fresh_cases=0,
            ),
        )
    )


def analyze():
    before = verify_qa_sources()
    config_ref, config, old = read_config()
    cv_ref, manifest_ref, cv, manifest = frozen_inputs(config)
    dry = prefix.read(prefix.reference(DRY))
    require(
        dry["complete"] is dry["passed"] is True
        and dry["config_ref"] == config_ref
        and dry["cv_ref"] == cv_ref
        and dry["manifest_ref"] == manifest_ref
        and dry["source_refs"] == source_refs(),
        "matching zero-model dry registration required",
    )
    require(not ROOT.exists(), "new non-overwriting supported-event output required")
    ROOT.mkdir(parents=True)
    result = dict(
        complete=False,
        passed=False,
        training_gate_passed=False,
        config_ref=config_ref,
        cv_ref=cv_ref,
        manifest_ref=manifest_ref,
        dry_ref=prefix.reference(DRY),
        qa_ref=prefix.reference(QA),
        new_fold_fits=0,
        full_fits=0,
        validation_future_reads=0,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        actual_old, actual_manifest_ref, actual_manifest, data = original.training_data()
        require(
            actual_old == old and actual_manifest_ref == manifest_ref and actual_manifest == manifest,
            "original training data must match the frozen manifest",
        )
        probability, maximum = reconstruct_oof(data, manifest_ref, cv, prefix.read)
        mask = future_closure_support(data["opening"], steps=config["support_future_steps"])
        supported = supported_probability(data["opening"], probability, steps=config["support_future_steps"])
        selection = select_threshold(data["opening"], data["truth"], supported, data["task"], old)
        independent = verify_supported_selection(
            data["opening"], data["truth"], probability, data["task"], mask, old, selection
        )
        prefix.checked(manifest["dataset_ref"])
        prefix.checked(cv_ref)
        prefix.checked(manifest_ref)
        for ref in cv["fold_refs"]:
            prefix.checked(ref)
        require(
            source_refs() == dry["source_refs"] and snapshot(REPO)["sources"] == before,
            "source or consumed bytes changed during selection",
        )
        result.update(
            complete=True,
            passed=True,
            training_gate_passed=selection["selected_threshold"] is not None,
            source_refs=source_refs(),
            support_mask=mask,
            support_counts={str(h): int(mask[:, :h].sum()) for h in (8, 16, 40)},
            max_oof_probability_difference=maximum,
            selection=selection,
            independent=independent,
            evidence_boundary="Data-informed training-only exploration on cached references, not physical tracking or grasp.",
        )
    finally:
        print(publish(ROOT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        analyze()


if __name__ == "__main__":
    main()
