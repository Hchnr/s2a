"""Fit once and score one frozen supported-event candidate on saved responses."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from s2a.geometry import pose9_to_transform
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.action_reference import decode_action_targets
from s2a.research.gripper.causal_opening_event import apply_opening_event, event_features, predict_probability
from s2a.research.gripper.causal_opening_event_metrics import evaluate, summarize
from s2a.research.gripper.supported_opening_event import future_closure_support, supported_probability
from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.training.verify_causal_opening_event import independent_features
from scripts.verification.wam.training.verify_supported_opening_event import verify_supported_selection
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require
from scripts.wam.training.native import causal_opening_event_study as original
from scripts.wam.training.native import supported_opening_event_study as study

REPO = prefix.REPO
ROOT = REPO / "outputs/supported_opening_event_final_20261111"
QA = REPO / "outputs/qa/supported_opening_event_validation_20261111/result.json"
SELECTION = REPO / "outputs/supported_opening_event_20261111/result.json"
SELECTION_SHA = "34d3c63489776d4ef509d1749216a21050ea3443bc56db92bec39d85243c0034"


def source_refs():
    return [
        prefix.reference(Path(__file__)),
        prefix.reference(REPO / "tests/research/gripper/test_supported_opening_event_validation.py"),
        *study.source_refs(),
    ]


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["all_tests_passed"] is True
        and qa["skipped_count"] == 0,
        "complete zero-skip CPU validation regression required",
    )
    before = snapshot(REPO)["sources"]
    require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU regression source identity")
    return before


def admit_selection(result, config_ref, cv_ref, manifest_ref, sources):
    require(
        result["complete"] is result["passed"] is result["training_gate_passed"] is True
        and result["config_ref"] == config_ref
        and result["cv_ref"] == cv_ref
        and result["manifest_ref"] == manifest_ref
        and result["source_refs"] == sources
        and result["selection"]["selected_threshold"] == 0.7
        and result["independent"]["selection"]["selected_threshold"] == 0.7
        and result["new_fold_fits"] == result["full_fits"] == result["validation_future_reads"] == 0
        and result["world_model_calls"] == result["physics_steps"] == result["fresh_cases"] == 0,
        "fixed training-only supported-event selection required",
    )
    return 0.7


def frozen_selection():
    config_ref, config, old = study.read_config()
    cv_ref, manifest_ref, _cv, manifest = study.frozen_inputs(config)
    selection_ref = {"path": str(SELECTION), "sha256": SELECTION_SHA}
    result = prefix.read(selection_ref)
    threshold = admit_selection(result, config_ref, cv_ref, manifest_ref, study.source_refs())
    return config, old, cv_ref, manifest_ref, manifest, selection_ref, threshold


def apply_supported(raw, probability, threshold, config):
    masked = supported_probability(raw, probability, steps=16, closed_below=config["closed_below"])
    return apply_opening_event(
        raw,
        masked,
        threshold,
        closed_below=config["closed_below"],
        commanded_opening=config["commanded_opening"],
    )


def fit():
    before = verify_qa_sources()
    config, old, cv_ref, manifest_ref, manifest, selection_ref, threshold = frozen_selection()
    require(not ROOT.exists(), "one new fit output and no automatic retry required")
    ROOT.mkdir(parents=True)
    registration_ref = publish(
        ROOT / "fit_registration.json",
        dict(
            selection_ref=selection_ref,
            manifest_ref=manifest_ref,
            cv_ref=cv_ref,
            qa_ref=prefix.reference(QA),
            source_refs=source_refs(),
            threshold=threshold,
            full_fit_budget=1,
            validation_future_reads=0,
        ),
    )
    result = dict(
        complete=False,
        passed=False,
        registration_ref=registration_ref,
        candidate_ref=None,
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
            "fixed training dataset required",
        )
        result["full_fits"] = 1
        coefficients, info, independent = original.fit_one(data, old, np.ones(300, dtype=bool))
        prefix.checked(manifest["dataset_ref"])
        prefix.checked(selection_ref)
        require(
            source_refs() == prefix.read(registration_ref)["source_refs"] and snapshot(REPO)["sources"] == before,
            "fit source or data changed",
        )
        candidate_ref = publish(
            ROOT / "candidate.json",
            dict(
                schema="s2a.supported-opening-event-candidate.v1",
                complete=True,
                registration_ref=registration_ref,
                selection_ref=selection_ref,
                manifest_ref=manifest_ref,
                threshold=threshold,
                support_future_steps=config["support_future_steps"],
                closed_below=old["closed_below"],
                commanded_opening=old["commanded_opening"],
                coefficients=coefficients,
                optimizer=info,
                independent_fit=independent,
                training_demos=30,
                training_queries=300,
                validation_future_reads=0,
                raw_position_rotation_unchanged=True,
            ),
        )
        result.update(complete=True, passed=True, candidate_ref=candidate_ref)
    finally:
        print(publish(ROOT / "fit.json", result), flush=True)


def admit_candidate(candidate, fit_result, selection_ref, manifest_ref, threshold):
    require(
        fit_result["complete"] is fit_result["passed"] is True
        and fit_result["full_fits"] == 1
        and fit_result["validation_future_reads"] == 0
        and fit_result["candidate_ref"] is not None
        and candidate["schema"] == "s2a.supported-opening-event-candidate.v1"
        and candidate["complete"] is True
        and candidate["registration_ref"] == fit_result["registration_ref"]
        and candidate["selection_ref"] == selection_ref
        and candidate["manifest_ref"] == manifest_ref
        and candidate["threshold"] == threshold == 0.7
        and candidate["support_future_steps"] == 16
        and candidate["closed_below"] == 0.35
        and candidate["commanded_opening"] == 0.1
        and candidate["training_demos"] == 30
        and candidate["training_queries"] == 300
        and candidate["validation_future_reads"] == 0,
        "one frozen supported-event candidate required",
    )


def validate():
    before = verify_qa_sources()
    _config, old, _cv_ref, manifest_ref, manifest, selection_ref, threshold = frozen_selection()
    fit_ref = prefix.reference(ROOT / "fit.json")
    fit_result = prefix.read(fit_ref)
    candidate = prefix.read(fit_result["candidate_ref"])
    admit_candidate(candidate, fit_result, selection_ref, manifest_ref, threshold)
    registration = prefix.read(fit_result["registration_ref"])
    require(registration["source_refs"] == source_refs(), "frozen fit and validation sources required")
    start_ref = publish(
        ROOT / "validation_started.json",
        dict(fit_ref=fit_ref, candidate_ref=fit_result["candidate_ref"], threshold=threshold, budget=48),
    )
    result = dict(
        complete=False,
        passed=False,
        validation_gate_passed=False,
        fit_ref=fit_ref,
        candidate_ref=fit_result["candidate_ref"],
        start_ref=start_ref,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        evidence = Evidence()
        score_ref = original.configured_ref(old, "validation_score")
        score = evidence.json(score_ref)
        numerical = evidence.json(score["numerical_ref"])
        require(
            score["schema_version"] == "s2a.aggregate-state-score.v1"
            and score["complete"]
            is score["passed"]
            is numerical["complete"]
            is numerical["passed"]
            is numerical["numerical_contract_passed"]
            is numerical["lifecycle_passed"]
            is True
            and len(numerical["generated_refs"]) == 96,
            "qualified saved validation responses required",
        )
        prepared = evidence.json(score["prepared_ref"])
        current = evidence.json(prepared["current_inputs_ref"])
        anchors = {item["name"]: item for item in current["current_inputs"]}
        original_rows = {row["name"]: row for row in score["state_rows"]}
        archives, openings, truths, probabilities, tasks, episodes, seeds, names = {}, [], [], [], [], [], [], []
        max_features = 0.0
        for item in numerical["generated_refs"]:
            if item["variant"] != "raw":
                continue
            anchor, row = anchors[item["anchor_name"]], original_rows[item["name"]]
            episode = row["episode"]
            if episode not in archives:
                with np.load(evidence.path(score["future_archive_refs"][episode]), allow_pickle=False) as z:
                    archives[episode] = {
                        key: z[key].copy() for key in ("state128", "sim_states", "sim_time", "action_valid")
                    }
            data, frame = archives[episode], anchor["anchor_frame"]
            states = data["state128"][frame : frame + 41]
            require(
                states.shape == (41, 128)
                and np.array_equal(states[0], anchor["measured_state128"])
                and hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest()
                == anchor["current_sim_state_sha256"]
                and data["sim_time"][frame] == anchor["current_time_s"]
                and data["action_valid"][frame : frame + 40].all()
                and np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, atol=1e-9, rtol=0),
                "validation current and future identity",
            )
            action = np.asarray(evidence.json(item["response_ref"])["outputs"][0]["content"]["action"])
            measured = pose9_to_transform(states[0, :9])
            targets, opening = decode_action_targets(measured, action, horizon=40)
            np.testing.assert_allclose(opening - states[1:, 9], row["errors"]["wam"]["opening"], atol=1e-12, rtol=0)
            features = event_features(measured, states[0, 9], targets, opening)
            max_features = max(
                max_features,
                float(np.max(abs(features - independent_features(measured, states[0, 9], targets, opening)))),
            )
            openings.append(opening)
            truths.append(states[1:, 9])
            probabilities.append(predict_probability(features, candidate["coefficients"]))
            tasks.append(row["task_index"])
            episodes.append(episode)
            seeds.append(row["sampling_seed"])
            names.append(item["name"])
        require(
            len(openings) == 48 and len(anchors) == 24 and len(archives) == 6 and max_features <= 1e-12,
            "all saved validation requests and independent features required",
        )
        validation_ids = {(int(name[4]), "demo_" + name.rsplit("_", 1)[1]) for name in archives}
        _cfg, data_ref, _manifest, training = original.training_data()
        training_ids = {(int(t), str(d)) for t, d in zip(training["task"], training["demo"], strict=True)}
        require(
            data_ref == manifest_ref and validation_ids == VALIDATION_DEMO_KEYS and not validation_ids & training_ids,
            "whole validation demos must be excluded from training",
        )
        raw, truth, probability, task_ids = map(np.asarray, (openings, truths, probabilities, tasks))
        support = future_closure_support(raw, steps=16, closed_below=old["closed_below"])
        masked = supported_probability(raw, probability, steps=16, closed_below=old["closed_below"])
        scored = evaluate(raw, truth, masked, task_ids, threshold, old)
        single_config = dict(old, thresholds=[threshold])
        singleton = dict(thresholds=[scored], selected_threshold=threshold if scored["eligible"] else None)
        independent = verify_supported_selection(raw, truth, probability, task_ids, support, single_config, singleton)
        corrected = apply_supported(raw, probability, threshold, old)
        by_episode = {
            episode: {
                str(h): summarize(
                    raw[np.asarray(episodes) == episode],
                    truth[np.asarray(episodes) == episode],
                    corrected[np.asarray(episodes) == episode],
                    h,
                    closed_below=old["closed_below"],
                )
                for h in (8, 16, 40)
            }
            for episode in sorted(set(episodes))
        }
        by_seed = {
            str(seed): {
                str(h): summarize(
                    raw[np.asarray(seeds) == seed],
                    truth[np.asarray(seeds) == seed],
                    corrected[np.asarray(seeds) == seed],
                    h,
                    closed_below=old["closed_below"],
                )
                for h in (8, 16, 40)
            }
            for seed in sorted(set(seeds))
        }
        evidence.finish()
        prefix.checked(manifest["dataset_ref"])
        prefix.checked(selection_ref)
        prefix.checked(fit_result["candidate_ref"])
        require(
            source_refs() == registration["source_refs"] and snapshot(REPO)["sources"] == before,
            "validation source or consumed bytes changed",
        )
        result.update(
            complete=True,
            passed=True,
            validation_gate_passed=scored["eligible"],
            state_gate_evaluated=False,
            eligible_for_development=False,
            score_ref=score_ref,
            requests=48,
            episodes=episodes,
            seeds=seeds,
            names=names,
            raw_opening=raw,
            truth_opening=truth,
            probability=probability,
            support_mask=support,
            candidate_opening=corrected,
            event_metrics=scored,
            by_episode=by_episode,
            by_seed=by_seed,
            independent=independent,
            max_independent_feature_difference=max_features,
            changed_endpoints={str(h): int(np.count_nonzero(raw[:, :h] != corrected[:, :h])) for h in (8, 16, 40)},
            max_opening_change=float(np.max(abs(raw - corrected))),
            input_sha256=evidence.inputs,
            raw_position_rotation_unchanged=True,
            evidence_boundary="Offline event-label scoring on previously viewed saved responses, not physical tracking, grasp, task success or 054 qualification.",
        )
    finally:
        print(publish(ROOT / "validation.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fit", action="store_true")
    group.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.fit:
        fit()
    else:
        validate()


if __name__ == "__main__":
    main()
