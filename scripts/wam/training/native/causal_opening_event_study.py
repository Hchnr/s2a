"""Prepare, train and validate one causal opening-event candidate on saved data."""

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

from s2a.geometry import pose9_to_transform
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.references.action_reference import decode_action_targets
from s2a.research.gripper.causal_opening_event import (
    apply_opening_event,
    event_features,
    fit_event_model,
    group_folds,
    predict_probability,
)
from s2a.research.gripper.causal_opening_event_metrics import evaluate, select_threshold, summarize
from s2a.research.wam.training.clean.clean_training_aggregation import admit_aggregation
from s2a.research.wam.training.clean.clean_training_data import VALIDATION_DEMO_KEYS
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import registered_feedback_phase
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.training.verify_causal_opening_event import (
    independent_features,
    verify_fit,
    verify_selection,
)
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require
from scripts.wam.training.native.cached_endpoint_calibration import raw_c0_action, training_archive

REPO = prefix.REPO
CONFIG = REPO / "configs/wam/training/causal_opening_event_20261111.json"
PLAN = REPO / "plans/v0.0.1/wam/training/causal_opening_event_20261111.md"
ROOT = REPO / "outputs/causal_opening_event_20261111"
DRY = REPO / "outputs/causal_opening_event_dry_20261111/result.json"
QA = REPO / "outputs/qa/causal_opening_event_study_20261111/result.json"
TEST = REPO / "tests/research/gripper/test_causal_opening_event_study.py"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            TEST,
            REPO / "src/s2a/research/gripper/causal_opening_event.py",
            REPO / "src/s2a/research/gripper/causal_opening_event_metrics.py",
            REPO / "scripts/verification/wam/training/verify_causal_opening_event.py",
            REPO / "scripts/wam/training/native/cached_endpoint_calibration.py",
            REPO / "tests/research/gripper/test_causal_opening_event.py",
            REPO / "tests/research/gripper/test_causal_opening_event_metrics.py",
            REPO / "tests/research/gripper/test_causal_opening_event_independent.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    require(
        config["schema"] == "s2a.causal-opening-event.v1"
        and config["output"] == "outputs/causal_opening_event_20261111"
        and config["seed"] == 42
        and config["training_demos"] == 30
        and config["training_anchors"] == 150
        and config["training_requests"] == 300
        and config["folds"] == 5
        and config["horizon"] == 40
        and config["dt_s"] == 0.05
        and config["closed_below"] == 0.35
        and config["commanded_opening"] == 0.1
        and config["positive_weight"] == 2.0
        and config["ridge"] == 0.003
        and config["max_iterations"] == 400
        and config["gradient_tolerance"] == 1e-6
        and config["thresholds"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
        and config["mae_ratio_limit"] == 1.05
        and config["mae_additive_limit"] == 0.01
        and config["pooled_fpr_increase"] == 0.01
        and config["task_fpr_increase"] == 0.02
        and config["minimum_h8_added_true_positives"] == 5
        and config["minimum_tasks_improved"] == 2
        and config["workers"] == 5
        and config["world_model_calls"] == config["physics_steps"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "registered opening event study differs",
    )
    return ref, config


def configured_ref(config, key):
    ref = {"path": str(REPO / config[key]), "sha256": config[key + "_sha256"]}
    prefix.checked(ref)
    return ref


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "complete CPU regression is required",
    )
    before = snapshot(REPO)["sources"]
    require(before == prefix.read(qa["source_before_ref"])["sources"], "CPU regression source identity")
    return before


def dry_run():
    config_ref, config = read_config()
    evidence = Evidence()
    registration_ref = configured_ref(config, "fit_registration")
    _previous, prepared, _refs, collections = admit_aggregation(
        evidence.json(registration_ref), read_json=evidence.json
    )
    identities = {
        (a["source_identity"]["task_index"], a["source_identity"]["demo"]) for a in prepared["training_anchors"]
    }
    require(len(identities) == 30 and not identities & VALIDATION_DEMO_KEYS, "training split differs")
    require(len(prepared["training_anchors"]) == 150 and len(collections[0]["requests"]) == 300, "D0 denominator")
    group_folds(sorted(identities), config["seed"])
    evidence.finish()
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                registration_ref=registration_ref,
                source_refs=source_refs(),
                training_future_reads=0,
                validation_future_reads=0,
                model_initializations=0,
                physics_steps=0,
            ),
        )
    )


def prepare():
    config_ref, config = read_config()
    before = verify_qa_sources()
    dry = prefix.read(prefix.reference(DRY))
    require(
        dry["complete"] is dry["passed"] is True
        and dry["config_ref"] == config_ref
        and dry["source_refs"] == source_refs(),
        "frozen metadata dry run",
    )
    require(not ROOT.exists(), "new study output root required")
    evidence = Evidence()
    registration_ref = configured_ref(config, "fit_registration")
    _previous, prepared, _refs, collections = admit_aggregation(
        evidence.json(registration_ref), read_json=evidence.json
    )
    training = collections[0]
    require(training["round_name"] == "D0" and len(training["requests"]) == 300, "complete D0 capture")
    normal = evidence.json(training["normalization_ref"])
    anchors = {a["name"]: a for a in prepared["training_anchors"]}
    identities = {(a["source_identity"]["task_index"], a["source_identity"]["demo"]) for a in anchors.values()}
    require(len(anchors) == 150 and len(identities) == 30 and not identities & VALIDATION_DEMO_KEYS, "training split")
    assignments = group_folds(sorted(identities), config["seed"])
    archives, counts = {}, Counter()
    features, openings, truths, folds, tasks, demos, names, seeds = [], [], [], [], [], [], [], []
    max_feature_difference = 0.0
    for request in training["requests"]:
        anchor = anchors[request["anchor_name"]]
        name = anchor["name"]
        if name not in archives:
            archives[name] = training_archive(evidence, anchor)
        states = archives[name]
        counts[name] += 1
        action = raw_c0_action(evidence, request, normal)
        measured = pose9_to_transform(states[0, :9])
        targets, opening = decode_action_targets(measured, action, horizon=40)
        x = event_features(measured, states[0, 9], targets, opening)
        independent = independent_features(measured, states[0, 9], targets, opening)
        max_feature_difference = max(max_feature_difference, float(np.max(abs(x - independent))))
        identity = anchor["source_identity"]
        task, demo = identity["task_index"], identity["demo"]
        features.append(x)
        openings.append(opening)
        truths.append(states[1:, 9])
        folds.append(assignments[(task, demo)])
        tasks.append(task)
        demos.append(demo)
        names.append(name)
        seeds.append(request["sampling_seed"])
    require(
        len(archives) == 150 and set(counts.values()) == {2} and max_feature_difference <= 1e-12,
        "complete and independently reconstructed training features",
    )
    source = source_refs()
    for ref in source:
        evidence.path(ref)
    evidence.finish()
    require(snapshot(REPO)["sources"] == before, "source bytes changed during preparation")
    ROOT.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "training_data.npz"
    with data_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            features=np.asarray(features),
            opening=np.asarray(openings),
            truth=np.asarray(truths),
            fold=np.asarray(folds),
            task=np.asarray(tasks),
            demo=np.asarray(demos),
            anchor=np.asarray(names),
            seed=np.asarray(seeds),
        )
    data_ref = prefix.reference(data_path)
    manifest = dict(
        complete=True,
        passed=True,
        config_ref=config_ref,
        registration_ref=registration_ref,
        dataset_ref=data_ref,
        source_refs=source,
        input_sha256=evidence.inputs,
        training_requests=300,
        training_anchors=150,
        training_demos=30,
        rows=12000,
        validation_future_reads=0,
        max_independent_feature_difference=max_feature_difference,
        fold_assignments=[dict(task=t, demo=d, fold=assignments[(t, d)]) for t, d in sorted(assignments)],
        model_initializations=0,
        physics_steps=0,
        fresh_cases=0,
    )
    print(publish(ROOT / "training_manifest.json", manifest))


def training_data():
    config_ref, config = read_config()
    manifest_ref = prefix.reference(ROOT / "training_manifest.json")
    manifest = prefix.read(manifest_ref)
    require(
        manifest["complete"] is manifest["passed"] is True
        and manifest["config_ref"] == config_ref
        and manifest["source_refs"] == source_refs()
        and manifest["training_requests"] == 300
        and manifest["rows"] == 12000
        and manifest["validation_future_reads"] == 0,
        "frozen complete training data required",
    )
    with np.load(prefix.checked(manifest["dataset_ref"]), allow_pickle=False) as z:
        data = {
            key: z[key].copy() for key in ("features", "opening", "truth", "fold", "task", "demo", "anchor", "seed")
        }
    require(
        data["features"].shape == (300, 40, 44)
        and data["opening"].shape == data["truth"].shape == (300, 40)
        and all(data[k].shape == (300,) for k in ("fold", "task", "demo", "anchor", "seed"))
        and np.isfinite(data["features"]).all()
        and np.isfinite(data["opening"]).all()
        and np.isfinite(data["truth"]).all()
        and np.all((data["opening"] >= 0) & (data["opening"] <= 1))
        and np.all((data["truth"] >= 0) & (data["truth"] <= 1)),
        "training array shapes, masks or range differ",
    )
    identities = {(int(task), str(demo)) for task, demo in zip(data["task"], data["demo"], strict=True)}
    assignments = group_folds(sorted(identities), config["seed"])
    require(
        np.array_equal(
            data["fold"], [assignments[(int(t), str(d))] for t, d in zip(data["task"], data["demo"], strict=True)]
        )
        and all(np.count_nonzero(data["fold"] == k) == 60 for k in range(5)),
        "all observations from one demo must remain in its registered fold",
    )
    return config, manifest_ref, manifest, data


def fit_one(data, config, selected):
    x = data["features"][selected].reshape(-1, 44)
    y = (data["truth"][selected].reshape(-1) < config["closed_below"]).astype(float)
    coefficients, info = fit_event_model(
        x,
        y,
        positive_weight=config["positive_weight"],
        ridge=config["ridge"],
        max_iterations=config["max_iterations"],
        gradient_tolerance=config["gradient_tolerance"],
    )
    independent = verify_fit(x, y, coefficients, config, info)
    return coefficients, info, independent


def worker_fold(fold):
    config, manifest_ref, manifest, data = training_data()
    require(type(fold) is int and 0 <= fold < 5, "registered fold required")
    training, held = data["fold"] != fold, data["fold"] == fold
    coefficients, info, independent = fit_one(data, config, training)
    probability = predict_probability(data["features"][held], coefficients)
    independent_probability = expit(data["features"][held] @ coefficients)
    difference = float(np.max(abs(probability - independent_probability)))
    require(difference <= 1e-12, "independent held-fold probability")
    prefix.checked(manifest["dataset_ref"])
    print(
        publish(
            ROOT / "cv" / f"fold_{fold}.json",
            dict(
                complete=True,
                passed=True,
                fold=fold,
                training_manifest_ref=manifest_ref,
                train_queries=int(training.sum()),
                held_queries=int(held.sum()),
                held_indices=np.flatnonzero(held),
                coefficients=coefficients,
                optimizer=info,
                independent_fit=independent,
                held_probability=probability,
                max_probability_difference=difference,
                validation_future_reads=0,
            ),
        )
    )


def train():
    before = verify_qa_sources()
    config, manifest_ref, manifest, data = training_data()
    cv = ROOT / "cv"
    require(not cv.exists(), "new cross-validation output required")
    cv.mkdir()
    registration_ref = publish(
        cv / "registration.json", dict(training_manifest_ref=manifest_ref, source_refs=source_refs())
    )
    specs = [
        dict(
            key=f"fold_{i}",
            argv=[
                sys.executable,
                "-m",
                "scripts.wam.training.native.causal_opening_event_study",
                "--worker-fold",
                str(i),
            ],
            log=cv / f"fold_{i}.log",
        )
        for i in range(5)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=config["worker_timeout_seconds"])
    for item in lifecycle["workers"]:
        item["log_ref"] = prefix.reference(item.pop("log"))
    workers_ref = publish(cv / "workers.json", lifecycle)
    result = dict(
        complete=False,
        passed=False,
        training_gate_passed=False,
        registration_ref=registration_ref,
        workers_ref=workers_ref,
        automatic_retry=False,
        validation_future_reads=0,
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    try:
        registered_feedback_phase.require_closed_workers(lifecycle, [spec["key"] for spec in specs])
        refs = [prefix.reference(cv / f"fold_{i}.json") for i in range(5)]
        folds = [prefix.read(ref) for ref in refs]
        probability = np.empty((300, 40))
        covered = np.zeros(300, dtype=bool)
        for i, fold in enumerate(folds):
            indices = np.asarray(fold["held_indices"], dtype=int)
            require(
                fold["complete"] is fold["passed"] is True
                and fold["fold"] == i
                and fold["training_manifest_ref"] == manifest_ref
                and len(indices) == 60
                and np.array_equal(indices, np.flatnonzero(data["fold"] == i))
                and not covered[indices].any(),
                "one out-of-fold prediction per registered query",
            )
            probability[indices] = fold["held_probability"]
            covered[indices] = True
        require(covered.all() and np.isfinite(probability).all(), "all training queries covered")
        selection = select_threshold(data["opening"], data["truth"], probability, data["task"], config)
        independent_selection = verify_selection(
            data["opening"], data["truth"], probability, data["task"], config, selection
        )
        threshold = selection["selected_threshold"]
        candidate_ref = None
        if threshold is not None:
            coefficients, info, independent = fit_one(data, config, np.ones(300, dtype=bool))
            candidate_ref = publish(
                ROOT / "candidate.json",
                dict(
                    schema="s2a.causal-opening-event-candidate.v1",
                    complete=True,
                    training_manifest_ref=manifest_ref,
                    config_ref=manifest["config_ref"],
                    threshold=threshold,
                    coefficients=coefficients,
                    optimizer=info,
                    independent_fit=independent,
                    train_demos=30,
                    train_queries=300,
                    feature_scope="measured TCP/opening and already-generated H40 only",
                    position_rotation_changed=False,
                    validation_future_reads=0,
                ),
            )
        for path, digest in manifest["input_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        require(
            source_refs() == manifest["source_refs"] and snapshot(REPO)["sources"] == before,
            "training source/input bytes changed",
        )
        result.update(
            complete=True,
            passed=True,
            training_gate_passed=threshold is not None,
            fold_refs=refs,
            oof_probability=probability,
            selection=selection,
            independent_selection=independent_selection,
            final_fit_attempted=threshold is not None,
            candidate_ref=candidate_ref,
        )
    finally:
        print(publish(cv / "result.json", result), flush=True)


def validate():
    config, manifest_ref, manifest, training = training_data()
    cv_ref = prefix.reference(ROOT / "cv/result.json")
    cv = prefix.read(cv_ref)
    require(
        cv["complete"] is cv["passed"] is cv["training_gate_passed"] is True and cv["candidate_ref"] is not None,
        "only a training-qualified frozen candidate can be validated",
    )
    candidate = prefix.read(cv["candidate_ref"])
    require(
        candidate["schema"] == "s2a.causal-opening-event-candidate.v1"
        and candidate["complete"] is True
        and candidate["training_manifest_ref"] == manifest_ref
        and candidate["threshold"] == cv["selection"]["selected_threshold"],
        "frozen candidate identity",
    )
    evidence = Evidence()
    score_ref = configured_ref(config, "validation_score")
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
        "full independently qualified validation data required",
    )
    prepared = evidence.json(score["prepared_ref"])
    current = evidence.json(prepared["current_inputs_ref"])
    anchors = {a["name"]: a for a in current["current_inputs"]}
    original = {row["name"]: row for row in score["state_rows"]}
    archives, openings, truths, probabilities, tasks, episodes, seeds, names = {}, [], [], [], [], [], [], []
    maximum = 0.0
    for item in numerical["generated_refs"]:
        if item["variant"] != "raw":
            continue
        anchor, row = anchors[item["anchor_name"]], original[item["name"]]
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
            and hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest() == anchor["current_sim_state_sha256"]
            and data["sim_time"][frame] == anchor["current_time_s"]
            and data["action_valid"][frame : frame + 40].all()
            and np.allclose(np.diff(data["sim_time"][frame : frame + 41]), 0.05, atol=1e-9, rtol=0),
            "validation current/future identity",
        )
        action = np.asarray(evidence.json(item["response_ref"])["outputs"][0]["content"]["action"])
        measured = pose9_to_transform(states[0, :9])
        targets, opening = decode_action_targets(measured, action, horizon=40)
        np.testing.assert_allclose(opening - states[1:, 9], row["errors"]["wam"]["opening"], atol=1e-12, rtol=0)
        x = event_features(measured, states[0, 9], targets, opening)
        maximum = max(maximum, float(np.max(abs(x - independent_features(measured, states[0, 9], targets, opening)))))
        openings.append(opening)
        truths.append(states[1:, 9])
        probabilities.append(predict_probability(x, candidate["coefficients"]))
        tasks.append(row["task_index"])
        episodes.append(episode)
        seeds.append(row["sampling_seed"])
        names.append(item["name"])
    require(
        len(openings) == 48 and len(anchors) == 24 and len(archives) == 6 and maximum <= 1e-12,
        "all validation requests and independent features",
    )
    validation_ids = {(int(name[4]), "demo_" + name.rsplit("_", 1)[1]) for name in archives}
    training_ids = {(int(t), str(d)) for t, d in zip(training["task"], training["demo"], strict=True)}
    require(
        validation_ids == VALIDATION_DEMO_KEYS and not validation_ids & training_ids,
        "entire validation demos excluded from training",
    )
    raw, truth, probability, tasks = (
        np.asarray(openings),
        np.asarray(truths),
        np.asarray(probabilities),
        np.asarray(tasks),
    )
    threshold = candidate["threshold"]
    scored = evaluate(raw, truth, probability, tasks, threshold, config)
    only_threshold = dict(config, thresholds=[threshold])
    singleton = dict(thresholds=[scored], selected_threshold=threshold if scored["eligible"] else None)
    independent = verify_selection(raw, truth, probability, tasks, only_threshold, singleton)
    corrected = apply_opening_event(
        raw, probability, threshold, closed_below=config["closed_below"], commanded_opening=config["commanded_opening"]
    )
    by_episode = {
        episode: {
            str(h): summarize(
                raw[np.asarray(episodes) == episode],
                truth[np.asarray(episodes) == episode],
                corrected[np.asarray(episodes) == episode],
                h,
                closed_below=config["closed_below"],
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
                closed_below=config["closed_below"],
            )
            for h in (8, 16, 40)
        }
        for seed in sorted(set(seeds))
    }
    evidence.finish()
    for path, digest in manifest["input_sha256"].items():
        prefix.checked({"path": path, "sha256": digest})
    print(
        publish(
            ROOT / "validation.json",
            dict(
                complete=True,
                passed=True,
                validation_gate_passed=scored["eligible"],
                candidate_ref=cv["candidate_ref"],
                score_ref=score_ref,
                state_gate_evaluated=False,
                eligible_for_development=False,
                requests=48,
                episodes=episodes,
                seeds=seeds,
                names=names,
                raw_opening=raw,
                truth_opening=truth,
                probability=probability,
                candidate_opening=corrected,
                event_metrics=scored,
                by_episode=by_episode,
                by_seed=by_seed,
                independent_selection=independent,
                max_independent_feature_difference=maximum,
                input_sha256=evidence.inputs,
                raw_position_rotation_unchanged=True,
                new_model_calls=0,
                physics_steps=0,
                fresh_cases=0,
                evidence_boundary="Offline event-label scoring on saved raw responses; not physical tracking, grasp, task success or a 054 qualification.",
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--worker-fold", type=int)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.prepare:
        prepare()
    elif args.train:
        train()
    elif args.validate:
        validate()
    else:
        worker_fold(args.worker_fold)


if __name__ == "__main__":
    main()
