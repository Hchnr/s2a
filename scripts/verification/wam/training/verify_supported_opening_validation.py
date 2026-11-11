"""Independently reconstruct the saved supported-opening validation score."""

import argparse
import ast
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.training.verify_causal_opening_event import independent_features
from scripts.verification.wam.training.verify_wam_clean_training_capture import Evidence, require

REPO = prefix.REPO
CONFIG = REPO / "configs/wam/training/supported_opening_validation_audit_20261111.json"
PLAN = REPO / "plans/v0.0.1/wam/training/supported_opening_validation_audit_20261111.md"
ROOT = REPO / "outputs/supported_opening_validation_audit_20261111"
DRY = REPO / "outputs/supported_opening_validation_audit_dry_20261111/result.json"
QA = REPO / "outputs/qa/supported_opening_validation_audit_20261111/result.json"


def source_refs():
    return [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__),
            REPO / "tests/research/gripper/test_supported_opening_validation_check.py",
            REPO / "scripts/verification/wam/training/verify_causal_opening_event.py",
            REPO / "scripts/verification/wam/training/verify_wam_clean_training_capture.py",
        )
    ]


def read_config():
    prefix.environment()
    ref = prefix.reference(CONFIG)
    config = prefix.read(ref)
    require(
        config["schema"] == "s2a.supported-opening-validation-audit.v1"
        and config["output"] == "outputs/supported_opening_validation_audit_20261111"
        and config["threshold"] == 0.7
        and config["support_future_steps"] == 16
        and config["closed_below"] == 0.35
        and config["commanded_opening"] == 0.1
        and config["requests"] == 48
        and config["demos"] == 6
        and config["seeds"] == [42, 43]
        and config["full_fits"] == config["world_model_calls"] == config["physics_steps"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "fixed independent supported-event audit required",
    )
    return ref, config


def configured_ref(config, name):
    ref = {"path": str(REPO / config[name]), "sha256": config[name + "_sha256"]}
    prefix.checked(ref)
    return ref


def verify_qa_sources():
    qa = prefix.read(prefix.reference(QA))
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and qa["all_tests_passed"] is True
        and qa["skipped_count"] == 0,
        "complete zero-skip independent-audit CPU regression required",
    )
    before = snapshot(REPO)["sources"]
    require(before == prefix.read(qa["source_before_ref"])["sources"], "audit regression source identity")
    return before


def independent_pose(value):
    value = np.asarray(value, dtype=float)
    require(value.shape == (9,) and np.isfinite(value).all(), "finite position and rot6d required")
    x, y = value[3:6].copy(), value[6:9].copy()
    nx = np.linalg.norm(x)
    require(nx >= 1e-8, "first rotation column is degenerate")
    x /= nx
    y -= np.dot(x, y) * x
    ny = np.linalg.norm(y)
    require(ny >= 1e-8, "second rotation column is degenerate")
    y /= ny
    out = np.eye(4)
    out[:3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    out[:3, 3] = value[:3]
    return out


def independent_targets(anchor_state, action):
    action = np.asarray(action, dtype=float)
    require(
        action.ndim == 2
        and action.shape[0] >= 40
        and action.shape[1] == 128
        and np.isfinite(action).all()
        and np.all(action[:, 10:] == 0),
        "finite ten-slot Action128 reference required",
    )
    anchor = independent_pose(anchor_state)
    pose = anchor.copy()
    poses = []
    for row in action[:40]:
        pose = pose @ independent_pose(row[:9])
        poses.append(pose.copy())
    return anchor, np.asarray(poses), np.clip(action[:40, 9], 0, 1)


def independent_opening(raw, probability, config):
    raw, probability = np.asarray(raw, dtype=float), np.asarray(probability, dtype=float)
    require(
        raw.ndim == 2
        and raw.shape[1] == 40
        and probability.shape == raw.shape
        and np.isfinite(raw).all()
        and np.isfinite(probability).all()
        and np.all((raw >= 0) & (raw <= 1))
        and np.all((probability >= 0) & (probability <= 1)),
        "independent H40 opening and probability required",
    )
    support = np.zeros(raw.shape, dtype=bool)
    output = raw.copy()
    for row in range(len(raw)):
        for endpoint in range(40):
            support[row, endpoint] = any(
                raw[row, later] < config["closed_below"]
                for later in range(endpoint + 1, min(endpoint + config["support_future_steps"] + 1, 40))
            )
            if support[row, endpoint] and probability[row, endpoint] >= config["threshold"]:
                if raw[row, endpoint] >= config["closed_below"]:
                    output[row, endpoint] = config["commanded_opening"]
    return support, output


def independent_summary(raw, truth, candidate, indices, horizon, closed_below):
    counts = dict(positive=0, negative=0, raw_tp=0, candidate_tp=0, added_tp=0, raw_fp=0, candidate_fp=0, added_fp=0)
    raw_errors, candidate_errors = [], []
    for row in indices:
        for endpoint in range(horizon):
            target, before, after = truth[row, endpoint], raw[row, endpoint], candidate[row, endpoint]
            actual, old, new = target < closed_below, before < closed_below, after < closed_below
            counts["positive"] += int(actual)
            counts["negative"] += int(not actual)
            counts["raw_tp"] += int(actual and old)
            counts["candidate_tp"] += int(actual and new)
            counts["added_tp"] += int(actual and not old and new)
            counts["raw_fp"] += int(not actual and old)
            counts["candidate_fp"] += int(not actual and new)
            counts["added_fp"] += int(not actual and not old and new)
            raw_errors.append(abs(float(before - target)))
            candidate_errors.append(abs(float(after - target)))
    require(raw_errors and counts["negative"], "nonempty group with negative endpoints required")
    return dict(
        endpoints=len(raw_errors),
        positive_endpoints=counts["positive"],
        negative_endpoints=counts["negative"],
        raw_true_positives=counts["raw_tp"],
        candidate_true_positives=counts["candidate_tp"],
        added_true_positives=counts["added_tp"],
        raw_false_positives=counts["raw_fp"],
        candidate_false_positives=counts["candidate_fp"],
        added_false_positives=counts["added_fp"],
        raw_false_positive_rate=counts["raw_fp"] / counts["negative"],
        candidate_false_positive_rate=counts["candidate_fp"] / counts["negative"],
        raw_mae=math.fsum(raw_errors) / len(raw_errors),
        candidate_mae=math.fsum(candidate_errors) / len(raw_errors),
    )


def compare_summary(checked, recorded):
    require(set(checked) == set(recorded), "independent summary keys differ")
    maximum = 0.0
    for key, value in checked.items():
        difference = abs(value - recorded[key])
        maximum = max(maximum, difference)
        require(difference <= 1e-12, "independent opening summary differs")
    return float(maximum)


def check_gate(pooled, groups, old_config, recorded):
    components = {}
    for h in (8, 16, 40):
        item = pooled[str(h)]
        components[f"h{h}_mae"] = (
            item["candidate_mae"] <= old_config["mae_ratio_limit"] * item["raw_mae"]
            and item["candidate_mae"] <= item["raw_mae"] + old_config["mae_additive_limit"]
        )
        components[f"h{h}_pooled_fpr"] = (
            item["candidate_false_positive_rate"] <= item["raw_false_positive_rate"] + old_config["pooled_fpr_increase"]
        )
        if h in (8, 16):
            for task in (0, 1, 2):
                group = groups[str(task)][str(h)]
                components[f"h{h}_task{task}_fpr"] = (
                    group["candidate_false_positive_rate"]
                    <= group["raw_false_positive_rate"] + old_config["task_fpr_increase"]
                )
    improved = {
        str(task): any(groups[str(task)][str(h)]["added_true_positives"] > 0 for h in (8, 16)) for task in (0, 1, 2)
    }
    components["h8_added_true_positives"] = (
        pooled["8"]["added_true_positives"] >= old_config["minimum_h8_added_true_positives"]
    )
    components["tasks_improved"] = sum(improved.values()) >= old_config["minimum_tasks_improved"]
    require(
        recorded["components"] == components
        and recorded["tasks_improved"] == improved
        and recorded["eligible"] == all(components.values())
        and recorded["added_score"] == sum(pooled[str(h)]["added_true_positives"] for h in (8, 16, 40)),
        "independent fixed event gate differs",
    )
    return components


def check_worker(evidence, ref, key, output_ref):
    receipt = evidence.json(ref)
    require(receipt["launch_error"] is None and len(receipt["workers"]) == 1, "one completed worker required")
    worker = receipt["workers"][0]
    require(
        worker["key"] == key
        and worker["returncode"] == 0
        and worker["actual_wait"] is worker["reaped"] is True
        and worker["signal_sent"] is False,
        "real worker wait, exit and reap required",
    )
    logged = ast.literal_eval(evidence.path(worker["log_ref"]).read_text().strip())
    require(logged == output_ref, "worker log must identify exact published result")
    return dict(pid=worker["pid"], exit_code=worker["returncode"], actual_wait=True, reaped=True)


def dry_run():
    before = verify_qa_sources()
    config_ref, config = read_config()
    refs = {
        name: configured_ref(config, name)
        for name in ("selection", "candidate", "fit", "validation", "fit_workers", "validation_workers")
    }
    evidence = Evidence()
    fit_lifecycle = check_worker(evidence, refs["fit_workers"], "fit", refs["fit"])
    validation_lifecycle = check_worker(evidence, refs["validation_workers"], "validate", refs["validation"])
    for ref in refs.values():
        evidence.path(ref)
    evidence.finish()
    require(snapshot(REPO)["sources"] == before, "dry audit source bytes changed")
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                config_ref=config_ref,
                refs=refs,
                source_refs=source_refs(),
                fit_lifecycle=fit_lifecycle,
                validation_lifecycle=validation_lifecycle,
                validation_future_reads=0,
                new_model_calls=0,
                physics_steps=0,
            ),
        )
    )


def audit():
    before = verify_qa_sources()
    config_ref, config = read_config()
    refs = {
        name: configured_ref(config, name)
        for name in ("selection", "candidate", "fit", "validation", "fit_workers", "validation_workers")
    }
    dry_ref = prefix.reference(DRY)
    dry = prefix.read(dry_ref)
    require(
        dry["complete"] is dry["passed"] is True
        and dry["config_ref"] == config_ref
        and dry["refs"] == refs
        and dry["source_refs"] == source_refs(),
        "matching independent no-label dry admission required",
    )
    require(not ROOT.exists(), "new non-overwriting audit output required")
    ROOT.mkdir(parents=True)
    result = dict(
        complete=False,
        passed=False,
        config_ref=config_ref,
        dry_ref=dry_ref,
        qa_ref=prefix.reference(QA),
        full_fits=0,
        world_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    try:
        evidence = Evidence()
        selection, candidate, fit, validation = (
            evidence.json(refs[name]) for name in ("selection", "candidate", "fit", "validation")
        )
        require(
            selection["complete"] is selection["passed"] is selection["training_gate_passed"] is True
            and selection["selection"]["selected_threshold"] == candidate["threshold"] == config["threshold"]
            and candidate["selection_ref"] == refs["selection"]
            and candidate["schema"] == "s2a.supported-opening-event-candidate.v1"
            and candidate["support_future_steps"] == config["support_future_steps"]
            and candidate["closed_below"] == config["closed_below"]
            and candidate["commanded_opening"] == config["commanded_opening"]
            and candidate["complete"] is fit["complete"] is fit["passed"] is True
            and fit["candidate_ref"] == refs["candidate"]
            and fit["full_fits"] == 1
            and validation["candidate_ref"] == refs["candidate"]
            and validation["fit_ref"] == refs["fit"]
            and validation["complete"] is validation["passed"] is True
            and validation["requests"] == 48,
            "frozen training selection, fit and recorded score required",
        )
        fit_lifecycle = check_worker(evidence, refs["fit_workers"], "fit", refs["fit"])
        validation_lifecycle = check_worker(evidence, refs["validation_workers"], "validate", refs["validation"])
        evidence.mapping(validation["input_sha256"])
        fit_registration = evidence.json(fit["registration_ref"])
        source_config_ref = next(
            ref
            for ref in fit_registration["source_refs"]
            if ref["path"].endswith("/causal_opening_event_20261111.json")
        )
        old = evidence.json(source_config_ref)
        require(
            old["closed_below"] == config["closed_below"] and old["commanded_opening"] == config["commanded_opening"],
            "original event thresholds required",
        )
        score = evidence.json(validation["score_ref"])
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
            "qualified saved response lineage required",
        )
        prepared = evidence.json(score["prepared_ref"])
        current = evidence.json(prepared["current_inputs_ref"])
        anchors = {a["name"]: a for a in current["current_inputs"]}
        original_rows = {row["name"]: row for row in score["state_rows"]}
        archives, openings, truths, probabilities, tasks, episodes, seeds, names = {}, [], [], [], [], [], [], []
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
                "independent validation state and time identity",
            )
            action = np.asarray(evidence.json(item["response_ref"])["outputs"][0]["content"]["action"])
            measured, poses, opening = independent_targets(states[0, :9], action)
            np.testing.assert_allclose(opening - states[1:, 9], row["errors"]["wam"]["opening"], atol=1e-12, rtol=0)
            features = independent_features(measured, states[0, 9], poses, opening)
            probability = expit(features @ np.asarray(candidate["coefficients"]))
            openings.append(opening)
            truths.append(states[1:, 9])
            probabilities.append(probability)
            tasks.append(row["task_index"])
            episodes.append(episode)
            seeds.append(row["sampling_seed"])
            names.append(item["name"])
        require(
            len(openings) == config["requests"]
            and len(archives) == config["demos"]
            and sorted(set(seeds)) == config["seeds"]
            and set(tasks) == {0, 1, 2}
            and len(set(names)) == 48
            and names == validation["names"]
            and episodes == validation["episodes"]
            and seeds == validation["seeds"],
            "48 distinct saved requests and six validation demos required",
        )
        raw, truth, probability = map(np.asarray, (openings, truths, probabilities))
        support, candidate_opening = independent_opening(raw, probability, config)
        maximum_arrays = {}
        for name, value in (
            ("raw_opening", raw),
            ("truth_opening", truth),
            ("probability", probability),
            ("candidate_opening", candidate_opening),
        ):
            recorded = np.asarray(validation[name])
            require(recorded.shape == value.shape, "independent validation array shape")
            maximum_arrays[name] = float(np.max(abs(value - recorded)))
            require(maximum_arrays[name] <= 1e-12, "independent validation array differs")
        require(np.array_equal(support, validation["support_mask"]), "independent support mask differs")
        maximum_metrics = 0.0
        pooled, groups = {}, {}
        for h in (8, 16, 40):
            pooled[str(h)] = independent_summary(raw, truth, candidate_opening, range(48), h, config["closed_below"])
            maximum_metrics = max(
                maximum_metrics, compare_summary(pooled[str(h)], validation["event_metrics"]["pooled"][str(h)])
            )
        for task in (0, 1, 2):
            groups[str(task)] = {}
            indices = [i for i, value in enumerate(tasks) if value == task]
            for h in (8, 16, 40):
                group = independent_summary(raw, truth, candidate_opening, indices, h, config["closed_below"])
                groups[str(task)][str(h)] = group
                maximum_metrics = max(
                    maximum_metrics, compare_summary(group, validation["event_metrics"]["by_task"][str(task)][str(h)])
                )
        for values, saved in ((episodes, validation["by_episode"]), (seeds, validation["by_seed"])):
            for value in sorted(set(values)):
                indices = [i for i, found in enumerate(values) if found == value]
                for h in (8, 16, 40):
                    group = independent_summary(raw, truth, candidate_opening, indices, h, config["closed_below"])
                    maximum_metrics = max(maximum_metrics, compare_summary(group, saved[str(value)][str(h)]))
        components = check_gate(pooled, groups, old, validation["event_metrics"])
        require(
            validation["event_metrics"]["threshold"] == config["threshold"]
            and validation["validation_gate_passed"] == all(components.values())
            and validation["validation_gate_passed"] is False,
            "recorded fixed validation gate and negative outcome required",
        )
        evidence.finish()
        require(source_refs() == dry["source_refs"] and snapshot(REPO)["sources"] == before, "audit source changed")
        result.update(
            complete=True,
            passed=True,
            validation_gate_passed=False,
            requests=48,
            demos=6,
            seeds=config["seeds"],
            max_array_differences=maximum_arrays,
            max_metric_difference=maximum_metrics,
            supported_endpoints=int(support.sum()),
            failed_components=[name for name, passed in components.items() if not passed],
            fit_lifecycle=fit_lifecycle,
            validation_lifecycle=validation_lifecycle,
            owner_exit_record_available=False,
            input_sha256=evidence.inputs,
            evidence_boundary="Independent re-evaluation of the same previously viewed responses; not new closed-loop or physical evidence.",
        )
    finally:
        print(publish(ROOT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        audit()


if __name__ == "__main__":
    main()
