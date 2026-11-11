"""Apply one frozen calibration to previously audited reference timelines."""

import argparse
import sys
from pathlib import Path

import numpy as np

from s2a.geometry import matrix_to_rot6d, pose_error
from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.cached_endpoint_calibration import apply_calibration
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.verification.wam.training.verify_cached_endpoint_calibration import corrected as independent_corrected
from scripts.wam.tracking.preview import bind_bounded_release_shadow as binding

REPO = prefix.REPO
CONFIG = REPO / "configs/wam/training/cached_endpoint_shadow_20261111.json"
PLAN = REPO / "plans/v0.0.1/wam/training/cached_endpoint_shadow_20261111.md"
ROOT = REPO / "outputs/cached_endpoint_shadow_20261111"
DRY = REPO / "outputs/cached_endpoint_shadow_dry_20261111/result.json"
QA = REPO / "outputs/qa/cached_endpoint_shadow_fix_20261111/result.json"
TEST = REPO / "tests/research/wam/training/test_cached_endpoint_shadow.py"
MODEL = REPO / "src/s2a/research/wam/training/cached_endpoint_calibration.py"
AUDIT = REPO / "scripts/verification/wam/training/verify_cached_endpoint_calibration.py"


def load_inputs():
    config_ref = prefix.reference(CONFIG)
    config = prefix.read(config_ref)
    prefix.require(
        config["schema"] == "s2a.cached-endpoint-shadow.v1"
        and config["output"] == "outputs/cached_endpoint_shadow_20261111"
        and config["correct_cases"] == 18
        and config["correct_steps"] == 4907
        and config["raw_case_indices"] == [18, 20, 22, 24, 26, 28]
        and config["raw_steps_per_case"] == 600
        and config["raw_chunks_per_case"] == 75
        and config["workers"] == 32
        and config["seed"] == 42
        and config["world_model_calls"] == config["physics_steps"] == config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "registered shadow identity differs",
    )
    candidate_ref = {"path": str(REPO / config["candidate"]), "sha256": config["candidate_sha256"]}
    independent_ref = {"path": str(REPO / config["independent"]), "sha256": config["independent_sha256"]}
    candidate, audit = prefix.read(candidate_ref), prefix.read(independent_ref)
    prefix.require(
        candidate["complete"] is audit["complete"] is audit["passed"] is True
        and audit["candidate_ref"] == candidate_ref
        and candidate["training_requests"] == 300,
        "frozen independently audited candidate required",
    )
    for ref in candidate["source_refs"]:
        prefix.checked(ref)
    candidate_config = prefix.read(candidate["config_ref"])
    inputs = binding.load_inputs()
    selected = inputs["cases"][:18] + [inputs["cases"][i] for i in config["raw_case_indices"]]
    prefix.require(
        len(selected) == 24
        and all(item["family"] == "correct_reference" for item in selected[:18])
        and sum(item["steps"] for item in selected[:18]) == 4907
        and all(item["family"] == "development" and item["case_id"].endswith("_raw") for item in selected[18:]),
        "unchanged correct-reference and raw development denominators",
    )
    return dict(
        config_ref=config_ref,
        config=config,
        candidate_ref=candidate_ref,
        independent_ref=independent_ref,
        candidate=candidate,
        candidate_config=candidate_config,
        cases=selected,
    )


def inputs_receipt(inputs):
    return {key: inputs[key] for key in ("config_ref", "candidate_ref", "independent_ref", "cases")}


def source_refs():
    return [
        prefix.reference(path) for path in (CONFIG, PLAN, Path(__file__), TEST, MODEL, AUDIT, Path(binding.__file__))
    ]


def bounded_reference(item, chunk, measured_opening, candidate, config, tolerance):
    """Only pad a short correct-reference tail; retain its actual prefix."""
    anchor = np.asarray(chunk["anchor_tcp"], dtype=float)
    poses = np.asarray(chunk["effective_targets_tcp"], dtype=float)
    openings = np.asarray(chunk["openings"], dtype=float)
    count = len(openings)
    prefix.require(poses.shape == (count, 4, 4) and 1 <= count <= 40, "known reference horizon")
    if item["family"] == "development":
        prefix.require(
            count == 40
            and chunk["model_rotation_alpha"] == 1
            and not chunk["retiming"]
            and np.array_equal(poses, np.asarray(chunk["raw_targets_tcp"])),
            "full unmodified raw WAM reference required",
        )
    else:
        prefix.require(item["family"] == "correct_reference", "known reference family")
    padded_poses = np.concatenate((poses, np.repeat(poses[-1:], 40 - count, axis=0)))
    padded_opening = np.r_[openings, np.repeat(openings[-1], 40 - count)]
    updated, new_opening = apply_calibration(
        anchor,
        padded_poses,
        padded_opening,
        measured_opening,
        candidate["pose_coefficients"],
        candidate["opening_coefficients"],
        config,
    )
    state = np.r_[anchor[:3, 3], matrix_to_rot6d(anchor[:3, :3]), measured_opening]
    p, r, o = independent_corrected(
        state, padded_poses[:, :3, 3], padded_poses[:, :3, :3], padded_opening, candidate, config
    )
    difference = max(
        float(np.max(abs(p - updated[:, :3, 3]))),
        float(np.max(abs(r - updated[:, :3, :3]))),
        float(np.max(abs(o - new_opening))),
    )
    prefix.require(difference <= tolerance, "independent reference reconstruction differs")
    return poses, openings, updated[:count], new_opening[:count], difference


def shadow_case(item, chunks, rows, tool, candidate, config, tolerance):
    samples, queries = [], []
    for query, chunk in enumerate(chunks):
        step = chunk["step"]
        reference = bounded_reference(item, chunk, rows[step]["opening_before"], candidate, config, tolerance)
        raw, opening, corrected, adjusted, difference = reference
        consumed = min(8, len(rows) - step)
        raw_events, corrected_events = opening < 0.35, adjusted < 0.35
        queries.append(
            dict(
                step=step,
                available=len(opening),
                consumed=consumed,
                raw_close_any=bool(raw_events.any()),
                corrected_close_any=bool(corrected_events.any()),
                raw_consumed_close_any=bool(raw_events[:consumed].any()),
                corrected_consumed_close_any=bool(corrected_events[:consumed].any()),
                independent_max_abs=difference,
            )
        )
        for offset in range(consumed):
            row = rows[step + offset]
            actual = np.asarray(row["after_source"]) @ tool
            actual_opening = row["opening"] if item["family"] == "correct_reference" else row["opening_after"]
            raw_error, new_error = pose_error(raw[offset], actual), pose_error(corrected[offset], actual)
            delta = pose_error(corrected[offset], raw[offset])
            samples.append(
                dict(
                    step=step + offset,
                    raw_position=float(np.linalg.norm(raw_error[:3])),
                    corrected_position=float(np.linalg.norm(new_error[:3])),
                    raw_rotation=float(np.linalg.norm(raw_error[3:])),
                    corrected_rotation=float(np.linalg.norm(new_error[3:])),
                    raw_opening=float(opening[offset] - actual_opening),
                    corrected_opening=float(adjusted[offset] - actual_opening),
                    reference_position_delta=float(np.linalg.norm(delta[:3])),
                    reference_rotation_delta=float(np.linalg.norm(delta[3:])),
                    reference_opening_delta=float(adjusted[offset] - opening[offset]),
                )
            )
    prefix.require(
        len(samples) == len(rows) and [s["step"] for s in samples] == list(range(len(rows))), "endpoint coverage"
    )
    return dict(samples=samples, queries=queries)


def summarize(samples, queries):
    result = {"steps": len(samples), "queries": len(queries)}
    for mode in ("raw", "corrected"):
        p = np.asarray([sample[mode + "_position"] for sample in samples])
        r = np.asarray([sample[mode + "_rotation"] for sample in samples])
        g = np.asarray([sample[mode + "_opening"] for sample in samples])
        result[mode] = dict(
            position_rmse_m=float(np.sqrt(np.mean(p**2))),
            rotation_rmse_rad=float(np.sqrt(np.mean(r**2))),
            opening_mae=float(np.mean(abs(g))),
        )
    for key in ("reference_position_delta", "reference_rotation_delta", "reference_opening_delta"):
        values = np.abs([sample[key] for sample in samples])
        result[key] = {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(np.max(values)),
        }
    for scope in ("", "consumed_"):
        original, altered = "raw_" + scope + "close_any", "corrected_" + scope + "close_any"
        result[scope + "closure_queries"] = dict(
            raw=sum(q[original] for q in queries),
            corrected=sum(q[altered] for q in queries),
            added=sum(not q[original] and q[altered] for q in queries),
            removed=sum(q[original] and not q[altered] for q in queries),
        )
    result["independent_max_abs"] = max(q["independent_max_abs"] for q in queries)
    return result


def dry_run():
    inputs = load_inputs()
    counts = []
    for item in inputs["cases"]:
        chunks, rows, _ = binding.read_case(item)
        if item["family"] == "development":
            prefix.require(len(chunks) == 75 and len(rows) == 600, "complete raw WAM timeline")
        counts.append(dict(case_id=item["case_id"], chunks=len(chunks), steps=len(rows)))
    for ref in source_refs():
        prefix.checked(ref)
    print(
        publish(
            DRY,
            dict(
                complete=True,
                passed=True,
                inputs=inputs_receipt(inputs),
                counts=counts,
                source_refs=source_refs(),
                new_model_calls=0,
                physics_steps=0,
                fresh_cases=0,
            ),
        )
    )


def worker(index):
    registration = prefix.read(prefix.reference(ROOT / "registration.json"))
    prefix.require(registration["source_refs"] == source_refs(), "frozen shadow sources")
    receipt = registration["inputs"]
    item = receipt["cases"][index]
    candidate = prefix.read(receipt["candidate_ref"])
    config = prefix.read(candidate["config_ref"])
    shadow = prefix.read(receipt["config_ref"])
    chunks, rows, tool = binding.read_case(item)
    result = shadow_case(item, chunks, rows, tool, candidate, config, shadow["max_independent_difference"])
    for ref in item["refs"].values():
        prefix.checked(ref)
    print(
        publish(
            ROOT / f"case_{index:02d}.json",
            dict(
                complete=True,
                passed=True,
                case_id=item["case_id"],
                family=item["family"],
                input_refs=item["refs"],
                summary=summarize(**result),
                **result,
            ),
        )
    )


def execute():
    inputs = load_inputs()
    dry = prefix.read(prefix.reference(DRY))
    qa = prefix.read(prefix.reference(QA))
    prefix.require(
        dry["complete"] is dry["passed"] is qa["complete"] is qa["passed"] is True
        and dry["inputs"] == inputs_receipt(inputs)
        and dry["source_refs"] == source_refs()
        and qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True,
        "passing fixed-source dry and CPU QA required",
    )
    prefix.require(not ROOT.exists(), "new output root required")
    registration_ref = publish(
        ROOT / "registration.json", dict(inputs=inputs_receipt(inputs), source_refs=source_refs())
    )
    specs = [
        dict(
            key=str(i),
            argv=[
                sys.executable,
                "-m",
                "scripts.wam.tracking.references.shadow_cached_endpoint_calibration",
                "--worker",
                str(i),
            ],
            log=ROOT / f"worker_{i:02d}.log",
        )
        for i in range(24)
    ]
    lifecycle = run_owned_workers(specs, cwd=REPO, timeout_seconds=inputs["config"]["worker_timeout_seconds"])
    workers_ref = publish(ROOT / "workers.json", lifecycle)
    prefix.require(
        lifecycle["launch_error"] is None
        and len(lifecycle["workers"]) == 24
        and all(
            w["actual_wait"] and w["reaped"] and not w["signal_sent"] and w["returncode"] == 0
            for w in lifecycle["workers"]
        ),
        "all owned shadow workers must finish normally",
    )
    cases = [prefix.read(prefix.reference(ROOT / f"case_{i:02d}.json")) for i in range(24)]
    correct, raw = cases[:18], cases[18:]
    prefix.require(
        all(c["complete"] is c["passed"] is True for c in cases)
        and sum(c["summary"]["steps"] for c in correct) == 4907
        and sum(c["summary"]["steps"] for c in raw) == 3600
        and sum(c["summary"]["queries"] for c in raw) == 450,
        "complete registered denominators required",
    )
    for ref in source_refs():
        prefix.checked(ref)
    result = dict(
        complete=True,
        passed=True,
        registration_ref=registration_ref,
        workers_ref=workers_ref,
        case_refs=[prefix.reference(ROOT / f"case_{i:02d}.json") for i in range(24)],
        correct_reference=summarize(
            [s for c in correct for s in c["samples"]], [q for c in correct for q in c["queries"]]
        ),
        raw_development=summarize([s for c in raw for s in c["samples"]], [q for c in raw for q in c["queries"]]),
        per_case=[dict(case_id=c["case_id"], family=c["family"], summary=c["summary"]) for c in cases],
        new_model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        automatic_retry=False,
        evidence_boundary="Fixed historical states only; not counterfactual physical tracking, grasp, task success, or new WAM inference.",
    )
    print(publish(ROOT / "result.json", result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--dry-run", action="store_true")
    phase.add_argument("--execute", action="store_true")
    phase.add_argument("--worker", type=int)
    args = parser.parse_args()
    prefix.environment()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        prefix.require(0 <= args.worker < 24, "registered case index required")
        worker(args.worker)


if __name__ == "__main__":
    main()
