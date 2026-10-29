"""Own one CPU replay and preserve its failure, elapsed time and source closure."""

import argparse
import json
import os
import time
from pathlib import Path

from s2a.infrastructure.lifecycle.worker_handshake import owned_child
from scripts.maintenance.run_cpu_tests import CPU_ENV, publish, reference, snapshot
from scripts.tracking.trajectory_search.state_vector_fresh_diagnostic_admission import (
    CONFIG,
    OUTPUT,
    OWNER,
    PRIOR_REFS,
    QA_TARGETS,
    REPO,
    require,
    validate_spec,
)

DRY_OWNER = REPO / "outputs/state_vector_dual_recovery_fresh_diagnostic_v2_dry_20261029"


def load_bound(path, digest):
    require(path.is_relative_to(REPO / "outputs"), "Repository output evidence required")
    require(reference(path)["sha256"] == digest, "Pinned admission evidence SHA")
    return json.loads(path.read_text())


def verify_qa(qa, before):
    require(
        qa["complete"] is qa["passed"] is qa["source_unchanged"] is qa["actual_wait"] is qa["reaped"] is True
        and not qa["collection"]["collection_errors"]
        and qa["call_outcomes"] == {"passed": qa["collection_node_count"]}
        and qa["collection_node_count"] > 0
        and set(QA_TARGETS) <= set(qa["targets"]),
        "Closed relevant CPU regression without hidden skip or collection error",
    )
    recorded = load_bound(Path(qa["source_before_ref"]["path"]), qa["source_before_ref"]["sha256"])
    require(recorded["sources"] == before["sources"], "QA must bind the exact current sources and inputs")


def main(args):
    require(Path.cwd() == REPO, "Run in the current S2A repository")
    require(all(os.environ.get(k) == v for k, v in CPU_ENV.items()), "Explicit CPU-only environment")
    spec = validate_spec(json.loads(CONFIG.read_text()))
    root = DRY_OWNER if args.dry else OWNER
    require(not root.exists() and not OUTPUT.exists(), "Exclusive new owner and replay output; no automatic retry")
    before = snapshot(REPO)
    qa_path = REPO / args.qa
    qa = load_bound(qa_path, args.qa_sha)
    verify_qa(qa, before)
    if not args.dry:
        require(args.dry_sha is not None, "Pinned completed actual --dry admission required")
        dry = load_bound(DRY_OWNER / "result.json", args.dry_sha)
        require(
            dry["complete"] is dry["passed"] is dry["source_unchanged"] is dry["reaped"] is True
            and dry["returncode"] == 0
            and dry["config_ref"] == reference(CONFIG)
            and dry["qa_ref"] == reference(qa_path),
            "One successful dry of this exact configuration and QA",
        )
    old_refs = {k: reference(REPO / p) for k, (p, _) in PRIOR_REFS.items()}
    require(old_refs == {k: spec[k] for k in PRIOR_REFS}, "Original receipts remain exact")
    root.mkdir()
    before_ref = publish(root / "source_before.json", before)
    result = dict(
        complete=False,
        passed=False,
        dry=args.dry,
        config_ref=reference(CONFIG),
        plan_ref=spec["plan_ref"],
        qa_ref=reference(qa_path),
        source_before_ref=before_ref,
        original_refs=old_refs,
        automatic_retry=False,
        actual_wait=False,
        reaped=False,
        source_unchanged=False,
        returncode=None,
    )
    command = [
        str(REPO / "outputs/libero_env311/bin/python"),
        "-m",
        "scripts.tracking.trajectory_search.check_tracking_v5_state_vector_fresh_diagnostic",
    ]
    if args.dry:
        command.append("--dry")
    result["command"] = command
    environment = dict(os.environ, **CPU_ENV)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        environment.pop(key, None)
    ownership = root / "ownership"
    ownership.mkdir()
    started = time.monotonic()
    try:
        with (root / "command.log").open("x") as stream:
            with owned_child(command, environment=environment, cwd=REPO, folder=ownership, stream=stream) as (
                child,
                birth,
            ):
                result["child_birth"] = birth
                result["returncode"] = child.wait(timeout=spec["timeout_seconds"])
                result["actual_wait"] = True
        result["complete"] = True
        if args.dry:
            lines = (root / "command.log").read_text().splitlines()
            receipt = json.loads(lines[-1])
            require(
                receipt["passed"] is receipt["dry"] is True
                and receipt["responses"] == receipt["queries"] == receipt["qp_solves"] == 0,
                "Actual metadata-only dry completed",
            )
            result["dry_receipt"] = receipt
            result["passed"] = result["returncode"] == 0
        elif (OUTPUT / "verified.json").is_file():
            result["checker_ref"] = reference(OUTPUT / "verified.json")
            checked = json.loads((OUTPUT / "verified.json").read_text())
            result["checker_complete"] = checked["complete"]
            result["checker_passed"] = checked["passed"]
            result["counts"] = {
                key: checked[key]
                for key in (
                    "response_evaluations",
                    "verified_responses",
                    "numerical_inner_steps",
                    "query_count",
                    "qp_solves",
                    "nnls_fit_attempts",
                    "physics_steps",
                    "world_model_calls",
                    "fresh_cases",
                )
            }
            result["passed"] = result["returncode"] == 0 and checked["complete"] is checked["passed"] is True
    except BaseException as error:
        result["error"] = repr(error)
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        release = ownership / "release.json"
        if release.is_file():
            result["release_ref"] = reference(release)
            cleanup = json.loads(release.read_text())
            result["reaped"] = cleanup["reaped"]
            result["signal_sent"] = cleanup["signal_sent"]
            result["returncode"] = cleanup["returncode"]
        try:
            after = snapshot(REPO)
            result["source_after_ref"] = publish(root / "source_after.json", after)
            result["source_unchanged"] = before == after
            result["original_receipts_unchanged"] = old_refs == {
                k: reference(REPO / p) for k, (p, _) in PRIOR_REFS.items()
            }
        except BaseException as error:
            result["source_after_error"] = repr(error)
        result["passed"] = bool(
            result["passed"]
            and result["source_unchanged"]
            and result["reaped"]
            and result.get("original_receipts_unchanged")
            and not result.get("signal_sent")
        )
        publish(root / "result.json", result)
    print(json.dumps(dict(result_ref=reference(root / "result.json"), passed=result["passed"])), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--qa-sha", required=True)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--dry-sha")
    raise SystemExit(main(parser.parse_args()))
