"""Own one active-optimization transaction branch and its parity check."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_transaction_feedback_execution_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    prior_ref = reference(REPO / "outputs/tracking_margin_feedback_execution_20261027/registration.json")
    sources = dict(read(prior_ref)["source_sha256"])
    for path in [
        Path(__file__),
        REPO / "scripts/run_tracking_transaction_feedback.py",
        REPO / "scripts/check_tracking_transaction_feedback.py",
        REPO / "src/s2a/tracking_preview_transaction.py",
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prior_registration_ref=prior_ref,
            physics_budget=128,
            prefix_steps=72,
            branch_steps=56,
            case_index=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg, stages=[])
    try:
        for label, script, extra in [
            (
                "candidate",
                "run_tracking_transaction_feedback.py",
                ["--case-index", "0", "--horizon", "56", "--strategy", "tracking"],
            ),
            ("parity", "check_tracking_transaction_feedback.py", []),
        ]:
            stage = owned_stage(
                [str(REPO / "outputs/libero_env311/bin/python"), f"scripts/{script}", *extra],
                environment=cpu_environment(),
                directory=root / label,
                timeout_s=1800,
                source_sha256=sources,
            )
            result["stages"].append(dict(label=label, stage_ref=stage))
            print(json.dumps(dict(completed_stage=label)), flush=True)
        result.update(
            complete=True,
            passed=True,
            parity_ref=reference(
                REPO / "outputs/tracking_transaction_feedback_20261027/case00_tracking/transaction_parity.json"
            ),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
