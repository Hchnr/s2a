"""Own one full baseline port replay and its independent parity check."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage


def main():
    root = REPO / "outputs/tracking_correct_reference_ordered_port06_v2_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    (REPO / "outputs/tracking_correct_reference_ordered_20261027").mkdir(exist_ok=True)
    prepared_ref = reference(REPO / "outputs/tracking_correct_reference_20261027/prepared.json")
    sources = dict(read(prepared_ref)["source_sha256"])
    for path in (
        Path(__file__),
        REPO / "scripts/run_tracking_correct_reference_ordered.py",
        REPO / "scripts/check_tracking_reference_ordered_parity.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    registration = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prepared_ref=prepared_ref,
            physics_budget=334,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=registration, stages=[])
    try:
        for name, script, extra in (
            ("baseline", "run_tracking_correct_reference_ordered.py", ["--policy", "baseline"]),
            ("parity", "check_tracking_reference_ordered_parity.py", []),
        ):
            receipt = owned_stage(
                [str(REPO / "outputs/libero_env311/bin/python"), f"scripts/{script}", "--case-index", "6", *extra],
                environment=cpu_environment(),
                directory=root / name,
                timeout_s=1800,
                source_sha256=sources,
            )
            result["stages"].append(dict(name=name, stage_ref=receipt))
            print(json.dumps(dict(completed_stage=name)), flush=True)
        result.update(
            complete=True,
            passed=True,
            parity_ref=reference(
                REPO / "outputs/tracking_correct_reference_ordered_20261027/case06_baseline/port_parity.json"
            ),
        )
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)


if __name__ == "__main__":
    main()
