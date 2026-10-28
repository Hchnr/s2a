"""Watch one existing producer by kernel identity, then own remaining audits and closure."""

import argparse
import json
import time
from pathlib import Path

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import cpu_environment, owned_stage
from s2a.worker_handshake import identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--producer-pid", type=int, required=True)
    args = p.parse_args()
    root = REPO / "outputs/dependency_safe_qualification_supervisor_20261027"
    require(not root.exists(), "No overwrite or automatic retry")
    producer = REPO / "outputs/dependency_safe_correct_reference_execution_20261027"
    birth = identity(args.producer_pid)
    require(birth["command"][-1] == "scripts/execute_dependency_safe_correct_reference.py", "Exact producer command")
    require(birth["cwd"] == str(REPO), "Current repository producer")
    require(
        birth["environment"]["CUDA_VISIBLE_DEVICES"] == birth["environment"]["LD_LIBRARY_PATH"] == "", "CPU producer"
    )
    sources = dict(read(reference(producer / "registration.json"))["source_sha256"])
    for path in [
        Path(__file__),
        *[
            REPO / "scripts" / name
            for name in (
                "execute_dependency_safe_audits.py",
                "audit_dependency_safe_robot_inputs.py",
                "audit_dependency_safe_labels_images.py",
                "execute_dependency_safe_closure.py",
                "close_dependency_safe_matrix.py",
            )
        ],
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    remaining = [i for i in range(18) if i not in (6, 0, 1, 2, 3)]
    registration = publish(
        root / "registration.json",
        dict(
            producer_identity=birth,
            source_sha256=sources,
            remaining_cases=remaining,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            required_qualified_cases=18,
            timeout_s=7200,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=registration, stages=[])
    try:
        deadline = time.monotonic() + 7200
        while not (producer / "result.json").exists():
            now = identity(args.producer_pid)
            require(
                now["start_ticks"] == birth["start_ticks"] and now["command"] == birth["command"], "Same live producer"
            )
            require(time.monotonic() < deadline, "Producer observation deadline")
            time.sleep(2)
        owner_ref = reference(producer / "result.json")
        owner = read(owner_ref)
        require(owner["complete"] is owner["passed"] is owner["performance_qualified"] is True, "Full original18 gate")
        for label, command in [
            (
                "remaining_audits",
                [
                    str(REPO / ".venv/bin/python"),
                    "scripts/execute_dependency_safe_audits.py",
                    "--case-indices",
                    *map(str, remaining),
                    "--run-id",
                    "remaining_20261027",
                ],
            ),
            ("closure", [str(REPO / ".venv/bin/python"), "scripts/execute_dependency_safe_closure.py"]),
        ]:
            stage = owned_stage(
                command, environment=cpu_environment(), directory=root / label, timeout_s=3600, source_sha256=sources
            )
            result["stages"].append(dict(label=label, stage_ref=stage))
        verified_ref = reference(REPO / "outputs/dependency_safe_matrix_closed_20261027/verified.json")
        require(read(verified_ref)["complete"] is read(verified_ref)["passed"] is True, "Full closure")
        result.update(complete=True, passed=True, producer_result_ref=owner_ref, verified_ref=verified_ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    main()
