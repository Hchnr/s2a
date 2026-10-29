"""Own a registered paired fixed-reference closure intervention probe."""

import argparse
import json
import signal
from pathlib import Path

from run_tracking_preview_wam_development_v5 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import accept_detached_launch, launch_detached


def main():
    root = REPO / "outputs/tracking_v5_endpoint_velocity_audit_execution_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    prior_ref = reference(REPO / "outputs/tracking_preview_wam_development_v5_20261028/registration.json")
    prior = read(prior_ref)
    prepared = read(prior["prepared_ref"])
    sources = dict(prior["source_sha256"])
    for path in [
        Path(__file__),
        REPO / "scripts/audit_tracking_v5_endpoint_velocity.py",
        REPO / "src/s2a/velocity_braking_joint_plan.py",
        REPO / "src/s2a/velocity_braking_tracking_hybrid.py",
        REPO / "src/s2a/velocity_braking_tracking_native.py",
        REPO / "scripts/probe_tracking_v5_velocity_braking.py",
    ]:
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    for path, digest in sources.items():
        checked(dict(path=path, sha256=digest))
    reg = publish(
        root / "registration.json",
        dict(
            source_sha256=sources,
            prior_registration_ref=prior_ref,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        command = owned_command(
            [str(REPO / "outputs/libero_env311/bin/python"), "scripts/audit_tracking_v5_endpoint_velocity.py"],
            cpu_environment(prepared),
            root / "capture",
            timeout=1800,
        )
        ref = reference(REPO / "outputs/tracking_v5_endpoint_velocity_audit_20261028/verified.json")
        require(read(ref)["complete"] is read(ref)["passed"] is True, "All actually executed causal-history records")
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(complete=True, passed=True, owned_command=command, result_ref=ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    command = [str(REPO / ".venv/bin/python"), "scripts/execute_tracking_v5_endpoint_velocity.py"]
    directory = REPO / "outputs/tracking_v5_endpoint_velocity_audit_supervisor_20261028"
    sources = {}
    for path in (
        Path(__file__),
        REPO / "scripts/audit_tracking_v5_endpoint_velocity.py",
        REPO / "src/s2a/velocity_braking_joint_plan.py",
        REPO / "src/s2a/velocity_braking_tracking_hybrid.py",
        REPO / "src/s2a/velocity_braking_tracking_native.py",
    ):
        ref = reference(path)
        sources[ref["path"]] = ref["sha256"]
    if args.launch:
        print(json.dumps(launch_detached(command, directory=directory, source_sha256=sources)), flush=True)
    else:
        accept_detached_launch(directory, command=command, source_sha256=sources)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(f"owner signal{signum}")))
        main()
