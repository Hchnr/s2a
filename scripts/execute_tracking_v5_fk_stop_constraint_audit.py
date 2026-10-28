"""Own a CPU-only stopped-state constraint audit and two H1 diagnostic searches."""

import argparse
import hashlib
import json
import signal
import subprocess
from pathlib import Path

from run_tracking_preview_wam_development_v5 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require
from s2a.durable_stage_supervisor import accept_detached_launch, launch_detached


def main():
    root = REPO / "outputs/tracking_v5_fk_stop_constraint_audit_execution_20261028"
    require(not root.exists(), "No overwrite or automatic retry")
    prior_ref = reference(REPO / "outputs/tracking_preview_wam_development_v5_20261028/registration.json")
    prior = read(prior_ref)
    prepared = read(prior["prepared_ref"])
    sources = dict(prior["source_sha256"])
    migrations = []
    # Explicitly admit only the two committed fixture fixes; runtime bytes stay frozen.
    for name in ("test_native_training_continuation_preparation.py", "test_native_training_runner.py"):
        relative = "tests/" + name
        path = REPO / relative
        before = subprocess.check_output(["git", "show", "9db6efa:" + relative], cwd=REPO)
        after = subprocess.check_output(["git", "show", "2f09739:" + relative], cwd=REPO)
        require(hashlib.sha256(before).hexdigest() == sources[str(path)], "Exact frozen test baseline")
        current = reference(path)
        require(hashlib.sha256(after).hexdigest() == current["sha256"], "Exact committed fixture correction")
        migrations.append(
            dict(
                path=str(path),
                before_sha256=sources[str(path)],
                after_ref=current,
                before_commit="9db6efa",
                after_commit="2f09739",
                scope="CPU fixture corrections only; no frozen runtime change",
            )
        )
        sources[str(path)] = current["sha256"]
    for path in [
        Path(__file__),
        REPO / "scripts/probe_tracking_v5_fk_stop_constraint_audit.py",
        REPO / "src/s2a/velocity_braking_joint_plan.py",
        REPO / "src/s2a/past_error_joint_plan.py",
        REPO / "src/s2a/causal_joint_error.py",
        REPO / "src/s2a/causal_inner_error.py",
        REPO / "src/s2a/past_inner_error_joint_plan.py",
        REPO / "src/s2a/quality_capped_inner_joint_plan.py",
        REPO / "src/s2a/feasibility_first_joint_plan.py",
        REPO / "src/s2a/trajectory_capped_joint_plan.py",
        REPO / "src/s2a/long_trajectory_capped_joint_plan.py",
        REPO / "src/s2a/fk_pose_cap_scope_joint_plan.py",
        REPO / "src/s2a/joint_preview_constraint_audit.py",
        REPO / "src/s2a/fk_consistent_joint_preview.py",
        REPO / "src/s2a/robot_site_kinematics.py",
        REPO / "src/s2a/xml_site_kinematics.py",
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
            explicit_test_source_migrations=migrations,
            physics_budget=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        command = owned_command(
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                "scripts/probe_tracking_v5_fk_stop_constraint_audit.py",
            ],
            cpu_environment(prepared),
            root / "capture",
            timeout=1800,
        )
        ref = reference(REPO / "outputs/tracking_v5_fk_stop_constraint_audit_20261028/verified.json")
        require(
            read(ref)["complete"] is read(ref)["passed"] is True,
            "Saved H8 reconstruction and both H1 diagnostics closed",
        )
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
    command = [str(REPO / ".venv/bin/python"), "scripts/execute_tracking_v5_fk_stop_constraint_audit.py"]
    directory = REPO / "outputs/tracking_v5_fk_stop_constraint_audit_supervisor_20261028"
    sources = {}
    for path in (
        Path(__file__),
        REPO / "scripts/probe_tracking_v5_fk_stop_constraint_audit.py",
        REPO / "src/s2a/velocity_braking_joint_plan.py",
        REPO / "src/s2a/past_error_joint_plan.py",
        REPO / "src/s2a/causal_joint_error.py",
        REPO / "src/s2a/causal_inner_error.py",
        REPO / "src/s2a/past_inner_error_joint_plan.py",
        REPO / "src/s2a/quality_capped_inner_joint_plan.py",
        REPO / "src/s2a/feasibility_first_joint_plan.py",
        REPO / "src/s2a/trajectory_capped_joint_plan.py",
        REPO / "src/s2a/long_trajectory_capped_joint_plan.py",
        REPO / "src/s2a/fk_pose_cap_scope_joint_plan.py",
        REPO / "src/s2a/joint_preview_constraint_audit.py",
        REPO / "src/s2a/fk_consistent_joint_preview.py",
        REPO / "src/s2a/robot_site_kinematics.py",
        REPO / "src/s2a/xml_site_kinematics.py",
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
