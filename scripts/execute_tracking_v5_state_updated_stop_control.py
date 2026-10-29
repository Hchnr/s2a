"""Own the bounded CPU response diagnostic or its separate independent audit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from run_tracking_preview_wam_development_v5 import cpu_environment
from run_wam_readout import owned_command

from s2a.artifacts import REPO
from s2a.clean_training_runtime import checked, publish, read, reference, require


def main(stage):
    suffix = "" if stage == "diagnose" else "_check"
    root = REPO / f"outputs/tracking_v5_state_updated_stop_control{suffix}_execution_20261028"
    output = REPO / f"outputs/tracking_v5_state_updated_stop_control{suffix}_20261028/verified.json"
    script = "probe" if stage == "diagnose" else "check"
    entry = REPO / f"scripts/{script}_tracking_v5_state_updated_stop_control.py"
    require(not root.exists() and not output.parent.exists(), "No overwrite or automatic retry")
    prior_ref = reference(REPO / "outputs/tracking_preview_wam_development_v5_20261028/registration.json")
    prior = read(prior_ref)
    prepared = read(prior["prepared_ref"])
    sources = dict(prior["source_sha256"])
    migrations = []
    for name in ("test_native_training_continuation_preparation.py", "test_native_training_runner.py"):
        relative = "tests/" + name
        path = REPO / relative
        before = subprocess.check_output(["git", "show", "9db6efa:" + relative], cwd=REPO)
        after = subprocess.check_output(["git", "show", "2f09739:" + relative], cwd=REPO)
        require(hashlib.sha256(before).hexdigest() == sources[str(path)], "Exact frozen fixture baseline")
        current = reference(path)
        require(hashlib.sha256(after).hexdigest() == current["sha256"], "Exact previously committed fixture repair")
        migrations.append(
            dict(
                path=str(path),
                before_sha256=sources[str(path)],
                after_ref=current,
                before_commit="9db6efa",
                after_commit="2f09739",
            )
        )
        sources[str(path)] = current["sha256"]
    for path in [
        Path(__file__),
        entry,
        *[
            REPO / p
            for p in (
                "src/s2a/nonlinear_joint_response.py",
                "src/s2a/native_goal_joint_response.py",
                "src/s2a/nonlinear_single_period_plan.py",
                "src/s2a/causal_inner_error.py",
                "src/s2a/joint_preview_constraint_audit.py",
                "src/s2a/mujoco_robot_dynamics.py",
                "src/s2a/robot_site_kinematics.py",
                "src/s2a/xml_site_kinematics.py",
                "tests/test_nonlinear_joint_response.py",
                "tests/test_native_goal_joint_response.py",
                "tests/test_nonlinear_single_period_plan.py",
                "plans/v0.0.1/state_updated_stop_control_20261028.md",
                "plans/v0.0.1/state_updated_joint_response_20261028.md",
                "reports/state_updated_joint_response_source_audit_20261028.md",
            )
        ],
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
            stage=stage,
            physics_steps=0,
            world_model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )
    result = dict(complete=False, passed=False, registration_ref=reg)
    try:
        command = owned_command(
            [str(REPO / "outputs/libero_env311/bin/python"), str(entry)],
            cpu_environment(prepared),
            root / "capture",
            timeout=600,
        )
        ref = reference(output)
        require(read(ref)["complete"] is read(ref)["passed"] is True, "Bounded diagnostic/audit completed")
        for path, digest in sources.items():
            checked(dict(path=path, sha256=digest))
        result.update(complete=True, passed=True, owned_command=command, result_ref=ref)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        publish(root / "result.json", result)
    print(json.dumps(dict(complete=True, passed=True, stage=stage)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("diagnose", "check"))
    main(parser.parse_args().stage)
