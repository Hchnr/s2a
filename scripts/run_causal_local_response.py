"""096 causal prequential and fixed-anchor validation with actual owned waits."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import (
    accept_detached_launch,
    check_sources,
    cpu_environment,
    identity,
    launch_detached,
    owned_stage,
    publish,
    sha,
)

ROOT = REPO / "outputs/causal_local_response_20261025/owned"
TESTS = ["tests/test_causal_local_response.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise", "run"))
    args = parser.parse_args()
    env = cpu_environment()
    env["PYTHONPATH"] = str(REPO / "src") + ":" + str(REPO / "scripts")
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    source_path = ROOT / "source_sha256.json"
    if args.mode == "launch":
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [REPO / "plans/v0.0.1/096_causal_local_response_20261025.md", *[REPO / t for t in TESTS]]
        closure_path = REPO / "outputs/history_conditioned_arm_forward_20261025/closure_verified.json"
        closure = json.loads(closure_path.read_text())
        assert closure["complete"] and closure["numerically_verified"] and closure["owner_normal_wait_verified"]
        paths.append(closure_path)
        for path, signature in closure["source_sha256"].items():
            assert sha(path) == signature
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        checks = []
        for i, cmd in enumerate(commands):
            p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
            log = ROOT / f"admission{i:02d}.log"
            log.write_text(p.stdout + p.stderr)
            checks.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha(log)))
            if p.returncode:
                publish(ROOT / "admission_failure.json", dict(checks=checks, physics_steps=0, new_fits=0))
                raise RuntimeError("096 admission failed; no retry")
        check_sources(sources)
        publish(
            ROOT / "admission.json",
            dict(passed=True, checks=checks, source_sha256=sources, physics_steps=0, new_fits=0),
        )
        publish(source_path, sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads(source_path.read_text())
    check_sources(sources)
    if args.mode == "supervise":
        accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
        outcome = owned_stage(
            [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "run"],
            environment=env,
            directory=ROOT / "owner_wait",
            timeout_s=3600,
            source_sha256=sources,
        )
        publish(
            ROOT / "supervision_result.json",
            dict(owner_stage_ref=outcome, owner_actual_wait_recorded=True, automatic_retry=False),
        )
        return
    launch_path = ROOT / "owner_wait/launch.json"
    deadline = time.monotonic() + 10
    while not launch_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("096 owner launch missing")
        time.sleep(0.01)
    launch = json.loads(launch_path.read_text())
    actual = identity(os.getpid())
    assert all(actual[k] == launch["child"][k] for k in ("pid", "start_ticks"))
    stages = []
    for stage in ("holdout", "anchors"):
        ref = owned_stage(
            [str(REPO / ".venv/bin/python"), str(REPO / "scripts/execute_causal_local_check.py"), stage],
            environment=env,
            directory=ROOT / "stages" / stage,
            timeout_s=1800,
            source_sha256=sources,
        )
        stages.append(dict(stage=stage, stage_ref=ref))
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            passed=True,
            stages=stages,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="causal local holdout and anchor validation; no control qualification",
        ),
    )


if __name__ == "__main__":
    main()
