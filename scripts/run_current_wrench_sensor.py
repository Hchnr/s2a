"""102 six fixed native-prefix controller trace replays with actual owned waits."""

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

ROOT = REPO / "outputs/current_wrench_sensor_20261025"
TESTS = ["tests/test_controller_trace.py", "tests/test_current_wrench.py"]


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
        paths += [REPO / "plans/v0.0.1/102_current_wrench_sensor_contract_20261025.md", *[REPO / t for t in TESTS]]
        closure_path = REPO / "outputs/paired_arm_response_20261025/closure_verified.json"
        closure = json.loads(closure_path.read_text())
        assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
        paths.append(closure_path)
        for path, signature in closure["source_sha256"].items():
            assert sha(path) == signature
        for group in range(6):
            path = REPO / "outputs/paired_arm_response_20261025" / f"case{7 * group:02d}/collection_result.json"
            assert sha(path) == closure["source_sha256"][str(path.resolve())]
            paths.append(path)
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_current_wrench_sensor.py"),
                "--group",
                str(g),
                "--dry-run",
            ]
            for g in range(6)
        ]
        checks = []
        for i, cmd in enumerate(commands):
            p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
            log = ROOT / f"admission{i:02d}.log"
            log.write_text(p.stdout + p.stderr)
            checks.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha(log)))
            if p.returncode:
                publish(ROOT / "admission_failure.json", dict(checks=checks, physics_steps=0, new_fits=0))
                raise RuntimeError("102 admission failed; no retry")
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
            raise TimeoutError("102 owner launch missing")
        time.sleep(0.01)
    launch = json.loads(launch_path.read_text())
    actual = identity(os.getpid())
    assert all(actual[k] == launch["child"][k] for k in ("pid", "start_ticks"))
    stages = []
    for group in range(6):
        for script, directory, interpreter in [
            (
                "collect_current_wrench_sensor.py",
                f"orchestration/group{group:02d}",
                "outputs/libero_env311/bin/python",
            ),
            ("verify_current_wrench_sensor.py", f"verification/group{group:02d}", ".venv/bin/python"),
        ]:
            ref = owned_stage(
                [str(REPO / interpreter), str(REPO / "scripts" / script), "--group", str(group)],
                environment=env,
                directory=ROOT / directory,
                timeout_s=1800,
                source_sha256=sources,
            )
            stages.append(dict(group=group, script=script, stage_ref=ref))
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            passed=True,
            stages=stages,
            physics_steps=477,
            new_fits=0,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="six exact baseline replays with inner-controller observation; control not qualified",
        ),
    )


if __name__ == "__main__":
    main()
