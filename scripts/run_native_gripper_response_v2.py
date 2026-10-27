"""076 baseline-gated acquisition with actual waits for workers and execution owner."""

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

ROOT = REPO / "outputs/native_gripper_response_interventions_20261025_v2"
CONFIG = REPO / "configs/native_gripper_response_interventions_20261025_v2.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise", "run"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LP_NUM_THREADS="2",
        PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
    )
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    sources_path = ROOT / "source_sha256.json"
    if args.mode == "launch":
        registration = json.loads(CONFIG.read_text())
        assert len(registration["profiles"]) == 12 and registration["physics_steps"] == 1568
        assert sum(p["physics_steps"] for p in registration["profiles"]) == 1568
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            CONFIG,
            REPO / "plans/v0.0.1/077_native_gripper_response_sensor_repair_20261025.md",
            REPO / "tests/test_native_gripper_response.py",
            REPO / "tests/test_native_gripper_sensor_repair.py",
        ]
        for p in registration["profiles"]:
            for key in ("trajectory_ref", "case_ref"):
                ref = p[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
            prior = json.loads(Path(p["case_ref"]["path"]).read_text())["prepared_ref"]
            assert sha(prior["path"]) == prior["sha256"]
            paths.append(Path(prior["path"]))
        paths += [REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json"]
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        admission = []
        commands = [
            [
                str(REPO / ".venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                "tests/test_native_gripper_response.py",
                "tests/test_native_gripper_sensor_repair.py",
            ]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_native_gripper_response_v2.py"),
                "--case-index",
                str(i),
                "--dry-run",
            ]
            for i in range(12)
        ]
        commands += [[str(REPO / ".venv/bin/python"), "-c", "import verify_native_gripper_response_v2"]]
        for i, cmd in enumerate(commands):
            p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
            log = ROOT / f"admission{i:02d}.log"
            log.write_text(p.stdout + p.stderr)
            admission.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha(log)))
            if p.returncode:
                publish(ROOT / "admission_failure.json", dict(checks=admission, physics_steps=0))
                raise RuntimeError("076 admission failed; no retry")
        check_sources(sources)
        publish(ROOT / "admission.json", dict(passed=True, checks=admission, source_sha256=sources, physics_steps=0))
        publish(sources_path, sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads(sources_path.read_text())
    check_sources(sources)
    if args.mode == "supervise":
        accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
        outcome = owned_stage(
            [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "run"],
            environment=env,
            directory=ROOT / "owner_wait",
            timeout_s=12000,
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
            raise TimeoutError("Execution owner launch missing")
        time.sleep(0.01)
    launch = json.loads(launch_path.read_text())
    actual = identity(os.getpid())
    assert all(actual[k] == launch["child"][k] for k in ("pid", "start_ticks"))
    stages = []
    # All baseline replays and independent equivalence checks precede any intervention.
    for index in (0, 4, 8, 1, 2, 3, 5, 6, 7, 9, 10, 11):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_native_gripper_response_v2.py"),
            ("verify", ".venv/bin/python", "verify_native_gripper_response_v2.py"),
        ]:
            directory = (
                ROOT / "orchestration" / f"case{index:02d}"
                if kind == "collect"
                else ROOT / "verification" / f"case{index:02d}"
            )
            result = owned_stage(
                [str(REPO / python), str(REPO / "scripts" / script), "--case-index", str(index)],
                environment=env,
                directory=directory,
                timeout_s=1800,
                source_sha256=sources,
            )
            stage = json.loads(Path(result["path"]).read_text())
            stages.append(dict(case=index, kind=kind, stage_ref=result))
            if not stage["passed"]:
                publish(
                    ROOT / "execution_result.json",
                    dict(
                        complete=False,
                        passed=False,
                        failed_case=index,
                        failed_stage=kind,
                        stages=stages,
                        automatic_retry=False,
                    ),
                )
                return
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            passed=True,
            stages=stages,
            physics_steps=1568,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            task_success_qualification=False,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
