"""084 three-case offline native-gripper ablation physical qualification with owned worker/checker and owner waits."""

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

ROOT = REPO / "outputs/native_gripper_ablation_20261025"
CONFIG = REPO / "configs/native_gripper_ablation_20261025.json"
TESTS = [
    "tests/test_pose_velocity_inverse.py",
    "tests/test_pose_velocity_arm_replay.py",
    "tests/test_native_gripper_ablation_scope.py",
]


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
        prior_ref = registration["prior_closure_ref"]
        prior_closure = Path(prior_ref["path"])
        assert sha(prior_closure) == prior_ref["sha256"]
        prior = json.loads(prior_closure.read_text())
        assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
        prior_summary = Path(prior["summary_ref"]["path"])
        assert sha(prior_summary) == prior["summary_ref"]["sha256"]
        prior_config = REPO / "configs/pose_velocity_loaded_control_20261025.json"
        previous_registration = json.loads(prior_config.read_text())
        assert [p["prior_case_index"] for p in registration["profiles"]] == [3, 10, 12]
        for entry in registration["profiles"]:
            assert entry["profile"] == previous_registration["profiles"][entry["prior_case_index"]]
            assert entry["native_command_field"] == "sent_commands" and entry["gripper_channel"] == -1
        assert registration["seed"] == 42 and registration["physics_steps"] == 566
        assert sum(p["profile"]["physics_steps"] for p in registration["profiles"]) == 566
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            prior_config,
            prior_closure,
            prior_summary,
            CONFIG,
            REPO / "plans/v0.0.1/084_native_gripper_ablation_20261025.md",
            *[REPO / p for p in TESTS],
        ]
        for entry in registration["profiles"]:
            profile = entry["profile"]
            for key in ("trajectory_ref", "case_ref"):
                ref = profile[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
        paths += [
            REPO / "configs/matched_oracle_requalification_20261025.json",
            REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json",
        ]
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_native_gripper_ablation.py"),
                "--case-index",
                str(i),
                "--dry-run",
            ]
            for i in range(3)
        ]
        commands += [
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import verify_native_gripper_ablation; import native_gripper_ablation_acceptance_checker; import pose_velocity_arm_independent_replay",
            ]
        ]
        admission = []
        for i, cmd in enumerate(commands):
            p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
            log = ROOT / f"admission{i:02d}.log"
            log.write_text(p.stdout + p.stderr)
            admission.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha(log)))
            if p.returncode:
                publish(ROOT / "admission_failure.json", dict(checks=admission, physics_steps=0))
                raise RuntimeError("084 admission failed; no retry")
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
    for index in range(3):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_native_gripper_ablation.py"),
            ("verify", ".venv/bin/python", "verify_native_gripper_ablation.py"),
        ]:
            directory = ROOT / ("orchestration" if kind == "collect" else "verification") / f"case{index:02d}"
            result = owned_stage(
                [str(REPO / python), str(REPO / "scripts" / script), "--case-index", str(index)],
                environment=env,
                directory=directory,
                timeout_s=1800,
                source_sha256=sources,
            )
            stages.append(dict(case=index, kind=kind, stage_ref=result))
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            passed=True,
            scope="execution/independent mechanical checks; performance per-case and aggregate separate",
            stages=stages,
            physics_steps=566,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
