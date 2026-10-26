"""097 fixed147 training-prefix paired arm response collection with owned worker/checker and owner waits."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

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

ROOT = REPO / "outputs/training_response_coverage_20261025"
CONFIG = REPO / "configs/training_response_coverage_20261025.json"
TESTS = [
    "tests/test_paired_arm_response.py",
    "tests/test_xml_site_kinematics.py",
    "tests/test_native_gripper_response.py",
    "tests/test_native_gripper_sensor_repair.py",
    "tests/test_native_gripper_metadata.py",
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
        assert registration["seed"] == 42 and registration["physics_steps"] == 8281
        assert len(registration["profiles"]) == 21 and len(registration["trials"]) == 147
        assert registration["all_new_data_split"] == "fit"
        for path, signature in registration["source_sha256"].items():
            assert sha(path) == signature
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [CONFIG, REPO / "plans/v0.0.1/097_training_response_coverage_20261025.md", *[REPO / p for p in TESTS]]
        ref = registration["native_data_result_ref"]
        assert sha(ref["path"]) == ref["sha256"]
        source = json.loads(Path(ref["path"]).read_text())
        fit = {r["index"]: r for r in source["trajectories"] if r["split"] == "fit"}
        assert len(fit) == 24
        indices = []
        for group, profile in enumerate(registration["profiles"]):
            native = fit[profile["native_data_index"]]
            indices.append(native["index"])
            assert profile["trajectory_ref"] == native["source"]
            assert profile["trajectory_ref"]["sha256"] not in registration["excluded_086_trajectory_sha256"]
            for key in ("trajectory_ref", "case_ref"):
                ref = profile[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
            with np.load(profile["trajectory_ref"]["path"], allow_pickle=False) as data:
                close = int(np.flatnonzero(data["sent_commands"][:, -1] > 0)[0])
            for branch in range(7):
                trial = registration["trials"][group * 7 + branch]
                assert trial["profile_index"] == trial["anchor_group"] == group
                assert trial["anchor_cycle"] == close and trial["physics_steps"] == close + 1
                assert trial["translation_axis"] == (None if branch == 0 else (branch - 1) // 2)
                assert trial["normalized_command_offset"] == (0 if branch == 0 else (-0.1 if branch % 2 else 0.1))
        assert len(set(indices)) == 21
        assert [sum(p["task"] == i for p in registration["profiles"]) for i in range(3)] == [7, 7, 7]
        assert sum(t["physics_steps"] for t in registration["trials"]) == 8281
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_training_response_coverage.py"),
                "--case-index",
                str(i),
                "--dry-run",
            ]
            for i in range(147)
        ]
        commands += [
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import verify_training_response_coverage; import s2a.xml_site_kinematics",
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
                raise RuntimeError("097 admission failed; no retry")
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
    for index in range(147):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_training_response_coverage.py"),
            ("verify", ".venv/bin/python", "verify_training_response_coverage.py"),
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
            scope="native-prefix paired arm response data only; not task qualification",
            stages=stages,
            physics_steps=8281,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
            task_success_qualification=False,
        ),
    )


if __name__ == "__main__":
    main()
