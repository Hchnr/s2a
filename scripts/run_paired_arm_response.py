"""086 fixed42 native-prefix paired arm response collection with owned worker/checker and owner waits."""

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

ROOT = REPO / "outputs/paired_arm_response_20261025"
CONFIG = REPO / "configs/paired_arm_response_20261025.json"
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
        prior_closure = Path(registration["prior_closure_path"])
        prior = json.loads(prior_closure.read_text())
        assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
        prior_summary = Path(prior["summary_ref"]["path"])
        assert sha(prior_summary) == prior["summary_ref"]["sha256"]
        source_config = Path(registration["source_ref"]["path"])
        assert sha(source_config) == registration["source_ref"]["sha256"]
        previous = json.loads(source_config.read_text())
        fit_roster = Path(registration["fit_roster_ref"]["path"])
        assert sha(fit_roster) == registration["fit_roster_ref"]["sha256"]
        fit = {p["ref"]["sha256"] for p in json.loads(fit_roster.read_text())["trajectories"] if p["split"] == "fit"}
        assert registration["seed"] == 42 and registration["physics_steps"] == 3339
        assert len(registration["trials"]) == 42 and len(registration["profiles"]) == 3
        for pi, profile in enumerate(registration["profiles"]):
            source_profile = previous["profiles"][4 * pi]
            assert profile == {k: source_profile[k] for k in ("task", "record", "trajectory_ref", "case_ref")}
            assert profile["task"] == pi and profile["trajectory_ref"]["sha256"] in fit
            with np.load(profile["trajectory_ref"]["path"], allow_pickle=False) as data:
                u = data["sent_commands"][:, -1]
                close = int(np.flatnonzero(u > 0)[0])
                opening = int(np.flatnonzero((u < 0) & (np.arange(len(u)) > close))[0])
            for j in range(14):
                index = 14 * pi + j
                trial = registration["trials"][index]
                local = j % 7
                assert trial["profile_index"] == pi and trial["anchor_group"] == index // 7
                assert trial["anchor_cycle"] == (close if j < 7 else opening)
                assert trial["phase"] == ("first_close" if j < 7 else "first_open_after_close")
                assert trial["physics_steps"] == trial["anchor_cycle"] + 1
                if local == 0:
                    assert (
                        trial["branch"] == "baseline"
                        and trial["translation_axis"] is None
                        and trial["normalized_command_offset"] == 0
                    )
                else:
                    axis = (local - 1) // 2
                    sign = -1 if local % 2 else 1
                    assert trial["translation_axis"] == axis and trial["normalized_command_offset"] == sign * 0.1
                    assert trial["branch"] == str(axis) + ("_minus" if sign < 0 else "_plus")
        assert sum(p["physics_steps"] for p in registration["trials"]) == 3339
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            source_config,
            fit_roster,
            prior_closure,
            prior_summary,
            CONFIG,
            REPO / "plans/v0.0.1/086_paired_arm_response_20261025.md",
            *[REPO / p for p in TESTS],
        ]
        for entry in registration["profiles"]:
            profile = entry
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
                str(REPO / "scripts/collect_paired_arm_response.py"),
                "--case-index",
                str(i),
                "--dry-run",
            ]
            for i in range(42)
        ]
        commands += [
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import verify_paired_arm_response; import s2a.xml_site_kinematics",
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
                raise RuntimeError("086 admission failed; no retry")
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
    for index in range(42):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_paired_arm_response.py"),
            ("verify", ".venv/bin/python", "verify_paired_arm_response.py"),
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
            physics_steps=3339,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
            task_success_qualification=False,
        ),
    )


if __name__ == "__main__":
    main()
