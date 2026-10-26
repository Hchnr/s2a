"""087 owned CPU reconstruction and independent verification, no simulation or fit."""

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

ROOT = REPO / "outputs/native_arm_response_data_20261025"
CONFIG = REPO / "configs/native_arm_response_data_20261025.json"
TESTS = ["tests/test_xml_site_kinematics.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise", "run"))
    args = parser.parse_args()
    env = cpu_environment()
    env["PYTHONPATH"] = str(REPO / "src") + ":" + str(REPO / "scripts")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    source_path = ROOT / "source_sha256.json"
    if args.mode == "launch":
        config = json.loads(CONFIG.read_text())
        assert config["seed"] == 42
        assert len(config["trajectories"]) == 30
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [CONFIG, REPO / "plans/v0.0.1/087_native_arm_response_data_20261025.md", *[REPO / p for p in TESTS]]
        for key in ("roster_ref", "prior_closure_ref", "paired_analysis_ref"):
            ref = config[key]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
        closure = json.loads(Path(config["prior_closure_ref"]["path"]).read_text())
        assert (
            closure["complete"]
            and closure["data_qualified"]
            and closure["verified_cases"] == 42
            and closure["owner_normal_wait_verified"]
        )
        ref = closure["summary_ref"]
        assert sha(ref["path"]) == ref["sha256"]
        paths.append(Path(ref["path"]))
        roster = json.loads(Path(config["roster_ref"]["path"]).read_text())["trajectories"]
        assert [{k: p[k] for k in ("task", "split", "ref", "loaded_steps")} for p in config["trajectories"]] == roster
        assert sum(p["split"] == "fit" for p in roster) == 24 and sum(p["split"] == "holdout" for p in roster) == 6
        for entry in config["trajectories"]:
            for key in ("ref", "result_ref", "goal_log_ref"):
                ref = entry[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS],
            [
                str(REPO / ".venv/bin/python"),
                str(REPO / "scripts/reconstruct_native_arm_response_data.py"),
                "--check-index",
                "0",
            ],
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import reconstruct_native_arm_response_data; import verify_native_arm_response_data",
            ],
        ]
        checks = []
        for i, cmd in enumerate(commands):
            p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
            log = ROOT / f"admission{i:02d}.log"
            log.write_text(p.stdout + p.stderr)
            checks.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha(log)))
            if p.returncode:
                publish(ROOT / "admission_failure.json", dict(checks=checks, physics_steps=0, new_fits=0))
                raise RuntimeError("087 admission failed; no retry")
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
            raise TimeoutError("087 owner launch missing")
        time.sleep(0.01)
    launch = json.loads(launch_path.read_text())
    actual = identity(os.getpid())
    assert all(actual[k] == launch["child"][k] for k in ("pid", "start_ticks"))
    stages = []
    for script, directory in [
        ("reconstruct_native_arm_response_data.py", "orchestration/reconstruct"),
        ("verify_native_arm_response_data.py", "verification/data"),
    ]:
        ref = owned_stage(
            [str(REPO / ".venv/bin/python"), str(REPO / "scripts" / script)],
            environment=env,
            directory=ROOT / directory,
            timeout_s=1800,
            source_sha256=sources,
        )
        stages.append(dict(script=script, stage_ref=ref))
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
            scope="data reconstruction and independent checks only",
        ),
    )


if __name__ == "__main__":
    main()
