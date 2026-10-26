"""089 fixed selected-contrast CPU fit, independent verification and validation."""

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

ROOT = REPO / "outputs/contrast_conditioned_arm_forward_20261025"
CONFIG = REPO / "configs/contrast_conditioned_arm_forward_20261025.json"
TESTS = ["tests/test_state_conditioned_arm_forward.py"]


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
        assert config["seed"] == 42 and config["ridge"] == 0.001 and config["dt"] == 0.05
        assert (config["state_dimension"], config["design_dimension"], config["output_dimension"]) == (71, 288, 12)
        assert config["std_floor"] == 1e-8
        assert config["native_weight"] == config["contrast_weight"] == 0.5
        assert config["training_anchor_groups"] == [0, 3, 5] and config["validation_anchor_groups"] == [1, 2, 4]
        assert config["contrast_equations"] == 9
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            CONFIG,
            REPO / "plans/v0.0.1/089_contrast_conditioned_arm_forward_20261025.md",
            *[REPO / p for p in TESTS],
        ]
        for key in (
            "data_closure_ref",
            "data_result_ref",
            "paired_closure_ref",
            "paired_analysis_ref",
            "prior_fit_closure_ref",
            "statistics_ref",
        ):
            ref = config[key]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
        previous_fit = json.loads(Path(config["prior_fit_closure_ref"]["path"]).read_text())
        assert (
            previous_fit["complete"]
            and previous_fit["numerically_verified"]
            and previous_fit["owner_normal_wait_verified"]
        )
        closure = json.loads(Path(config["data_closure_ref"]["path"]).read_text())
        assert (
            closure["complete"]
            and closure["data_qualified"]
            and closure["verified_trajectories"] == 30
            and closure["owner_normal_wait_verified"]
        )
        paired = json.loads(Path(config["paired_closure_ref"]["path"]).read_text())
        assert (
            paired["complete"]
            and paired["data_qualified"]
            and paired["verified_cases"] == 42
            and paired["owner_normal_wait_verified"]
        )
        for prior in (closure, paired):
            ref = prior["summary_ref"]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
            for path, signature in prior["source_sha256"].items():
                assert sha(path) == signature
        analysis = json.loads(Path(config["paired_analysis_ref"]["path"]).read_text())
        assert analysis["complete"] and analysis["verified_anchor_groups"] == 6
        data = json.loads(Path(config["data_result_ref"]["path"]).read_text())
        assert data["complete"] and data["rows_by_split"] == dict(fit=4865, holdout=1315)
        assert sum(p["split"] == "fit" for p in data["trajectories"]) == 24
        for entry in data["trajectories"]:
            ref = entry["cache_ref"]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
            meta = Path(entry["source"]["path"]).parent / "result.json"
            assert sha(meta) == closure["source_sha256"][str(meta.resolve())]
            paths.append(meta)
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS],
            [str(REPO / ".venv/bin/python"), str(REPO / "scripts/check_arm_contrast_source.py")],
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import fit_contrast_conditioned_arm_forward; import verify_contrast_conditioned_arm_fit; import validate_contrast_conditioned_arm_forward",
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
                raise RuntimeError("089 admission failed; no retry")
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
            raise TimeoutError("089 owner launch missing")
        time.sleep(0.01)
    launch = json.loads(launch_path.read_text())
    actual = identity(os.getpid())
    assert all(actual[k] == launch["child"][k] for k in ("pid", "start_ticks"))
    stages = []
    for script, directory in [
        ("fit_contrast_conditioned_arm_forward.py", "orchestration/fit"),
        ("verify_contrast_conditioned_arm_fit.py", "verification/fit"),
        ("validate_contrast_conditioned_arm_forward.py", "verification/validation"),
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
            new_fits=1,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
            scope="single fit and independent numerical/performance validation; control not qualified",
        ),
    )


if __name__ == "__main__":
    main()
