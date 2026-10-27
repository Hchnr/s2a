"""205 fixed18-case physical qualification with owned worker/checker and owner waits."""

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

ROOT = REPO / "outputs/late_clipped_inverse_control_20261026"
CONFIG = REPO / "configs/late_clipped_inverse_control_registered_20261026.json"
TESTS = [
    "tests/test_velocity_model_planner.py",
    "tests/test_velocity_residual_response.py",
    "tests/test_velocity_model_control_reuse.py",
    "tests/test_velocity_model_checker_reuse.py",
    "tests/test_loaded_velocity_control.py",
    "tests/test_loaded_velocity_control_reuse.py",
    "tests/test_loaded_velocity_replay.py",
    "tests/test_loaded_velocity_checker_reuse.py",
    "tests/test_pose_velocity_inverse.py",
    "tests/test_pose_velocity_control_reuse.py",
    "tests/test_pose_velocity_arm_replay.py",
    "tests/test_pose_velocity_checker_reuse.py",
    "tests/test_causal_arm_residual.py",
    "tests/test_causal_arm_replay.py",
    "tests/test_causal_residual_scope.py",
    "tests/test_qualified_forward_inverse.py",
    "tests/test_qualified_inverse_independent.py",
    "tests/test_current_model_state.py",
    "tests/test_runtime_response_history.py",
    "tests/test_bounded_response_inverse.py",
    "tests/test_native_precision_goal.py",
    "tests/test_enumerated_response_inverse.py",
    "tests/test_qualified_enumerated_inverse.py",
    "tests/test_training_support.py",
    "tests/test_support_aware_inverse.py",
    "tests/test_residual_supported_inverse.py",
    "tests/test_loading_phase_inverse.py",
    "tests/test_rate_loading_inverse.py",
    "tests/test_tracking_guarded_rate.py",
    "tests/test_guarded_loading_inverse.py",
    "tests/test_direct_arm_inverse_inputs.py",
    "tests/test_direct_arm_inverse.py",
    "tests/test_direct_arm_inverse_runtime.py",
    "tests/test_late_clipped_arm_inverse.py",
    "tests/test_executed_command_residual.py",
    "tests/test_executed_residual_independent.py",
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
        assert not ROOT.exists()
        registration = json.loads(CONFIG.read_text())
        prior_config = json.loads((REPO / "configs/pose_velocity_loaded_control_20261025.json").read_text())
        assert registration["profiles"] == prior_config["profiles"]
        assert registration["seed"] == 42 and registration["physics_steps"] == 4907
        assert len(registration["profiles"]) == 18
        assert sum(p["physics_steps"] for p in registration["profiles"]) == 4907
        assert (
            registration["arm_translation_objective"]
            == "late_clipped_direct_translation_conditioned_on_desired_SITE_pose"
        )
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [CONFIG, REPO / "plans/v0.0.1/216_late_clipped_direct_inverse_20261026.md", *[REPO / p for p in TESTS]]
        for key, ref in registration.items():
            if not key.endswith("_ref") or not isinstance(ref, dict) or "sha256" not in ref:
                continue
            assert sha(ref["path"]) == ref["sha256"], key
            paths.append(Path(ref["path"]))
            artifact = json.loads(Path(ref["path"]).read_text())
            for path, signature in artifact.get("source_sha256", {}).items():
                assert sha(path) == signature
                paths.append(Path(path))
        for profile in registration["profiles"]:
            for key in ("trajectory_ref", "case_ref"):
                ref = profile[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
        binding = json.loads(Path(registration["direct_binding_ref"]["path"]).read_text())
        assert binding["complete"] and binding["independent_command_and_history"] and binding["rows"] == 164
        assert len(binding["corruptions_rejected"]) == 8
        runtime = json.loads(Path(registration["direct_inverse_runtime_ref"]["path"]).read_text())
        assert runtime["complete"] and runtime["rows"] == 6180 and runtime["independent_command_verified"]
        assert runtime["parameter_ref"] == registration["direct_inverse_parameter_ref"]
        feedback = json.loads(Path(registration["feedback_monotonicity_ref"]["path"]).read_text())
        assert feedback["complete"] and feedback["parameter_ref"] == registration["direct_inverse_parameter_ref"]
        assert feedback["late_clip_preprojection_symmetric_jacobian_lower"] > 0
        assert registration["arm_rotation_inverse_weight"] == 0.5
        prior_closure = REPO / "outputs/qualified_inverse_guarded_loading_20261025/closure_verified.json"
        prior = json.loads(prior_closure.read_text())
        assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
        check_sources(prior["source_sha256"])
        paths += [prior_closure, *map(Path, prior["source_sha256"])]
        sources = {str(p.resolve()): sha(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_late_clipped_inverse_control.py"),
                "--case-index",
                str(i),
                "--dry-run",
            ]
            for i in range(18)
        ]
        commands += [
            [
                str(REPO / ".venv/bin/python"),
                "-c",
                "import verify_late_clipped_inverse_control; import late_clipped_inverse_acceptance_checker; import late_clipped_inverse_arm_log_replay; import qualified_inverse_independent; import qualified_input_state_audit",
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
                raise RuntimeError("205 admission failed; no retry")
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
    for index in range(18):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_late_clipped_inverse_control.py"),
            ("verify", ".venv/bin/python", "verify_late_clipped_inverse_control.py"),
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
            physics_steps=4907,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
