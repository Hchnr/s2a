"""168 fixed18-case physical qualification with owned worker/checker and owner waits."""

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

ROOT = REPO / "outputs/qualified_inverse_supported_20261025"
CONFIG = REPO / "configs/qualified_inverse_supported_20261025.json"
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
        solver_ref = registration["enumerated_solver_ref"]
        assert sha(solver_ref["path"]) == solver_ref["sha256"]
        solver_check = json.loads(Path(solver_ref["path"]).read_text())
        assert solver_check["checked"] and [v["checked_rows"] for v in solver_check["results"]] == [164, 90]
        for source_path, signature in solver_check["source_sha256"].items():
            assert sha(source_path) == signature
        prior_closure = Path(registration["prior_execution_closure_path"])
        prior = json.loads(prior_closure.read_text())
        assert prior["complete"] and prior["verified_cases"] == 18 and prior["owner_normal_wait_verified"]
        prior_summary = Path(prior["summary_ref"]["path"])
        assert sha(prior_summary) == prior["summary_ref"]["sha256"]
        prior_config = REPO / "configs/pose_velocity_loaded_control_20261025.json"
        previous_registration = json.loads(prior_config.read_text())
        assert registration["profiles"] == previous_registration["profiles"]
        assert registration["seed"] == 42
        assert registration["arm_translation_objective"] == "position_squared_plus_dt_squared_velocity_squared"
        assert registration["arm_velocity_reference"] == "generated_SITE_central_anchor_and_backward_tail"
        assert registration["arm_rotation_inverse_weight"] == 0.5
        assert len(registration["profiles"]) == 18 and registration["physics_steps"] == 4907
        assert sum(p["physics_steps"] for p in registration["profiles"]) == 4907
        assert (
            registration["arm_residual_policy"]
            == "150 completed12 innovation supplied to frozen forward;no extra online fitted correction"
        )
        ablation_ref = registration["ablation_closure_ref"]
        assert sha(ablation_ref["path"]) == ablation_ref["sha256"]
        ablation = json.loads(Path(ablation_ref["path"]).read_text())
        assert ablation["complete"] and ablation["verified_cases"] == 3 and ablation["owner_normal_wait_verified"]
        assert sha(ablation["summary_ref"]["path"]) == ablation["summary_ref"]["sha256"]
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        support_ref = registration["arm_support_calibration_ref"]
        assert sha(support_ref["path"]) == support_ref["sha256"]
        calibration = json.loads(Path(support_ref["path"]).read_text())
        assert calibration["independent_dense_verified"] and calibration["evaluation_rows_used_for_calibration"] == 0
        assert calibration["parameter_ref"] == registration["arm_forward_parameter_ref"]
        points_ref = calibration["arrays_ref"]
        assert sha(points_ref["path"]) == points_ref["sha256"]
        paths += [Path(support_ref["path"]), Path(points_ref["path"])]
        for source_path, signature in calibration["source_sha256"].items():
            assert sha(source_path) == signature
            paths.append(Path(source_path))

        paths += [
            Path(ablation_ref["path"]),
            Path(ablation["summary_ref"]["path"]),
            prior_config,
            prior_closure,
            prior_summary,
            CONFIG,
            Path(registration["enumerated_solver_ref"]["path"]),
            REPO / "plans/v0.0.1/168_supported_control_execution_20261025.md",
            *[REPO / p for p in TESTS],
        ]
        for profile in registration["profiles"]:
            for key in ("trajectory_ref", "case_ref"):
                ref = profile[key]
                assert sha(ref["path"]) == ref["sha256"]
                paths.append(Path(ref["path"]))
        for key in (
            "response_parameters_ref",
            "planner_preflight_ref",
            "planner_independent_ref",
            "response_qualification_ref",
        ):
            ref = registration[key]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
            if key != "response_parameters_ref":
                artifact = json.loads(Path(ref["path"]).read_text())
                if key in ("planner_independent_ref", "response_qualification_ref"):
                    assert artifact["numerically_verified"]
                if key == "response_qualification_ref":
                    assert artifact["offline_performance_eligible"]
                if key == "planner_preflight_ref":
                    assert artifact["velocity_envelope"] == registration["gripper_velocity_envelope"]
                for path, signature in artifact["source_sha256"].items():
                    assert sha(path) == signature
                    paths.append(Path(path))
        for key in (
            "arm_forward_parameter_ref",
            "arm_forward_qualification_ref",
            "runtime_prefix_parity_ref",
            "native_precision_ref",
        ):
            ref = registration[key]
            assert sha(ref["path"]) == ref["sha256"]
            paths.append(Path(ref["path"]))
        precision = json.loads(Path(registration["native_precision_ref"]["path"]).read_text())
        assert precision["complete"] and precision["periods"] == 164 and precision["maximum_error"] <= 1e-9
        for path, signature in precision["source_sha256"].items():
            assert sha(path) == signature
            paths.append(Path(path))
        qualified = json.loads(Path(registration["arm_forward_qualification_ref"]["path"]).read_text())
        assert qualified["complete"] and qualified["offline_forward_eligible"]
        assert qualified["parameter_ref"] == registration["arm_forward_parameter_ref"]
        for path, signature in qualified["source_sha256"].items():
            assert sha(path) == signature
            paths.append(Path(path))
        parity = json.loads(Path(registration["runtime_prefix_parity_ref"]["path"]).read_text())
        assert parity["runtime_input_parity"] and parity["completed_history_periods"] == 48
        checker_path = REPO / "outputs/qualified_control_checker_admission_20261025/state_binding_verified.json"
        paths.append(checker_path)
        checker = json.loads(checker_path.read_text())
        assert checker["complete"] and checker["corruption_rejected"]
        for path, signature in checker["source_sha256"].items():
            assert sha(path) == signature
            paths.append(Path(path))
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
                str(REPO / "scripts/collect_qualified_inverse_supported.py"),
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
                "import verify_qualified_inverse_supported; import supported_acceptance_checker; import supported_arm_log_replay; import qualified_inverse_independent; import qualified_input_state_audit",
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
                raise RuntimeError("168 admission failed; no retry")
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
            ("collect", "outputs/libero_env311/bin/python", "collect_qualified_inverse_supported.py"),
            ("verify", ".venv/bin/python", "verify_qualified_inverse_supported.py"),
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
