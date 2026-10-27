"""205 fixed18-case physical qualification with owned worker/checker and owner waits."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from coupled_preview_execution_sources import prepare_sources

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

ROOT = REPO / "outputs/coupled_preview_control_20261026"
CONFIG = REPO / "configs/coupled_preview_control_registered_20261026.json"
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
    "tests/test_executed_command_residual.py",
    "tests/test_executed_residual_independent.py",
    "tests/test_full_coupled_pose_inverse.py",
    "tests/test_full_coupled_pose_inverse_sqp.py",
    "tests/test_full_pose_execution_sources_resolved.py",
    "tests/test_generated_reference_twist.py",
    "tests/test_coupled_pose_preview.py",
    "tests/test_coupled_pose_preview_independent.py",
    "tests/test_coupled_pose_preview_bounded.py",
    "tests/test_coupled_pose_preview_executable.py",
    "tests/test_coupled_pose_preview_runtime.py",
    "tests/test_current_coupled_response.py",
    "tests/test_coupled_response_history.py",
    "tests/test_coupled_rotation_goal.py",
    "tests/test_archived_forward_state.py",
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
        assert registration["arm_pose_objective"] == "four_step_world_tangent_pose_plus_dt_world_twist_squared"
        binding = json.loads(Path(registration["pose_preview_log_ref"]["path"]).read_text())
        assert binding["complete"] and binding["independent_preview_control_log"]
        assert binding["rows"] == binding["source_commit_cycles"] == 164
        assert len(binding["corruptions_rejected"]) == 22
        chain = json.loads(Path(registration["pose_preview_chain_ref"]["path"]).read_text())
        assert chain["complete"] and len(chain["results"]) == 18
        assert sum(item["periods"] for item in chain["results"]) == 4907
        assert max(item["maximum_kkt"] for item in chain["results"]) < 2e-6
        reader = json.loads(Path(registration["coupled_reader_ref"]["path"]).read_text())
        assert reader["complete"] and reader["current_dynamics_reader_verified"]
        supervised = json.loads(Path(registration["pose_preview_supervision_ref"]["path"]).read_text())
        assert supervised["complete"] and supervised["actual_owned_wait_recorded"]
        assert registration["source_provisional_inverse_weight"] == 0.5
        runtime_sources, provenance_sources, aliases = prepare_sources(registration, CONFIG, TESTS)
        sources = runtime_sources
        ROOT.mkdir(exist_ok=False)
        aliases_ref = publish(ROOT / "source_path_aliases.json", aliases)
        sources[aliases_ref["path"]] = aliases_ref["sha256"]
        provenance_ref = publish(ROOT / "provenance_source_sha256.json", provenance_sources)
        sources[provenance_ref["path"]] = provenance_ref["sha256"]
        commands = [
            [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "--basetemp", str(ROOT / "pytest_tmp"), *TESTS]
        ]
        commands += [
            [
                str(REPO / "outputs/libero_env311/bin/python"),
                str(REPO / "scripts/collect_coupled_preview_control.py"),
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
                "import verify_coupled_preview_control; import coupled_preview_acceptance_checker_bound; import coupled_preview_arm_log_replay_bound; import full_coupled_pose_independent; import qualified_inverse_independent; import qualified_input_state_audit",
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
                raise RuntimeError("232 admission failed; no retry")
        check_sources(sources)
        publish(ROOT / "admission.json", dict(passed=True, checks=admission, source_sha256=sources, physics_steps=0))
        publish(sources_path, sources)
        print(
            launch_detached(
                command, directory=ROOT / "supervisor", source_sha256=sources, environment=env, ready_timeout_s=300
            )
        )
        return
    sources = json.loads(sources_path.read_text())
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
    check_sources(sources)
    launch_path = ROOT / "owner_wait/launch.json"
    deadline = time.monotonic() + 300
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
            ("collect", "outputs/libero_env311/bin/python", "collect_coupled_preview_control.py"),
            ("verify", "outputs/libero_env311/bin/python", "verify_coupled_preview_control.py"),
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
