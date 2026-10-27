"""205 fixed18-case physical qualification with owned worker/checker and owner waits."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from audit_wam_position import Evidence

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
from s2a.owned_receipt_contract import normal_receipts

ROOT = REPO / "outputs/preview_control_serialized_20261026"
DATA = REPO / "outputs/preview_control_20261026"
CONFIG = REPO / "configs/preview_control_serialized_20261026.json"
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
    "tests/test_rate_preview_active_set.py",
    "tests/test_preview_translation_runtime.py",
    "tests/test_serialized_preview_audit.py",
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
        original = REPO / "configs/preview_control_registered_20261026.json"
        assert {k: v for k, v in registration.items() if k != "serialization_binding_ref"} == json.loads(
            original.read_text()
        )
        check_sources(json.loads((DATA / "source_sha256.json").read_text()))
        evidence = Evidence()
        origin = DATA / "orchestration/case00"
        normal_receipts(
            evidence, origin / "launch.json", origin / "exit.json", origin / "release.json", format="durable"
        )
        assert json.loads((origin / "result.json").read_text())["passed"]
        collection = json.loads((DATA / "case00/collection_result.json").read_text())
        assert collection["complete"] and sha(collection["arrays_ref"]["path"]) == collection["arrays_ref"]["sha256"]
        for failed in (DATA / "verification/case00", DATA / "owner_wait"):
            launch = json.loads((failed / "launch.json").read_text())["child"]
            stat = Path(f"/proc/{launch['pid']}/stat")
            assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != launch["start_ticks"]
            exited, released = (json.loads((failed / name).read_text()) for name in ("exit.json", "release.json"))
            assert exited["actual_wait"] and exited["returncode"] == 1
            assert released["reaped"] and released["actual_wait"] and released["returncode"] == 1
            assert not released["signal_sent"] and not released["kill_sent"]

        prior_config = json.loads((REPO / "configs/pose_velocity_loaded_control_20261025.json").read_text())
        assert registration["profiles"] == prior_config["profiles"]
        assert registration["seed"] == 42 and registration["physics_steps"] == 4907
        assert len(registration["profiles"]) == 18
        assert sum(p["physics_steps"] for p in registration["profiles"]) == 4907
        assert (
            registration["arm_translation_objective"] == "preview_sum_position_squared_plus_dt_squared_velocity_squared"
        )
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            CONFIG,
            original,
            DATA / "source_sha256.json",
            DATA / "case00/collection_result.json",
            Path(collection["arrays_ref"]["path"]),
            origin / "result.json",
            DATA / "verification/case00/command.log",
            *map(Path, evidence.hashes),
            *list((DATA / "verification/case00").glob("*.json")),
            *list((DATA / "owner_wait").glob("*.json")),
            REPO / "plans/v0.0.1/222_preview_serialization_recovery_20261026.md",
            *[REPO / p for p in TESTS],
        ]
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
        binding = json.loads(Path(registration["preview_log_binding_ref"]["path"]).read_text())
        assert binding["complete"] and binding["independent_preview_log_and_history"] and binding["rows"] == 164
        assert binding["source_commit_cycles"] == 164 and len(binding["corruptions_rejected"]) == 10
        runtime = json.loads(Path(registration["preview_runtime_ref"]["path"]).read_text())
        assert runtime["complete"] and runtime["rows"] == 164 and runtime["online_input_and_reference_binding"]
        qp = json.loads(Path(registration["preview_qp_ref"]["path"]).read_text())
        assert qp["complete"] and qp["rows"] == 164 and qp["max_stationarity_residual"] < 2e-6
        serialized = json.loads(Path(registration["serialization_binding_ref"]["path"]).read_text())
        assert (
            serialized["complete"]
            and serialized["json_roundtrip_rows"] == 164
            and len(serialized["corruptions_rejected"]) == 10
        )
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
                str(REPO / "scripts/collect_preview_control.py"),
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
                "import verify_preview_control_serialized; import preview_acceptance_checker_serialized; import preview_arm_log_replay_serialized; import serialized_rate_preview_independent; import qualified_inverse_independent; import qualified_input_state_audit",
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
                raise RuntimeError("221 admission failed; no retry")
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
    stages = [
        dict(
            case=0,
            kind="collect",
            reused_collection=True,
            stage_ref=dict(
                path=str(DATA / "orchestration/case00/result.json"),
                sha256=sha(DATA / "orchestration/case00/result.json"),
            ),
        )
    ]
    for index in range(18):
        for kind, python, script in [
            ("collect", "outputs/libero_env311/bin/python", "collect_preview_control.py"),
            ("verify", ".venv/bin/python", "verify_preview_control_serialized.py"),
        ]:
            if index == 0 and kind == "collect":
                continue
            directory = (
                DATA / ("orchestration" if kind == "collect" else "verification_serialized") / f"case{index:02d}"
            )
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
            newly_owned_stages=35,
            reused_physics_steps=164,
            new_physics_steps=4743,
            preserved_prior_checker_owner_failure=True,
            model_calls=0,
            fits=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
