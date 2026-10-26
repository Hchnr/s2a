"""239 one-shot durable native/JSON/ODE admission for the full-pose proposal."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import (
    accept_detached_launch,
    check_sources,
    cpu_environment,
    launch_detached,
    owned_stage,
    publish,
    sha,
)

ROOT = REPO / "outputs/full_coupled_all_records_startup_20261026"
PROOF = REPO / "outputs/full_coupled_all_records_checked_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        prior = REPO / "outputs/current_coupled_rotation_supervision_20261026"
        execution = json.loads((prior / "execution_result.json").read_text())
        assert execution["complete"] and execution["actual_owned_wait_recorded"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((prior / "audit" / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 0
            if name == "release.json":
                assert receipt["reaped"]
        first = REPO / "outputs/full_coupled_chain_supervision_20261026"
        initial = json.loads((first / "execution_result.json").read_text())
        assert initial["complete"] and initial["actual_owned_wait_recorded"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((first / "admission" / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 0
            if name == "release.json":
                assert receipt["reaped"]
        failed = REPO / "outputs/full_coupled_all_records_supervision_20261026/admission"
        child = json.loads((failed / "launch.json").read_text())["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((failed / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 1
            if name == "release.json":
                assert receipt["reaped"]
        startup = REPO / "outputs/full_coupled_all_records_supervision_checked_20261026/supervisor"
        interrupted = json.loads((startup / "launcher_result.json").read_text())
        assert not interrupted["complete"] and not interrupted["passed"]
        assert interrupted["owned_cleanup"]["reaped"] and interrupted["owned_cleanup"]["returncode"] == -15
        child = json.loads((startup / "launch.json").read_text())["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
        paths = [path for directory in ("src", "scripts") for path in (REPO / directory).rglob("*.py")]
        paths.extend(
            [
                REPO / "plans/v0.0.1/239_nonlinear_coupled_pose_inverse_20261026.md",
                REPO / "tests/test_full_coupled_pose_inverse.py",
                REPO / "tests/test_generated_reference_twist.py",
                prior / "execution_result.json",
                startup / "launcher_result.json",
                startup / "launch.json",
                REPO / "plans/v0.0.1/241_full_pose_startup_deadline_20261026.md",
                failed / "exit.json",
                failed / "release.json",
                REPO / "outputs/full_coupled_all_records_20261026/failure.json",
                REPO / "plans/v0.0.1/240_full_pose_matrix_environment_admission_20261026.md",
                first / "execution_result.json",
                first / "admission/exit.json",
                first / "admission/release.json",
                REPO / "outputs/full_coupled_recorded_chain_20261026/independent_verified.json",
                prior / "audit/exit.json",
                prior / "audit/release.json",
                REPO / "outputs/current_coupled_rotation_tracking_20261026/verified.json",
                REPO / "outputs/current_coupled_control_20261026/closure_verified.json",
            ]
        )
        sources = {str(path): sha(path) for path in paths}
        ROOT.mkdir()
        publish(ROOT / "source_sha256.json", sources)
        print(
            launch_detached(
                command, directory=ROOT / "supervisor", source_sha256=sources, environment=env, ready_timeout_s=300
            )
        )
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    stage = owned_stage(
        [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/admit_full_coupled_all_records_checked.py"),
        ],
        environment=env,
        directory=ROOT / "admission",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert (
        proof["complete"]
        and proof["independent_full_pose_inverse_and_native_history"]
        and proof["rows"] == 4907
        and proof["cases"] == 18
    )
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            actual_owned_wait_recorded=True,
            stage_ref=stage,
            proof_ref=dict(path=str(PROOF), sha256=sha(PROOF)),
            physics_steps=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
