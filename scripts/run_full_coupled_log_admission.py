"""242 one-shot durable full JSON/native control-log admission; no physics."""

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

ROOT = REPO / "outputs/full_coupled_log_supervision_20261026"
PROOF = REPO / "outputs/full_coupled_control_log_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        first = REPO / "outputs/full_coupled_chain_supervision_20261026"
        result = json.loads((first / "execution_result.json").read_text())
        assert result["complete"] and result["actual_owned_wait_recorded"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((first / "admission" / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 0
            if name == "release.json":
                assert receipt["reaped"]
        paths = [path for directory in ("src", "scripts") for path in (REPO / directory).rglob("*.py")]
        paths.extend(
            [
                REPO / "plans/v0.0.1/242_full_pose_control_log_binding_20261026.md",
                REPO / "tests/test_full_coupled_pose_inverse.py",
                REPO / "tests/test_generated_reference_twist.py",
                first / "execution_result.json",
                first / "admission/exit.json",
                first / "admission/release.json",
                REPO / "outputs/full_coupled_recorded_chain_20261026/independent_verified.json",
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
        [str(REPO / "outputs/libero_env311/bin/python"), str(REPO / "scripts/admit_full_coupled_control_log.py")],
        environment=env,
        directory=ROOT / "verification",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert proof["complete"] and proof["independent_full_pose_control_log"] and proof["rows"] == 164
    assert proof["source_commit_cycles"] == 164 and len(proof["corruptions_rejected"]) == 20
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
