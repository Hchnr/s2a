"""244 fixed18 SQP admission after the recorded optimizer failure; no retries."""

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

ROOT = REPO / "outputs/full_pose_sqp_matrix_supervision_20261026"
PROOF = REPO / "outputs/full_pose_sqp_all_records_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        prior = REPO / "outputs/full_coupled_all_records_startup_20261026/admission"
        child = json.loads((prior / "launch.json").read_text())["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((prior / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 1
            if name == "release.json":
                assert receipt["reaped"]
        failure = json.loads((REPO / "outputs/full_coupled_all_records_checked_20261026/failure.json").read_text())
        assert failure["error_type"] == "RuntimeError" and "not locally stationary" in failure["error"]
        paths = [path for directory in ("src", "scripts") for path in (REPO / directory).rglob("*.py")]
        paths += [
            REPO / "plans/v0.0.1/244_full_pose_stationarity_and_source_audit_20261026.md",
            REPO / "tests/test_full_coupled_pose_inverse_sqp.py",
            REPO / "tests/fixtures/full_pose_case04_step71.json",
            prior / "launch.json",
            prior / "exit.json",
            prior / "release.json",
            REPO / "outputs/full_coupled_all_records_checked_20261026/failure.json",
            REPO / "outputs/full_pose_stationarity_failure_20261026/diagnosis.json",
        ]
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
        [str(REPO / "outputs/libero_env311/bin/python"), str(REPO / "scripts/admit_full_pose_sqp_all_records.py")],
        environment=env,
        directory=ROOT / "admission",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert proof["complete"] and proof["solver_method"] == "SLSQP" and proof["rows"] == 4907 and proof["cases"] == 18
    assert proof["source_graph_complete_start_and_end_verified"] and proof["max_independent_stationarity"] < 2e-6
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
