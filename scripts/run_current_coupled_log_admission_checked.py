"""234 one-shot durable supervision of the full JSON control-log admission."""

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

ROOT = REPO / "outputs/current_coupled_log_supervision_checked_20261026"
PROOF = REPO / "outputs/current_coupled_control_log_checked_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        prior = REPO / "outputs/coupled_recorded_chain_supervision_checked_20261026"
        execution = json.loads((prior / "execution_result.json").read_text())
        assert execution["complete"] and execution["actual_owned_wait_recorded"]
        for name in ("exit.json", "release.json"):
            receipt = json.loads((prior / "verification" / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 0
            if name == "release.json":
                assert receipt["reaped"]
        failed = REPO / "outputs/current_coupled_log_supervision_20261026/verification"
        for name in ("exit.json", "release.json"):
            receipt = json.loads((failed / name).read_text())
            assert receipt["actual_wait"] and receipt["returncode"] == 1
            if name == "release.json":
                assert receipt["reaped"]
        launch = json.loads((failed / "launch.json").read_text())["child"]
        stat = Path(f"/proc/{launch['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != launch["start_ticks"]
        paths = [p for directory in ("src", "scripts") for p in (REPO / directory).rglob("*.py")]
        paths.extend(
            [
                REPO / "plans/v0.0.1/234_coupled_control_log_binding_20261026.md",
                prior / "execution_result.json",
                failed / "exit.json",
                failed / "release.json",
                REPO / "outputs/current_coupled_control_log_20261026/failure.json",
                REPO / "plans/v0.0.1/235_coupled_log_comparison_repair_20261026.md",
                prior / "verification/exit.json",
                prior / "verification/release.json",
                REPO / "outputs/current_coupled_recorded_chain_checked_20261026/independent_verified.json",
                REPO / "outputs/current_coupled_reader_admission_20261026/verified.json",
            ]
        )
        sources = {str(p): sha(p) for p in paths}
        ROOT.mkdir()
        publish(ROOT / "source_sha256.json", sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    check_sources(sources)
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    stage = owned_stage(
        [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/admit_current_coupled_control_log_checked.py"),
        ],
        environment=env,
        directory=ROOT / "verification",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert proof["complete"] and proof["independent_full_control_log"] and proof["rows"] == 164
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
