"""231 durable one-shot supervision for the230 recorded-state admission."""

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

ROOT = REPO / "outputs/coupled_recorded_chain_supervision_checked_20261026"
PROOF = REPO / "outputs/current_coupled_recorded_chain_checked_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        prior = REPO / "outputs/coupled_recorded_chain_supervision_20261026/verification"
        launch = json.loads((prior / "launch.json").read_text())["child"]
        stat = Path(f"/proc/{launch['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != launch["start_ticks"]
        exited, released = (json.loads((prior / name).read_text()) for name in ("exit.json", "release.json"))
        assert (
            exited["returncode"] == released["returncode"] == 1
            and exited["actual_wait"]
            and released["actual_wait"]
            and released["reaped"]
        )
        paths = [
            Path(__file__).resolve(),
            REPO / "scripts/admit_coupled_recorded_chain_checked.py",
            REPO / "scripts/current_coupled_independent.py",
            REPO / "src/s2a/archived_forward_state.py",
            REPO / "scripts/admit_coupled_recorded_chain.py",
            REPO / "plans/v0.0.1/233_archival_quaternion_normalization_20261026.md",
            REPO / "outputs/current_coupled_recorded_chain_20261026/failure.json",
            prior / "launch.json",
            prior / "exit.json",
            prior / "release.json",
            REPO / "src/s2a/current_coupled_response.py",
            REPO / "src/s2a/current_coupled_state.py",
            REPO / "src/s2a/coupled_response_history.py",
            REPO / "outputs/current_coupled_reader_admission_20261026/verified.json",
            REPO / "plans/v0.0.1/230_coupled_recorded_chain_20261026.md",
            REPO / "plans/v0.0.1/231_coupled_chain_durable_supervision_20261026.md",
        ]
        sources = {str(p): sha(p) for p in paths}
        ROOT.mkdir()
        publish(
            ROOT / "prior_failure.json",
            dict(
                status="original check failed and was actually reaped;new corrected source",
                prior_pid=launch["pid"],
                prior_start_ticks=launch["start_ticks"],
                prior_returncode=1,
                automatic_retry=False,
            ),
        )
        publish(ROOT / "source_sha256.json", sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    check_sources(sources)
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    stage = owned_stage(
        [str(REPO / "outputs/libero_env311/bin/python"), str(REPO / "scripts/admit_coupled_recorded_chain_checked.py")],
        environment=env,
        directory=ROOT / "verification",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert proof["complete"] and proof["independent_coupled_inverse_and_history"] and proof["rows"] == 164
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
