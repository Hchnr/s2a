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

ROOT = REPO / "outputs/coupled_recorded_chain_supervision_20261026"
PROOF = REPO / "outputs/current_coupled_recorded_chain_20261026/independent_verified.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        assert not Path("/proc/364867").exists()
        paths = [
            Path(__file__).resolve(),
            REPO / "scripts/admit_coupled_recorded_chain.py",
            REPO / "scripts/current_coupled_independent.py",
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
            ROOT / "prior_interruption.json",
            dict(
                status="interrupted with missing process/handle; no exit receipt",
                prior_pid=364867,
                prior_session=28601,
                process_absent=True,
                result_root_absent=True,
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
        [str(REPO / "outputs/libero_env311/bin/python"), str(REPO / "scripts/admit_coupled_recorded_chain.py")],
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
