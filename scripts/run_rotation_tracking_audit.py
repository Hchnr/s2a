"""Durably run the interrupted238 read-only audit once, without automatic retries."""

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

ROOT = REPO / "outputs/current_coupled_rotation_supervision_20261026"
PROOF = REPO / "outputs/current_coupled_rotation_tracking_20261026/verified.json"
AUDIT = REPO / "scripts/audit_coupled_rotation_tracking.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not PROOF.parent.exists()
        matches = []
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(
                arg.decode(errors="replace").endswith("/audit_coupled_rotation_tracking.py")
                or arg == b"scripts/audit_coupled_rotation_tracking.py"
                for arg in argv
            ):
                matches.append(proc.name)
        assert not matches, matches
        paths = [
            Path(__file__).resolve(),
            AUDIT,
            REPO / "plans/v0.0.1/238_full_coupled_pose_inverse_diagnosis_20261026.md",
            REPO / "outputs/current_coupled_control_20261026/closure_verified.json",
            REPO / "src/s2a/durable_stage_supervisor.py",
            REPO / "scripts/audit_wam_position.py",
        ]
        sources = {str(path): sha(path) for path in paths}
        ROOT.mkdir()
        publish(
            ROOT / "prior_interruption.json",
            dict(
                prior_session_id=16844,
                status="session missing;matching audit process absent;no proof directory",
                prior_source_ref=dict(path=str(AUDIT), sha256=sha(AUDIT)),
                replacement_scope="same read-only audit under durable supervision",
                automatic_retry=False,
                physics_steps=0,
            ),
        )
        publish(ROOT / "source_sha256.json", sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    check_sources(sources)
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    stage = owned_stage(
        [str(REPO / ".venv/bin/python"), str(AUDIT)],
        environment=env,
        directory=ROOT / "audit",
        timeout_s=1800,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads(PROOF.read_text())
    assert proof["complete"] and proof["rows"] == 4907 and len(proof["cases"]) == 18
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
