"""Close the fixed18 run once its existing owner is normally reaped; never restart it."""

import argparse
import json
import time
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

RUN = REPO / "outputs/full_coupled_control_20261026"
ROOT = REPO / "outputs/full_coupled_closure_supervision_20261026"
CLOSURE = REPO / "scripts/close_full_coupled_control.py"


def matching_process(child):
    path = Path(f"/proc/{child['pid']}/stat")
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return int(fields[19]) == child["start_ticks"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not (RUN / "closure_verified.json").exists()
        sources = json.loads((RUN / "source_sha256.json").read_text())
        owner = json.loads((RUN / "owner_wait/launch.json").read_text())["child"]
        if not matching_process(owner):
            assert (RUN / "supervision_result.json").exists()
        sources.update(
            {
                str(path): sha(path)
                for path in (
                    Path(__file__).resolve(),
                    CLOSURE,
                    RUN / "source_sha256.json",
                    RUN / "owner_wait/launch.json",
                )
            }
        )
        check_sources(sources)
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
    owner = json.loads((RUN / "owner_wait/launch.json").read_text())["child"]
    deadline = time.monotonic() + 12000
    while True:
        live = matching_process(owner)
        if not live:
            # Terminal evidence must agree; missing receipts are not a successful run.
            released = RUN / "owner_wait/release.json"
            if released.exists():
                receipt = json.loads(released.read_text())
                if receipt["actual_wait"] and receipt["reaped"] and receipt["returncode"] != 0:
                    publish(
                        ROOT / "upstream_failure.json",
                        dict(complete=False, owner=owner, release=receipt, automatic_retry=False, new_physics_steps=0),
                    )
                    raise RuntimeError("Existing control owner failed; no restart or qualification")
            if (RUN / "supervision_result.json").exists():
                break
            supervisor = json.loads((RUN / "supervisor/launch.json").read_text())["child"]
            if not matching_process(supervisor):
                raise RuntimeError("Owner and supervisor absent without normal supervision result")
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing run observation deadline reached; no restart")
        time.sleep(2)
    check_sources(sources)
    stage = owned_stage(
        [str(REPO / ".venv/bin/python"), str(CLOSURE)],
        environment=env,
        directory=ROOT / "closure",
        timeout_s=3600,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = json.loads((RUN / "closure_verified.json").read_text())
    assert proof["complete"] and proof["verified_cases"] == 18 and proof["owner_normal_wait_verified"]
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            actual_owned_wait_recorded=True,
            stage_ref=stage,
            closure_ref=dict(path=str(RUN / "closure_verified.json"), sha256=sha(RUN / "closure_verified.json")),
            performance_qualified=proof["performance_qualified"],
            new_physics_steps=0,
            model_calls=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
