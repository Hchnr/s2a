"""Continue only cases01-17 after a verified checker-only repair; never recollect case00."""

import argparse
import json
import os
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

ROOT = REPO / "outputs/coupled_preview_continuation_20261026"
RUN = REPO / "outputs/coupled_preview_control_20261026"
REPAIR = REPO / "outputs/coupled_preview_verification_repair_20261026"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "supervise", "run"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(
        PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LP_NUM_THREADS="2",
    )
    supervisor = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists()
        proof = json.loads((REPAIR / "verified.json").read_text())
        assert proof["complete"] and proof["actual_owned_wait_recorded"] and proof["reuses_original_collection"]
        assert sha(proof["proof_ref"]["path"]) == proof["proof_ref"]["sha256"]
        e = Evidence()
        normal_receipts(
            e,
            REPAIR / "case00/launch.json",
            REPAIR / "case00/exit.json",
            REPAIR / "case00/release.json",
            format="durable",
        )
        assert not any((RUN / f"case{i:02d}").exists() for i in range(1, 18))
        sources = dict(proof["source_sha256"])
        for path in (
            Path(__file__).resolve(),
            REPAIR / "verified.json",
            RUN / "case00/independent_verified.json",
            REPO / "scripts/close_coupled_preview_continuation.py",
            REPO / "scripts/supervise_coupled_preview_continuation_closure.py",
            REPO / "scripts/launch_coupled_preview_continuation.py",
        ):
            sources[str(path)] = sha(path)
        sources.update(e.hashes)
        check_sources(sources)
        ROOT.mkdir()
        publish(ROOT / "source_sha256.json", sources)
        print(
            launch_detached(
                supervisor, directory=ROOT / "supervisor", source_sha256=sources, environment=env, ready_timeout_s=300
            ),
            flush=True,
        )
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    if args.mode == "supervise":
        accept_detached_launch(ROOT / "supervisor", command=supervisor, source_sha256=sources)
        stage = owned_stage(
            [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "run"],
            environment=env,
            directory=ROOT / "owner_wait",
            timeout_s=12000,
            source_sha256=sources,
        )
        publish(
            ROOT / "supervision_result.json",
            dict(owner_stage_ref=stage, owner_actual_wait_recorded=True, automatic_retry=False),
        )
        return
    check_sources(sources)
    deadline = time.monotonic() + 300
    while not (ROOT / "owner_wait/launch.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child = json.loads((ROOT / "owner_wait/launch.json").read_text())["child"]
    actual = identity(os.getpid())
    assert all(actual[k] == child[k] for k in ("pid", "start_ticks"))
    stages = []
    for index in range(1, 18):
        for kind, script in [
            ("collect", "collect_coupled_preview_control.py"),
            ("verify", "verify_coupled_preview_control_registered.py"),
        ]:
            directory = RUN / ("orchestration" if kind == "collect" else "verification") / f"case{index:02d}"
            stage = owned_stage(
                [
                    str(REPO / "outputs/libero_env311/bin/python"),
                    str(REPO / "scripts" / script),
                    "--case-index",
                    str(index),
                ],
                environment=env,
                directory=directory,
                timeout_s=1800,
                source_sha256=sources,
            )
            stages.append(dict(case=index, kind=kind, stage_ref=stage))
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            passed=True,
            stages=stages,
            reused_case00=True,
            remaining_cases=list(range(1, 18)),
            new_physics_steps=4743,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
