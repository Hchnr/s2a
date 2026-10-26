"""243 wait on the two existing admissions, then launch fixed18 control once."""

import argparse
import json
import time
from pathlib import Path

from audit_wam_position import Evidence

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
from s2a.owned_receipt_contract import normal_receipts

ROOT = REPO / "outputs/full_coupled_control_handoff_20261026"
CONTROL = REPO / "outputs/full_coupled_control_20261026"
CONFIG = REPO / "configs/full_coupled_control_registered_20261026.json"
UPSTREAM = (
    (
        REPO / "outputs/full_coupled_all_records_startup_20261026",
        "admission",
        REPO / "outputs/full_coupled_all_records_checked_20261026/independent_verified.json",
        "independent_full_pose_inverse_and_native_history",
        4907,
    ),
    (
        REPO / "outputs/full_coupled_log_supervision_20261026",
        "verification",
        REPO / "outputs/full_coupled_control_log_20261026/independent_verified.json",
        "independent_full_pose_control_log",
        164,
    ),
)


def matching_live(child):
    path = Path(f"/proc/{child['pid']}/stat")
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return int(fields[19]) == child["start_ticks"] and fields[0] not in ("Z", "X")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not CONTROL.exists() and not CONFIG.exists()
        paths = [path for directory in ("src", "scripts") for path in (REPO / directory).rglob("*.py")]
        paths += [
            REPO / "plans/v0.0.1/242_full_pose_control_log_binding_20261026.md",
            REPO / "plans/v0.0.1/243_full_pose_control_handoff_20261026.md",
        ]
        handles = []
        for root, stage, _, _, _ in UPSTREAM:
            path = root / stage / "launch.json"
            paths.append(path)
            child = json.loads(path.read_text())["child"]
            supervisor = json.loads((root / "supervisor/launch.json").read_text())["child"]
            assert matching_live(child) or (root / "execution_result.json").exists()
            handles.append(dict(root=str(root), stage=stage, child=child, supervisor=supervisor))
        sources = {str(path): sha(path) for path in paths}
        ROOT.mkdir()
        publish(ROOT / "upstream_handles.json", handles)
        publish(ROOT / "source_sha256.json", sources)
        print(
            launch_detached(
                command, directory=ROOT / "supervisor", source_sha256=sources, environment=env, ready_timeout_s=300
            )
        )
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    handles = json.loads((ROOT / "upstream_handles.json").read_text())
    deadline = time.monotonic() + 12000
    for spec, handle in zip(UPSTREAM, handles, strict=True):
        root, stage, proof_path, required, rows = spec
        child, supervisor = handle["child"], handle["supervisor"]
        while True:
            if not matching_live(child):
                release_path = root / stage / "release.json"
                if release_path.exists():
                    release = json.loads(release_path.read_text())
                    if release["actual_wait"] and release["reaped"] and release["returncode"] != 0:
                        publish(
                            ROOT / "upstream_failure.json",
                            dict(
                                complete=False, root=str(root), release=release, automatic_retry=False, physics_steps=0
                            ),
                        )
                        raise RuntimeError("Existing upstream admission failed; no restart or physics")
                if (root / "execution_result.json").exists():
                    break
                if not matching_live(supervisor):
                    raise RuntimeError("Existing upstream supervisor absent without normal result")
            if time.monotonic() > deadline:
                raise TimeoutError("Upstream observation deadline; no restart or physics")
            time.sleep(2)
        e = Evidence()
        receipt_dir = root / stage
        recorded = normal_receipts(
            e, receipt_dir / "launch.json", receipt_dir / "exit.json", receipt_dir / "release.json", format="durable"
        )["child"]
        assert recorded["pid"] == child["pid"] and recorded["start_ticks"] == child["start_ticks"]
        assert not matching_live(child)
        completed = e.json(root / "execution_result.json")
        assert completed["complete"] and completed["actual_owned_wait_recorded"]
        proof = e.json(proof_path)
        assert proof["complete"] and proof[required] and proof["rows"] == rows
        e.recheck()
    check_sources(sources)
    registration = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/register_full_coupled_control.py")],
        environment=env,
        directory=ROOT / "registration",
        timeout_s=3600,
        source_sha256=sources,
    )
    control_launch = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_full_coupled_control.py"), "launch"],
        environment=env,
        directory=ROOT / "control_launch",
        timeout_s=7200,
        source_sha256=sources,
    )
    owner_path = CONTROL / "owner_wait/launch.json"
    control_supervisor = json.loads((CONTROL / "supervisor/launch.json").read_text())["child"]
    deadline = time.monotonic() + 3600
    while not owner_path.exists():
        if not matching_live(control_supervisor):
            raise RuntimeError("Control supervisor ended before its real owner launch")
        if time.monotonic() > deadline:
            raise TimeoutError("Control owner launch observation deadline; no restart")
        time.sleep(2)
    closure_launch = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/supervise_full_coupled_closure.py"), "launch"],
        environment=env,
        directory=ROOT / "closure_launch",
        timeout_s=900,
        source_sha256=sources,
    )
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            scope="registered fixed18 owner and closure supervisor launched;not a performance qualification",
            registration_stage_ref=registration,
            control_launch_stage_ref=control_launch,
            closure_launch_stage_ref=closure_launch,
            config_ref=dict(path=str(CONFIG), sha256=sha(CONFIG)),
            control_result_pending=True,
            automatic_retry=False,
            model_calls=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
