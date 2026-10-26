"""246 repair only the confirmed pre-physics relative-source admission failure."""

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

ROOT = REPO / "outputs/full_coupled_sqp_admission_repair_20261026"
CONTROL = REPO / "outputs/full_coupled_sqp_control_20261026"
CONFIG = REPO / "configs/full_coupled_sqp_control_registered_20261026.json"
UPSTREAM = (
    (
        REPO / "outputs/full_pose_sqp_matrix_supervision_20261026",
        "admission",
        REPO / "outputs/full_pose_sqp_all_records_20261026/independent_verified.json",
        "independent_full_pose_inverse_and_native_history",
        4907,
    ),
    (
        REPO / "outputs/full_pose_sqp_log_supervision_20261026",
        "verification",
        REPO / "outputs/full_pose_sqp_control_log_20261026/independent_verified.json",
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


def audit_prior_admission():
    e = Evidence()
    prior_root = REPO / "outputs/full_coupled_sqp_handoff_20261026"
    registration = prior_root / "registration"
    old = normal_receipts(
        e, registration / "launch.json", registration / "exit.json", registration / "release.json", format="durable"
    )["child"]
    assert not matching_live(old)
    failed = prior_root / "control_launch"
    launch = e.json(failed / "launch.json")
    child = launch["child"]
    assert not matching_live(child)
    for name in ("exit.json", "release.json"):
        receipt = e.json(failed / name)
        assert receipt["actual_wait"] and receipt["returncode"] == 1
        if name == "release.json":
            assert receipt["reaped"] and not receipt["signal_sent"] and not receipt["kill_sent"]
            assert receipt["pid"] == child["pid"] and receipt["start_ticks"] == child["start_ticks"]
    assert b"assert Path(name).is_absolute()" in e.read(failed / "command.log")
    assert not CONTROL.exists()
    result = e.json(registration / "result.json")
    assert result["passed"]
    config = e.json(CONFIG)
    assert config["physics_steps"] == 4907 and len(config["profiles"]) == 18
    e.recheck()
    return dict(
        complete=True,
        original_registration_normal_wait=True,
        original_prephysics_failure_actual_wait_one_reaped=True,
        original_failed_child=child,
        registration_stage_ref=dict(path=str(registration / "result.json"), sha256=sha(registration / "result.json")),
        config_ref=dict(path=str(CONFIG), sha256=sha(CONFIG)),
        source_sha256=e.hashes,
        new_physics_steps=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "supervise"))
    args = parser.parse_args()
    env = cpu_environment()
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2")
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), "supervise"]
    if args.mode == "launch":
        assert not ROOT.exists() and not CONTROL.exists() and CONFIG.exists()
        prior = audit_prior_admission()
        paths = [path for directory in ("src", "scripts") for path in (REPO / directory).rglob("*.py")]
        paths += [
            REPO / "plans/v0.0.1/244_full_pose_stationarity_and_source_audit_20261026.md",
            REPO / "plans/v0.0.1/245_sqp_control_binding_and_provenance_20261026.md",
            REPO / "plans/v0.0.1/246_relative_provenance_admission_repair_20261026.md",
            REPO / "tests/test_full_pose_execution_sources_resolved.py",
            CONFIG,
            *[Path(name) for name in prior["source_sha256"]],
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
        prior_ref = publish(ROOT / "prior_admission_audit.json", prior)
        sources[prior_ref["path"]] = prior_ref["sha256"]
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
        assert proof["solver_method"] == "SLSQP"
        if rows == 4907:
            assert proof["cases"] == 18 and proof["source_graph_complete_start_and_end_verified"]
            assert proof["max_independent_stationarity"] < 2e-6
        else:
            assert proof["source_commit_cycles"] == 164 and len(proof["corruptions_rejected"]) == 20
            assert proof["max_stationarity_residual"] < 2e-6
        e.recheck()
    check_sources(sources)
    prior = audit_prior_admission()
    registration = prior["registration_stage_ref"]
    control_launch = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_full_coupled_sqp_control_resolved.py"), "launch"],
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
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/supervise_full_coupled_sqp_closure.py"), "launch"],
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
