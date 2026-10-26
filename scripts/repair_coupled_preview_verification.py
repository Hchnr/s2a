"""Repair only the stale required-runtime roster; reuse the completed first trajectory."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha
from s2a.owned_receipt_contract import normal_receipts

ROOT = REPO / "outputs/coupled_preview_verification_repair_20261026"
RUN = REPO / "outputs/coupled_preview_control_20261026"


def absent(child):
    path = Path(f"/proc/{child['pid']}/stat")
    assert not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]


def main():
    assert not ROOT.exists() and not (RUN / "case00/independent_verified.json").exists()
    e = Evidence()
    for directory in (RUN / "owner_wait", RUN / "verification/case00"):
        launch, exited, released = (e.json(directory / name) for name in ("launch.json", "exit.json", "release.json"))
        assert exited["actual_wait"] and exited["returncode"] == 1
        assert released["actual_wait"] and released["reaped"] and released["returncode"] == 1
        assert not released["signal_sent"] and not released["kill_sent"]
        assert all(launch["child"][k] == released[k] for k in ("pid", "start_ticks"))
        absent(launch["child"])
    directory = RUN / "orchestration/case00"
    absent(
        normal_receipts(
            e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
        )["child"]
    )
    sources = json.loads((RUN / "source_sha256.json").read_text())
    for path in (
        Path(__file__).resolve(),
        REPO / "scripts/verify_coupled_preview_control_registered.py",
        REPO / "scripts/coupled_preview_acceptance_checker_registered.py",
    ):
        sources[str(path)] = sha(path)
    sources.update(e.hashes)
    check_sources(sources)
    ROOT.mkdir()
    publish(ROOT / "source_sha256.json", sources)
    env = cpu_environment()
    env.update(
        PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LP_NUM_THREADS="2",
    )
    stage = owned_stage(
        [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/verify_coupled_preview_control_registered.py"),
            "--case-index",
            "0",
        ],
        environment=env,
        directory=ROOT / "case00",
        timeout_s=1800,
        source_sha256=sources,
    )
    check_sources(sources)
    proof = RUN / "case00/independent_verified.json"
    result = json.loads(proof.read_text())
    assert result["complete"] and result["mechanical_labels_images_passed"]
    publish(
        ROOT / "verified.json",
        dict(
            complete=True,
            actual_owned_wait_recorded=True,
            stage_ref=stage,
            proof_ref=dict(path=str(proof), sha256=sha(proof)),
            source_sha256=sources,
            performance_qualified=result["performance_qualified"],
            new_physics_steps=0,
            reuses_original_collection=True,
            automatic_retry=False,
        ),
    )
    print("original case00 independently verified without recollection", flush=True)


if __name__ == "__main__":
    main()
