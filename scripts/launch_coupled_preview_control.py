"""Own complete preview admission, then arrange persistent physical closure without retries."""

import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha

ROOT = REPO / "outputs/coupled_preview_launch_supervision_20261026"
RUN = REPO / "outputs/coupled_preview_control_20261026"


def main():
    assert not ROOT.exists() and not RUN.exists()
    paths = [path for folder in ("src/s2a", "scripts") for path in (REPO / folder).glob("*.py")]
    paths.extend(
        [
            REPO / "configs/coupled_preview_control_registered_20261026.json",
            REPO / "plans/v0.0.1/248_coupled_preview_physical_20261026.md",
        ]
    )
    sources = {str(path): sha(path) for path in paths}
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
    admission = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_coupled_preview_control.py"), "launch"],
        environment=env,
        directory=ROOT / "admission",
        timeout_s=7200,
        source_sha256=sources,
    )
    owner = json.loads((RUN / "owner_wait/launch.json").read_text())["child"]
    stat = Path(f"/proc/{owner['pid']}/stat")
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        assert int(fields[19]) == owner["start_ticks"] and fields[0] not in ("Z", "X")
    else:
        assert (RUN / "supervision_result.json").exists()
    closure = owned_stage(
        [str(REPO / ".venv/bin/python"), str(REPO / "scripts/supervise_coupled_preview_closure.py"), "launch"],
        environment=env,
        directory=ROOT / "closure_launch",
        timeout_s=1800,
        source_sha256=sources,
    )
    check_sources(sources)
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=True,
            actual_owned_wait_recorded=True,
            admission_ref=admission,
            closure_launch_ref=closure,
            physical_owner=owner,
            performance_qualified=False,
            automatic_retry=False,
        ),
    )
    print("preview physical owner and independent closure launched", flush=True)


if __name__ == "__main__":
    main()
