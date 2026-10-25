"""Own nine fixed063 physical training recovery trials; no model calls."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import (
    accept_detached_launch,
    cpu_environment,
    launch_detached,
    owned_stage,
    publish,
    sha,
)

ROOT = REPO / "outputs/gripper_reference_preview_20261025"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    env = cpu_environment()
    env.update(
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        OPENBLAS_NUM_THREADS="8",
        OMP_NUM_THREADS="8",
        LP_NUM_THREADS="2",
        PYTHONPATH=str(REPO / "src") + ":" + str(REPO / "scripts"),
    )
    command = [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve())]
    source_path = ROOT / "source_sha256.json"
    if args.launch:
        ROOT.mkdir(exist_ok=False)
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [
            REPO / "configs/training_state_recovery_qualification_20261025.json",
            REPO / "plans/v0.0.1/063_training_state_recovery_qualification_20261025.md",
        ]
        registration = json.loads((REPO / "configs/training_state_recovery_qualification_20261025.json").read_text())
        prepared_path = (
            REPO
            / "outputs/wam_stability_20261021/clean_residual_training_20261023/recovery_20261023/preparation/prepared.json"
        )
        prepared = json.loads(prepared_path.read_text())
        forbidden = set(prepared["exclusion"]["forbidden_initial_state_sha256"])
        if any(entry["record"]["initial_state_sha256"] in forbidden for entry in registration["records"]):
            raise ValueError("Training recovery overlaps excluded initial states")
        if registration["physics_steps"] != 2307 or registration["base_x_offsets_m"] != [0, 0.02, -0.02]:
            raise ValueError("Fixed063 budget differs")
        paths += [prepared_path, REPO / "configs/matched_oracle_requalification_20261025.json"]
        for entry in registration["records"]:
            for key in ("trajectory_ref", "case_ref"):
                path = Path(entry[key]["path"])
                if sha(path) != entry[key]["sha256"]:
                    raise ValueError("Training source changed")
                paths.append(path)
        paths += [
            REPO / "configs/training_recovery_delay_isolation_20261025.json",
            REPO / "plans/v0.0.1/064_training_recovery_delay_isolation_20261025.md",
        ]
        preview_config = REPO / "configs/gripper_reference_preview_20261025.json"
        profile = json.loads(preview_config.read_text())
        if profile["profiles"] != [[0, 0], [0, 6], [1, 6], [2, 0], [2, 6]] or profile["physics_steps"] != 948:
            raise ValueError("Fixed066 cohort differs")
        prior = REPO / "outputs/training_recovery_delay_isolation_20261025/independent_verified.json"
        evidence = json.loads(prior.read_text())
        if not evidence["complete"] or not evidence["mechanical_labels_images_passed"]:
            raise ValueError("064 completed baseline required")
        paths += [preview_config, prior, REPO / "plans/v0.0.1/066_gripper_reference_preview_20261025.md"]
        sources = {str(p): sha(p) for p in paths}
        publish(source_path, sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads(source_path.read_text())
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    results = []
    for index in range(5):
        record = f"case{index:02d}"
        cmd = [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/collect_gripper_reference_preview.py"),
            "--case-index",
            str(index),
        ]
        result = owned_stage(
            cmd, environment=env, directory=ROOT / "orchestration" / record, timeout_s=1800, source_sha256=sources
        )
        results.append(json.loads(Path(result["path"]).read_text()))
        if not results[-1]["passed"]:
            break
    publish(
        ROOT / "execution_result.json",
        dict(
            complete=len(results) == 5,
            passed=len(results) == 5 and all(x["passed"] for x in results),
            scope="Execution receipts only; independent task/geometry audit pending",
            stages=results,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
