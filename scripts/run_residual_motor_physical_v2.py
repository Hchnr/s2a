"""Own the fixed three CPU oracle qualifications; no retries or model calls."""

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

ROOT = REPO / "outputs/residual_motor_physical_20261025_v2"


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
        admission_path = REPO / "outputs/residual_motor_physical_admission_20261025_v3/admission.json"
        admission = json.loads(admission_path.read_text())
        if not admission["passed"]:
            raise ValueError("074 admission failed")
        for path, signature in admission["source_sha256"].items():
            if sha(path) != signature:
                raise ValueError("074 admitted source or input changed")
        if ROOT.exists() and {x.name for x in ROOT.iterdir()} != {"admission_failure.json"}:
            raise ValueError("Refuse to overwrite an existing execution")
        ROOT.mkdir(exist_ok=True)
        paths = list((REPO / "src/s2a").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
        paths += [admission_path]
        paths += [
            REPO / "configs/residual_motor_physical_20261025_v2.json",
            REPO / "plans/v0.0.1/074_residual_motor_physical_qualification_20261025.md",
        ]
        config = json.loads((REPO / "configs/residual_motor_physical_20261025_v2.json").read_text())
        if (
            len(config["profiles"]) != 18
            or config["physics_steps"] != 4907
            or sum(p["physics_steps"] for p in config["profiles"]) != 4907
        ):
            raise ValueError("Fixed074 budget differs")
        prior = Path(config["prior_ref"]["path"])
        if (
            sha(prior) != config["prior_ref"]["sha256"]
            or not json.loads(prior.read_text())["execution_and_independent_checks_passed"]
        ):
            raise ValueError("067 completed independent audit missing")
        parameter_ref = config["residual_parameters_ref"]
        parameter_path = Path(parameter_ref["path"])
        qualification = REPO / "outputs/residual_motor_prior_20261025/independent_verified.json"
        verified = json.loads(qualification.read_text())
        if (
            sha(parameter_path) != parameter_ref["sha256"]
            or not verified["numerically_verified"]
            or not verified["offline_performance_eligible"]
        ):
            raise ValueError("073 offline qualification missing or parameters changed")
        for path, signature in verified["input_sha256"].items():
            if sha(path) != signature:
                raise ValueError("073 independent verification inputs changed")
        paths += [prior, parameter_path, qualification, REPO / "configs/matched_oracle_requalification_20261025.json"]
        for profile in config["profiles"]:
            for key in ("trajectory_ref", "case_ref"):
                path = Path(profile[key]["path"])
                if sha(path) != profile[key]["sha256"]:
                    raise ValueError("Registered reference source changed")
                paths.append(path)
        sources = {str(p): sha(p) for p in paths}
        publish(source_path, sources)
        print(launch_detached(command, directory=ROOT / "supervisor", source_sha256=sources, environment=env))
        return
    sources = json.loads(source_path.read_text())
    accept_detached_launch(ROOT / "supervisor", command=command, source_sha256=sources)
    results = []
    for index in range(18):
        record = f"case{index:02d}"
        cmd = [
            str(REPO / "outputs/libero_env311/bin/python"),
            str(REPO / "scripts/collect_residual_motor_physical_v2.py"),
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
            complete=len(results) == 18,
            passed=len(results) == 18 and all(x["passed"] for x in results),
            scope="Execution receipts only; independent task/geometry audit pending",
            stages=results,
            model_calls=0,
            fresh_cases=0,
            automatic_retry=False,
        ),
    )


if __name__ == "__main__":
    main()
