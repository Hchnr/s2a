"""Freeze and execute074 CPU admission checks before any physical launch."""

import json
import os
import subprocess
from pathlib import Path

from s2a.artifacts import sha256


def main():
    root = Path("outputs/residual_motor_physical_admission_20261025")
    root.mkdir(exist_ok=False)
    registration = json.loads(Path("configs/residual_motor_physical_20261025.json").read_text())
    assert len(registration["profiles"]) == 18 and registration["physics_steps"] == 4907
    paths = list(Path("src/s2a").glob("*.py")) + list(Path("scripts").glob("*.py"))
    tests = [
        "tests/test_residual_motor_control_reuse.py",
        "tests/test_residual_motor_checker_reuse.py",
        "tests/test_residual_motor_inverse.py",
        "tests/test_state_conditioned_gripper_inverse.py",
    ]
    paths += [Path(p) for p in tests] + [
        Path("configs/residual_motor_physical_20261025.json"),
        Path("plans/v0.0.1/074_residual_motor_physical_qualification_20261025.md"),
    ]
    for p in registration["profiles"]:
        for key in ("trajectory_ref", "case_ref"):
            assert sha256(p[key]["path"]) == p[key]["sha256"]
            paths.append(Path(p[key]["path"]))
    ref = registration["residual_parameters_ref"]
    assert sha256(ref["path"]) == ref["sha256"]
    paths.append(Path(ref["path"]))
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:scripts",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        OPENBLAS_NUM_THREADS="1",
    )
    commands = [[".venv/bin/python", "-m", "pytest", "-q", *tests]]
    commands += [
        [
            "outputs/libero_env311/bin/python",
            "scripts/collect_residual_motor_physical.py",
            "--case-index",
            str(i),
            "--dry-run",
        ]
        for i in range(18)
    ]
    commands += [
        [
            "outputs/libero_env311/bin/python",
            "-c",
            "import residual_motor_acceptance_checker; import residual_motor_scalar_replay; import verify_residual_motor_physical_case",
        ]
    ]
    receipts = []
    for i, cmd in enumerate(commands):
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (root / f"check{i:02d}.log").write_text(p.stdout + p.stderr)
        receipts.append(dict(command=cmd, returncode=p.returncode, log_sha256=sha256(root / f"check{i:02d}.log")))
        if p.returncode:
            raise RuntimeError(f"Admission failed check{i}: {p.stderr[-1000:]}")
    assert all(sha256(p) == sha for p, sha in hashes.items())
    output = dict(passed=True, checks=receipts, source_sha256=hashes, model_calls=0, physics_steps=0, fresh_cases=0)
    (root / "admission.json").write_text(json.dumps(output, indent=2) + "\n")
    print("passed", len(receipts), "checks;18 cases;4907 registered physics steps;0 executed")


if __name__ == "__main__":
    main()
