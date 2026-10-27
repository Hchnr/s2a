"""Audit each074 completed trial once; do not launch or retry physical execution."""

import json
import os
import subprocess
import time
from pathlib import Path

from s2a.artifacts import sha256

ROOT = Path("outputs/residual_motor_physical_20261025_v2")
OWNER_PID = 3496816
OWNER_TICKS = 583073895


def main():
    audit = ROOT / "independent_followup"
    audit.mkdir(exist_ok=False)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="src:scripts",
        OPENBLAS_NUM_THREADS="1",
    )
    registration = json.loads(Path("configs/residual_motor_physical_20261025_v2.json").read_text())
    receipts = []
    for index in range(18):
        label = f"case{index:02d}"
        result_path = ROOT / label / "independent_verified.json"
        while not (ROOT / "orchestration" / label / "result.json").exists():
            terminal = ROOT / "execution_result.json"
            if terminal.exists():
                raise RuntimeError(f"Execution terminated without {label}; no automatic retry")
            identity = Path(f"/proc/{OWNER_PID}/stat")
            if not identity.exists() or int(identity.read_text().rsplit(")", 1)[1].split()[19]) != OWNER_TICKS:
                raise RuntimeError("Original owner missing/replaced before terminal receipt; inspect manually")
            time.sleep(5)
        if not result_path.exists():
            cmd = [".venv/bin/python", "scripts/verify_residual_motor_physical_case_v2.py", "--case-index", str(index)]
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)
            (audit / f"{label}.log").write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"{label} independent verification failed; log retained; no retry")
        verified = json.loads(result_path.read_text())
        assert verified["complete"] and verified["mechanical_labels_images_passed"]
        for path, signature in verified["input_sha256"].items():
            assert sha256(path) == signature, path
        row = dict(
            case=index,
            reference=registration["profiles"][index]["trajectory_ref"],
            verification_ref=dict(path=str(result_path.resolve()), sha256=sha256(result_path)),
            outcome=verified["control"]["outcome"],
            tracking=verified["control"]["tracking"],
            performance_qualified=verified["performance_qualified"],
        )
        receipts.append(row)
        with (audit / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(label, "verified", row["performance_qualified"], flush=True)
    while not (ROOT / "execution_result.json").exists():
        time.sleep(1)
    execution = json.loads((ROOT / "execution_result.json").read_text())
    assert execution["passed"] and execution["complete"]
    for path, signature in json.loads((ROOT / "source_sha256.json").read_text()).items():
        assert sha256(path) == signature, path
    summary = dict(
        complete=True,
        execution_and_independent_checks_passed=True,
        performance_qualified=all(r["performance_qualified"] for r in receipts),
        candidate_stable=sum(r["outcome"]["stable"] for r in receipts),
        candidate_trials=18,
        physics_steps=4907,
        cases=receipts,
        source_roster_unchanged=True,
        model_calls=0,
        fresh_cases=0,
        owner_release_audit_pending=True,
        followup_source_sha256=sha256(__file__),
    )
    with (ROOT / "regression_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print("complete stable", summary["candidate_stable"], "/18", flush=True)


if __name__ == "__main__":
    main()
