"""A failed empty collection must never claim that all tests passed."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def test_missing_test_path_reports_no_tests_as_failed():
    root = Path(__file__).resolve().parents[3]
    missing = root / "tests/__missing_cpu_runner_test__.py"
    assert not missing.exists()
    with TemporaryDirectory(prefix="cpu-runner-test-", dir=root / "outputs/qa") as directory:
        output = Path(directory) / "nested_cpu_run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.maintenance.run_cpu_tests",
                "--jobs",
                "1",
                "--output",
                str(output.relative_to(root)),
                str(missing.relative_to(root)),
            ],
            cwd=root,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (output / "result.json").is_file(), result.stdout + result.stderr
        receipt = json.loads((output / "result.json").read_text())
    assert result.returncode == 1
    assert receipt["passed"] is False and receipt["all_tests_passed"] is False
    assert receipt["collection"]["returncode"] != 0
    assert receipt["collection_node_count"] == receipt["call_count"] == 0
    assert receipt["actual_wait"] is receipt["reaped"] is True
