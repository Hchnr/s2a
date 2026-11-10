"""Recovery must preserve original evidence and exclude still-live original workers."""

from pathlib import Path

import pytest

from scripts.simulation.runtime import bind_causal_event_bridge_factorial as binding
from scripts.simulation.runtime import prepare_causal_event_factorial_recovery as recovery


def test_recovery_binds_interrupted_logs_and_existing_references():
    inputs = binding.load_inputs()
    receipt, document = recovery.admit(inputs)
    assert receipt["prior_final_exit_and_cost_known"] is False
    assert all(row["control_completed"] == 64 for row in receipt["prior_last_progress"].values())
    assert all(item["checked_endpoints"] == 40 for item in receipt["independent_component_checks"].values())
    assert set(document["references"]) == set(binding.MODES)


def test_original_process_scan_excludes_recovery_and_unrelated_processes(tmp_path):
    for pid, argv in {
        10: ["python", "-m", recovery.MODULE, "--producer"],
        11: ["python", "-m", recovery.MODULE, "--worker", "checker", "--recovery"],
        12: ["python", "-m", "other.module", "--worker"],
    }.items():
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    assert recovery.original_process_ids(tmp_path) == [10]


def test_log_progress_rejects_unregistered_or_incomplete_counts(tmp_path):
    path = Path(tmp_path) / "worker.log"
    path.write_text('{"phase":"producer","branch":"bridge_pose_old_opening","control_completed":63}\n')
    with pytest.raises((ValueError, KeyError)):
        recovery.last_progress(path, "bridge_pose_old_opening")
