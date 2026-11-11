"""The reset roster cannot silently change task, seed, or reference variant."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime.prepare_reset_development import validate_case_identity


@pytest.mark.parametrize(
    "index,task,seed,variant",
    [(0, 8, 42, "raw"), (3, 8, 43, "terminal058"), (6, 9, 43, "raw"), (9, 6, 42, "terminal058")],
)
def test_case_identity_binds_task_seed_and_variant(index, task, seed, variant):
    case = {
        "case_index": index,
        "case_id": f"task{index // 4}_init0_s{seed}_{variant}",
        "sampling_seed": seed,
        "variant": variant,
    }
    manifest = {"record": {"task_id": task}, "runtime_config": {"task_id": task, "seed": seed}}
    validate_case_identity(index, case, manifest)
    wrong = copy.deepcopy(manifest)
    wrong["runtime_config"]["seed"] = 43 if seed == 42 else 42
    with pytest.raises(ValueError, match="case, task, seed"):
        validate_case_identity(index, case, wrong)
    wrong_case = dict(case, variant="other")
    with pytest.raises(ValueError, match="case, task, seed"):
        validate_case_identity(index, wrong_case, manifest)


def test_invalid_roster_index_is_rejected():
    with pytest.raises(ValueError, match="twelve"):
        validate_case_identity(True, {}, {})
    with pytest.raises(ValueError, match="twelve"):
        validate_case_identity(12, {}, {})


def test_source_loader_does_not_import_simulation_before_osmesa_setup():
    code = (
        "import sys; from scripts.simulation.runtime import prepare_reset_development; "
        "assert not any(name in sys.modules for name in ('mujoco', 'OpenGL', 'robosuite', 'libero'))"
    )
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
