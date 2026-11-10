"""An independent checker rejects changed source identity and component mixing."""

import copy
import os
import subprocess
import sys

import pytest

from scripts.simulation.runtime.construct_causal_reference_factorial import construct_reference
from scripts.verification.reference_ports.check_causal_reference_factorial import verify_reference
from tests.research.native_runtime.test_causal_reference_factorial import fixture


@pytest.mark.parametrize("mode", ["bridge_pose_old_opening", "old_pose_bridge_opening"])
def test_checks_every_component_at_its_original_absolute_endpoint(mode):
    *refs, kwargs = fixture(mode)
    produced = construct_reference(*refs, **kwargs)
    assert verify_reference(*refs, produced, **kwargs) == {
        "passed": True,
        "checked_endpoints": 40,
        "old_aligned_horizon": 24,
    }


@pytest.mark.parametrize("corruption", ["time", "mode", "horizon", "pose", "opening", "tail", "source"])
def test_rejects_altered_component_or_provenance(corruption):
    *refs, kwargs = fixture("bridge_pose_old_opening")
    produced = copy.deepcopy(construct_reference(*refs, **kwargs))
    if corruption == "time":
        produced["origin_step"] += 1
    elif corruption == "mode":
        produced["mode"] = "old_pose_bridge_opening"
    elif corruption == "horizon":
        produced["old_aligned_horizon"] -= 1
    elif corruption == "pose":
        produced["poses"][8, 0, 3] += 0.001
    elif corruption == "opening":
        produced["openings"][8] += 0.001
    elif corruption == "tail":
        produced["openings"][24] += 0.001
    else:
        refs[1][24] += 0.001
    with pytest.raises(ValueError):
        verify_reference(*refs, produced, **kwargs)


def test_checker_import_does_not_load_intervention_producer():
    program = (
        "import sys; from scripts.verification.reference_ports import check_causal_reference_factorial; "
        "assert 'scripts.simulation.runtime.construct_causal_reference_factorial' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
