"""Independent SLERP audit must reject tampered reference evidence."""

import copy
import os
import subprocess
import sys

import pytest
from scipy.spatial.transform import Rotation

from scripts.verification.reference_ports.check_causal_reference_bridge import verify_reference_bridge
from tests.research.wam.tracking.preview.test_causal_reference_bridge import bridge, references


def verify(refs, result):
    return verify_reference_bridge(*refs, result, old_step=344, latest_step=360, start_offset=8, transition_steps=8)


def test_independent_slerp_checks_all_endpoints():
    refs = references()
    result = verify(refs, bridge(refs))
    assert result["passed"] and result["checked_endpoints"] == 40
    assert result["maximum_pose_element_difference"] < 1e-14
    assert result["maximum_opening_difference"] == 0


@pytest.mark.parametrize(
    "corruption",
    ["old_step", "origin_step", "start", "weight", "translation", "rotation", "opening", "prefix", "suffix"],
)
def test_rejects_changed_times_weights_or_saved_reference(corruption):
    refs = references()
    changed = copy.deepcopy(bridge(refs))
    if corruption == "old_step":
        changed["old_step"] += 1
    elif corruption == "origin_step":
        changed["origin_step"] += 1
    elif corruption == "start":
        changed["start_offset"] -= 1
    elif corruption == "weight":
        changed["weights"][8] += 0.125
    elif corruption == "translation":
        changed["poses"][8, 0, 3] += 0.001
    elif corruption == "rotation":
        changed["poses"][8, :3, :3] = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    elif corruption == "opening":
        changed["openings"][8] += 0.01
    elif corruption == "prefix":
        changed["poses"][0, 0, 3] += 1e-12
    else:
        changed["openings"][-1] += 1e-12
    with pytest.raises(ValueError):
        verify(refs, changed)


def test_rejects_changed_original_source_chunk():
    refs = list(references())
    produced = bridge(refs)
    refs[0] = refs[0].copy()
    refs[0][24, 0, 3] += 0.001
    with pytest.raises(ValueError):
        verify(refs, produced)


def test_independent_checker_does_not_import_the_producer():
    producer = "s2a.research.wam.tracking.preview.causal_reference_bridge"
    program = (
        "import sys; from scripts.verification.reference_ports import check_causal_reference_bridge; "
        f"assert {producer!r} not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": ""},
        capture_output=True,
    )
