"""028 actual CPU path with nontrivial center/scale, without WAM weights."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def module():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "scaled_precision_test", REPO / "scripts/probe_wam_scaled_readout_quantization.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def model():
    weights = np.zeros((1281, 10), np.float64)
    weights[0, 0], weights[-1, 1] = 0.125, -0.25
    return dict(weights=weights, center=np.full(1280, 0.125, np.float64), scale=np.asarray(2.0, np.float64))


def test_registered_cases_preserve_actual_parameters_and_bf16_representable_inputs():
    probe = module()
    sigmas = np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32).astype(np.float64)
    actual = model()
    cases = list(probe.case_inputs(actual, sigmas))
    assert len(cases) == len({case["name"] for case in cases}) == 20
    assert [case["name"] for case in cases[15:]] == [f"sigma{i}_candidate" for i in range(5)]
    for case in cases:
        for key in ("hidden", "base"):
            assert not np.any(case[key].view(np.uint32) & 0xFFFF)
        if case["name"].endswith("candidate"):
            for key in actual:
                np.testing.assert_array_equal(case[key], actual[key])
        else:
            assert case["scale"] == 3
            np.testing.assert_array_equal(case["center"][:2], [0.5, -0.25])


def test_real_cpu_hook_matches_independent_normalized_design_and_tail_bits():
    pytest.importorskip("torch")
    probe = module()
    cases = list(
        probe.case_inputs(
            model(), [float(np.float32(0.2)), 0.5, float(np.float32(0.8)), 0.12809006869792938, 0.9990000128746033]
        )
    )
    for case in (cases[0], cases[2], cases[15]):
        actual = probe.run_case(case, "cpu")
        np.testing.assert_array_equal(actual[:, :10], probe.expected_bits(case))
        expected_tail = (case["base"].view(np.uint32)[:, 10:] >> 16).astype(np.uint16).view(np.int16)
        np.testing.assert_array_equal(actual[:, 10:], expected_tail)
    assert probe.primitive_values().shape == (10,)
