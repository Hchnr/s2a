"""026 precision reference and real CPU hook checks without CUDA or WAM."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("probe_wam_flow_readout_quantization")


def test_independent_rounding_retains_midpoint_direction_subnormals_and_signed_zero(probe):
    values = [1 + 2**-8 + 2**-30, -1 - 2**-8 - 2**-30, 2**-134 + 2**-160, -(2**-134) - 2**-160, -0.0]
    expected = np.asarray([0x3F81, 0xBF81, 1, 0x8001, 0x8000], np.uint16).view(np.int16)
    np.testing.assert_array_equal(probe.bf16_bits(values), expected)


def test_precision_cases_include_actual_float32_sigmas_and_complete1280_geometry(probe):
    cases = list(probe.case_inputs())
    assert len(cases) == 15
    assert len({case["name"] for case in cases}) == 15
    sigmas = sorted({float(case["sigma_actual"][0, 0]) for case in cases})
    assert float(np.float32(0.12809006869792938)) in sigmas
    assert float(np.float32(0.2)) in sigmas and float(np.float32(0.8)) in sigmas
    assert all(case["weights"].shape == (1281, 10) and case["hidden"].shape == (2, 1280) for case in cases)


def test_actual_cpu_flow_hook_all_boundary_cases_match_independent_math(probe):
    pytest.importorskip("torch")
    for case in probe.case_inputs():
        actual = probe.run_case(case, "cpu")
        expected = probe.expected_bits(case["hidden"], case["base"], case["weights"], case["sigma_actual"])
        np.testing.assert_array_equal(actual[:, :10], expected)
        np.testing.assert_array_equal(actual[:, 10:], probe.bf16_bits(case["base"][:, 10:]))
