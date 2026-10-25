import numpy as np
import pytest

from s2a.gripper_reference_preview import opening_preview


def test_only_already_generated_monotonic_opening_can_advance():
    generated = np.array([0.2, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    original = generated.copy()
    result = opening_preview(generated, 0, preview_steps=6)
    assert result["reference"] == 1 and result["selected_index"] == 6
    np.testing.assert_array_equal(generated, original)
    assert opening_preview(generated[:3], 0, preview_steps=6)["reference"] == 0.3
    generated[6] = 0.8
    assert opening_preview(generated, 0, preview_steps=2)["reference"] == 0.3


@pytest.mark.parametrize("values", [[0.8, 0.6, 0.4], [0.4, 0.2, 0.5], [0.4, 0.4, 0.4]])
def test_closing_reversal_and_flat_reference_preserved(values):
    result = opening_preview(values, 0, preview_steps=6)
    assert not result["advanced"] and result["reference"] == values[0]


def test_zero_preview_and_final_endpoint_keep_original():
    assert opening_preview([0.2, 0.8], 0, preview_steps=0)["reference"] == 0.2
    assert opening_preview([0.2, 0.8], 1, preview_steps=6)["reference"] == 0.8


def test_invalid_unavailable_or_nonfinite_inputs_rejected():
    for values, index, steps in [
        ([], 0, 6),
        ([0.2], 1, 6),
        ([np.nan], 0, 6),
        ([1.1], 0, 6),
        ([0.2], True, 6),
        ([0.2], 0, -1),
    ]:
        with pytest.raises(ValueError):
            opening_preview(values, index, preview_steps=steps)
