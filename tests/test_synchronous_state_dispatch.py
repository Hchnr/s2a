"""Exercise every actual variant, including the first adapted request boundary."""

import pytest

from s2a.synchronous_state_dispatch import select_readout


def test_all_registered_variants_select_exact_supplied_objects():
    zero, probe = object(), object()
    assert select_readout("raw", zero=zero, probe=probe) is None
    assert select_readout("zero", zero=zero, probe=probe) is zero
    assert select_readout("probe052", zero=zero, probe=probe) is probe


def test_full_matrix_first_adapted_boundary_is_dispatchable():
    zero, probe = object(), object()
    variants = ["zero"] * 3 + [v for _ in range(48) for v in ("raw", "probe052")]
    selected = [select_readout(v, zero=zero, probe=probe) for v in variants]
    assert selected[3] is None and selected[4] is probe
    assert sum(value is probe for value in selected) == 48


@pytest.mark.parametrize("variant", ["w2", "probe", "", None])
def test_old_or_unregistered_identity_rejected(variant):
    with pytest.raises(ValueError):
        select_readout(variant, zero=object(), probe=object())
