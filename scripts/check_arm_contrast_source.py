"""089 actual selected-pair assembly parity before the only parameter fit."""

import numpy as np
from arm_contrast_independent import equations
from arm_contrast_training import contrast_rows
from audit_wam_position import Evidence


def main():
    e = Evidence()
    config = e.json("configs/contrast_conditioned_arm_forward_20261025.json")
    ref = config["statistics_ref"]
    e.check(ref["path"], ref["sha256"])
    p = e.json(ref["path"])
    x, y, b, records = contrast_rows(config, e, p)
    xx, yy, bb, ids = equations(config, e, p)
    np.testing.assert_allclose(x, xx, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(y, yy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b, bb, rtol=1e-10, atol=1e-10)
    assert x.shape == (9, 288) and y.shape == b.shape == (9, 12)
    assert ids == [(r["anchor_group"], r["axis"], r["minus_case"], r["plus_case"]) for r in records]
    e.recheck()
    print("089 actual9training contrasts independently matched; held anchors not loaded; new_fits0")


if __name__ == "__main__":
    main()
