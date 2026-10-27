"""088 real first-trajectory feature parity before the unique fit; no training."""

import json
from pathlib import Path

import numpy as np
from arm_forward_independent import conditions, feature_matrix, nominal_prediction

from s2a.artifacts import sha256
from s2a.state_conditioned_arm_forward import design, nominal, raw_condition


def main():
    config = json.loads(Path("configs/state_conditioned_arm_forward_20261025.json").read_text())
    ref = config["data_result_ref"]
    assert sha256(ref["path"]) == ref["sha256"]
    entry = json.loads(Path(ref["path"]).read_text())["trajectories"][0]
    ref = entry["cache_ref"]
    assert sha256(ref["path"]) == ref["sha256"]
    contract = json.loads((Path(entry["source"]["path"]).parent / "result.json").read_text())["controller_contract"]
    p = dict(
        state_mean=np.zeros(71),
        state_scale=np.ones(71),
        translation_mean=np.zeros(3),
        translation_scale=np.ones(3) * 0.05,
        dt=0.05,
        kp=contract["kp"],
        kd=contract["kd"],
    )
    with np.load(ref["path"], allow_pickle=False) as data:
        np.testing.assert_allclose(raw_condition(data), conditions(data), rtol=0, atol=1e-12)
        np.testing.assert_allclose(design(data, p), feature_matrix(data, p), rtol=0, atol=1e-12)
        np.testing.assert_allclose(nominal(data, p), nominal_prediction(data, p), rtol=0, atol=1e-12)
        print("088 real feature parity rows", len(data["step"]), "new_fits0")


if __name__ == "__main__":
    main()
