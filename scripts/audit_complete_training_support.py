"""165 recover all fitted condition bases, including207 contrast equations."""

import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish, sha
from s2a.joint_wrench_condition_forward import raw_condition


def main():
    root = REPO / "outputs/complete_training_support_20261025"
    if root.exists():
        raise FileExistsError(root)
    parameter_path = REPO / "outputs/left_table_fit_20261025/parameters.json"
    result_path = REPO / "outputs/left_table_forward_equations_20261025/result.json"
    parameters = json.loads(parameter_path.read_text())
    result = json.loads(result_path.read_text())
    ref = result["training_ref"]
    assert sha(ref["path"]) == ref["sha256"]
    with np.load(ref["path"], allow_pickle=False) as cache:
        x = cache["features"].copy()
    assert x.shape == (5072, 1128)
    phi = x[:4865, :282].copy()
    contrasts = []
    for index, row in enumerate(x[4865:]):
        assert np.max(np.abs(row[:282])) < 1e-12
        blocks = row[282:].reshape(3, 282)
        active = np.flatnonzero(np.abs(blocks[:, 0]) > 1e-12)
        assert len(active) == 1
        axis = int(active[0])
        assert np.max(np.abs(np.delete(blocks, axis, axis=0))) < 1e-10
        contrasts.append(blocks[axis] / blocks[axis, 0])
        assert axis == result["contrast_sources"][index]["axis"]
    phi = np.vstack([phi, contrasts])
    ztrain = np.c_[phi[:, 1:84], phi[:, 212:218]]
    phase = ztrain[:, :83] @ np.asarray(parameters["frequencies"]).T
    joint_phase = ztrain @ np.asarray(parameters["joint_frequencies"]).T
    reconstructed = np.c_[
        np.ones(len(phi)),
        ztrain[:, :83],
        np.cos(phase) / 8,
        np.sin(phase) / 8,
        ztrain[:, 83:],
        np.cos(joint_phase) / np.sqrt(32),
        np.sin(joint_phase) / np.sqrt(32),
    ]
    np.testing.assert_allclose(reconstructed, phi, atol=1e-10, rtol=0)
    assert ztrain.shape == (5072, 89)
    lower, upper = ztrain.min(axis=0), ztrain.max(axis=0)
    case = REPO / "outputs/qualified_inverse_enumerated_20261025/case00"
    verified = case / "independent_verified.json"
    assert json.loads(verified.read_text())["mechanical_labels_images_passed"]
    path = next(case.rglob("steps.jsonl"))
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        data = {k: np.asarray(v) for k, v in row["qualified_forward_input"].items()}
        z = (raw_condition(data)[0] - np.asarray(parameters["state_mean"])) / np.asarray(parameters["state_scale"])
        distances = np.linalg.norm(ztrain - z, axis=1)
        nearest = int(np.argmin(distances))
        excess = np.maximum(lower - z, z - upper)
        rows.append(
            dict(
                step=row["step"],
                nearest_training_row=nearest,
                nearest_kind="natural" if nearest < 4865 else "contrast",
                nearest_standardized_distance=float(distances[nearest]),
                outside_indices=np.flatnonzero(excess > 1e-10).tolist(),
                max_standardized_range_excess=float(max(0, excess.max())),
            )
        )
    sources = [
        Path(__file__).resolve(),
        parameter_path,
        result_path,
        Path(ref["path"]),
        verified,
        path,
        REPO / "src/s2a/joint_wrench_condition_forward.py",
        REPO / "src/s2a/nonlinear_conditioned_arm_forward.py",
        REPO / "src/s2a/wrench_conditioned_arm_forward.py",
    ]
    publish(
        root / "result.json",
        dict(
            complete=True,
            training_conditions=5072,
            natural_rows=4865,
            contrast_equations=207,
            basis_reconstruction_checked=True,
            rows=rows,
            standardized_lower=lower.tolist(),
            standardized_upper=upper.tolist(),
            source_sha256={str(p): sha(p) for p in sources},
            physics_steps=0,
            fits=0,
            control_qualified=False,
            limitation="Fit-equation condition support; marginal membership is not joint coverage or physical derivative proof",
        ),
    )
    print(json.dumps([r for r in rows if r["step"] in (70, 74, 78, 80, 84, 90)]))


if __name__ == "__main__":
    main()
