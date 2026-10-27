"""210 one fixed direct inverse residual fit, independent SVD and action diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
from direct_arm_inverse_independent import features
from scipy.linalg import cho_factor, cho_solve, lstsq

from s2a.artifacts import REPO
from s2a.direct_arm_inverse import feature_matrix
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha

ROOT = REPO / "outputs/direct_arm_inverse_fit_20261026"
DATA = REPO / "outputs/direct_arm_inverse_data_checked_20261026"


def system():
    record = json.loads((DATA / "result.json").read_text())
    for key in ("training_ref", "parameter_ref"):
        ref = record[key]
        assert sha(ref["path"]) == ref["sha256"]
    params = json.loads(Path(record["parameter_ref"]["path"]).read_text())
    with np.load(record["training_ref"]["path"], allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    assert arrays["values"].shape == (5369, 89) and arrays["target"].shape == arrays["nominal"].shape == (5369, 3)
    return record, params, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "fit", "verify"])
    args = parser.parse_args()
    if args.mode == "run":
        assert not ROOT.exists()
        proof_path = DATA / "independent_verified.json"
        proof = json.loads(proof_path.read_text())
        assert proof["complete"] and proof["independent_input_and_nominal_verified"]
        check_sources(proof["source_sha256"])
        paths = [
            Path(__file__).resolve(),
            proof_path,
            DATA / "result.json",
            DATA / "parameters.json",
            REPO / "plans/v0.0.1/210_direct_arm_inverse_fit_20261026.md",
            REPO / "src/s2a/direct_arm_inverse.py",
            REPO / "scripts/direct_arm_inverse_independent.py",
        ]
        sources = dict(proof["source_sha256"])
        sources.update({str(p.resolve()): sha(p) for p in paths})
        ROOT.mkdir()
        publish(ROOT / "source_sha256.json", sources)
        stages = []
        for mode in ("fit", "verify"):
            stages.append(
                owned_stage(
                    [str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()), mode],
                    environment=cpu_environment(),
                    directory=ROOT / mode,
                    timeout_s=1800,
                    source_sha256=sources,
                )
            )
        check_sources(sources)
        publish(
            ROOT / "execution_result.json",
            dict(complete=True, stages=stages, new_fits=1, physics_steps=0, control_qualified=False),
        )
        return
    check_sources(json.loads((ROOT / "source_sha256.json").read_text()))
    record, params, data = system()
    target = data["target"] - data["nominal"]
    weight = data["weights"]
    penalty = np.full(218, 0.001)
    penalty[0] = 0
    if args.mode == "fit":
        x = feature_matrix(data["values"], params)
        a = x * np.sqrt(weight)[:, None]
        b = target * np.sqrt(weight)[:, None]
        coef = cho_solve(cho_factor(a.T @ a + np.diag(penalty)), a.T @ b)
        publish(
            ROOT / "parameters.json",
            dict(
                params, coefficients=coef.tolist(), source_sha256=json.loads((ROOT / "source_sha256.json").read_text())
            ),
        )
    else:
        produced = json.loads((ROOT / "parameters.json").read_text())
        for key, value in params.items():
            assert produced[key] == value
        coef = np.asarray(produced["coefficients"])
        x = features(data["values"], params)
        np.testing.assert_allclose(x, feature_matrix(data["values"], params), atol=1e-10, rtol=0)
        a = x * np.sqrt(weight)[:, None]
        b = target * np.sqrt(weight)[:, None]
        independent, _, rank, _ = lstsq(
            np.vstack([a, np.diag(np.sqrt(penalty))]), np.vstack([b, np.zeros((218, 3))]), lapack_driver="gelsd"
        )
        np.testing.assert_allclose(coef, independent, atol=1e-10, rtol=0)
        gradient = a.T @ (a @ coef - b) + penalty[:, None] * coef
        assert rank == 218 and np.max(np.abs(gradient)) < 1e-10
        predicted = np.clip(
            data["holdout_nominal"] + features(data["holdout_values"], params) @ coef,
            params["translation_input_min"],
            params["translation_input_max"],
        )
        diagnostics = []
        start = 0
        for row in record["holdout_records"]:
            end = start + row["rows"]
            truth = data["holdout_target"][start:end]
            model = float(np.sqrt(np.mean(np.sum((predicted[start:end] - truth) ** 2, axis=1))))
            baseline = float(np.sqrt(np.mean(np.sum((data["holdout_nominal"][start:end] - truth) ** 2, axis=1))))
            diagnostics.append(
                dict(
                    native_index=row["native_index"],
                    rows=row["rows"],
                    normalized_action_rmse=model,
                    nominal_action_rmse=baseline,
                )
            )
            start = end
        assert start == 1315
        publish(
            ROOT / "independent_verified.json",
            dict(
                complete=True,
                numerically_verified=True,
                rank=int(rank),
                max_coefficient_difference=float(np.max(np.abs(coef - independent))),
                max_gradient=float(np.max(np.abs(gradient))),
                parameter_ref=dict(path=str(ROOT / "parameters.json"), sha256=sha(ROOT / "parameters.json")),
                holdout_action_diagnostics=diagnostics,
                action_error_is_control_success=False,
                control_qualified=False,
            ),
        )
        print(json.dumps(diagnostics))


if __name__ == "__main__":
    main()
