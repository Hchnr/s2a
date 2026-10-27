"""139 one diagonal-quadratic action fit and independent augmented SVD via owned stages."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq

from s2a.artifacts import sha256
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish

ROOT = Path("outputs/quadratic_action_fit_20261025")
DATA = Path("outputs/quadratic_action_equations_20261025/result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "fit", "verify"))
    args = parser.parse_args()
    if args.mode == "run":
        data = json.loads(DATA.read_text())
        assert (
            data["complete"]
            and data["rows"] == 5261
            and data["contrast_equations"] == data["curvature_equations"] == 198
        )
        assert data["design_dimension"] == 1974 and data["new_fits"] == 0
        check_sources(data["source_sha256"])
        paths = [
            DATA,
            Path(data["parameter_ref"]["path"]),
            Path(data["training_ref"]["path"]),
            Path("plans/v0.0.1/139_quadratic_action_fit_20261025.md"),
            Path("tests/test_diagonal_quadratic_action_forward.py"),
        ]
        paths += list(Path("src/s2a").glob("*.py")) + list(Path("scripts").glob("*.py"))
        assert sha256(data["training_ref"]["path"]) == data["training_ref"]["sha256"]
        sources = {str(p.resolve()): sha256(p) for p in paths}
        ROOT.mkdir(exist_ok=False)
        publish(ROOT / "source_sha256.json", sources)
        refs = []
        for mode in ("fit", "verify"):
            refs.append(
                owned_stage(
                    [str(Path(".venv/bin/python").resolve()), str(Path(__file__).resolve()), mode],
                    environment=cpu_environment(),
                    directory=ROOT / mode,
                    timeout_s=1800,
                    source_sha256=sources,
                )
            )
        check_sources(sources)
        publish(
            ROOT / "execution_result.json",
            dict(
                complete=True,
                stage_refs=refs,
                new_fits=1,
                independent_numerical_solves=1,
                performance_qualified=False,
                physics_steps=0,
                model_calls=0,
                fresh_cases=0,
            ),
        )
        print("139 unique fit and independent SVD complete; physical performance not yet validated")
        return
    sources = json.loads((ROOT / "source_sha256.json").read_text())
    check_sources(sources)
    data = json.loads(DATA.read_text())
    ref = data["training_ref"]
    assert sha256(ref["path"]) == ref["sha256"]
    with np.load(ref["path"], allow_pickle=False) as a:
        x, y, w = a["features"].copy(), (a["target"] - a["nominal"]).copy(), a["weights"].copy()
    assert x.shape == (5261, 1974) and y.shape == (5261, 12)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.all(w > 0)
    np.testing.assert_allclose([w[:4865].sum(), w[4865:].sum()], [0.5, 0.5], rtol=0, atol=1e-12)
    penalty = np.full(1974, 0.001)
    penalty[0] = 0
    if args.mode == "fit":
        publish(ROOT / "fit_attempt.json", dict(new_fits=1, automatic_retry=False))
        gram = x.T @ (w[:, None] * x) + np.diag(penalty)
        beta = cho_solve(cho_factor(gram, lower=True), x.T @ (w[:, None] * y))
        bound_ref = data["parameter_ref"]
        assert sha256(bound_ref["path"]) == bound_ref["sha256"]
        bound = json.loads(Path(bound_ref["path"]).read_text())
        bound.update(
            coefficients=beta.tolist(),
            ridge=0.001,
            dt=0.05,
            representation="physical_world_response",
            design_dimension=1974,
        )
        publish(ROOT / "parameters.json", bound)
    else:
        augmented = np.vstack([np.sqrt(w)[:, None] * x, np.diag(np.sqrt(penalty))])
        rhs = np.vstack([np.sqrt(w)[:, None] * y, np.zeros((1974, 12))])
        independent, _, rank, _ = lstsq(augmented, rhs, lapack_driver="gelsd")
        beta = np.asarray(json.loads((ROOT / "parameters.json").read_text())["coefficients"])
        difference = float(np.max(np.abs(beta - independent)))
        gradient = float(np.max(np.abs(x.T @ (w[:, None] * (x @ beta - y)) + penalty[:, None] * beta)))
        assert rank == 1974 and difference <= 1e-8 and gradient <= 1e-10
        publish(
            ROOT / "independent_verified.json",
            dict(
                complete=True,
                numerically_verified=True,
                rank=rank,
                max_coefficient_difference=difference,
                maximum_regularized_gradient=gradient,
                parameter_ref=dict(
                    path=str((ROOT / "parameters.json").resolve()), sha256=sha256(ROOT / "parameters.json")
                ),
            ),
        )
        print("rank", rank, "coefficient_difference", difference, "gradient", gradient)
    check_sources(sources)


if __name__ == "__main__":
    main()
