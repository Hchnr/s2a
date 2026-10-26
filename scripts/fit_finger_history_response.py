"""185 intervention-only slopes then natural bias; independent augmented SVD checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha

ROOT = REPO / "outputs/finger_history_fit_20261025"


def system():
    record = json.loads((REPO / "outputs/finger_history_equations_20261025/result.json").read_text())
    assert record["complete"] and record["rows"] == 5081 and record["contrast_equations"] == 216
    ref = record["training_ref"]
    assert sha(ref["path"]) == ref["sha256"]
    with np.load(ref["path"], allow_pickle=False) as data:
        x, w, target = (data[k].copy() for k in ("features", "weights", "target"))
    assert x.shape == (5081, 1284) and w.shape == target.shape == (5081,)
    return x, w, target, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "fit", "verify"])
    args = parser.parse_args()
    if args.mode == "run":
        assert not ROOT.exists()
        proof = REPO / "outputs/finger_history_equations_20261025/independent_verified.json"
        verified = json.loads(proof.read_text())
        assert verified["complete"] and verified["independent_equations_verified"]
        check_sources(verified["source_sha256"])
        paths = [
            proof,
            Path(__file__).resolve(),
            REPO / "plans/v0.0.1/198_finger_history_response_candidate_20261025.md",
            REPO / "outputs/left_table_forward_equations_20261025/result.json",
            REPO / "outputs/finger_history_equations_20261025/result.json",
        ]
        for p in paths[-2:]:
            artifact = json.loads(p.read_text())
            for key in ("training_ref", "labels_ref"):
                if key in artifact:
                    ref = artifact[key]
                    assert sha(ref["path"]) == ref["sha256"]
                    paths.append(Path(ref["path"]))
        sources = dict(verified["source_sha256"])
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
            dict(complete=True, stages=stages, new_fits=2, physics_steps=0, control_qualified=False),
        )
        return
    check_sources(json.loads((ROOT / "source_sha256.json").read_text()))

    x, w, target, _ = system()
    slope_x = x[4865:, 321:]
    assert np.max(np.abs(x[4865:, :321])) == 0
    slope_a = slope_x * np.sqrt(w[4865:])[:, None]
    slope_b = target[4865:] * np.sqrt(w[4865:])
    slope_penalty = np.full(963, 0.001)
    bias_a = x[:4865, :321] * np.sqrt(w[:4865])[:, None]
    bias_penalty = np.full(321, 0.001)
    bias_penalty[0] = 0
    if args.mode == "fit":
        slopes = cho_solve(cho_factor(slope_a.T @ slope_a + np.diag(slope_penalty)), slope_a.T @ slope_b)
        bias_b = (target[:4865] - x[:4865, 321:] @ slopes) * np.sqrt(w[:4865])
        bias = cho_solve(cho_factor(bias_a.T @ bias_a + np.diag(bias_penalty)), bias_a.T @ bias_b)
        coef = np.r_[bias, slopes]
        publish(
            ROOT / "parameters.json",
            dict(
                coefficients=coef.tolist(),
                design="146 frozen1284;185 intervention slopes then natural bias",
                output="delta(q_left-q_right) metres",
                nominal="zero",
                ridge=0.001,
                source_sha256=json.loads((ROOT / "source_sha256.json").read_text()),
            ),
        )
    else:
        coef = np.asarray(json.loads((ROOT / "parameters.json").read_text())["coefficients"])
        slopes, _, rank_slope, _ = lstsq(
            np.vstack([slope_a, np.diag(np.sqrt(slope_penalty))]), np.r_[slope_b, np.zeros(963)], lapack_driver="gelsd"
        )
        bias_b = (target[:4865] - x[:4865, 321:] @ slopes) * np.sqrt(w[:4865])
        bias, _, rank_bias, _ = lstsq(
            np.vstack([bias_a, np.diag(np.sqrt(bias_penalty))]), np.r_[bias_b, np.zeros(321)], lapack_driver="gelsd"
        )
        independent = np.r_[bias, slopes]
        np.testing.assert_allclose(coef, independent, atol=1e-10, rtol=0)
        gradient_slope = slope_a.T @ (slope_a @ coef[321:] - slope_b) + slope_penalty * coef[321:]
        actual_bias_b = (target[:4865] - x[:4865, 321:] @ coef[321:]) * np.sqrt(w[:4865])
        gradient_bias = bias_a.T @ (bias_a @ coef[:321] - actual_bias_b) + bias_penalty * coef[:321]
        gradient = np.r_[gradient_bias, gradient_slope]
        assert rank_slope == 963 and rank_bias == 321 and np.max(np.abs(gradient)) < 1e-10
        publish(
            ROOT / "independent_verified.json",
            dict(
                complete=True,
                numerically_verified=True,
                slope_rank=int(rank_slope),
                bias_rank=int(rank_bias),
                max_coefficient_difference=float(np.max(np.abs(coef - independent))),
                max_gradient=float(np.max(np.abs(gradient))),
                parameter_ref=dict(path=str(ROOT / "parameters.json"), sha256=sha(ROOT / "parameters.json")),
                control_qualified=False,
            ),
        )


if __name__ == "__main__":
    main()
