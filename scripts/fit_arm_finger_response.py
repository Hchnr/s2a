"""181 fixed1128 design, one aperture output, owned fit and independent SVD."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import check_sources, cpu_environment, owned_stage, publish, sha

ROOT = REPO / "outputs/arm_finger_linear_fit_20261025"


def system():
    equations = json.loads((REPO / "outputs/left_table_forward_equations_20261025/result.json").read_text())
    labels = json.loads((REPO / "outputs/arm_finger_response_labels_20261025/result.json").read_text())
    with np.load(equations["training_ref"]["path"], allow_pickle=False) as cache:
        x, w = cache["features"].copy(), cache["weights"].copy()
    with np.load(labels["labels_ref"]["path"], allow_pickle=False) as cache:
        target = cache["fit_target"].copy()
    assert x.shape == (5072, 1128) and target.shape == w.shape == (5072,)
    penalty = np.ones(1128) * 0.001
    penalty[0] = 0
    return x, w, target, penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "fit", "verify"])
    args = parser.parse_args()
    if args.mode == "run":
        assert not ROOT.exists()
        proof = REPO / "outputs/arm_finger_labels_independent_20261025/verified.json"
        verified = json.loads(proof.read_text())
        assert verified["complete"] and verified["independent_raw_state_xml_verified"]
        check_sources(verified["source_sha256"])
        paths = [
            proof,
            Path(__file__).resolve(),
            REPO / "plans/v0.0.1/181_arm_finger_linear_fit_20261025.md",
            REPO / "outputs/left_table_forward_equations_20261025/result.json",
            REPO / "outputs/arm_finger_response_labels_20261025/result.json",
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
            dict(complete=True, stages=stages, new_fits=1, physics_steps=0, control_qualified=False),
        )
        return
    check_sources(json.loads((ROOT / "source_sha256.json").read_text()))
    x, w, target, penalty = system()
    a, b = x * np.sqrt(w)[:, None], target * np.sqrt(w)
    if args.mode == "fit":
        coef = cho_solve(cho_factor(a.T @ a + np.diag(penalty)), a.T @ b)
        publish(
            ROOT / "parameters.json",
            dict(
                coefficients=coef.tolist(),
                design="146 frozen1128",
                output="delta(q_left-q_right) metres",
                nominal="zero",
                ridge=0.001,
                source_sha256=json.loads((ROOT / "source_sha256.json").read_text()),
            ),
        )
    else:
        coef = np.asarray(json.loads((ROOT / "parameters.json").read_text())["coefficients"])
        independent, _, rank, _ = lstsq(
            np.vstack([a, np.diag(np.sqrt(penalty))]), np.r_[b, np.zeros(1128)], lapack_driver="gelsd"
        )
        np.testing.assert_allclose(coef, independent, atol=1e-10, rtol=0)
        gradient = a.T @ (a @ coef - b) + penalty * coef
        assert rank == 1128 and np.max(np.abs(gradient)) < 1e-10
        publish(
            ROOT / "independent_verified.json",
            dict(
                complete=True,
                numerically_verified=True,
                rank=int(rank),
                max_coefficient_difference=float(np.max(np.abs(coef - independent))),
                max_gradient=float(np.max(np.abs(gradient))),
                parameter_ref=dict(path=str(ROOT / "parameters.json"), sha256=sha(ROOT / "parameters.json")),
                control_qualified=False,
            ),
        )


if __name__ == "__main__":
    main()
