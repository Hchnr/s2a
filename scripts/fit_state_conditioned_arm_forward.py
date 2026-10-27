"""088 unique24-trajectory weighted ridge fit; holdout/probes are never read as rows."""

import json
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from scipy.linalg import cho_factor, cho_solve

from s2a.artifacts import sha256
from s2a.state_conditioned_arm_forward import design, nominal, raw_condition, targets

CONFIG = Path("configs/state_conditioned_arm_forward_20261025.json")
ROOT = Path("outputs/state_conditioned_arm_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/state_conditioned_arm_forward.py")
    config = e.json(CONFIG)
    assert config["seed"] == 42 and config["ridge"] == 0.001
    for key in ("data_closure_ref", "data_result_ref", "paired_closure_ref", "paired_analysis_ref"):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    closure = e.json(config["data_closure_ref"]["path"])
    assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
    source = e.json(config["data_result_ref"]["path"])
    fit = [p for p in source["trajectories"] if p["split"] == "fit"]
    assert len(fit) == 24
    ROOT.mkdir(exist_ok=True)
    assert not any((ROOT / name).exists() for name in ("parameters.json", "fit_result.json", "fit_cache.npz"))
    data_sets = []
    raw = []
    translations = []
    weights = []
    groups = []
    sources = []
    contract = None
    for entry in fit:
        ref = entry["cache_ref"]
        e.check(ref["path"], ref["sha256"])
        meta = Path(entry["source"]["path"]).parent / "result.json"
        e.check(meta, closure["source_sha256"][str(meta.resolve())])
        current = e.json(meta)["controller_contract"]
        if contract is None:
            contract = current
        else:
            assert contract == current
        with np.load(ref["path"], allow_pickle=False) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        n = len(data["step"])
        data_sets.append(data)
        raw.append(raw_condition(data))
        translations.append(data["action_goal_delta"][:, :3])
        weights.extend([1 / (24 * n)] * n)
        groups.extend([entry["index"]] * n)
        sources.append(dict(index=entry["index"], rows=n, cache_ref=ref))
    raw = np.vstack(raw)
    translations = np.vstack(translations)
    weights = np.array(weights)
    groups = np.array(groups)
    assert len(raw) == 4865 and abs(weights.sum() - 1) < 1e-12

    def stats(values):
        mean = np.sum(values * weights[:, None], axis=0)
        scale = np.sqrt(np.sum((values - mean) ** 2 * weights[:, None], axis=0))
        scale[scale < 1e-8] = 1
        return mean.tolist(), scale.tolist()

    state_mean, state_scale = stats(raw)
    translation_mean, translation_scale = stats(translations)
    parameters = dict(
        schema="s2a.state-conditioned-arm-forward.v1",
        seed=42,
        ridge=0.001,
        dt=0.05,
        kp=contract["kp"],
        kd=contract["kd"],
        state_mean=state_mean,
        state_scale=state_scale,
        translation_mean=translation_mean,
        translation_scale=translation_scale,
    )
    x = np.vstack([design(data, parameters) for data in data_sets])
    base = np.vstack([nominal(data, parameters) for data in data_sets])
    target = np.vstack([targets(data) for data in data_sets])
    residual = target - base
    penalty = np.ones(288)
    penalty[0] = 0
    gram = x.T @ (weights[:, None] * x) + 0.001 * np.diag(penalty)
    with (ROOT / "fit_attempt.json").open("x") as stream:
        json.dump(
            dict(
                attempt=1,
                rows=4865,
                fit_trajectories=24,
                ridge=0.001,
                solver="weighted normal Cholesky",
                automatic_retry=False,
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    coefficients = cho_solve(cho_factor(gram, lower=True), x.T @ (weights[:, None] * residual))
    assert coefficients.shape == (288, 12) and np.isfinite(coefficients).all()
    parameters["coefficients"] = coefficients.tolist()
    with (ROOT / "fit_cache.npz").open("xb") as stream:
        np.savez(
            stream,
            raw=raw,
            translation=translations,
            features=x,
            target=target,
            nominal=base,
            residual_target=residual,
            weights=weights,
            groups=groups,
        )
    with (ROOT / "parameters.json").open("x") as stream:
        json.dump(parameters, stream, indent=2, allow_nan=False)
        stream.write("\n")
    e.recheck()
    result = dict(
        complete=True,
        rows=4865,
        fit_trajectories=24,
        fit_sources=sources,
        new_fits=1,
        parameter_ref=dict(path=str((ROOT / "parameters.json").resolve()), sha256=sha256(ROOT / "parameters.json")),
        cache_ref=dict(path=str((ROOT / "fit_cache.npz").resolve()), sha256=sha256(ROOT / "fit_cache.npz")),
        source_sha256=e.hashes,
        holdout_rows_used=0,
        probe_rows_used=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "fit_result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("088 unique fit4865rows24trajectories; numerical/performance qualification pending")


if __name__ == "__main__":
    main()
