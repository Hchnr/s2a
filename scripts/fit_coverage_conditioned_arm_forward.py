"""098 causal-history conditional fit with the fixed72 paired equations."""

import json
from pathlib import Path

import numpy as np
from arm_coverage_training import contrast_rows
from audit_wam_position import Evidence
from scipy.linalg import cho_factor, cho_solve

from s2a.artifacts import sha256
from s2a.history_conditioned_arm_forward import design, nominal, raw_condition, targets, with_history

CONFIG = Path("configs/coverage_conditioned_arm_forward_20261025.json")
ROOT = Path("outputs/coverage_conditioned_arm_forward_20261025")


def main():
    e = Evidence()
    e.read(__file__)
    e.read("src/s2a/state_conditioned_arm_forward.py")
    e.read("src/s2a/history_conditioned_arm_forward.py")
    e.read("scripts/arm_coverage_training.py")
    config = e.json(CONFIG)
    assert config["seed"] == 42 and config["ridge"] == 0.001
    for key in (
        "data_closure_ref",
        "data_result_ref",
        "paired_closure_ref",
        "paired_analysis_ref",
        "prior_fit_closure_ref",
        "statistics_ref",
        "additional_closure_ref",
    ):
        ref = config[key]
        e.check(ref["path"], ref["sha256"])
    closure = e.json(config["data_closure_ref"]["path"])
    assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
    source = e.json(config["data_result_ref"]["path"])
    fit = [p for p in source["trajectories"] if p["split"] == "fit"]
    assert len(fit) == 24
    additional = e.json(config["additional_closure_ref"]["path"])
    assert additional["complete"] and additional["data_qualified"] and additional["verified_cases"] == 147
    assert additional["owner_normal_wait_verified"]
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
        data = with_history(data, dict(dt=0.05, kp=contract["kp"], kd=contract["kd"]))
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

    assert config["native_weight"] == config["contrast_weight"] == 0.5
    assert config["training_anchor_groups"] == [0, 3, 5] and config["validation_anchor_groups"] == [1, 2, 4]

    def stats(values):
        mean = np.sum(values * weights[:, None], axis=0)
        scale = np.sqrt(np.sum((values - mean) ** 2 * weights[:, None], axis=0))
        scale[scale < 1e-8] = 1
        return mean.tolist(), scale.tolist()

    state_mean, state_scale = stats(raw)
    translation_mean, translation_scale = stats(translations)
    old_stats = e.json(config["statistics_ref"]["path"])
    for key, value in (
        ("state_mean", state_mean),
        ("state_scale", state_scale),
        ("translation_mean", translation_mean),
        ("translation_scale", translation_scale),
    ):
        np.testing.assert_allclose(value, old_stats[key], rtol=0, atol=1e-12)
    state_mean, state_scale = old_stats["state_mean"], old_stats["state_scale"]
    translation_mean, translation_scale = old_stats["translation_mean"], old_stats["translation_scale"]
    assert len(state_mean) == 83
    parameters = dict(
        schema="s2a.coverage-conditioned-arm-forward.v1",
        seed=42,
        ridge=0.001,
        dt=0.05,
        kp=contract["kp"],
        kd=contract["kd"],
        state_mean=state_mean,
        state_scale=state_scale,
        translation_mean=translation_mean,
        translation_scale=translation_scale,
        native_weight=0.5,
        contrast_weight=0.5,
        contrast_equations=72,
        training_anchor_groups=[0, 3, 5],
        validation_anchor_groups=[1, 2, 4],
    )
    x = np.vstack([design(data, parameters) for data in data_sets])
    base = np.vstack([nominal(data, parameters) for data in data_sets])
    target = np.vstack([targets(data) for data in data_sets])
    cx, cy, cb, contrast_sources = contrast_rows(config, e, parameters)
    native_rows = len(x)
    x = np.vstack([x, cx])
    target = np.vstack([target, cy])
    base = np.vstack([base, cb])
    weights = np.r_[weights * 0.5, np.full(72, 0.5 / 72)]
    groups = np.r_[groups, [row["equation_group"] for row in contrast_sources]]
    row_kind = np.r_[np.zeros(native_rows, dtype=np.int64), np.ones(72, dtype=np.int64)]
    assert (
        x.shape == (4937, 336)
        and abs(weights[:native_rows].sum() - 0.5) < 1e-12
        and abs(weights[native_rows:].sum() - 0.5) < 1e-12
    )
    residual = target - base
    penalty = np.ones(336)
    penalty[0] = 0
    gram = x.T @ (weights[:, None] * x) + 0.001 * np.diag(penalty)
    with (ROOT / "fit_attempt.json").open("x") as stream:
        json.dump(
            dict(
                attempt=1,
                rows=4937,
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
    assert coefficients.shape == (336, 12) and np.isfinite(coefficients).all()
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
            row_kind=row_kind,
        )
    with (ROOT / "parameters.json").open("x") as stream:
        json.dump(parameters, stream, indent=2, allow_nan=False)
        stream.write("\n")
    e.recheck()
    result = dict(
        complete=True,
        rows=4937,
        native_rows=4865,
        contrast_equations=72,
        contrast_sources=contrast_sources,
        fit_trajectories=24,
        fit_sources=sources,
        new_fits=1,
        parameter_ref=dict(path=str((ROOT / "parameters.json").resolve()), sha256=sha256(ROOT / "parameters.json")),
        cache_ref=dict(path=str((ROOT / "fit_cache.npz").resolve()), sha256=sha256(ROOT / "fit_cache.npz")),
        source_sha256=e.hashes,
        holdout_rows_used=0,
        probe_rows_used=144,
        held_anchor_groups_used=0,
        physics_steps=0,
        model_calls=0,
        fresh_cases=0,
    )
    with (ROOT / "fit_result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("098 unique fit4865native+72contrast; qualification pending")


if __name__ == "__main__":
    main()
