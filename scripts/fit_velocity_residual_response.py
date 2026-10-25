"""079 training only; requires complete078 qualification before any fit."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.velocity_residual_response import design_features, motor_after_native_period

COLLECTION = Path("outputs/native_gripper_response_interventions_20261025_v3")
ROOT = Path("outputs/velocity_residual_response_20261025")
ROSTER = Path("outputs/gripper_forward_identification_20261025/fit_result.json")
VELOCITY = Path("outputs/gripper_velocity_state_audit_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    closure_path = COLLECTION / "closure_verified.json"
    closure = json.loads(closure_path.read_text())
    assert closure["collection_qualified"] and closure["owner_normal_wait_verified"] and closure["verified_cases"] == 12
    paths = [
        closure_path,
        ROSTER,
        VELOCITY / "audit.json",
        VELOCITY / "velocity_traces.npz",
        Path(__file__),
        Path("src/s2a/velocity_residual_response.py"),
        Path("plans/v0.0.1/079_velocity_residual_response_validation_20261025.md"),
    ]
    hashes = {str(p): digest(p) for p in paths}
    velocity_audit = json.loads((VELOCITY / "audit.json").read_text())
    for path, sha in velocity_audit["source_sha256"].items():
        assert digest(path) == sha
    index = {d["ref"]["path"]: i for i, d in enumerate(velocity_audit["cases"])}
    entries = [d for d in json.loads(ROSTER.read_text())["trajectories"] if d["split"] == "fit"]
    assert len(entries) == 24
    datasets = []
    raw = []
    target = []
    weights = []
    groups = []

    def append(entry, q, v, m, commands, start, stop, total_weight):
        first = len(raw)
        for k in range(start, stop):
            nextmotor = motor_after_native_period(m[k], float(commands[k]))
            np.testing.assert_allclose(nextmotor, m[k + 1], rtol=0, atol=1e-12)
            raw.append([q[k], m[k], nextmotor, v[k]])
            target.append([q[k + 1] - q[k] - v[k], v[k + 1] - v[k]])
            weights.append(total_weight / (stop - start))
            groups.append(len(datasets))
        datasets.append(
            dict(
                **entry,
                row_start=first,
                row_stop=len(raw),
                source_start=start,
                source_stop=stop,
                total_weight=total_weight,
            )
        )

    with np.load(VELOCITY / "velocity_traces.npz", allow_pickle=False) as velocity:
        for entry in entries:
            ref = entry["ref"]
            assert digest(ref["path"]) == ref["sha256"]
            hashes[ref["path"]] = ref["sha256"]
            with np.load(ref["path"], allow_pickle=False) as data:
                q = data["opening"]
                u = data["sent_commands"][:, -1]
                off = data["gripper_current_action_offsets"]
                flat = data["gripper_current_action_flat"]
                m = [float(flat[off[k]]) for k in range(len(q))]
                v = 0.05 * velocity[f"case{index[ref['path']]:02d}_velocity"]
                append(dict(kind="original_native_fit", ref=ref, task=entry["task"]), q, v, m, u, 0, len(u), 0.5 / 24)
    for case in (1, 3, 5, 7, 9, 11):
        folder = COLLECTION / f"case{case:02d}"
        verification = folder / "independent_verified.json"
        result_path = folder / "collection_result.json"
        verified = json.loads(verification.read_text())
        result = json.loads(result_path.read_text())
        assert verified["mechanical_verified"]
        for path in [verification, result_path]:
            hashes[str(path)] = digest(path)
        ref = result["arrays_ref"]
        assert digest(ref["path"]) == ref["sha256"]
        hashes[ref["path"]] = ref["sha256"]
        profile = result["profile"]
        anchor = profile["anchor_cycle"]
        assert profile["branch"] in ("open", "close")
        with np.load(ref["path"], allow_pickle=False) as data:
            q = data["opening"]
            v = 0.05 * (data["gripper_qvel"][:, 0] - data["gripper_qvel"][:, 1]) / 0.079
            off = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            m = [float(flat[off[k]]) for k in range(len(q))]
            u = data["sent_commands"][:, -1]
            append(
                dict(kind="new_intervention_fit", ref=ref, task=profile["task"], case=case),
                q,
                v,
                m,
                u,
                anchor,
                anchor + 24,
                0.5 / 6,
            )
    raw, target, weights, groups = np.array(raw), np.array(target), np.array(weights), np.array(groups)
    assert len(datasets) == 30 and sum(d["kind"] == "new_intervention_fit" for d in datasets) == 6
    np.testing.assert_allclose(weights.sum(), 1, rtol=0, atol=1e-12)
    mean = (weights[:, None] * raw).sum(axis=0)
    scale = np.maximum(np.sqrt((weights[:, None] * (raw - mean) ** 2).sum(axis=0)), 1e-8)
    x = design_features(raw, mean, scale)
    penalty = np.eye(15) * 0.001
    penalty[-1, -1] = 0
    coef = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty, x.T @ (weights[:, None] * target))
    assert np.isfinite(coef).all() and all(digest(p) == s for p, s in hashes.items())
    ROOT.mkdir(exist_ok=False)
    np.savez(ROOT / "fit_cache.npz", raw=raw, target=target, weights=weights, groups=groups)
    parameters = dict(
        seed=42,
        ridge=0.001,
        prior="constant_current_velocity",
        velocity_units="normalized_opening_per_20Hz_period",
        mean=mean.tolist(),
        scale=scale.tolist(),
        coefficients=coef.tolist(),
    )
    (ROOT / "parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    result = dict(
        complete=True,
        validation_executed=False,
        physical_qualification=False,
        source_sha256=hashes,
        datasets=datasets,
        rows=len(raw),
        parameters_sha256=digest(ROOT / "parameters.json"),
        cache_sha256=digest(ROOT / "fit_cache.npz"),
        new_fits=1,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    (ROOT / "fit_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("singlefit rows", len(raw), "newfit cases1/3/5/7/9/11 only;validation not executed")


if __name__ == "__main__":
    main()
