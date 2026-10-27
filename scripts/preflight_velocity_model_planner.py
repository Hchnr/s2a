"""080 frozen-model planning feasibility/latency on known response-validation states."""

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from s2a.velocity_model_planner import VelocityModelPlanner, fit_velocity_envelope

ROOT = Path("outputs/velocity_model_planner_preflight_20261025")
MODEL = Path("outputs/velocity_residual_response_20261025")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parameter_path = MODEL / "parameters.json"
    cache_path = MODEL / "fit_cache.npz"
    validation_path = MODEL / "independent_validation_verified.json"
    validation = json.loads(validation_path.read_text())
    assert validation["numerically_verified"] and validation["offline_performance_eligible"]
    fit = json.loads((MODEL / "fit_result.json").read_text())
    assert digest(parameter_path) == fit["parameters_sha256"] and digest(cache_path) == fit["cache_sha256"]
    with np.load(cache_path, allow_pickle=False) as data:
        envelope = fit_velocity_envelope(data["raw"], data["target"])
    planner = VelocityModelPlanner(json.loads(parameter_path.read_text()), velocity_envelope=envelope)
    velocity_root = Path("outputs/gripper_velocity_state_audit_20261025")
    audit_path = velocity_root / "audit.json"
    traces_path = velocity_root / "velocity_traces.npz"
    audit = json.loads(audit_path.read_text())
    index = {row["ref"]["path"]: i for i, row in enumerate(audit["cases"])}
    paths = [
        parameter_path,
        cache_path,
        validation_path,
        MODEL / "fit_result.json",
        audit_path,
        traces_path,
        Path(__file__),
        Path("src/s2a/velocity_model_planner.py"),
        Path("src/s2a/velocity_residual_response.py"),
    ]
    hashes = {str(p): digest(p) for p in paths}
    queries = []
    elapsed = []
    with np.load(traces_path, allow_pickle=False) as traces:
        for row in validation["cases"]:
            ref = row["ref"]
            assert digest(ref["path"]) == ref["sha256"]
            hashes[ref["path"]] = ref["sha256"]
            with np.load(ref["path"], allow_pickle=False) as data:
                q = data["opening"].copy()
                off = data["gripper_current_action_offsets"]
                flat = data["gripper_current_action_flat"]
                motors = [flat[off[k] : off[k + 1]].copy() for k in range(len(q))]
                if "gripper_qvel" in data:
                    v = 0.05 * (data["gripper_qvel"][:, 0] - data["gripper_qvel"][:, 1]) / 0.079
                else:
                    v = 0.05 * traces[f"case{index[ref['path']]:02d}_velocity"]
            for k in (0, (len(q) - 1) // 2, len(q) - 9):
                reference = q[k + 1 : k + 41]
                if not queries:
                    for _ in range(2):
                        planner.plan(float(q[k]), motors[k], float(v[k]), reference)
                start = time.perf_counter()
                result = planner.plan(float(q[k]), motors[k], float(v[k]), reference)
                duration = time.perf_counter() - start
                elapsed.append(duration)
                queries.append(
                    dict(
                        ref=ref,
                        cycle=k,
                        current_opening=float(q[k]),
                        current_motor=motors[k].tolist(),
                        current_period_velocity=float(v[k]),
                        generated_reference=reference.tolist(),
                        plan=asdict(result),
                        elapsed_s=duration,
                    )
                )
    assert all(digest(p) == sha for p, sha in hashes.items())
    ROOT.mkdir(exist_ok=False)
    output = dict(
        complete=True,
        feasible_queries=len(queries),
        velocity_envelope=envelope,
        queries=queries,
        latency_s=dict(median=float(np.median(elapsed)), p95=float(np.quantile(elapsed, 0.95)), maximum=max(elapsed)),
        source_sha256=hashes,
        new_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
        real_time_qualification=False,
    )
    payload = json.dumps(output, indent=2, allow_nan=False) + "\n"
    with (ROOT / "preflight.json").open("x") as stream:
        stream.write(payload)
    print("feasible queries", len(queries), "envelope", envelope, "latency_s", output["latency_s"])


if __name__ == "__main__":
    main()
