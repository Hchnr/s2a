"""Run new per-step checker on one fully qualified historical 35-node capture."""

import os
from pathlib import Path

from verify_wam_clean_training_capture import Evidence, require
from verify_wam_terminal_points import compact_points

from s2a.clean_training_runtime import publish, reference
from s2a.terminal_candidate import candidate_head


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    output = Path("outputs/terminal_sampler_preparation_20261025")
    output.mkdir(exist_ok=True)
    require(not (output / "historical_point_regression_v2.json").exists(), "Immutable output required")
    e = Evidence()
    proof_ref = reference(Path("outputs/terminal_solver_convergence_20261025/fit_verified.json"))
    policy, params_ref = candidate_head(proof_ref, read_json=e.json)
    root = Path("outputs/synchronous_state_probe052_continuation_20261024")
    numerical = e.json(reference(root / "numerical_verified.json"))
    require(
        numerical["passed"]
        is numerical["complete"]
        is numerical["numerical_contract_passed"]
        is numerical["lifecycle_passed"]
        is True,
        "Qualified historical result required",
    )
    sampled = e.json(reference(root / "state_sampling/result.json"))
    generation = next(g for g in sampled["generations"] if g["variant"] == "probe052")
    require(
        any(g["response_ref"] == generation["response_ref"] for g in numerical["generated_refs"]),
        "Historical generation not qualified",
    )
    ledger = e.json(generation["compact_ledger_ref"])
    header = e.json(ledger["request_ref"])
    trace = e.json(generation["trace_ref"])
    base = (policy.baseline.coefficients, policy.baseline.center, policy.baseline.scale)
    directory = Path(generation["compact_ledger_ref"]["path"]).parent.parent
    _, changed = compact_points(e, ledger, header, directory, trace, head=(base, base))
    e.finish()
    publish(
        output / "historical_point_regression_v2.json",
        dict(
            passed=True,
            complete=True,
            proof_ref=proof_ref,
            params_ref=params_ref,
            historical_generation=generation["name"],
            historical_nodes_verified=35,
            changed_active_values=changed,
            input_sha256=e.inputs,
            source_refs=[
                reference(Path(__file__)),
                reference(Path("scripts/verify_wam_terminal_points.py")),
                reference(Path("src/s2a/terminal_candidate.py")),
                reference(Path("src/s2a/terminal_capture.py")),
            ],
            boundary="Historical052 selected in both checker branches; validates inherited full math on actual arrays, not new058 deployment.",
            new_model_calls=0,
            physics_steps=0,
            fresh_cases=0,
        ),
    )


if __name__ == "__main__":
    main()
