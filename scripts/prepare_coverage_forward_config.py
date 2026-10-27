"""098 bind fixed protocol only after complete097 data/lifecycle qualification."""

from pathlib import Path

from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def reference(path):
    return dict(path=str(path.resolve()), sha256=sha256(path))


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/098_coverage_conditioned_arm_forward_20261025.md")
    additional = Path("outputs/training_response_coverage_20261025/closure_verified.json")
    closed = e.json(additional)
    assert closed["complete"] and closed["data_qualified"] and closed["verified_cases"] == 147
    assert closed["owner_normal_wait_verified"] and closed["physics_steps"] == 8281
    for path, signature in closed["source_sha256"].items():
        e.check(path, signature)
    config = e.json("configs/history_conditioned_arm_forward_20261025.json")
    prior = Path("outputs/history_conditioned_arm_forward_20261025/closure_verified.json")
    c = e.json(prior)
    assert c["complete"] and c["numerically_verified"] and c["owner_normal_wait_verified"]
    params = prior.parent / "parameters.json"
    fit = e.json(prior.parent / "fit_result.json")
    e.check(params, fit["parameter_ref"]["sha256"])
    config.update(
        output_dir="outputs/coverage_conditioned_arm_forward_20261025",
        contrast_equations=72,
        additional_anchor_groups=21,
        total_equations=4937,
        probe_rows_used=144,
        additional_closure_ref=reference(additional),
        statistics_ref=reference(params),
        prior_fit_closure_ref=reference(prior),
    )
    e.recheck()
    config["binding_source_sha256"] = e.hashes
    write_once("configs/coverage_conditioned_arm_forward_20261025.json", config)
    print("098 bound complete097 qualification; parameter_fits0")


if __name__ == "__main__":
    main()
