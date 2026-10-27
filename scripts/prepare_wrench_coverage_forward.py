"""107 bind fully qualified106 data while preserving the104 model specification."""

from pathlib import Path

from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def reference(path):
    return dict(path=str(path.resolve()), sha256=sha256(path))


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/107_wrench_coverage_forward_20261025.md")
    old = Path("configs/wrench_conditioned_arm_forward_20261025.json")
    config = e.json(old)
    closure_path = Path("outputs/wrench_coverage_interventions_20261025/closure_verified.json")
    closure = e.json(closure_path)
    assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
    assert closure["verified_cases"] == 147 and closure["physics_steps"] == 13209
    for path, digest in closure["source_sha256"].items():
        e.check(path, digest)
    prior_path = Path("outputs/wrench_conditioned_arm_forward_20261025/closure_verified.json")
    prior = e.json(prior_path)
    assert prior["complete"] and prior["numerically_verified"] and prior["owner_normal_wait_verified"]
    original_fit = e.json("outputs/wrench_conditioned_arm_forward_20261025/fit_result.json")
    original_parameters = original_fit["parameter_ref"]
    e.check(original_parameters["path"], original_parameters["sha256"])
    config.update(
        output_dir="outputs/wrench_coverage_forward_20261025",
        contrast_equations=135,
        coverage_closure_ref=reference(closure_path),
        source_protocol_ref=reference(old),
        prior_fit_closure_ref=reference(prior_path),
        original_condition_statistics_unchanged=True,
        original_parameter_ref=original_parameters,
        planned_total_equations=5000,
    )
    assert config["native_weight"] == config["contrast_weight"] == 0.5
    assert config["ridge"] == 0.001 and config["design_dimension"] == 872
    e.recheck()
    config["binding_source_sha256"] = e.hashes
    write_once("configs/wrench_coverage_forward_20261025.json", config)
    print("107 qualified106 bound;135 contrasts/5000 total equations; new_fits0")


if __name__ == "__main__":
    main()
