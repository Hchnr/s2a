"""116 bind fully qualified115 data to the unchanged109 forward specification."""

from pathlib import Path

from audit_wam_position import Evidence
from collect_wam_native_training import write_once

from s2a.artifacts import sha256


def reference(path):
    return dict(path=str(path.resolve()), sha256=sha256(path))


def main():
    e = Evidence()
    e.read(__file__)
    e.read("plans/v0.0.1/116_condition_span_forward_20261025.md")
    old = Path("configs/joint_wrench_condition_forward_20261025.json")
    config = e.json(old)
    data_path = Path("outputs/condition_span_interventions_20261025/closure_verified.json")
    closure = e.json(data_path)
    assert closure["complete"] and closure["data_qualified"] and closure["owner_normal_wait_verified"]
    assert closure["verified_cases"] == 147 and closure["physics_steps"] == 10626
    for path, digest in closure["source_sha256"].items():
        e.check(path, digest)
    prior_path = Path("outputs/joint_wrench_condition_forward_20261025/closure_verified.json")
    prior = e.json(prior_path)
    assert prior["complete"] and prior["numerically_verified"] and prior["owner_normal_wait_verified"]
    fit = e.json("outputs/joint_wrench_condition_forward_20261025/fit_result.json")
    parameters = fit["parameter_ref"]
    e.check(parameters["path"], parameters["sha256"])
    config.update(
        output_dir="outputs/condition_span_forward_20261025",
        contrast_equations=198,
        planned_total_equations=5063,
        span_closure_ref=reference(data_path),
        source_protocol_ref=reference(old),
        prior_fit_closure_ref=reference(prior_path),
        original_parameter_ref=parameters,
    )
    assert config["native_weight"] == config["contrast_weight"] == 0.5
    assert config["design_dimension"] == 1128 and config["ridge"] == 0.001
    e.recheck()
    config["binding_source_sha256"] = e.hashes
    write_once("configs/condition_span_forward_20261025.json", config)
    print("116 qualified115 bound;198contrasts/5063equations;new_fits0")


if __name__ == "__main__":
    main()
