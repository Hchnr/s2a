"""Standalone full first-candidate mechanical replay, without whole-service qualification."""

import os
from pathlib import Path

from audit_wam_position import Evidence
from wam_position_acceptance_checker import source_proof, verify_case

from s2a.clean_training_runtime import publish, reference, require


def main():
    require(all(os.environ.get(k) == "" for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")), "CPU required")
    root = Path("outputs/terminal_development_20261025").resolve()
    output = root / "first_candidate_mechanical_witness.json"
    require(not output.exists(), "Immutable witness required")
    e = Evidence()
    config_ref = reference(root / "development/configs/task0_init0_s42_terminal058.json")
    config = e.json(config_ref["path"])
    fixture = e.json(config["fixture"])
    records = [r for r in fixture["records"] if r["name"] == "task0_init0"]
    require(len(records) == 1, "One exact first raw fixture required")
    directory = (
        root / "development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42"
    )
    result = dict(
        passed=False,
        complete=False,
        partial_development=True,
        config_ref=config_ref,
        new_model_calls=0,
        new_physics_steps=0,
        fresh_cases=0,
    )
    try:
        source_proof(e)
        mechanical = verify_case(
            config,
            records[0],
            config["variants"][0],
            directory,
            e,
            source="wam",
            checkpoint="8400_masked",
            full=True,
            sampling_seed=42,
            first_query_validator=lambda *a, **k: dict(
                standalone_mechanics_only=True, whole_service_binding_verified=False
            ),
        )
        mechanical.pop("first_query003_regression")
        e.recheck()
        result.update(passed=True, complete=True, mechanical=mechanical, input_sha256=e.hashes)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result.update(
            source_ref=reference(Path(__file__)), whole_development_qualified=False, candidate_stability_qualified=False
        )
        publish(output, result)


if __name__ == "__main__":
    main()
