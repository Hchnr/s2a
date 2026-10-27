"""Export only audited current endpoints to a separate model-input manifest."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.synchronous_validation import checked, publish, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.verified.read_text())
    if not value["passed"] or not value["complete"] or not value["eligible_for_model_validation"]:
        raise ValueError("Complete independent physical audit and six stable native controls required")
    if value["native_stable_cases"] != 6 or len(value["current_model_inputs"]) != 24:
        raise ValueError("Fixed current-input denominator changed")
    allowed = {
        "name",
        "task",
        "anchor_frame",
        "image_ref",
        "current_time_s",
        "current_sim_state_sha256",
        "measured_state128",
        "state_mask128",
        "initial_sync_certified",
    }
    inputs = value["current_model_inputs"]
    names = set()
    for record in inputs:
        if set(record) != allowed or not record["initial_sync_certified"] or record["name"] in names:
            raise ValueError("Current-only schema or unique identity changed")
        names.add(record["name"])
        state = np.asarray(record["measured_state128"])
        mask = np.asarray(record["state_mask128"])
        if (
            state.shape != (128,)
            or not np.isfinite(state).all()
            or mask.dtype != np.bool_
            or not np.array_equal(mask, np.arange(128) < 10)
        ):
            raise ValueError("Current measured-state/mask contract changed")
        checked(record["image_ref"])
    result = {
        "schema_version": "s2a.synchronous-validation-current-inputs.v1",
        "passed": True,
        "complete": True,
        "script_ref": reference(Path(__file__)),
        "source_audit_ref": reference(args.verified),
        "source_prepared_ref": value["prepared_ref"],
        "current_inputs": inputs,
        "coordinate_contract": {
            "state128": "base-frame TCP xyz, rotation columns1/2, observed aperture; active10",
            "position_unit": "m",
            "time_unit": "s",
            "physical_dt_s": 0.05,
        },
        "future_archive_refs_exported": 0,
        "expert_commands_exported": 0,
        "task_predicates_exported": 0,
        "model_sampling_authorized": False,
        "fresh_authorized": False,
    }
    print(json.dumps(publish(args.output, result)), flush=True)


if __name__ == "__main__":
    main()
