"""Check the registered case08 arm history at its measured step80 endpoint."""

import json
import os
import traceback

import numpy as np

from s2a.research.joint_control.preview.ordered_correct_arm_history import restore_ordered_arm_history
from scripts.simulation.runtime.rebuild_ordered_correct_gripper import CONFIG, ROOT, load_inputs, read_checked, sha256

OUTPUT = ROOT / "outputs/ordered_correct_arm_rebuild_20261111"


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only reconstruction requires empty CUDA and LD")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {"complete": False, "passed": False, "model_calls": 0, "physics_steps": 0, "fresh_cases": 0}
    try:
        inputs = load_inputs()
        config = inputs["config"]
        all_rows = read_checked(ROOT / config["rows"], config["rows_sha256"])
        prior, preaction = inputs["rows"][-1], all_rows[80]
        if prior["step"] != 79 or preaction["step"] != 80:
            raise ValueError("Original completed and pre-action row clocks differ")
        site, twist = preaction["site_pose"], preaction["site_twist"]
        tracking_before = preaction["refinement"]["history_before"]
        control = read_checked(inputs["refs"]["controller"]["path"], inputs["refs"]["controller"]["sha256"])
        coupled, tracking, difference = restore_ordered_arm_history(
            prior, site, twist, control["contract"], tracking_before
        )
        if (
            coupled.completed_periods != 80
            or tracking.history.completed_periods != 80
            or tracking.phase != "ready"
            or tracking.history.pending_plan is not None
            or not np.array_equal(tracking.history.last_plan, prior["refinement"]["planned_commands"])
        ):
            raise ValueError("Original completed arm history was not restored")
        result.update(
            complete=True,
            passed=True,
            completed_periods=80,
            max_innovation_difference=difference,
            previous_command=prior["command"],
            warm_start_horizon=len(tracking.history.last_plan),
            config_sha256=sha256(CONFIG),
            script_sha256=sha256(__file__),
            implementation_sha256=sha256(
                ROOT / "src/s2a/research/joint_control/preview/ordered_correct_arm_history.py"
            ),
            input_refs=inputs["refs"],
            automatic_retry=False,
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        with (OUTPUT / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("input_refs", "traceback")}), flush=True)


if __name__ == "__main__":
    main()
