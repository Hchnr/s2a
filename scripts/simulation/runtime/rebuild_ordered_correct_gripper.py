"""Run the registered CPU-only case08 completed-gripper-history audit."""

import hashlib
import json
import os
import traceback
from pathlib import Path

from s2a.research.gripper.ordered_correct_history import reconstruct_ordered_gripper

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/tracking/gripper/ordered_correct_history_rebuild_20261111.json"
OUTPUT = ROOT / "outputs/ordered_correct_history_rebuild_20261111"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checked(path, digest):
    path = Path(path)
    if not path.resolve().is_relative_to(ROOT) or sha256(path) != digest:
        raise ValueError(f"Input path or SHA256 differs: {path}")
    return json.loads(path.read_text())


def read_ref(ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("An exact input reference is required")
    return read_checked(ref["path"], ref["sha256"])


def load_inputs():
    config = json.loads(CONFIG.read_text())
    if (
        config["schema"] != "s2a.ordered-correct-gripper-rebuild.v1"
        or ROOT / config["output"] != OUTPUT
        or [config[k] for k in ("case_index", "seed", "prefix_steps", "model_calls", "physics_steps", "fresh_cases")]
        != [8, 42, 80, 0, 0, 0]
        or config["comparison_atol"] != 1e-10
        or config["automatic_retry"] is not False
    ):
        raise ValueError("Registered reconstruction scope differs")
    prepared = read_checked(ROOT / config["prepared"], config["prepared_sha256"])
    entry = prepared["profiles"][8]
    if entry["case_index"] != 8 or entry["profile"]["physics_steps"] != 182:
        raise ValueError("Original correct-reference case differs")
    base, control, tool = (read_ref(entry[k]) for k in ("execution_config_ref", "controller_ref", "tool_ref"))
    parameters = read_ref(base["response_parameters_ref"])
    rows = read_checked(ROOT / config["rows"], config["rows_sha256"])
    chunks = read_checked(ROOT / config["chunks"], config["chunks_sha256"])
    if len(rows) != 182 or len(chunks) != 23:
        raise ValueError("Original complete case denominator differs")
    return {
        "config": config,
        "rows": rows[:80],
        "chunks": chunks[:10],
        "tool": tool["source_from_tcp"],
        "parameters": parameters,
        "velocity_envelope": base["gripper_velocity_envelope"],
        "gripper_config": control["gripper_config"],
        "refs": {
            "execution_config": entry["execution_config_ref"],
            "controller": entry["controller_ref"],
            "tool": entry["tool_ref"],
            "parameters": base["response_parameters_ref"],
        },
    }


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only reconstruction requires empty CUDA and LD")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = {"complete": False, "passed": False, "model_calls": 0, "physics_steps": 0, "fresh_cases": 0}
    try:
        inputs = load_inputs()
        controller, history = reconstruct_ordered_gripper(
            inputs["rows"],
            inputs["chunks"],
            inputs["tool"],
            inputs["parameters"],
            velocity_envelope=inputs["velocity_envelope"],
            gripper_config=inputs["gripper_config"],
            progress=lambda value: result.update(progress=value),
        )
        if history["steps"] != 80 or history["candidate_sequences"] != history["planner_calls"] * 3**8:
            raise ValueError("The complete original prefix and planner budget must be accounted for")
        result.update(
            complete=True,
            passed=True,
            config_sha256=sha256(CONFIG),
            script_sha256=sha256(__file__),
            implementation_sha256=sha256(ROOT / "src/s2a/research/gripper/ordered_correct_history.py"),
            input_refs=inputs["refs"],
            history=history,
            automatic_retry=False,
        )
        del controller
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
