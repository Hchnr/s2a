"""Fixed validation identities and current-only input boundaries for protocol053."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

ROSTER = (
    (0, "demo_16", 95),
    (0, "demo_32", 92),
    (1, "demo_4", 158),
    (1, "demo_27", 177),
    (2, "demo_14", 106),
    (2, "demo_24", 104),
)


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def checked(ref):
    path = Path(ref["path"])
    if reference(path) != ref:
        raise ValueError("Registered file changed: " + str(path))
    return path


def publish(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)

    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=convert, allow_nan=False)
        stream.write("\n")
    return reference(path)


def validate_records(records, forbidden_hashes):
    if len(records) != len(ROSTER):
        raise ValueError("All six fixed validation cases required")
    identities = set(forbidden_hashes)
    for record, expected in zip(records, ROSTER, strict=True):
        task, demo, count = expected
        if (record["task_index"], record["demo"], record["native_steps"]) != expected:
            raise ValueError("Fixed validation task/demo/length changed")
        if record["anchor_frames"] != [j * (count - 40) // 3 for j in range(4)]:
            raise ValueError("Four fixed complete H40 anchors required")
        state = np.asarray(record["initial_state"], dtype=np.float64)
        identity = hashlib.sha256(state.tobytes()).hexdigest()
        if state.shape != (79,) or identity != record["initial_state_sha256"] or identity in identities:
            raise ValueError("Initial state identity/shape or isolation changed")
        identities.add(identity)
        if hashlib.sha256(record["model_xml"].encode()).hexdigest() != record["model_xml_sha256"]:
            raise ValueError("Model XML changed")
        if (task, record["task_id"]) not in ((0, 8), (1, 9), (2, 6)):
            raise ValueError("Task controller environment changed")
        if not record["name"] == f"task{task}_validation_{demo}":
            raise ValueError("Case name changed")


def load_prepared(ref):
    value = json.loads(checked(ref).read_text())
    if Path(value["root"]).resolve() != REPO / "outputs/synchronous_validation_20261024":
        raise ValueError("Validation output root changed")
    if value["budget"] != {
        "environments": 6,
        "physics_steps": 1212,
        "current_canvas_images": 24,
        "future_rgb_frames": 0,
        "model_calls": 0,
        "fresh_cases": 0,
    }:
        raise ValueError("Validation budget changed")
    validate_records(value["records"], value["forbidden_initial_hashes"])
    for mapping in [value["source_sha256"], value["snapshot_sha256"], value["input_sha256"]]:
        for filename, digest in mapping.items():
            if sha256(filename) != digest:
                raise ValueError("Preparation source/input changed: " + filename)
    for record in value["records"]:
        path = checked({"path": record["expert_path"], "sha256": record["expert_sha256"]})
        native = np.load(path, allow_pickle=False)
        if native.shape != (record["native_steps"], 7) or native.dtype != np.float64 or not np.isfinite(native).all():
            raise ValueError("Native commands changed")
    return value


def current_model_input(anchor, data, *, task, case_name):
    """Export measured current endpoint only; future arrays never enter this record."""
    frame = anchor["frame_index"]
    receipt = anchor["current_endpoint"]
    if receipt["sim_time"] != float(data["sim_time"][frame]):
        raise ValueError("Current image/state clock differs")
    if receipt["sim_state_sha256"] != hashlib.sha256(data["sim_states"][frame].tobytes()).hexdigest():
        raise ValueError("Current image/state identity differs")
    return {
        "name": f"{case_name}_a{frame}",
        "task": task,
        "anchor_frame": frame,
        "image_ref": anchor["canvas_png"],
        "current_time_s": receipt["sim_time"],
        "current_sim_state_sha256": receipt["sim_state_sha256"],
        "measured_state128": data["state128"][frame].tolist(),
        "state_mask128": data["state_mask"][frame].tolist(),
        "initial_sync_certified": True,
    }
