"""Read-only, provenance-aligned extraction of the selected DROID episode."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from s2a.artifacts import code_fingerprint, sha256, write_json

R2_COLUMNS = [
    "action",
    "action_dim_mask",
    "action_control_mask",
    "action_valid",
    "observation.state",
    "observation.state_mask",
    "timestamp",
    "episode_index",
    "frame_index",
    "source_frame_index",
    "raw_frame_index",
    "source_physical_timestamp",
    "transition_dt",
    "physical_time_valid",
    "nominal_transition_dt",
    "robot_timestamp_seconds",
    "robot_timestamp_nanos",
    "gripper_action_timestamp",
    "gripper_action_timestamp_mask",
]
R1_COLUMNS = [
    "frame_index",
    "observation.state.cartesian_position",
    "observation.state.joint_positions",
    "observation.state.joint_velocities",
    "observation.state.gripper_position",
    "action.cartesian_position",
    "action.cartesian_velocity",
    "action.joint_position",
    "action.joint_velocity",
    "action.gripper_position",
    "action.gripper_velocity",
    "action_valid",
]
META_COLUMNS = [
    "episode_index",
    "length",
    "training_eligible",
    "action_domain",
    "source_episode_id",
    "data_path",
    "processing_path",
    "processing_index",
    "state_reference",
    "eef_gripper_source",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_episode(path, columns, episode):
    table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode)])
    require(len(table) > 0, f"episode {episode} absent from {path}")
    return {key: np.asarray(table[key].to_pylist()) for key in columns}


def episode_metadata(root, episode, columns):
    # Metadata file selection is explicit and configurable; never scan videos/data shards.
    table = pq.read_table(root, columns=columns, filters=[("episode_index", "=", episode)])
    require(len(table) == 1, f"episode metadata must be unique: {episode}")
    return table.to_pylist()[0]


def pose_transition_mask(rows):
    state = rows["observation.state_mask"][:, :9].all(axis=1)
    mask = (
        rows["action_valid"].astype(bool)
        & rows["action_dim_mask"][:, :9].all(axis=1)
        & rows["physical_time_valid"].astype(bool)
        & state
        & (rows["transition_dt"] > 0)
    )
    # Even action_valid=true cannot make the terminal observation a transition.
    mask[:-1] &= state[1:] & (np.diff(rows["frame_index"]) == 1)
    mask[-1] = False
    return mask


def align_original(rows, original):
    frames = original["frame_index"]
    require(len(np.unique(frames)) == len(frames), "duplicate R1 frame IDs")
    require(len(np.unique(rows["frame_index"])) == len(rows["frame_index"]), "duplicate R2 frame IDs")
    require(np.array_equal(rows["raw_frame_index"], rows["source_frame_index"]), "raw/source provenance conflict")
    lookup = {int(f): i for i, f in enumerate(frames)}
    require(all(int(f) in lookup for f in rows["source_frame_index"]), "missing R1 provenance frame")
    indices = [lookup[int(f)] for f in rows["source_frame_index"]]
    return {key: value[indices] for key, value in original.items()}


def split_indices(frame_ids, interval, history=0, lag=0, horizon=1):
    """All input/history frames and endpoint labels stay inside [lo, hi].

    Transitions are [lo,hi); their boundary observation at hi is allowed.
    Returns command indices; labels occur lag rows later.
    """
    f = np.asarray(frame_ids)
    require(history >= 0 and lag >= 0 and horizon >= 1, "invalid temporal support")
    lo, hi = interval
    result = []
    for i in range(history, len(f) - lag - horizon):
        support = f[i - history : i + lag + horizon + 1]
        if support[0] >= lo and support[-1] <= hi and np.all(np.diff(support) == 1):
            result.append(i)
    return np.asarray(result, dtype=int)


def extract(config, out):
    sources, selection = config["sources"], config["selection"]
    r2, r1 = Path(sources["action128_root"]), Path(sources["original_root"])
    files = config["files"]
    paths = {
        "catalog": Path(sources["catalog"]),
        "r2_metadata": r2 / files["r2_metadata"],
        "r1_metadata": r1 / files["r1_metadata"],
        "r2_numeric": r2 / files["r2_numeric"],
        "r1_numeric": r1 / files["r1_numeric"],
        "state_contract": r2 / "meta/eef_state_contract.json",
        "tool_contract": r2 / "meta/eef_tools/droid_robotiq2f85.json",
        "time_contract": r2 / "meta/eef_time.json",
        "receipt": r2 / files["receipt"],
        "processing": r2 / files["processing"],
    }
    ep, ep1 = selection["episode_index"], selection["original_episode_index"]
    meta = episode_metadata(paths["r2_metadata"], ep, META_COLUMNS)
    orig_meta = episode_metadata(paths["r1_metadata"], ep1, ["episode_id", "length"])
    catalog = pq.read_table(paths["catalog"], columns=["data"]).to_pylist()
    require(
        any(Path(json.loads(c["data"])["root"]).resolve() == r2.resolve() for c in catalog), "catalog root mismatch"
    )
    require(meta["data_path"] == files["r2_numeric"], "numeric path differs from metadata")
    require(meta["training_eligible"] or not selection["require_training_eligible"], "episode ineligible")
    require(meta["action_domain"] == sources["action_domain"], "action domain mismatch")
    require(orig_meta["episode_id"] == sources["source_episode_id"], "original episode ID mismatch")
    with paths["processing"].open() as stream:
        for i, line in enumerate(stream):
            if i == meta["processing_index"]:
                processing = json.loads(line)
                break
        else:
            raise ValueError("missing processing record")
    binding = processing["source_processing"]
    require(
        binding["source_episode"] == ep1 and Path(binding["source_dataset"]).resolve() == r1.resolve(),
        "R1 processing binding mismatch",
    )
    receipt = json.loads(paths["receipt"].read_text())
    receipt_ep = [e for e in receipt["episodes"] if e["episode_index"] == ep]
    require(len(receipt_ep) == 1, "receipt episode missing/duplicated")
    receipt_ep = receipt_ep[0]
    require(receipt_ep["pose_source"]["reference_link"] == "panda_link0", "unknown pose reference")
    require(
        receipt_ep["gripper_source"]["kind"] == "native_driver_feedback_and_absolute_command",
        "unknown gripper semantics",
    )
    rows = read_episode(paths["r2_numeric"], R2_COLUMNS, ep)
    original = read_episode(paths["r1_numeric"], R1_COLUMNS, ep1)
    order = np.argsort(rows["frame_index"])
    rows = {key: value[order] for key, value in rows.items()}
    original = align_original(rows, original)
    mask = pose_transition_mask(rows)
    require(len(mask) == meta["length"] == orig_meta["length"], "episode length mismatch")
    start, end = selection["frame_start_inclusive"], selection["frame_end_exclusive"]
    selected = (rows["frame_index"] >= start) & (rows["frame_index"] < end)
    require(selected.sum() == end - start and mask[selected].all(), "invalid or missing selected pose transitions")
    require(rows["observation.state_mask"][selected, :10].all(), "invalid selected state")
    require(rows["action_dim_mask"][selected, 9].all(), "invalid selected gripper command")
    require(
        rows["action_control_mask"][selected, 9].all() and not rows["action_control_mask"][:, :9].any(),
        "unexpected arm/gripper control mask",
    )
    dt = rows["transition_dt"][:-1]
    require(np.allclose(dt, np.diff(rows["source_physical_timestamp"]), atol=1e-9, rtol=0), "physical dt mismatch")
    require((dt > 0).all(), "nonpositive physical clock")
    require(rows["transition_dt"][-1] == 0 and not rows["physical_time_valid"][-1], "invalid terminal time contract")
    g_cmd = np.max(np.abs(rows["action"][:, 9] - (1 - original["action.gripper_position"])))
    g_state = np.max(np.abs(rows["observation.state"][:, 9] - (1 - original["observation.state.gripper_position"])))
    require(max(g_cmd, g_state) < 1e-6, "gripper mapping mismatch")
    keep = (rows["frame_index"] >= max(0, start - selection["history_frames"])) & (rows["frame_index"] <= end)
    arrays = {key.replace(".", "_"): value[keep] for key, value in rows.items()}
    arrays.update({"r1_" + key.replace(".", "_"): value[keep] for key, value in original.items()})
    arrays["pose_transition_valid"] = mask[keep]
    tool = json.loads(paths["tool_contract"].read_text())
    arrays["H_FC"] = np.array(tool["H_source_from_tcp"])
    for key, value in arrays.items():
        require(
            value.dtype != object and (not np.issubdtype(value.dtype, np.number) or np.isfinite(value).all()),
            f"invalid/nonfinite array {key}",
        )
    np.savez_compressed(out / "sample.npz", **arrays)
    audit = {
        "episode_rows": len(mask),
        "episode_valid_pose_transitions": int(mask.sum()),
        "episode_valid_gripper_commands": int(rows["action_dim_mask"][:, 9].sum()),
        "selected_transitions": int(selected.sum()),
        "stored_observations_with_history": int(keep.sum()),
        "dt_min_median_max_s": [float(np.min(dt)), float(np.median(dt)), float(np.max(dt))],
        "physical_interval_s": rows["source_physical_timestamp"][[start, end]],
        "media_interval_s": rows["timestamp"][[start, end]],
        "gripper_command_max_error": float(g_cmd),
        "gripper_state_max_error": float(g_state),
        "terminal_action_valid": bool(rows["action_valid"][-1]),
        "terminal_pose_valid": bool(mask[-1]),
        "checks_passed": True,
    }
    manifest = {
        "config": config,
        "code": code_fingerprint(),
        "files": {
            key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "sample_sha256": sha256(out / "sample.npz"),
        "columns": {"r2": R2_COLUMNS, "r1": R1_COLUMNS},
        "episode_metadata": meta,
        "original_metadata": orig_meta,
        "processing": processing,
        "receipt_episode": {k: receipt_ep[k] for k in ["episode_index", "rows", "pose_source", "gripper_source"]},
        "contracts": {
            key: json.loads(paths[key].read_text()) for key in ["state_contract", "tool_contract", "time_contract"]
        },
        "unknowns": [
            "robot serial",
            "historical controller version/config",
            "command-state acquisition offset",
            "feedback freshness/status",
            "contact annotations",
            "independent simulator/hardware",
        ],
        "evidence_level": "read_only_log_extraction",
    }
    write_json(out / "data_audit.json", audit)
    write_json(out / "sample_manifest.json", manifest)
    return arrays, audit
