"""Join historical LIBERO fixtures to current R1/R2 by command content, not episode number.

Run export with the simulator interpreter (h5py), verify with the base interpreter
(PyArrow). No environment modification or Cosmos training reader is needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.geometry import inverse, pose_error, rot6d_to_matrix, transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/libero_provenance_20261019.json")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", choices=["export", "verify"], required=True)
    args = p.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    f = json.loads(Path(c["fixture"]).read_text())
    if args.stage == "export":
        import h5py

        names = [r["demo"] for r in f["episodes"]] + c["additional_demos"]
        with h5py.File(f["source_hdf5"]) as source:
            rows = {name: source["data"][name]["actions"][:].tolist() for name in names}
            observations = {
                name: {
                    key: source["data"][name]["obs"][key][:].tolist() for key in ("ee_pos", "ee_ori", "gripper_states")
                }
                for name in names
            }
        write_json(
            out / "native_bridge.json",
            dict(
                actions=rows,
                observations=observations,
                code=code_fingerprint(),
                source_hdf5_sha256=sha256(f["source_hdf5"]),
            ),
        )
        return
    import pyarrow.parquet as pq

    bridge = json.loads((out / "native_bridge.json").read_text())
    r1 = Path(c["r1_root"])
    r2 = Path(c["r2_root"])
    meta1 = r1 / "meta/episodes/chunk-000/file-000.parquet"
    meta2 = r2 / "meta/episodes/chunk-000/file-000.parquet"
    episodes = [r for r in pq.read_table(meta1).to_pylist() if c["task"] in r["tasks"]]
    inputs = {
        str(meta1): sha256(meta1),
        str(meta2): sha256(meta2),
        str(out / "native_bridge.json"): sha256(out / "native_bridge.json"),
    }
    groups = {}
    for chunk, index in sorted({(r["data/chunk_index"], r["data/file_index"]) for r in episodes}):
        path = r1 / f"data/chunk-{chunk:03d}/file-{index:03d}.parquet"
        ids = [r["episode_index"] for r in episodes if r["data/chunk_index"] == chunk and r["data/file_index"] == index]
        data = pq.read_table(
            path, columns=["episode_index", "action.native_control"], filters=[("episode_index", "in", ids)]
        ).to_pylist()
        inputs[str(path)] = sha256(path)
        for row in data:
            groups.setdefault(row["episode_index"], []).append(row["action.native_control"])
    current = pq.read_table(
        meta2, columns=["episode_index", "source_episode_id", "training_eligible", "data_role", "data_path"]
    ).to_pylist()
    rows = []
    tool_path = Path(c["tool_contract"])
    h = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
    inputs[str(tool_path)] = sha256(tool_path)
    numeric = {}
    for name, actions in bridge["actions"].items():
        a = np.asarray(actions)
        rot = Rotation.from_rotvec(a[:, 3:6]).as_matrix()
        projected = np.column_stack((a[:, :3], rot[:, :, 0], rot[:, :, 1], (1 - a[:, 6]) / 2))
        matches = []
        for index, values in groups.items():
            values = np.asarray(values)
            if values.shape == projected.shape:
                error = float(np.max(abs(values - projected)))
                if error < c["max_native_projection_error"]:
                    matches.append((index, error))
        if len(matches) > 1:
            raise ValueError("ambiguous command-content identity")
        sources = []
        if matches:
            index, error = matches[0]
            for row in current:
                source_id = json.loads(row["source_episode_id"]).get("source_id")
                if source_id == str(r1) + ":" + str(index):
                    sources.append(row)
        geometry = []
        for metadata in sources:
            path = r2 / metadata["data_path"]
            if str(path) not in numeric:
                ids = [
                    r["episode_index"]
                    for r in current
                    if r["data_path"] == metadata["data_path"]
                    and json.loads(r["source_episode_id"]).get("source_id", "").split(":")[-1]
                    in {str(k) for k in groups}
                ]
                numeric[str(path)] = pq.read_table(
                    path,
                    columns=[
                        "episode_index",
                        "frame_index",
                        "action",
                        "observation.state",
                        "action_dim_mask",
                        "observation.state_mask",
                    ],
                    filters=[("episode_index", "in", ids)],
                ).to_pylist()
                inputs[str(path)] = sha256(path)
            published = sorted(
                [r for r in numeric[str(path)] if r["episode_index"] == metadata["episode_index"]],
                key=lambda r: r["frame_index"],
            )
            labels = bridge["observations"][name]
            expected = transform(np.array(labels["ee_pos"]), Rotation.from_rotvec(labels["ee_ori"]).as_matrix()) @ h
            values = np.array([r["observation.state"] for r in published])
            if len(values) != len(expected):
                raise ValueError("source provenance row count changed")
            state_valid = np.array([r["observation.state_mask"][:9] for r in published]).all(axis=1)
            state_error = pose_error(
                expected[state_valid], transform(values[state_valid, :3], rot6d_to_matrix(values[state_valid, 3:9]))
            )
            delta = inverse(expected[:-1]) @ expected[1:]
            action = np.array([r["action"] for r in published[:-1]])
            mask = np.array([r["action_dim_mask"] for r in published[:-1]])
            valid = mask[:, :9].all(axis=1)
            action_error = pose_error(delta[valid], transform(action[valid, :3], rot6d_to_matrix(action[valid, 3:9])))
            grip = np.array(labels["gripper_states"])[1:]
            expected_opening = np.clip((grip[:, 0] - grip[:, 1] - 0.001) / 0.079, 0, 1)
            geometry.append(
                dict(
                    r2_episode=metadata["episode_index"],
                    valid_states=int(state_valid.sum()),
                    valid_transitions=int(valid.sum()),
                    state_position_max_m=float(np.max(np.linalg.norm(state_error[:, :3], axis=1))),
                    state_rotation_max_rad=float(np.max(np.linalg.norm(state_error[:, 3:], axis=1))),
                    action_translation_max_m=float(np.max(np.linalg.norm(action_error[:, :3], axis=1))),
                    action_rotation_max_rad=float(np.max(np.linalg.norm(action_error[:, 3:], axis=1))),
                    gripper_endpoint_max_abs=float(np.max(abs(action[mask[:, 9], 9] - expected_opening[mask[:, 9]]))),
                )
            )
        rows.append(
            dict(
                demo=name,
                frames=len(a),
                raw_native_float32_sha256=hashlib.sha256(a.astype(np.float32).tobytes()).hexdigest(),
                r1_episode=matches[0][0] if matches else None,
                projection_max_abs=matches[0][1] if matches else None,
                current_r2=sources,
                geometry=geometry,
                training_consumption_verified=False,
            )
        )
    result = dict(
        seed=c["seed"],
        code=code_fingerprint(),
        inputs_sha256=inputs,
        source_hdf5_sha256=bridge["source_hdf5_sha256"],
        current_same_task_r1_episodes=len(episodes),
        cases=rows,
        matched=sum(r["r1_episode"] is not None for r in rows),
        notes=[
            "R1 native_control is a 10D rot6d/opening projection, not the raw seven controller slots",
            "Current training_eligible metadata does not prove the frozen run consumed an episode",
            "Missing same-task match is unknown, not evidence of model holdout",
            "Historical fixture integers are not current release episode IDs",
        ],
    )
    write_json(out / "source_identity.json", result)
    print(json.dumps(dict(matched=result["matched"], total=len(rows))))


if __name__ == "__main__":
    main()
