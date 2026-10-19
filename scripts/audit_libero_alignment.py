"""Read the configured training catalog and prepare exact published camera/label pairs.

Uses plain Parquet and PyAV, never a Cosmos training dataloader. Run in the
base environment; simulator/HDF5 dependencies are not needed for this stage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def decode_frame(path, timestamp):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
        nearest = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            if nearest is None or abs(actual - timestamp) < nearest[0]:
                nearest = abs(actual - timestamp), frame.to_ndarray(format="rgb24"), actual
            if actual >= timestamp:
                break
    if nearest is None or nearest[0] > 0.025001:
        raise ValueError(f"no 20Hz video frame near {timestamp}: {path}")
    return nearest[1], dict(requested_time_s=timestamp, decoded_time_s=nearest[2], error_s=nearest[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    dest = output / "alignment"
    dest.mkdir(exist_ok=False)
    policy = json.loads(Path(c["policy_config"]).read_text())
    provenance = json.loads(Path(c["provenance_config"]).read_text())
    fixture = json.loads(Path(policy["fixture"]).read_text())
    identity = json.loads(Path(c["source_identity"]).read_text())
    config_path = Path(policy["checkpoint_run"]) / "config.yaml"
    training = yaml.safe_load(config_path.read_text())
    specs = training["dataloader_train"]["dataset_specs"]
    configured = [s for s in specs if Path(s["path"]) == Path(c["training_catalog"])]
    if len(configured) != 1:
        raise ValueError("catalog is not uniquely referenced by the frozen training configuration")
    catalog = pq.read_table(c["training_catalog"]).to_pylist()
    if len(catalog) != 1:
        raise ValueError("expected one LIBERO full-training catalog row")
    data = json.loads(catalog[0]["data"])
    root = Path(data["root"])
    if root != Path(provenance["r2_root"]) or data["mode"] != "wam" or data["use_state"]:
        raise ValueError("training catalog root, mode or state-input contract differs")
    meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pq.read_table(meta_path).to_pylist()
    inputs = {str(p): sha256(p) for p in (config_path, Path(c["training_catalog"]), meta_path)}
    for key in ("source_identity", "native_bridge", "published_bridge", "policy_config", "provenance_config"):
        inputs[c[key]] = sha256(c[key])
    inputs[policy["fixture"]] = sha256(policy["fixture"])
    columns = [
        "frame_index",
        "action",
        "action_dim_mask",
        "action_valid",
        "observation.state",
        "observation.state_mask",
    ]
    cases = []
    for trial in c["trials"]:
        demo = fixture["episodes"][trial]["demo"]
        match = next(r for r in identity["cases"] if r["demo"] == demo)
        if len(match["current_r2"]) != 1:
            raise ValueError("training image probe requires a unique source match")
        episode = next(r for r in metadata if r["episode_index"] == match["current_r2"][0]["episode_index"])
        if not all(episode[k] == v for k, v in data["episode_filter"].items()):
            raise ValueError("probe episode is not eligible under the actual catalog filter")
        if episode["action_domain"] != policy["action_domain"] or episode["fps"] != policy["control_fps"]:
            raise ValueError("training/evaluation action domain or FPS differs")
        path = root / episode["data_path"]
        rows = pq.read_table(
            path, columns=columns, filters=[("episode_index", "=", episode["episode_index"])]
        ).to_pylist()
        rows.sort(key=lambda r: r["frame_index"])
        if [r["frame_index"] for r in rows] != list(range(episode["length"])):
            raise ValueError("published episode frames are not complete and ordered")
        inputs[str(path)] = inputs.get(str(path)) or sha256(path)
        view_rows, images = [], {}
        for role in ("auxiliary_0", "left_wrist"):
            view = next(v for v in episode["video_views"] if v["role"] == role)
            path = root / view["path"]
            timestamp = view["from_timestamp"] + c["training_frame_index"] / episode["fps"]
            pixels, decoded = decode_frame(path, timestamp)
            images[role] = pixels
            inputs[str(path)] = inputs.get(str(path)) or sha256(path)
            view_rows.append(dict(role=role, path=str(path), **decoded))
        image_path = dest / f"trial{trial}_cameras.npz"
        np.savez_compressed(image_path, **images)
        cases.append(
            dict(
                trial=trial,
                demo=demo,
                r1_episode=match["r1_episode"],
                r2_episode=episode["episode_index"],
                training_eligible=True,
                training_consumption_verified=False,
                cameras_npz=str(image_path),
                cameras_sha256=sha256(image_path),
                views=view_rows,
                rows=rows,
                source_episode_id=episode["source_episode_id"],
            )
        )
        print(json.dumps(dict(trial=trial, demo=demo, r2_episode=episode["episode_index"], decoded=True)), flush=True)
    original = json.loads(catalog[0]["provenance"])["extra"]
    source_hashes = dict(
        catalog_info_sha256=original.get("source_info_sha256"),
        current_info_sha256=sha256(root / "meta/info.json"),
    )
    result = dict(
        seed=c["seed"],
        code=code_fingerprint(),
        config=c,
        inputs_sha256=inputs,
        catalog_row=catalog[0],
        catalog_data=data,
        configured_sampling_ratio=configured[0]["ratio"],
        source_hashes=source_hashes,
        consumption_trace_config=training["trainer"]["callbacks"].get("load_balance_trace", {}),
        cases=cases,
        evidence="Exact catalog-referenced published camera/label pairs; not physical initial-state recovery",
    )
    write_json(dest / "matched_targets.json", result)


if __name__ == "__main__":
    main()
