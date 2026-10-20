"""Freeze episode-disjoint, multi-phase published camera/state samples."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.checkpoint_audit import phase_anchors
from s2a.libero_simulation import canvas


def read_episode_video(path, start, count, fps):
    """One seek per camera; verify each requested sample against decoded PTS."""
    wanted = start + np.arange(count) / fps
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            i = round((timestamp - start) * fps)
            if 0 <= i < count and abs(timestamp - wanted[i]) < 0.5 / fps:
                error = abs(timestamp - wanted[i])
                if i not in found or error < found[i][0]:
                    found[i] = error, frame.to_ndarray(format="rgb24")
            if timestamp > wanted[-1] + 1 / fps:
                break
    if set(found) != set(range(count)):
        raise ValueError(f"missing video frames: {path}")
    return np.stack([found[i][1] for i in range(count)]), max(v[0] for v in found.values())


def caption_renderer(path):
    """Execute a pinned pure renderer only; never import a training reader."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_caption")
    scope = dict(
        json=json,
        math=math,
        CatalogSchemaError=ValueError,
        stable_hash=lambda x: hashlib.blake2b(json.dumps(x, sort_keys=True).encode(), digest_size=16).hexdigest(),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["render_caption"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    dest = output / "samples"
    dest.mkdir(exist_ok=False)
    root = Path(c["r2_root"])
    metadata = root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [
        "episode_index",
        "length",
        "fps",
        "training_eligible",
        "data_role",
        "caption_path",
        "caption_index",
        "source_episode_id",
        "source_row_start",
        "source_row_end",
        "data_path",
        "video_views",
        "embodiment_name",
    ]
    episodes = pq.read_table(metadata, columns=columns).to_pylist()
    captions = {
        str(p.relative_to(root)): [json.loads(x) for x in p.read_text().splitlines()]
        for p in root.glob("captions/chunk-*/file-*.jsonl")
    }
    render_path = Path(c["cosmos_repo"]) / "cosmos_framework/data/generator/action/unified_action128.py"
    render = caption_renderer(render_path)
    hashes = {str(p): sha256(p) for p in [Path(args.config), metadata, render_path, Path(c["training_catalog"])]}
    rng = np.random.default_rng(args.seed)
    selected = []
    for task_index, task in enumerate(c["tasks"]):
        candidates = [
            r
            for r in episodes
            if r["training_eligible"]
            and r["length"] >= 60
            and r["fps"] == c["control_fps"]
            and task in captions[r["caption_path"]][r["caption_index"]].get("source", {}).get("original_tasks", [])
        ]
        candidates.sort(key=lambda r: r["episode_index"])
        used_sources = set()
        for i in rng.permutation(len(candidates)):
            r = candidates[i]
            if r["source_episode_id"] in used_sources:
                continue
            split = "diagnostic" if len(used_sources) < c["offline_diagnostic_episodes_per_task"] else "study_heldout"
            selected.append(dict(task_index=task_index, task=task, split=split, metadata=r))
            used_sources.add(r["source_episode_id"])
            if len(used_sources) == c["offline_episodes_per_task"]:
                break
        if len(used_sources) != c["offline_episodes_per_task"]:
            raise ValueError("not enough independent source episodes")
    write_json(dest / "selection.json", dict(seed=args.seed, code=code_fingerprint(), selection=selected))
    rows_by_episode = {}
    for relative in sorted({x["metadata"]["data_path"] for x in selected}):
        path = root / relative
        ids = [x["metadata"]["episode_index"] for x in selected if x["metadata"]["data_path"] == relative]
        rows = pq.read_table(
            path,
            filters=[("episode_index", "in", ids)],
            columns=[
                "episode_index",
                "frame_index",
                "timestamp",
                "observation.state",
                "observation.state_mask",
                "action",
                "action_dim_mask",
                "action_valid",
            ],
        ).to_pylist()
        hashes[str(path)] = sha256(path)
        for i in ids:
            rows_by_episode[i] = sorted([r for r in rows if r["episode_index"] == i], key=lambda r: r["frame_index"])
    cases = []
    for item in selected:
        ep = item["metadata"]
        index = ep["episode_index"]
        rows = rows_by_episode[index]
        if [r["frame_index"] for r in rows] != list(range(ep["length"])):
            raise ValueError("incomplete source episode")
        states = np.asarray([r["observation.state"] for r in rows])
        state_mask = np.asarray([r["observation.state_mask"] for r in rows], dtype=bool)
        if not state_mask[:, :10].all():
            raise ValueError("missing active state dimensions")
        np.testing.assert_allclose(np.diff([r["timestamp"] for r in rows]), 1 / ep["fps"], atol=1e-6)
        cameras, errors = {}, {}
        for role in ("auxiliary_0", "left_wrist"):
            view = next(v for v in ep["video_views"] if v["role"] == role)
            path = root / view["path"]
            cameras[role], errors[role] = read_episode_video(path, view["from_timestamp"], len(rows), ep["fps"])
            if str(path) not in hashes:
                hashes[str(path)] = sha256(path)
        arrays_path = dest / f"episode{index}.npz"
        np.savez_compressed(
            arrays_path,
            states=states,
            state_mask=state_mask,
            actions=np.asarray([r["action"] for r in rows]),
            action_mask=np.asarray([r["action_dim_mask"] for r in rows]),
            action_valid=np.asarray([r["action_valid"] for r in rows]),
            **cameras,
        )
        caption_path = root / ep["caption_path"]
        hashes[str(caption_path)] = hashes.get(str(caption_path)) or sha256(caption_path)
        caption = captions[ep["caption_path"]][ep["caption_index"]]
        for phase in phase_anchors(states):
            frame = phase["anchor_frame"]
            name = f"task{item['task_index']}_ep{index}_{phase['phase']}"
            image_path = dest / f"{name}.png"
            Image.fromarray(canvas([cameras[r][frame] for r in ("auxiliary_0", "left_wrist")])).save(image_path)
            prompts = {}
            for horizon in c["offline_horizons"]:
                prompts[str(horizon)] = render(
                    caption,
                    start=frame / ep["fps"],
                    duration=horizon / ep["fps"],
                    fps=ep["fps"],
                    action_frequency_hz=ep["fps"],
                    width=640,
                    height=960,
                    embodiment=ep["embodiment_name"],
                    camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                    seed=args.seed,
                )
            cases.append(
                dict(
                    name=name,
                    task=item["task"],
                    task_index=item["task_index"],
                    split=item["split"],
                    episode=index,
                    source_episode_id=ep["source_episode_id"],
                    pretraining_consumption="unknown",
                    **phase,
                    arrays_path=str(arrays_path),
                    arrays_sha256=sha256(arrays_path),
                    image_path=str(image_path),
                    image_sha256=sha256(image_path),
                    published_prompts=prompts,
                    valid_future_steps=len(rows) - frame - 1,
                    camera_pts_errors_s=errors,
                )
            )
        print(json.dumps(dict(episode=index, split=item["split"], phases=4, pts_error=errors)), flush=True)
    write_json(
        dest / "manifest.json",
        dict(
            config=c,
            code=code_fingerprint(),
            inputs_sha256=hashes,
            cases=cases,
            language_diagnostic="published caption may contain privileged demonstration descriptions; never count as pure policy success",
        ),
    )


if __name__ == "__main__":
    main()
