"""Frozen WAM queries on exact catalog camera frames; offline target diagnostics only."""

import argparse
import ast
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.libero_simulation import canvas, generate_from_canvas
from s2a.trajectory_audit import trajectory_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    matched_path = output / "alignment/matched_targets.json"
    matched = json.loads(matched_path.read_text())
    p = json.loads((output / "serving_config.json").read_text())
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(p["server_url"] + "/info", timeout=15)
        response.raise_for_status()
        info = response.json()
    expected = Path(p["checkpoint_run"]) / "checkpoints" / p["checkpoint_iteration"] / "model"
    if Path(info["checkpoint_path"]).resolve() != expected.resolve():
        raise ValueError("server checkpoint differs")
    dest = output / "training_image_queries"
    dest.mkdir(exist_ok=False)
    source = Path(p["cosmos_repo"]) / "cosmos_framework/data/generator/action/unified_action128.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_resized_camera")
    # Execute only the pinned pure resize function, not a training module/loader.
    scope = dict(np=np, math=math, torch=torch, F=torch.nn.functional)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    inputs = {str(matched_path): sha256(matched_path), str(source): sha256(source)}
    cases = []
    for record in matched["cases"]:
        trial = record["trial"]
        cameras_path = Path(record["cameras_npz"])
        if sha256(cameras_path) != record["cameras_sha256"]:
            raise ValueError("prepared source camera hash differs")
        with np.load(cameras_path) as arrays:
            images = [arrays[role] for role in ("auxiliary_0", "left_wrist")]
        pixels = canvas(images)
        official = torch.empty((3, 1, 960, 640), dtype=torch.uint8)
        for image, start in zip(images, (0, 480), strict=True):
            scope["_copy_resized_camera"]([image], official[:, :, start : start + 480])
        official[:, :, 478:482] = 0
        official = official[:, 0].permute(1, 2, 0).numpy()
        parity = int(np.max(abs(pixels.astype(int) - official.astype(int))))
        if parity != 0:
            raise ValueError("S2A canvas differs from the configured training resize")
        previous = Path(json.loads(Path(c["policy_config"]).read_text())["output_dir"])
        live_path = previous / "policy/baseline" / f"trial{trial}/input_0000.png"
        live = np.asarray(Image.open(live_path).convert("RGB"))
        Image.fromarray(np.concatenate([pixels, live], axis=1)).save(dest / f"trial{trial}_source_vs_sim.png")
        for horizon in c["training_horizons"]:
            sub = dest / f"trial{trial}_h{horizon}"
            sub.mkdir()
            config = {**p, "action_chunk_size": horizon}
            write_json(sub / "manifest.json", dict(code=code_fingerprint(), config=config, server_info=info))
            started = time.monotonic()
            action, observation_hash = generate_from_canvas(config, pixels, sub, 0)
            elapsed = time.monotonic() - started
            rows = record["rows"][c["training_frame_index"] : c["training_frame_index"] + horizon]
            if len(rows) != horizon:
                raise ValueError("not enough matched published labels")
            diagnostics = trajectory_diagnostics(
                action, np.asarray([r["action"] for r in rows]), np.asarray([r["action_dim_mask"] for r in rows])
            )
            result = dict(
                trial=trial,
                demo=record["demo"],
                r2_episode=record["r2_episode"],
                horizon=horizon,
                source_frame=c["training_frame_index"],
                canvas_training_parity_max_abs=parity,
                source_vs_initial_sim_pixel_mae=float(np.mean(abs(pixels.astype(float) - live))),
                source_vs_initial_sim_are_paired=False,
                observation_sha256=observation_hash,
                request_elapsed_s=elapsed,
                diagnostics=diagnostics,
                evidence="Offline WAM prediction on exact published camera frame against matched R2 labels; no task score",
            )
            write_json(sub / "diagnostics.json", result)
            cases.append(result)
            print(json.dumps(result), flush=True)
    write_json(dest / "summary.json", dict(code=code_fingerprint(), inputs_sha256=inputs, cases=cases))


if __name__ == "__main__":
    main()
