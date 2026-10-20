"""CPU-only audit of actual HTTP batch text and first-frame pixels."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    from cosmos_framework.inference.action_serving import prepare_wam_request
    from cosmos_framework.inference.args import OmniSampleOverrides
    from omegaconf import OmegaConf

    output = root / "caption_pixel_audit"
    output.mkdir(exist_ok=False)
    config = OmegaConf.create(
        yaml.safe_load((Path(c["checkpoints"][-1]["run"]) / "config.yaml").read_text())["model"]["config"]
    )
    spec = OmegaConf.to_container(config.action128_spec, resolve=True)
    samples_path = root / "samples/manifest.json"
    samples = json.loads(samples_path.read_text())["cases"]
    template_path = next((root / "state_queries/8800").glob("*_h16_task_only_s42/request.json"))
    template = json.loads(template_path.read_text())
    rows = []
    for sample in samples:
        pixels = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        for horizon in c["offline_horizons"]:
            for seed in c["sampling_seeds"]:
                request = dict(
                    template,
                    prompt=sample["published_prompts"][str(horizon)],
                    vision_path=str(Path(sample["image_path"]).absolute()),
                    action_chunk_size=horizon,
                    num_frames=horizon + 1,
                    seed=seed,
                )
                _, batch = prepare_wam_request(
                    OmniSampleOverrides.model_validate(request), spec, config, output, "video"
                )
                actual_pixels = batch["video"][0][0][:, 0].permute(1, 2, 0).numpy()
                row = dict(
                    sample=sample["name"],
                    horizon=horizon,
                    sampling_seed=seed,
                    caption_bytes_equal=batch["ai_caption"][0] == request["prompt"],
                    caption_json_equal=json.loads(batch["ai_caption"][0]) == json.loads(request["prompt"]),
                    first_frame_pixels_equal=bool(np.array_equal(pixels, actual_pixels)),
                    raw_action_dim=int(batch["raw_action_dim"][0]),
                    action_shape=list(batch["action"][0][0].shape),
                )
                rows.append(row)
    write_json(
        output / "result.json",
        dict(
            code=code_fingerprint(),
            samples_sha256=sha256(samples_path),
            cases=rows,
            interpretation="Uses the actual stock HTTP preparer, not assemble_batch in isolation; full JSON is parsed and preserved.",
        ),
    )
    if not all(r["caption_bytes_equal"] and r["first_frame_pixels_equal"] for r in rows):
        raise ValueError("stock HTTP text/pixel contract differs from frozen samples")
    print(json.dumps(dict(cases=len(rows), all_text_equal=True, all_pixels_equal=True)))


if __name__ == "__main__":
    main()
