"""CPU actual HTTP preparation for all022 pairs; never load model weights."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from run_wam_short_replan import cpu_environment, verify_hashes, write_new
from serve_wam_static_command import align_empty_plan

from s2a.artifacts import REPO, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and not args.output.exists()
    assert args.output.resolve().is_relative_to(REPO / "outputs")
    assert sha256(args.prepared) == args.prepared_sha256
    protocol = json.loads(args.prepared.read_text())
    verify_hashes(protocol["input_sha256"])
    import torch
    import yaml
    from cosmos_framework.inference.action_serving import prepare_wam_request
    from cosmos_framework.inference.args import OmniSampleOverrides
    from omegaconf import OmegaConf

    config = json.loads(Path(protocol["configs"]["reproduction"]["path"]).read_text())
    model_config = OmegaConf.create(
        yaml.safe_load((Path(config["checkpoints"][0]["run"]) / "config.yaml").read_text())["model"]["config"]
    )
    spec = OmegaConf.to_container(model_config.action128_spec, resolve=True)
    args.output.mkdir(parents=True)
    rows = []
    try:
        for left, right in zip(protocol["requests"][:48], protocol["requests"][48:], strict=True):
            assert left["mode"] == "reproduction" and right["mode"] == "expanded"
            l_request, r_request = left["request"], right["request"]
            assert {k: v for k, v in l_request.items() if k not in {"prompt", "name"}} == {
                k: v for k, v in r_request.items() if k not in {"prompt", "name"}
            }
            batches, identities, aligned = [], [], []
            for item in (left, right):
                prepared, batch = prepare_wam_request(
                    OmniSampleOverrides.model_validate(item["request"]), spec, model_config, args.output, "video"
                )
                aligned.append(align_empty_plan(batch, expanded=item["mode"] == "expanded"))
                batches.append(batch)
                identities.append((prepared.action_dim_mask, prepared.seed, prepared.guidance, prepared.shift))
            lc, rc = [json.loads(batch["ai_caption"][0]) for batch in batches]
            assert {k: v for k, v in lc.items() if k != "command"} == {k: v for k, v in rc.items() if k != "command"}
            assert lc["command"] == l_request["prompt"]
            assert rc["command"] == json.loads(protocol["commands"][lc["task_title"]])["command"]
            assert lc["task_plan"] == rc["task_plan"] == [] and "current_subtask" not in lc
            assert identities[0] == identities[1] == ([True] * 10 + [False] * 118, l_request["seed"], 1.0, 5.0)
            assert torch.equal(batches[0]["video"][0][0], batches[1]["video"][0][0])
            source_pixels = np.asarray(Image.open(l_request["vision_path"]).convert("RGB"))
            actual = batches[0]["video"][0][0][:, 0].permute(1, 2, 0).numpy()
            assert np.array_equal(source_pixels, actual)
            for key in ("domain_id", "normalization_domain_id", "conditioning_fps", "conditioning_fps_action"):
                assert repr(batches[0][key]) == repr(batches[1][key])
            assert list(batches[0]["action"][0][0].shape) == list(batches[1]["action"][0][0].shape) == [40, 128]
            rows.append(
                dict(
                    reproduction_name=l_request["name"],
                    expanded_name=r_request["name"],
                    seed=l_request["seed"],
                    baseline_caption=lc,
                    expanded_caption=rc,
                    only_command_different=True,
                    first_frame_source_pixels_exact=True,
                    complete_packed_video_equal=True,
                    domain_mask_fps_shape_equal=True,
                    schema_alignment=aligned,
                )
            )
        verify_hashes(protocol["input_sha256"])
    except BaseException as exc:
        write_new(args.output / "error.json", dict(error=repr(exc), verified_pairs=len(rows), rows=rows))
        raise
    result = dict(
        schema_version="s2a.static-command-actual-cpu-preparation.v1",
        complete=True,
        verified_pairs=48,
        prepared=dict(path=str(args.prepared.resolve()), sha256=args.prepared_sha256),
        rows=rows,
        input_sha256=dict(protocol["input_sha256"], **{str(args.prepared.resolve()): args.prepared_sha256}),
        new_model_requests=0,
        new_physics_steps=0,
        model_weights_loaded=False,
    )
    write_new(args.output / "result.json", result)
    print(
        json.dumps(
            dict(path=str(args.output / "result.json"), sha256=sha256(args.output / "result.json"), verified_pairs=48)
        )
    )


if __name__ == "__main__":
    main()
