"""Recompute frozen WAM flow objectives and compare training/inference packing."""

import argparse
import copy
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, load_config, write_json
from s2a.libero_simulation import canvas
from s2a.wam_runtime import load_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--checkpoint", default="8800")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--horizon", type=int, choices=(16, 40), default=16)
    parser.add_argument("--published-caption", action="store_true")
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs")
    tag = args.checkpoint + (f"_h{args.horizon}_published" if args.published_caption else "")
    output = root / "numerics" / tag
    output.mkdir(parents=True, exist_ok=False)
    service = load_service(config, args.checkpoint, output)
    import torch
    from cosmos_framework.data.generator.sequence_packing.static_layout import (
        PACKED_SEQUENCE_KEY,
        StaticPackingConfig,
        bind_static_packed_sequence,
        build_static_packed_sequence,
    )
    from cosmos_framework.inference.args import OmniSampleOverrides
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_generation_caption
    from cosmos_framework.utils import misc

    model = service.model
    cases = json.loads((root / "samples/manifest.json").read_text())["cases"]
    phase = "approach" if args.published_caption else "closing_proxy"
    selected = [next(x for x in cases if x["task_index"] == i and x["phase"] == phase) for i in range(3)]
    horizon = args.horizon
    options = model._input_sequence_options()
    geometry = StaticPackingConfig(
        **{f.name: options[f.name] for f in fields(StaticPackingConfig) if f.name in options}
    )
    write_json(
        output / "manifest.json", dict(code=code_fingerprint(), cases=selected, config=config, runtime=service.info)
    )
    records = []
    with torch.inference_mode():
        for sample in selected:
            directory = output / sample["name"]
            directory.mkdir()
            request_path = root / "state_queries/8800" / f"{sample['name']}_h16_task_only_s42/request.json"
            # Build from an already frozen request without depending on query completion.
            if not request_path.exists():
                request_path = next((root / "state_queries/8800").glob("*_h16_task_only_s42/request.json"))
            request = json.loads(request_path.read_text())
            request.update(vision_path=str(Path(sample["image_path"]).absolute()), prompt=sample["task"])
            request.update(action_chunk_size=horizon, num_frames=horizon + 1)
            prepared, raw_batch = service.prepare(OmniSampleOverrides.model_validate(request), directory)
            if args.published_caption:
                raw_batch["ai_caption"] = [sample["published_prompts"][str(horizon)]]
            inference = model._prepare_inference_data(misc.to(copy.deepcopy(raw_batch), device="cuda"), [args.seed])
            plans, infer_clean, text, _, initial, _, _, has_action = inference
            offset = infer_clean.x0_tokens_vision[0].numel()
            initial_action = initial[0][offset:].reshape(horizon, 128)
            audit = dict(
                sample=sample["name"],
                stock_initial_inactive_rms=float(initial_action[:, 10:].square().mean().sqrt()),
                stock_batch_has_loss_mask="action_loss_mask" in raw_batch,
                inference_caption=raw_batch["ai_caption"][0],
            )
            write_json(directory / "initial_noise.json", audit)
            with np.load(sample["arrays_path"]) as data:
                frame = sample["anchor_frame"]
                images = [
                    canvas([data[r][j] for r in ("auxiliary_0", "left_wrist")])
                    for j in range(frame, frame + horizon + 1)
                ]
                physical = torch.tensor(data["actions"][frame : frame + horizon], dtype=torch.float32)
                valid = torch.tensor(data["action_mask"][frame : frame + horizon], dtype=torch.bool)
                temporal_valid = torch.tensor(data["action_valid"][frame : frame + horizon], dtype=torch.bool)
            valid &= temporal_valid.reshape(-1, 1)
            domain = raw_batch["normalization_domain_id"][0]
            transform = service.spec["transforms"][domain]
            normalized = physical.clone()
            mask = torch.tensor(transform["normalization_mask"], dtype=torch.bool)
            normalized[:, mask] = (
                normalized[:, mask] * torch.tensor(transform["scale"])[mask] + torch.tensor(transform["bias"])[mask]
            )
            audit["clipped_label_elements"] = int(
                ((normalized[:, mask] < transform["clip_min"]) | (normalized[:, mask] > transform["clip_max"])).sum()
            )
            normalized[:, mask] = normalized[:, mask].clamp(transform["clip_min"], transform["clip_max"])
            normalized[~valid] = 0
            training = copy.deepcopy(raw_batch)
            training[model.input_video_key] = [[torch.tensor(np.stack(images)).permute(3, 0, 1, 2)]]
            training["action"] = [[normalized]]
            # Same bytes for numerical parity; full published training text is a separate diagnostic.
            ids, _ = tokenize_generation_caption(training["ai_caption"][0], model.vlm_tokenizer, is_video=True)
            train_tokens = [ids]
            training["text_token_ids"] = [[torch.tensor(train_tokens[0])]]
            item = dict(
                sequence_plan=plans[0],
                text_token_ids=train_tokens[0],
                video=training[model.input_video_key][0][0],
                action=normalized,
                action_loss_mask=valid,
                vision_loss_mask=torch.ones((1 + horizon // 4, 60, 40), dtype=torch.bool),
                domain_id=raw_batch["domain_id"][0],
                raw_action_dim=128,
                conditioning_fps=20.0,
                conditioning_fps_action=20.0,
                decode_metadata=raw_batch["decode_metadata"][0],
            )
            static = build_static_packed_sequence([item], special_tokens=model.llm_special_tokens, config=geometry)
            training[PACKED_SEQUENCE_KEY] = static
            training = misc.to(training, device="cuda")
            prepared_train = model._prepare_training_data(training, iteration=8800)
            train_text, train_plans, clean = prepared_train[:3]
            audit.update(
                token_ids_identical=train_tokens == text == train_text,
                condition_latent_max_abs=float(
                    (clean.x0_tokens_vision[0][:, :, :1] - infer_clean.x0_tokens_vision[0][:, :, :1]).abs().max()
                ),
                normalized_action_max_abs=float((clean.x0_tokens_action[0] - normalized.cuda()).abs().max()),
                action_valid_elements=int(valid.sum()),
                flow_schedule=str(model.config.rectified_flow_training_config),
                losses=[],
            )
            for sigma in (0.2, 0.5, 0.8):
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                timestep = torch.tensor([[sigma * 1000]], device="cuda")
                packed = bind_static_packed_sequence(
                    copy.deepcopy(static),
                    clean,
                    timestep.cpu(),
                    latent_patch_size=geometry.latent_patch_size,
                    vision_spatial_position_mode=geometry.vision_spatial_position_mode,
                )
                packed.to_cuda()
                noised = model._add_noise_to_input(clean, packed, timestep / 1000)
                model._replace_clean_with_noised(packed, noised)
                prediction = model.denoise(data_batch_packed=packed)
                total, losses = model._compute_losses(prediction, packed, noised, timestep, False)
                flat = [
                    torch.cat(
                        (noised.xt_tokens_vision[0].flatten().float(), noised.xt_tokens_action[0].flatten().float())
                    )
                ]
                velocity = model._get_velocity(
                    noise_x=flat,
                    timestep=timestep.flatten(),
                    text_tokens=train_text,
                    sequence_plans=train_plans,
                    gen_data_clean=clean,
                    has_noisy_actions=has_action,
                )
                inferred = velocity[0][offset:].reshape(horizon, 128)
                trained = prediction["preds_action"][0]
                targets = noised.vt_target_action[0]
                active = valid.cuda()
                mse = (trained.float() - targets.float()).square()
                row = dict(
                    sigma=sigma,
                    total_loss=float(total),
                    losses={k: v.float().cpu().tolist() for k, v in losses.items()},
                    active_action_mse=float(mse[active].mean()),
                    all_slot_masked_mse=float(mse.masked_fill(~active, 0).mean()),
                    train_inactive_input_max_abs=float(noised.xt_tokens_action[0][:, 10:].abs().max()),
                    velocity_active_max_abs=float((inferred - trained)[active].abs().max()),
                )
                # A paired counterfactual changes only inactive network input slots.
                perturbed = flat[0].clone()
                perturbed[offset:].reshape(horizon, 128)[:, 10:] = initial_action[:, 10:] * sigma
                stock_like = model._get_velocity(
                    noise_x=[perturbed],
                    timestep=timestep.flatten(),
                    text_tokens=train_text,
                    sequence_plans=train_plans,
                    gen_data_clean=clean,
                    has_noisy_actions=has_action,
                )[0][offset:].reshape(horizon, 128)
                row["inactive_noise_active_velocity_rmse"] = float(
                    (stock_like - inferred)[active].square().mean().sqrt()
                )
                row["inactive_noise_active_target_mse"] = float((stock_like - targets)[active].square().mean())
                audit["losses"].append(row)
                write_json(directory / "result.json", audit)
                print(json.dumps(dict(sample=sample["name"], **row)), flush=True)
            records.append(audit)
            write_json(output / "results.json", records)
    write_json(output / "complete.json", dict(complete=True, samples=len(records)))


if __name__ == "__main__":
    main()
