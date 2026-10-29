"""Opt-in, auditable model instrumentation; external Cosmos sources stay read-only.

This module imports model/packing utilities only when invoked in the separate
Cosmos runtime. It never imports a training dataset or dataloader.
"""

import json
import types
from pathlib import Path


def install_active_slot_mask(model, mask):
    """Keep inactive slots zero at initialization, every forward and its velocity.

    The caller supplies the published slot mask. Normalization masks are not
    interchangeable with valid-dimension masks. Hooks preserve active slots and
    random-number draws, so paired seeds remain paired.
    """
    import torch

    active = torch.as_tensor(mask, dtype=torch.bool)
    if active.shape != (128,) or not active.any():
        raise ValueError("requires an explicit nonempty Action128 dimension mask")
    original_prepare = model._prepare_inference_data
    original_denoise = model.denoise
    evidence = []

    def prepare(self, *args, **kwargs):
        result = original_prepare(*args, **kwargs)
        _, clean, _, _, noise, _, _, has_action = result
        if not has_action or len(noise) != 1 or clean.num_vision_items_per_sample is not None:
            raise ValueError("mask audit supports one WAM sample with one stitched view")
        offset = clean.x0_tokens_vision[0].numel()
        action = noise[0][offset:].reshape(clean.x0_tokens_action[0].shape)
        valid = active.to(action.device)
        before = float(action[:, ~valid].float().square().mean().sqrt()) if (~valid).any() else 0.0
        action[:, ~valid] = 0
        evidence.append(dict(stage="initial_noise", inactive_rms_before=before, inactive_rms_after=0.0))
        return result

    def denoise(self, *args, **kwargs):
        packed = kwargs["data_batch_packed"]
        if packed.action is None:
            raise ValueError("mask instrumentation requires action tokens")
        for token in packed.action.tokens:
            token[:, ~active.to(token.device)] = 0
        result = original_denoise(*args, **kwargs)
        result["preds_action"] = [x.masked_fill(~active.to(x.device), 0) for x in result["preds_action"]]
        return result

    model._prepare_inference_data = types.MethodType(prepare, model)
    model.denoise = types.MethodType(denoise, model)
    return evidence


def load_service(config, checkpoint, output):
    """Load the stock frozen service with its published model configuration."""
    from cosmos_framework.inference.common.init import init_script

    init_script()
    import torch
    from cosmos_framework.inference.action_server import (
        DistributedActionService,
        deployment_plan,
        initialize_single_rank_distributed,
        model_overrides,
        parser,
    )
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from torch import distributed as dist

    ck = next(x for x in config["checkpoints"] if x["name"] == checkpoint)
    args = parser().parse_args(
        [
            "--checkpoint-path",
            str(Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"),
            "--output-dir",
            str(output),
            "--attention-backend",
            "sdpa",
            "--no-compile",
            "--no-reuse-small-action-metadata",
        ]
    )
    torch.cuda.set_device(0)
    initialize_single_rank_distributed()
    group = dist.new_group(backend="gloo")
    plan = deployment_plan(args)
    setup = OmniSetupOverrides(
        checkpoint_path=plan["checkpoint_path"],
        config_file=plan["config_file"],
        output_dir=output,
        experiment_overrides=model_overrides(args),
        use_ema_weights=False,
        use_torch_compile=False,
        guardrails=False,
        dp_shard_size=1,
        dp_replicate_size=1,
        cp_size=1,
        cfgp_size=1,
    ).build_setup(world_size=1)
    service = DistributedActionService(OmniInference.create(setup), group, plan)
    (output / "runtime.json").write_text(json.dumps(plan, indent=2) + "\n")
    return service
