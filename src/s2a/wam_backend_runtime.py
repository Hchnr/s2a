"""Isolated013 loader using the checkpoint's saved Flash3-video backend.

The frozen original loader stays unchanged. Its construction sequence is copied
literally except for the fixed attention-backend argument. Decoder compilation,
inference padding, small-action reuse, EMA and parallelism retain their values.
The backend itself uses Flash3 for VIDEO and Flex for UND/ACTION; its internal
Flex kernel compilation is part of that backend, independent of decoder compile.
"""

import json
from pathlib import Path


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
            "flash3_video",
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
