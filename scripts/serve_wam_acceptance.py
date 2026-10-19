"""Explicit S2A-owned serving interventions for checkpoint acceptance."""

import argparse
import json
import types
from pathlib import Path

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.wam_runtime import install_active_slot_mask, load_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--active-slot-mask", action="store_true")
    parser.add_argument("--exact-published-caption", action="store_true")
    parser.add_argument("--warm-start-contract-run", type=Path)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    output = root / "intervention_services" / str(args.port)
    output.mkdir(parents=True, exist_ok=False)
    service = load_service(c, args.checkpoint, output)
    import uvicorn
    import yaml
    from cosmos_framework.inference.action_serving import create_app

    intervention = dict(
        active_slot_mask=args.active_slot_mask,
        exact_published_caption=args.exact_published_caption,
        warm_start_contract_run=str(args.warm_start_contract_run) if args.warm_start_contract_run else None,
        code=code_fingerprint(),
    )
    if args.warm_start_contract_run:
        from cosmos_framework.checkpoint.action128 import checkpoint_contract, validate_contract_extension
        from cosmos_framework.model.generator.mot.action128 import Action128Normalization

        model = service.model
        path = args.warm_start_contract_run / "config.yaml"
        current = yaml.safe_load(path.read_text())["model"]["config"]
        spec = current["action128_spec"]
        spatial = current["diffusion_expert_config"]["vision_spatial_position_mode"]
        old = checkpoint_contract(model)
        validate_contract_extension(
            old, dict(version=1, spec=spec, vision_spatial_position_mode=spatial), allow_normalization_migration=True
        )
        # Official warm_start skips only these four parameter-free contract buffers.
        # All loaded parameters retain object identity, storage and version.
        before = {name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()}
        model.net.action_normalization = Action128Normalization(spec, spatial).cuda()
        model.config.action128_spec = spec
        after = {name: (id(p), p.data_ptr(), p._version) for name, p in model.named_parameters()}
        if before != after:
            raise RuntimeError("warm-start contract migration changed learned parameters")
        service.spec = spec
        intervention.update(
            new_contract_config_sha256=sha256(path),
            old_contract=old,
            current_contract=checkpoint_contract(model),
            parameters_unchanged=True,
        )
    if args.active_slot_mask:
        evidence = install_active_slot_mask(service.model, [True] * 10 + [False] * 118)
    else:
        evidence = []
    original_prepare = service.prepare

    def prepare(self, request, output_dir):
        prepared, batch = original_prepare(request, output_dir)
        if prepared.action_dim_mask != [True] * 10 + [False] * 118:
            raise ValueError("this acceptance service is restricted to the LIBERO ten-slot contract")
        exact = False
        if args.exact_published_caption and request.prompt.lstrip().startswith("{"):
            caption = json.loads(request.prompt)
            required = {
                "task_title",
                "command",
                "embodiment",
                "duration",
                "fps",
                "action_frequency_hz",
                "camera_layout",
            }
            if not required.issubset(caption) or caption["action_frequency_hz"] != prepared.action_fps:
                raise ValueError("invalid explicit training-caption diagnostic")
            batch["ai_caption"] = [request.prompt]
            exact = True
        write_json(
            output_dir / "acceptance_input.json",
            dict(
                caption=batch["ai_caption"],
                exact_published_caption=exact,
                active_slot_mask=args.active_slot_mask,
                seed=prepared.seed,
                audit_stage="before_model",
            ),
        )
        return prepared, batch

    service.prepare = types.MethodType(prepare, service)
    service.info["acceptance_intervention"] = intervention
    service.info["port"] = args.port
    write_json(output / "intervention.json", intervention)
    try:
        uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, workers=1)
    finally:
        write_json(output / "mask_evidence.json", evidence)


if __name__ == "__main__":
    main()
