"""029 pure synthetic qualification of the real UniPC conversion arithmetic."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from prepare_wam_flow_readout import GPU_UUID, reference, require
from prepare_wam_free_trajectory import read_prepared

from s2a.artifacts import sha256
from s2a.worker_handshake import write_once


def precision_payload(name):
    index = np.arange(256)
    if name == "signed_normal":
        sample = ((index % 65 - 32) / 8).astype(np.float32)
        velocity = ((index * 7 % 97 - 48) / 16).astype(np.float32)
    elif name == "boundaries_subnormal":
        values = np.asarray(
            [
                0.0,
                -0.0,
                2**-133,
                -(2**-133),
                2**-126,
                -(2**-126),
                1 + 2**-8,
                -1 - 2**-8,
                1 + 2**-8 + 2**-23,
                -1 - 2**-8 - 2**-23,
                256,
                -256,
                0.5,
                -0.5,
                2**-20,
                -(2**-20),
            ],
            np.float32,
        )
        sample = np.resize(values, 256)
        velocity = np.resize(values[::-1], 256)
    else:
        raise ValueError("unregistered precision payload")
    return sample.reshape(2, 128), velocity.reshape(2, 128)


def observe_conversion(scheduler, sample, velocity):
    """Observe the product made inside the real call; return all objects intact."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    products = []

    class ProductObserver(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.mul.Tensor and any(a is velocity for a in args):
                products.append((result.detach().cpu().float().numpy().copy(), str(result.dtype)))
            return result

    with ProductObserver():
        converted = scheduler.convert_model_output(velocity, sample=sample)
    require(len(products) == 1, "actual conversion must make its one observed sigma-times-velocity product")
    return converted, products[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and os.environ.get("LD_LIBRARY_PATH") == "",
        "029 precision requires only physicalGPU6 and emptyLD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    output = Path(prepared["output_root"]) / "precision"
    output.mkdir(exist_ok=False)
    write_once(output / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.free-trajectory-synthetic-precision.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(args.prepared),
        seed=42,
        physical_gpu=6,
        cuda_visible_devices="6",
        model_weights_loaded=0,
        wam_denoise_calls=0,
        fit_groups=0,
        episode_reads=0,
        physics_steps=0,
        cases=[],
    )
    try:
        import torch

        require(torch.cuda.device_count() == 1, "029 precision CUDA mapping differs")
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = str(properties.uuid)
        require(
            actual_uuid.removeprefix("GPU-").lower() == GPU_UUID.removeprefix("GPU-").lower(),
            "029 precision actualGPU UUID differs",
        )
        result.update(actual_gpu_uuid=actual_uuid, torch_version=torch.__version__)
        source = (
            Path(prepared["config"]["cosmos_repo"])
            / "cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py"
        )
        require(sha256(source) == prepared["source_sha256"][str(source)], "029 real scheduler source differs")
        spec = importlib.util.spec_from_file_location("s2a_029_frozen_scheduler", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        arrays = {}
        for node in prepared["precision_nodes"]:
            for dtype_name in prepared["config"]["precision_velocity_dtypes"]:
                dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[dtype_name]
                for payload in prepared["config"]["precision_payloads"]:
                    name = f"step{node['step_index']}_{dtype_name.removeprefix('torch.')}_{payload}"
                    sample_np, velocity_np = precision_payload(payload)
                    metadata = dict(
                        name=name,
                        node=node,
                        payload=payload,
                        velocity_dtype=dtype_name,
                        sample_dtype="torch.float32",
                        source_conversion_calls_per_device=1,
                    )
                    arrays[f"{name}_sample"] = sample_np
                    arrays[f"{name}_velocity_input"] = velocity_np
                    for device in ("cpu", "cuda"):
                        scheduler = module.FlowUniPCMultistepScheduler()
                        scheduler.set_timesteps(35, device=device, shift=5.0)
                        scheduler._step_index = node["step_index"]
                        sigma = scheduler.sigmas[scheduler.step_index]
                        require(
                            float(sigma) == node["sigma_actual"]
                            and int(scheduler.timesteps[scheduler.step_index]) == node["network_timestep"],
                            "029 actual synthetic scheduler grid differs",
                        )
                        sample = torch.tensor(sample_np, device=device)
                        velocity = torch.tensor(velocity_np, device=device).to(dtype)
                        before = (sample._version, velocity._version)
                        converted, (product, product_dtype) = observe_conversion(scheduler, sample, velocity)
                        require(before == (sample._version, velocity._version), "029 conversion mutated its input")
                        arrays[f"{name}_{device}_velocity"] = velocity.detach().cpu().float().numpy()
                        arrays[f"{name}_{device}_product"] = product
                        arrays[f"{name}_{device}_x0"] = converted.detach().cpu().float().numpy()
                        arrays[f"{name}_{device}_sigma"] = sigma.detach().cpu().numpy()
                        metadata[device] = dict(
                            product_dtype=product_dtype,
                            x0_dtype=str(converted.dtype),
                            sigma_dtype=str(sigma.dtype),
                            sigma_device=str(sigma.device),
                            actual_input_unchanged=True,
                        )
                    for suffix in ("velocity", "product", "x0", "sigma"):
                        require(
                            np.array_equal(
                                arrays[f"{name}_cpu_{suffix}"].view(np.uint32),
                                arrays[f"{name}_cuda_{suffix}"].view(np.uint32),
                            ),
                            f"029 actual CPU/CUDA {suffix} bits differ",
                        )
                    result["cases"].append(metadata)
        require(len(result["cases"]) == 20, "029 synthetic20 coverage differs")
        archive = output / "actual_outputs.npz"
        np.savez(archive, **arrays)
        result.update(
            complete=True,
            passed=True,
            actual_outputs=reference(archive),
            worker_source=reference(__file__),
            scheduler_source=reference(source),
            actual_scheduler_conversion_calls=40,
            cpu_cuda_full128_bitwise_equal=True,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(output / "worker_result.json", result)
    print(json.dumps(dict(result=reference(output / "worker_result.json"))))


if __name__ == "__main__":
    main()
