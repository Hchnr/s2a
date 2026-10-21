"""026 synthetic CPU/CUDA precision worker; never loads WAM or episode data."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from run_wam_flow_readout import read_prepared, reference, require

from s2a.artifacts import REPO, sha256
from s2a.wam_flow_readout import FlowReadoutHook, FlowResidualReadout
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext, _bfloat16_rne
from s2a.worker_handshake import write_once


def expected_bits(hidden, base, weights, sigma):
    """Separate NumPy arithmetic reference for the registered operation order."""
    phi = np.column_stack((hidden.astype(np.float64) / np.sqrt(hidden.shape[1]), np.ones(len(hidden))))
    active = base[:, :10].astype(np.float64) + (-phi / np.asarray(sigma, np.float64)) @ weights
    return bf16_bits(active)


def bf16_bits(values):
    value = np.asarray(values, np.float64)
    _, exponent = np.frexp(np.abs(value))
    spacing = np.ldexp(np.ones(value.shape), np.maximum(exponent - 8, -133))
    rounded = np.copysign(np.rint(np.abs(value) / spacing) * spacing, value).astype(np.float32)
    return (rounded.view(np.uint32) >> 16).astype(np.uint16).view(np.int16)


def case_inputs():
    half, epsilon, spacing = 2**-8, 2**-30, 2**-133
    scenarios = (
        (
            "positive",
            1.0,
            [
                half - epsilon,
                half,
                half + epsilon,
                -half + epsilon,
                -half,
                -half - epsilon,
                3 * half - epsilon,
                3 * half,
                3 * half + epsilon,
                0.0,
            ],
        ),
        (
            "negative",
            -1.0,
            [
                -half + epsilon,
                -half,
                -half - epsilon,
                half - epsilon,
                half,
                half + epsilon,
                -3 * half + epsilon,
                -3 * half,
                -3 * half - epsilon,
                -0.0,
            ],
        ),
        (
            "subnormal",
            0.0,
            [
                spacing / 2 - 2**-160,
                spacing / 2,
                spacing / 2 + 2**-160,
                -spacing / 2 + 2**-160,
                -spacing / 2,
                -spacing / 2 - 2**-160,
                1.5 * spacing,
                2.5 * spacing,
                2**-126 - spacing / 2,
                -0.0,
            ],
        ),
    )
    for index, sigma in enumerate(np.asarray([0.2, 0.5, 0.8, 0.12809006869792938, 0.9990000128746033], np.float32)):
        for name, scalar, bias in scenarios:
            hidden = np.zeros((2, 1280), np.float32)
            hidden[1, :2] = [0.5, -0.25]
            base = np.full((2, 128), scalar, np.float32)
            base[:, 10:] = np.arange(118, dtype=np.float32)
            base[:, 10] = -0.0
            weights = np.zeros((1281, 10), np.float64)
            weights[-1] = -np.asarray(bias, np.float64) * float(sigma)
            weights[0, 0], weights[1, 1] = 2**-20, -(2**-21)
            yield dict(
                name=f"sigma{index}_{name}",
                hidden=hidden,
                base=base,
                weights=weights,
                sigma_actual=np.full((2, 1), float(sigma), np.float64),
            )


def tensor_bits(value):
    return value.detach().cpu().contiguous().view(__import__("torch").int16).numpy().copy()


def run_case(case, device):
    import torch

    class Head(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.register_buffer("base", base)
            self.last = None

        def forward(self, hidden, ids):
            self.last = self.base.clone()
            return self.last

    context = DomainContext(LIBERO_DOMAIN, 31, 0, ACTIVE_MASK)
    other = DomainContext("synthetic/other/full/domain", 2, 0, ACTIVE_MASK)
    readout = FlowResidualReadout(case["weights"])
    hidden, base = (torch.tensor(case[key], dtype=torch.bfloat16, device=device) for key in ("hidden", "base"))
    ids = torch.zeros(len(base), dtype=torch.long, device=device)
    original_hidden, original_base = tensor_bits(hidden), tensor_bits(base)
    output = readout.apply_torch(hidden, base, context, ids, case["sigma_actual"])
    require(
        np.array_equal(tensor_bits(hidden), original_hidden) and np.array_equal(tensor_bits(base), original_base),
        "026 synthetic input mutated",
    )
    require(np.array_equal(tensor_bits(output[:, 10:]), original_base[:, 10:]), "026 synthetic inactive118 changed")
    head = Head(base).eval()
    hook = FlowReadoutHook(head, readout)
    with hook.request(context), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        hooked = head(hidden, ids)
    require(np.array_equal(tensor_bits(output), tensor_bits(hooked)), "026 actual hook differs from direct application")
    with hook.request(other), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        unchanged = head(hidden, ids)
        require(
            unchanged is head.last and np.array_equal(tensor_bits(unchanged), original_base),
            "026 other domain bypass changed",
        )
    hook.readout = FlowResidualReadout.zero()
    with hook.request(context), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        identity = head(hidden, ids)
        require(
            identity is head.last and np.array_equal(tensor_bits(identity), original_base), "026 zero bypass changed"
        )
    hook.remove()
    return tensor_bits(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and os.environ.get("LD_LIBRARY_PATH") == "",
        "026 precision worker requires only physicalGPU6 and emptyLD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    root = Path(prepared["output_root"]) / "precision"
    root.mkdir()
    write_once(root / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.flow-readout-synthetic-cpu-cuda-precision.v1",
        prepared_ref=reference(args.prepared),
        complete=False,
        passed=False,
        seed=42,
        physical_gpu=6,
        core_sha256=sha256(REPO / "src/s2a/wam_flow_readout.py"),
        model_weights_loaded=0,
        wam_denoise_calls=0,
        fit_groups=0,
        episode_reads=0,
        physics_steps=0,
    )
    try:
        import torch

        torch.cuda.set_device(0)
        torch.manual_seed(42)
        properties = torch.cuda.get_device_properties(0)
        require(
            str(properties.uuid).removeprefix("GPU-").lower() == "7f176867-170c-b508-5c89-1cd819ad3b8e",
            "026 actual CUDA device UUID differs",
        )
        arrays, cases = {}, []
        with torch.inference_mode():
            for case in case_inputs():
                name = case["name"]
                for key in ("hidden", "base", "weights", "sigma_actual"):
                    arrays[f"{name}_{key}"] = case[key]
                actual = {device: run_case(case, device) for device in ("cpu", "cuda:0")}
                expected = expected_bits(case["hidden"], case["base"], case["weights"], case["sigma_actual"])
                require(np.array_equal(actual["cpu"], actual["cuda:0"]), "026 synthetic CPU/CUDA bitwise mismatch")
                require(
                    np.array_equal(actual["cpu"][:, :10], expected),
                    "026 synthetic arithmetic disagrees with independent NumPy reference",
                )
                arrays[f"{name}_cpu_bits"], arrays[f"{name}_cuda_bits"] = actual["cpu"], actual["cuda:0"]
                cases.append(
                    dict(
                        name=name,
                        cpu_cuda_bitwise_equal=True,
                        actual_hook_matches_apply=True,
                        zero_original_object=True,
                        other_domain_original_object=True,
                        other118_preserved=True,
                        input_tensors_unchanged=True,
                    )
                )
            primitive = np.asarray(
                [
                    0.0,
                    -0.0,
                    2**-134 - 2**-160,
                    2**-134,
                    2**-134 + 2**-160,
                    -(2**-134) + 2**-160,
                    -(2**-134),
                    -(2**-134) - 2**-160,
                    1 + 2**-8 + 2**-30,
                    -1 - 2**-8 - 2**-30,
                ],
                np.float64,
            )
            arrays["primitive_values"] = primitive
            for device in ("cpu", "cuda:0"):
                actual = tensor_bits(_bfloat16_rne(torch.tensor(primitive, dtype=torch.float64, device=device)))
                require(np.array_equal(actual, bf16_bits(primitive)), "026 primitive midpoint/subnormal RNE differs")
                arrays[f"primitive_{'cpu' if device == 'cpu' else 'cuda'}_bits"] = actual
        torch.cuda.synchronize()
        archive = root / "actual_outputs.npz"
        with archive.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        read_prepared(args.prepared, args.prepared_sha256)
        result.update(
            complete=True,
            passed=True,
            cpu_cuda_bitwise_equal=True,
            cases=cases,
            actual_outputs=reference(archive),
            actual_gpu_uuid=str(properties.uuid),
            actual_cuda_index=0,
            cuda_visible_devices="6",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            actual_gpu_name=properties.name,
            worker_source=reference(__file__),
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write_once(root / "worker_result.json", result)
    print(json.dumps(dict(result=reference(root / "worker_result.json"))))


if __name__ == "__main__":
    main()
