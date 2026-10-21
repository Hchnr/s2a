"""028 synthetic CPU/CUDA qualification of frozen027 centering and scaling."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from probe_wam_flow_readout_quantization import bf16_bits, tensor_bits
from probe_wam_flow_readout_quantization import case_inputs as flow_cases
from run_wam_scaled_state import read_prepared, reference, require

from s2a.artifacts import REPO, sha256
from s2a.wam_flow_readout import FlowReadoutHook
from s2a.wam_readout import ACTIVE_MASK, LIBERO_DOMAIN, DomainContext, _bfloat16_rne
from s2a.wam_scaled_readout import ScaledFlowReadout
from s2a.worker_handshake import write_once


def case_inputs(model, sigmas):
    for case in flow_cases():
        center = np.zeros(1280, np.float64)
        center[:2] = [0.5, -0.25]
        yield dict(case, center=center, scale=np.asarray(3.0, np.float64))
    rng = np.random.default_rng(42)
    for index, sigma in enumerate(sigmas):
        yield dict(
            name=f"sigma{index}_candidate",
            hidden=rng.integers(-32, 33, (2, 1280)).astype(np.float32) / 128,
            base=rng.integers(-16, 17, (2, 128)).astype(np.float32) / 16,
            weights=model["weights"].copy(),
            center=model["center"].copy(),
            scale=np.asarray(model["scale"], np.float64).copy(),
            sigma_actual=np.full((2, 1), sigma, np.float64),
        )


def expected_bits(case):
    phi = np.column_stack(
        ((case["hidden"].astype(np.float64) - case["center"]) / case["scale"], np.ones(len(case["hidden"])))
    )
    active = case["base"][:, :10].astype(np.float64) + (-phi / case["sigma_actual"]) @ case["weights"]
    return bf16_bits(active)


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
    readout = ScaledFlowReadout(case["weights"], case["center"], case["scale"])
    hidden, base = (torch.tensor(case[key], dtype=torch.bfloat16, device=device) for key in ("hidden", "base"))
    ids = torch.zeros(len(base), dtype=torch.long, device=device)
    original_hidden, original_base = tensor_bits(hidden), tensor_bits(base)
    output = readout.apply_torch(hidden, base, context, ids, case["sigma_actual"])
    require(
        np.array_equal(tensor_bits(hidden), original_hidden) and np.array_equal(tensor_bits(base), original_base),
        "028 synthetic input mutated",
    )
    require(np.array_equal(tensor_bits(output[:, 10:]), original_base[:, 10:]), "028 inactive118 changed")
    head = Head(base).eval()
    hook = FlowReadoutHook(head, readout)
    with hook.request(context), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        hooked = head(hidden, ids)
    require(np.array_equal(tensor_bits(output), tensor_bits(hooked)), "028 actual hook differs from apply")
    with hook.request(other), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        unchanged = head(hidden, ids)
        require(
            unchanged is head.last and np.array_equal(tensor_bits(unchanged), original_base),
            "028 domain bypass changed",
        )
    hook.readout = ScaledFlowReadout.zero(center=case["center"], scale=case["scale"])
    with hook.request(context), hook._velocity_step(float(case["sigma_actual"][0, 0]), {}):
        identity = head(hidden, ids)
        require(
            identity is head.last and np.array_equal(tensor_bits(identity), original_base), "028 zero bypass changed"
        )
    hook.remove()
    return tensor_bits(output)


def primitive_values():
    return np.asarray(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "6" and os.environ.get("LD_LIBRARY_PATH") == "",
        "028 precision requires physicalGPU6 and emptyLD",
    )
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    model_ref = prepared["adapted_model_ref"]
    with np.load(model_ref["path"], allow_pickle=False) as archive:
        require(set(archive.files) == {"weights", "center", "scale"}, "028 actual candidate parameter roster differs")
        model = {key: archive[key].copy() for key in archive.files}
    require(
        model["weights"].shape == (1281, 10)
        and model["center"].shape == (1280,)
        and model["scale"].shape == ()
        and all(v.dtype == np.float64 and np.isfinite(v).all() for v in model.values())
        and model["scale"] > 0,
        "028 actual candidate precision/geometry differs",
    )
    root = Path(prepared["output_root"]) / "precision"
    root.mkdir()
    write_once(root / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.scaled-readout-synthetic-cpu-cuda-precision.v1",
        prepared_ref=reference(args.prepared),
        candidate_model_ref=model_ref,
        complete=False,
        passed=False,
        seed=42,
        physical_gpu=6,
        core_sha256=sha256(REPO / "src/s2a/wam_scaled_readout.py"),
        model_weights_loaded=0,
        candidate_models_loaded=1,
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
            "028 actual CUDA device UUID differs",
        )
        arrays, cases = {}, []
        with torch.inference_mode():
            for case in case_inputs(model, prepared["config"]["precision_actual_sigmas"]):
                name = case["name"]
                for key in ("hidden", "base", "weights", "center", "scale", "sigma_actual"):
                    arrays[f"{name}_{key}"] = case[key]
                actual = {device: run_case(case, device) for device in ("cpu", "cuda:0")}
                require(np.array_equal(actual["cpu"], actual["cuda:0"]), "028 CPU/CUDA bitwise mismatch")
                require(
                    np.array_equal(actual["cpu"][:, :10], expected_bits(case)),
                    "028 independent NumPy arithmetic differs",
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
            primitive = primitive_values()
            arrays["primitive_values"] = primitive
            for device in ("cpu", "cuda:0"):
                bits = tensor_bits(_bfloat16_rne(torch.tensor(primitive, dtype=torch.float64, device=device)))
                require(np.array_equal(bits, bf16_bits(primitive)), "028 primitive RNE differs")
                arrays[f"primitive_{'cpu' if device == 'cpu' else 'cuda'}_bits"] = bits
        require(len(cases) == 20, "028 precision case budget differs")
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
