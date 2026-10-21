"""029 independent CPU preparation, mixed-precision and trajectory audit.

No producer observer, Torch, Cosmos, training loader or fitting code is used.
Preparation/precision never open EP/PNG files. Labels are opened only after the
complete released collection, exact raw replay and all per-call contracts pass.
The diagnostic cannot authorize a fit, physics or fresh evaluation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_free_trajectory_diagnostic_20261021.json"
CONFIG_SHA = "a271cacaf0e011f0dd497173437f09cf3f256cb384e66bdf9d8a71928dd2cb73"
PLAN = REPO / "plans/v0.0.1/029_free_trajectory_diagnostic_20261021.md"
PLAN_SHA = "3616046f32d6baa6939b02ecab619c31297c9b187cc26b50520f9a38a8b4b731"
ADDENDUM = REPO / "plans/v0.0.1/029_scalar_promotion_addendum_20261021.md"
ADDENDUM_SHA = "a83792831cd9aefd31adfb3b6f92827171411b087cd91d084f8fd98a5142ca6d"
SOURCE_ADDENDUM = REPO / "plans/v0.0.1/029_source_revision_addendum_20261021.md"
SOURCE_ADDENDUM_SHA = "4abe4742eaee961d371f129031d79fa41d3c3a1ea8252defa5b37c4da74b7946"
SCALAR_UNIT = (
    REPO / "outputs/wam_stability_20261021/free_trajectory_arithmetic_unit_observer/model_python_cpu_scalar.json"
)
SCALAR_UNIT_SHA = "20622da8808d62e8d45db32993dcf138593f34b8d97404bbf3c5d9db758eb018"
BACKEND_CONTRACT = dict(
    cpu_bfloat16="sigma_bf16_then_product_fp32_then_bf16_then_sub_fp32",
    cuda_bfloat16="sigma_fp32_then_product_fp32_then_bf16_then_sub_fp32",
    float32="product_fp32_then_sub_fp32",
)
HELPER = REPO / "scripts/verify_wam_scaled_state.py"
HELPER_SHA = "75da0451cd5d00465a09a45dfbff0212be01e98fdda15f4d2e3046507826ca82"
NEW_SOURCES = (
    "src/s2a/wam_trajectory_capture.py",
    "scripts/collect_wam_free_trajectory.py",
    "tests/test_wam_trajectory_capture.py",
    "scripts/verify_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_verification.py",
    "scripts/prepare_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_preparation.py",
    "scripts/run_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_runner.py",
    "scripts/probe_wam_free_trajectory_precision.py",
    "tests/test_wam_free_trajectory_precision.py",
)
STATE_REFS = ("prepared", "sampling", "verification", "owner", "final_peer")


def frozen_helpers():
    if sha256(HELPER) != HELPER_SHA:
        raise ValueError("frozen028 independent helper changed")
    spec = importlib.util.spec_from_file_location("frozen028_trajectory_helpers", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATE = frozen_helpers()
SCALED, FLOW, OLD = STATE.SCALED, STATE.FLOW, STATE.OLD
require, numeric, exact = OLD.require, OLD.numeric, OLD.exact
reference = STATE.reference


def bits_equal(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(
        actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
        f"{name}: actual dtype/shape/bits differ",
    )


def observation_paths(source):
    return SCALED.forbidden_observations(source)


def metadata_hashes(mapping, forbidden, evidence):
    """Never hash raw observations, including PNG paths absent from the roster."""
    skipped = {}
    for filename, digest in mapping.items():
        path = Path(filename).resolve()
        if str(path) in forbidden or path.suffix.lower() == ".png":
            skipped[str(path)] = digest
        else:
            evidence.check(path, digest)
    return skipped


def source_closure(prepared, source, path):
    expected = (
        set(source["source_sha256"])
        | {str(REPO / p) for p in NEW_SOURCES}
        | {str(CONFIG), str(PLAN), str(ADDENDUM), str(SOURCE_ADDENDUM)}
    )
    require(
        len(source["source_sha256"]) == 96
        and len(expected) == 111
        and set(prepared["source_sha256"]) == expected
        and all(prepared["source_sha256"].get(k) == v for k, v in source["source_sha256"].items()),
        "029 exact111 source closure changed or omitted required entrypoints",
    )
    require(
        prepared["source_sha256"][str(CONFIG)] == CONFIG_SHA
        and prepared["source_sha256"][str(PLAN)] == PLAN_SHA
        and prepared["source_sha256"][str(ADDENDUM)] == ADDENDUM_SHA
        and prepared["source_sha256"][str(SOURCE_ADDENDUM)] == SOURCE_ADDENDUM_SHA
        and prepared["source_sha256"][str(HELPER)] == HELPER_SHA,
        "029 frozen registration/helper changed",
    )
    # Unlike027/028,029 explicitly numbers the sorted complete source table.
    expected_snapshots = {
        str(Path(path).resolve().parent / "source_snapshot" / f"{i:03d}_{Path(p).name}"): prepared["source_sha256"][p]
        for i, p in enumerate(sorted(expected))
    }
    require(prepared["snapshot_sha256"] == expected_snapshots, "029 source snapshots are not the111 exact sources")


def inherited_state(values, refs):
    source, sampling, audit, owner, peer = (values[k] for k in STATE_REFS)
    require(
        source["schema_version"] == "s2a.scaled-state-preparation.v1"
        and sampling["schema_version"] == "s2a.scaled-state-sampling.v1"
        and sampling["complete"] is sampling["passed"] is True
        and sampling["prepared_ref"] == refs["prepared"]
        and sampling["denoise_calls"] == 3465
        and len(sampling["generations"]) == 96,
        "029 lacks complete original028 runtime",
    )
    require(
        audit["passed"] is audit["sampling_verified"] is True
        and audit["prepared"] == refs["prepared"]
        and audit["input_sha256"].get(refs["sampling"]["path"]) == refs["sampling"]["sha256"]
        and audit["sampling_gate"]["passed"] is False
        and audit["eligible_for_development_physics_registration"] is audit["physics_authorized"] is False,
        "029 changed original028 failed state gate",
    )
    require(
        owner["complete"] is owner["passed"] is True
        and owner["producer_result"] == refs["sampling"]
        and owner["prepared_ref"] == refs["prepared"]
        and owner["automatic_retry"] is False
        and owner["eligible_for_development_physics_registration"] is owner["physics_authorized"] is False
        and len(owner["owned_commands"]) == 2,
        "029 original028 owner/lifecycle incomplete",
    )
    require(
        peer["complete"] is peer["passed"] is True
        and peer["prepared"] == refs["prepared"]
        and peer["sampling_result"] == refs["sampling"]
        and peer["verification"] == refs["verification"]
        and peer["owner_result"] == refs["owner"]
        and peer["frozen_sources_verified"] == peer["source_snapshots_verified"] == 96
        and peer["original8400_weight_stats_verified"] == 163
        and peer["source_before_after_unchanged"] is True,
        "029 final028 peer binding differs",
    )


def roster_contract(prepared, source, sampling):
    config = prepared["config"]
    anchors = [a for a in source["anchors"] if a["split"] == "validation"]
    require(
        prepared["anchors"] == anchors
        and len(anchors) == 24
        and list(dict.fromkeys(a["episode"] for a in anchors)) == config["validation_episodes"]
        and all(a["valid_future_steps"] == 40 for a in anchors),
        "029 changed six validation EP/four complete anchors",
    )
    schedule, times = FLOW.solver_schedule()
    nodes = [
        dict(step_index=i, sigma_actual=float(schedule[i]), network_timestep=int(times[i])) for i in (0, 19, 29, 33, 34)
    ]
    require(prepared["precision_nodes"] == nodes, "029 actual35/shift5 precision nodes differ")
    original = [entry for entry in sampling["generations"] if entry["variant"] == "raw"]
    require(
        [(e["anchor_name"], e["sampling_seed"]) for e in original]
        == [(a["name"], seed) for a in anchors for seed in (42, 43)],
        "029 inherited raw48 anchor/seed order differs",
    )
    expected_free, expected_teacher = [], []
    for i, old in enumerate(original):
        member = {
            k: v for k, v in old.items() if k not in ("response", "conditioning", "initial", "actual_sigma_trace")
        }
        member.update(
            generation_index=i,
            source_generation_index=old["generation_index"],
            cache={k: old[k] for k in ("response", "conditioning", "initial", "actual_sigma_trace")},
        )
        expected_free.append(member)
        for node in nodes[1:4]:
            expected_teacher.append(
                dict(
                    call_index=len(expected_teacher),
                    name=f"{old['name']}_teacher_step{node['step_index']}",
                    anchor_name=old["anchor_name"],
                    sampling_seed=old["sampling_seed"],
                    free_generation_index=i,
                    **node,
                )
            )
    require(
        prepared["free_generations"] == expected_free and prepared["teacher_calls"] == expected_teacher,
        "029 raw48/pairedteacher144 grid is incomplete, reordered or changed",
    )


def verify_preparation(path, evidence):
    prepared = evidence.json(path)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(HELPER, HELPER_SHA)
    require(
        prepared["schema_version"] == "s2a.free-trajectory-preparation.v1"
        and prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == reference(PLAN)
        and prepared.get("source_revision_ref") == dict(path=str(SOURCE_ADDENDUM), sha256=SOURCE_ADDENDUM_SHA)
        and prepared["output_root"] == str(REPO / (config["output_dir"] + "_source_v2")),
        "029 prepared registration/config/output changed",
    )
    evidence.check(SOURCE_ADDENDUM, SOURCE_ADDENDUM_SHA)
    arithmetic_admission(prepared, evidence)
    refs = {
        key: dict(path=str(REPO / config[f"source_state_{key}"]), sha256=config[f"source_state_{key}_sha256"])
        for key in STATE_REFS
    }
    require(all(prepared[f"source_state_{k}_ref"] == ref for k, ref in refs.items()), "029 changed original028 refs")
    values = {key: evidence.ref(ref) for key, ref in refs.items()}
    inherited_state(values, refs)
    source = values["prepared"]
    forbidden = observation_paths(source)
    declarations = {}
    for value in values.values():
        for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
            declarations.update(metadata_hashes(value.get(key, {}), forbidden, evidence))
    require(prepared["inherited_observation_sha256"] == declarations, "029 inherited observation declarations changed")
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        require(
            not metadata_hashes(prepared[key], forbidden, evidence),
            "029 actual hashed inputs wrongly include original EP/PNG",
        )
    source_closure(prepared, source, path)
    for key in (
        "source_prepared_ref",
        "source_features_ref",
        "scaled_prepared_ref",
        "scaled_fit_ref",
        "scaled_fit_verification_ref",
        "adapted_model_ref",
        "domain",
        "checkpoint",
        "weight_files",
        "external_cosmos",
        "conservative_source_exclusion",
    ):
        require(prepared[key] == source[key], f"029 changed inherited {key}")
    roster_contract(prepared, source, values["sampling"])
    head = subprocess.check_output(["git", "-C", config["cosmos_repo"], "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", config["cosmos_repo"], "status", "--porcelain", "--untracked-files=no"], text=True
    )
    require(
        head == "8ed68a9722a14e8f6c21fa0c6c700e1617b7b73e"
        and not dirty.strip()
        and prepared["external_cosmos"] == dict(head=head, tracked_source_clean=True),
        "029 external tracked tree must remain clean8ed",
    )
    for name in ("prepared", "fit", "fit_verification"):
        require(
            prepared[f"scaled_{name}_ref"]
            == dict(path=str(REPO / config[f"source_scaled_{name}"]), sha256=config[f"source_scaled_{name}_sha256"]),
            "029 changed the frozen027 ancestor identity",
        )
    fit, fit_audit = (evidence.ref(prepared[f"scaled_{name}_ref"]) for name in ("fit", "fit_verification"))
    require(
        fit["complete"] is fit["passed"] is True
        and fit_audit["passed"] is fit_audit["fit_verified"] is True
        and fit["teacher_gate"]["passed"] is fit_audit["teacher_gate"]["passed"] is True
        and fit["eligible_for_sampling"] is fit_audit["eligible_for_sampling"] is False
        and prepared["adapted_model_ref"] == fit["models"]["adapt"]
        and fit_audit["input_sha256"].get(prepared["scaled_fit_ref"]["path"]) == prepared["scaled_fit_ref"]["sha256"],
        "029 replaced027 coefficients or upgraded its original admission",
    )
    for i, command in enumerate(values["owner"]["owned_commands"]):
        STATE.owned_exit(command, evidence, gpu=i == 0)
    require(len(prepared["weight_files"]) == 163, "029 original8400 weight roster differs")
    for filename, saved in prepared["weight_files"].items():
        stat = Path(filename).stat()
        require((stat.st_size, stat.st_mtime_ns) == (saved["bytes"], saved["mtime_ns"]), "029 frozen8400 stat changed")
    STATE.read_model(prepared["adapted_model_ref"], evidence)
    old = evidence.ref(prepared["source_prepared_ref"])
    config_path = str(Path(config["checkpoint"]["run"]) / "config.yaml")
    spec = OLD.yaml.safe_load(evidence.check(config_path, old["input_sha256"][config_path]).read_text())["model"][
        "config"
    ]["action128_spec"]
    require(
        spec["registry"]["domains"].index(OLD.DOMAIN) == 31 and spec["registry"]["domain_projection_ids"][31] == 0,
        "029 normalization31/projection0 registry changed",
    )
    return prepared, spec["transforms"][31]


def arithmetic_admission(prepared, evidence):
    require(
        prepared["numerical_addendum_ref"] == dict(path=str(ADDENDUM), sha256=ADDENDUM_SHA)
        and prepared["scalar_promotion_unit_ref"] == dict(path=str(SCALAR_UNIT), sha256=SCALAR_UNIT_SHA)
        and prepared["backend_arithmetic_contract"] == BACKEND_CONTRACT,
        "029 preregistered backend arithmetic/addendum changed",
    )
    evidence.check(ADDENDUM, ADDENDUM_SHA)
    unit = evidence.json(SCALAR_UNIT, SCALAR_UNIT_SHA)
    require(
        unit["torch_version"] == "2.10.0+cu130"
        and unit["all_bfloat16_scalar_first_exact"] is True
        and unit["environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and all(
            unit[k] == 0
            for k in (
                "original_scheduler_calls",
                "model_weights_loaded",
                "wam_denoise_calls",
                "fit_groups",
                "episode_reads",
                "physics_steps",
            )
        ),
        "029 CPU-only scalar observation scope differs",
    )
    expected_cases = [
        (sigma, payload)
        for sigma in prepared["config"]["selected_actual_sigmas"]
        for payload in ("signed_normal", "boundaries_subnormal")
    ]
    require(len(unit["cases"]) == len(expected_cases) == 6, "029 actual CPU scalar observation six-case grid differs")
    for case, (sigma, payload) in zip(unit["cases"], expected_cases, strict=True):
        sample, velocity = precision_payload(payload)
        velocity = OLD.quantize_bfloat16(velocity)
        product, _ = mixed_convert(sample, velocity, np.asarray(sigma, np.float32), "torch.bfloat16", "cpu")
        require(
            case["sigma_actual"] == sigma
            and case["payload"] == payload
            and case["bf16_scalar_first_max_abs_difference"] == 0
            and case["actual_product_sha256"] == hashlib.sha256(product.tobytes()).hexdigest(),
            "029 actual CPU scalar product bytes differ from independently fixed gold",
        )
    torch_root = Path(prepared["config"]["cosmos_repo"]) / ".venv/lib/python3.13/site-packages/torch"
    required = [
        str(torch_root / suffix)
        for suffix in (
            "version.py",
            "include/ATen/native/cuda/Loops.cuh",
            "include/ATen/TensorIterator.h",
            "include/ATen/OpMathType.h",
        )
    ]
    require(
        [ref["path"] for ref in prepared["torch_arithmetic_inputs"]] == required,
        "029 arithmetic input source roster changed",
    )
    for ref in prepared["torch_arithmetic_inputs"]:
        evidence.check(ref["path"], ref["sha256"])


def mixed_convert(sample, velocity, sigma, velocity_dtype, backend="cuda"):
    """Preregistered backend-specific scalar promotion and FP32 subtraction.

    CPU casts the scalar to the BF16 common dtype. CUDA's opmath scalar path
    retains the FP32 scalar. Both round the product to FP32 before its BF16
    output rounding. Each fixed gold must pass actual20-case qualification;
    cross-backend equality is only reported. The FP64 ideal is kept separate.
    """
    sample, velocity = numeric(sample, "sample"), numeric(velocity, "velocity")
    sigma = numeric(sigma, "sigma")
    require(
        sample.dtype == velocity.dtype == sigma.dtype == np.float32 and sigma.shape == (),
        "actual convert input dtypes differ",
    )
    require(sample.shape == velocity.shape and float(sigma) > 0, "actual convert shape/positive sigma differs")
    require(backend in ("cpu", "cuda"), "029 unregistered arithmetic backend")
    if velocity_dtype == "torch.bfloat16":
        bits_equal(velocity, OLD.quantize_bfloat16(velocity), "actual BF16 velocity lossless export")
        scalar = OLD.quantize_bfloat16(sigma) if backend == "cpu" else sigma
        product = OLD.quantize_bfloat16((velocity.astype(np.float64) * float(scalar)).astype(np.float32))
    else:
        require(velocity_dtype == "torch.float32", "unsupported actual velocity dtype")
        product = (velocity.astype(np.float64) * float(sigma)).astype(np.float32)
    result = (sample - product).astype(np.float32)
    return product, result


def precision_payload(name):
    index = np.arange(256)
    if name == "signed_normal":
        sample = ((index % 65 - 32) / 8).astype(np.float32)
        velocity = ((index * 7 % 97 - 48) / 16).astype(np.float32)
    else:
        require(name == "boundaries_subnormal", "029 unregistered precision payload")
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
        sample, velocity = np.resize(values, 256), np.resize(values[::-1], 256)
    return sample.reshape(2, 128), velocity.reshape(2, 128)


def precision_arrays(arrays, cases, nodes):
    expected = [
        (node, dtype, payload)
        for node in nodes
        for dtype in ("torch.bfloat16", "torch.float32")
        for payload in ("signed_normal", "boundaries_subnormal")
    ]
    require(len(cases) == len(expected) == 20, "029 actual conversion20 coverage differs")
    required, ideal_errors, equalities = set(), [], []
    for case, (node, dtype, payload) in zip(cases, expected, strict=True):
        name = f"step{node['step_index']}_{dtype.removeprefix('torch.')}_{payload}"
        require(
            case["name"] == name
            and case["node"] == node
            and case["payload"] == payload
            and case["velocity_dtype"] == dtype
            and case["sample_dtype"] == "torch.float32"
            and case["source_conversion_calls_per_device"] == 1,
            "029 actual precision case identity/delegation differs",
        )
        sample, uncast = precision_payload(payload)
        inputs = dict(sample=sample, velocity_input=uncast)
        velocity = OLD.quantize_bfloat16(uncast) if dtype == "torch.bfloat16" else uncast
        sigma = np.asarray(node["sigma_actual"], np.float32)
        for key, expected_value in inputs.items():
            key = f"{name}_{key}"
            required.add(key)
            bits_equal(arrays[key], expected_value, "029 precision fixture input")
        for device in ("cpu", "cuda"):
            product, x0 = mixed_convert(sample, velocity, sigma, dtype, device)
            meta = case[device]
            require(
                meta["product_dtype"] == dtype
                and meta["x0_dtype"] == meta["sigma_dtype"] == "torch.float32"
                # Frozen UniPC keeps its sigma table on CPU even when its
                # integer timestep table and sample/velocity are on CUDA.
                and meta["sigma_device"] == "cpu"
                and meta["actual_input_unchanged"] is True,
                "029 actual scalar promotion/product/subtraction dtype differs",
            )
            for key, gold in dict(velocity=velocity, product=product, x0=x0, sigma=sigma).items():
                key = f"{name}_{device}_{key}"
                required.add(key)
                bits_equal(arrays[key], gold, "029 actual conversion full128 bits")
            ideal = sample.astype(np.float64) - float(sigma) * velocity.astype(np.float64)
            difference = x0.astype(np.float64) - ideal
            ideal_errors.append(
                dict(
                    name=name,
                    backend=device,
                    max_abs=float(np.max(np.abs(difference))),
                    mse=float(np.mean(difference**2)),
                )
            )
        equality = {
            suffix: arrays[f"{name}_cpu_{suffix}"].tobytes() == arrays[f"{name}_cuda_{suffix}"].tobytes()
            for suffix in ("velocity", "product", "x0", "sigma")
        }
        require(case["cpu_cuda_bits_equal"] == equality, "029 cross-backend difference report differs from saved bits")
        equalities.append(equality)
    require(set(arrays) == required, "029 precision archive contains omitted/extra inputs or outputs")
    return dict(
        ideal_fp64_difference=ideal_errors,
        cpu_cuda_full128_bitwise_equal=all(all(item.values()) for item in equalities),
        cpu_cuda_equal_cases={
            key: sum(item[key] for item in equalities) for key in ("velocity", "product", "x0", "sigma")
        },
        cross_backend_equality_is_admission_gate=False,
    )


def prepared_ref(prepared):
    return reference(Path(prepared["output_root"]) / "preparation/prepared.json")


def verify_precision(prepared, path, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.free-trajectory-synthetic-precision.v1"
        and actual["complete"] is actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == prepared_ref(prepared)
        and actual["seed"] == 42
        and actual["physical_gpu"] == 6
        and actual["cuda_visible_devices"] == "6"
        and actual["actual_gpu_uuid"].removeprefix("GPU-").lower() == STATE.GPU_UUID.removeprefix("GPU-").lower(),
        "029 partial/rejected precision or actualGPU identity differs",
    )
    require(
        all(
            actual[k] == 0
            for k in ("model_weights_loaded", "wam_denoise_calls", "fit_groups", "episode_reads", "physics_steps")
        )
        and actual["actual_scheduler_conversion_calls"] == 40
        and isinstance(actual["cpu_cuda_full128_bitwise_equal"], bool),
        "029 synthetic qualification exceeded budget",
    )
    require(
        actual["backend_arithmetic_contract"] == BACKEND_CONTRACT
        and actual["numerical_addendum_ref"] == prepared["numerical_addendum_ref"]
        and actual["torch_version"] == "2.10.0+cu130"
        and actual["torch_git_version"] == "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
        and actual["cuda_version"] == "13.0",
        "029 actual runtime differs from registered arithmetic qualification",
    )
    for key, suffix in (
        ("worker_source", "scripts/probe_wam_free_trajectory_precision.py"),
        ("scheduler_source", "cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py"),
    ):
        ref = actual[key]
        require(
            ref["path"].endswith(suffix) and prepared["source_sha256"].get(ref["path"]) == ref["sha256"],
            "029 precision source is not frozen",
        )
        evidence.check(ref["path"], ref["sha256"])
    identity = STATE.public_lifecycle(actual, prepared, "precision", evidence)
    archive = evidence.check(actual["actual_outputs"]["path"], actual["actual_outputs"]["sha256"])
    with np.load(archive, allow_pickle=False) as data:
        diagnostics = precision_arrays({k: data[k] for k in data.files}, actual["cases"], prepared["precision_nodes"])
    require(
        actual["cpu_cuda_full128_bitwise_equal"] == diagnostics["cpu_cuda_full128_bitwise_equal"],
        "029 aggregate cross-backend report differs",
    )
    return dict(
        precision_verified=True,
        backend_gold_verified=True,
        eligible_for_collection=True,
        actual_conversion_cases=20,
        actual_scheduler_conversion_calls=40,
        actual_owned_model_exit=identity,
        **diagnostics,
    )


def trace_contract(trace, planned, prepared):
    # The original independent026 check is reused with only029's additional
    # actual conversion counter removed from its exact step dictionary.
    stripped = dict(trace, steps=[{k: v for k, v in s.items() if k != "conversion_calls"} for s in trace["steps"]])
    FLOW.sigma_trace(stripped, planned["name"], planned["sampling_seed"], prepared)
    require(
        all(s.get("conversion_calls") == 1 for s in trace["steps"]), "029 extra/missing actual scheduler conversion"
    )


def tensor_contract(metadata, arrays, *, teacher=False):
    shapes = dict(
        hidden=(40, 1280),
        base=(40, 128),
        projection_ids=(40,),
        pred_action=(40, 128),
        packed_action=(40, 128),
        network_timestep=(1, 1),
    )
    for key, shape in shapes.items():
        require(arrays[key].shape == shape, f"029 actual {key} shape differs")
    require(
        arrays["projection_ids"].dtype == np.int64 and not np.any(arrays["projection_ids"]),
        "029 actual shared head projection differs",
    )
    require(arrays["network_timestep"].dtype == np.int64, "029 sampler network timestep is not actual int64")
    for key in ("hidden", "base", "packed_action", "packed_vision", "pred_action"):
        require(metadata["original_dtypes"][key] == "torch.bfloat16", f"029 actual {key} dtype differs")
        bits_equal(arrays[key], OLD.quantize_bfloat16(arrays[key]), f"029 lossless BF16 {key}")
    require(arrays["packed_vision"].shape == (1, 48, 11, 60, 40), "029 actual latent vision layout differs")
    bits_equal(arrays["pred_action"][:, :10], arrays["base"][:, :10], "029 zero head active output")
    require(
        not np.any(arrays["pred_action"][:, 10:]) and not np.any(arrays["packed_action"][:, 10:]),
        "029 inactive118 action/velocity is nonzero",
    )
    exact(arrays["condition_action"], np.zeros((40, 1)), "029 no future action conditioning")
    for kind in ("vision", "action"):
        key = f"time_{kind}"
        require(
            arrays[key].size > 0 and metadata["original_dtypes"][key] == "torch.float32",
            "029 final packed time dtype/shape differs",
        )
        bits_equal(
            arrays[key], np.full_like(arrays[key], metadata["network_timestep"]), "029 actual final packed network time"
        )
    OLD.context_check(metadata["context"])
    require(
        int(arrays["network_timestep"].item()) == metadata["network_timestep"], "029 saved actual network time differs"
    )
    if not teacher:
        offset = int(np.prod(arrays["packed_vision"].shape))
        require(
            metadata["action_offset"] == offset and metadata["vision_shape"] == list(arrays["packed_vision"].shape),
            "029 original flat solver layout differs",
        )
        require(
            arrays["sample"].shape == arrays["velocity"].shape == (offset + 5120,),
            "029 original flat sample/velocity dimensions differ",
        )
        require(
            arrays["sample"].dtype == np.float32 and metadata["original_dtypes"]["sample"] == "torch.float32",
            "029 sample is not original FP32 solver latent",
        )
        bits_equal(
            arrays["sample_action"], arrays["sample"][offset:].reshape(40, 128), "029 original sample action view"
        )
        bits_equal(
            arrays["velocity_action"],
            arrays["velocity"][offset:].reshape(40, 128),
            "029 actual returned velocity action view",
        )
        bits_equal(
            arrays["velocity_action"], arrays["pred_action"], "029 original actual single-conditional action velocity"
        )
        bits_equal(
            arrays["packed_action"], OLD.quantize_bfloat16(arrays["sample_action"]), "029 actual packed action cast"
        )
        bits_equal(
            arrays["packed_vision"],
            OLD.quantize_bfloat16(arrays["sample"][:offset].reshape(arrays["packed_vision"].shape)),
            "029 actual packed vision cast",
        )
        _, converted = mixed_convert(
            arrays["sample_action"],
            arrays["velocity_action"],
            np.asarray(metadata["sigma_actual"], np.float32),
            metadata["original_dtypes"]["velocity"],
        )
        bits_equal(arrays["converted_action"], converted, "029 actual scheduler converted action")
        require(
            metadata["conversion_sample_is_original_storage_view"] is True
            and metadata["conversion_returned_original_object"] is True
            and metadata["conversion_calls"] == 1,
            "029 observer changed the actual conversion delegation",
        )


def point_identity(metadata, planned, evidence, *, teacher=False):
    require(
        metadata["step_index"] == planned["step_index"]
        and metadata["sigma_actual"] == planned["sigma_actual"]
        and metadata["network_timestep"] == planned["network_timestep"],
        "029 selected point uses wrong actual sigma/step/network time",
    )
    for key in ("head_calls", "denoise_calls") if teacher else ("head_calls", "denoise_calls", "velocity_delegations"):
        require(metadata[key] == 1, "029 selected point repeated/missing original neural delegation")
    for key in ("rng_before", "rng_after"):
        OLD.rng_signature(metadata[key], evidence)


def collection_contract(actual, prepared):
    require(
        actual["schema_version"] == "s2a.free-trajectory-collection.v1"
        and actual["complete"] is actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == prepared_ref(prepared),
        "029 partial/rejected collection or wrong preparation",
    )
    require(
        actual["all_original_parameters_frozen"] is actual["original_parameters_require_grad_false"] is True
        and actual["wam_model_initializations"] == 1
        and actual["denoise_calls"] == 1824
        and len(actual["free_generations"]) == 48
        and len(actual["teacher_calls"]) == 144,
        "029 complete48/144/1824 budget or parameter freeze differs",
    )
    for key in ("fit_groups", "train_episode_reads", "physics_steps", "fresh_cases"):
        require(actual[key] == 0, "029 exceeded no-fit/no-physics diagnostic budget")
    require(actual["head_call_counts"] == dict(free=1680, teacher=144), "029 actual head counts differ")
    labels = [f"free:{i}" for i in range(48) for _ in range(35)] + [f"teacher:{i}" for i in range(144)]
    require(
        actual["denoise_ledger"] == [dict(index=i, label=label, completed=True) for i, label in enumerate(labels)],
        "029 actual1824 boundary ledger incomplete/repeated/reordered",
    )
    require(
        actual["runtime"]["checkpoint_path"] == prepared["checkpoint"] and actual["runtime"]["ema_weights"] is False,
        "029 actual checkpoint changed",
    )


def precision_admission(prepared, precision_path, verified_path, evidence):
    precision, verified = evidence.json(precision_path), evidence.json(verified_path)
    require(
        precision["complete"] is precision["passed"] is True
        and precision["prepared_ref"] == prepared_ref(prepared)
        and verified["schema_version"] == "s2a.free-trajectory-independent-verification.v1"
        and verified["passed"] is verified["precision_verified"] is verified["eligible_for_collection"] is True
        and verified["backend_gold_verified"] is True
        and verified["phase"] == "precision"
        and verified["prepared"] == prepared_ref(prepared)
        and verified["input_sha256"].get(str(Path(precision_path).resolve())) == sha256(precision_path)
        and verified["eligible_for_fit"] is verified["physics_authorized"] is False,
        "029 lacks actual precision independent admission",
    )
    STATE.public_lifecycle(precision, prepared, "precision", evidence)
    owner = evidence.json(Path(prepared["output_root"]) / "orchestration/precision/result.json")
    require(
        owner["complete"] is owner["passed"] is True
        and len(owner["owned_commands"]) == 2
        and owner["prepared_ref"] == prepared_ref(prepared)
        and owner["producer_result"] == reference(precision_path)
        and owner["automatic_retry"] is False,
        "029 prior precision owner is incomplete",
    )
    for i, command in enumerate(owner["owned_commands"]):
        STATE.owned_exit(command, evidence, gpu=i == 0)
    return reference(precision_path), reference(verified_path)


def collection_records(prepared, actual, transform, evidence):
    """Complete protocol only: deliberately no EP/PNG/future-label access."""
    complete = evidence.ref(actual["free_complete_ref"])
    require(
        complete["complete"] is complete["passed"] is True
        and complete["event_index"] == 0
        and complete["prepared_ref"] == prepared_ref(prepared)
        and complete["free_generations"] == actual["free_generations"]
        and complete["denoise_calls"] == 1680
        and complete["denoise_ledger"] == actual["denoise_ledger"][:1680]
        and complete["all_original_parameters_frozen"] is complete["original_parameters_require_grad_false"] is True
        and complete["teacher_calls"] == complete["future_source_opens"] == 0,
        "029 future source admission lacks complete immutable48raw manifest",
    )
    anchors = {a["name"]: a for a in prepared["anchors"]}
    free_points = {}
    for planned, recorded in zip(prepared["free_generations"], actual["free_generations"], strict=True):
        require(all(recorded[k] == v for k, v in planned.items()), "029 raw roster/cache source changed")
        anchor = anchors[planned["anchor_name"]]
        # Existing pure026 condition/initial checker expects its original step
        # schema; inspect029's extended trace separately below instead.
        response, condition = evidence.ref(recorded["response"]), evidence.ref(recorded["conditioning"])
        OLD.context_check(condition["context"])
        require(
            condition["inference_only_current_image"] is True
            and condition["model_runtime"]["checkpoint_path"] == prepared["checkpoint"]
            and condition["model_runtime"]["ema_weights"] is False,
            "029 raw actual current observation/weights differ",
        )
        caption = json.loads(condition["caption"])
        require(
            caption["task_title"] == caption["command"] == anchor["task"]
            and caption.get("task_plan", []) == []
            and "current_subtask" not in caption,
            "029 raw input is not actual task-only",
        )
        require(
            response["outputs"][0]["content"]["action128"]["conditioning_caption_sha256"]
            == hashlib.sha256(condition["caption"].encode()).hexdigest(),
            "029 response caption binding differs",
        )
        OLD.response_actions(response, planned["request"], prepared["checkpoint"], transform)
        prior = evidence.ref(planned["cache"]["response"])
        for key in ("action", "normalized_action"):
            first = response["outputs"][0]["content"]
            second = prior["outputs"][0]["content"]
            bits_equal(
                np.asarray(first["action128"][key] if key == "normalized_action" else first[key]),
                np.asarray(second["action128"][key] if key == "normalized_action" else second[key]),
                f"029 full128 raw replay {key}",
            )
        initial_meta, initial = OLD.bundle(recorded["initial"], evidence)
        prior_meta, prior_initial = OLD.bundle(planned["cache"]["initial"], evidence)
        current_signature = OLD.initial_signature(initial_meta, initial)
        prior_signature = OLD.initial_signature(prior_meta, prior_initial)
        FLOW.require_same_initial(current_signature, prior_signature)
        require(current_signature["caption"] == condition["caption"], "029 actual initial/condition text differs")
        trace_contract(evidence.ref(recorded["actual_sigma_trace"]), planned, prepared)
        require(len(recorded["points"]) == 3, "029 raw trajectory omitted one selected point")
        for ref, node in zip(recorded["points"], prepared["precision_nodes"][1:4], strict=True):
            metadata, arrays = OLD.bundle(ref, evidence)
            point_identity(metadata, node, evidence)
            require(
                all(metadata[k] == v for k, v in planned.items())
                and metadata["conditioning_ref"] == recorded["conditioning"],
                "029 actual point lost its request/condition identity",
            )
            tensor_contract(metadata, arrays)
            require(
                OLD.rng_signature(metadata["rng_before"], evidence)
                == OLD.rng_signature(metadata["rng_after"], evidence),
                "029 raw observer/model unexpectedly consumed RNG",
            )
            free_points[(planned["generation_index"], node["step_index"])] = ref
    require(
        len(free_points) == 144 and len(actual["teacher_source_opens"]) == 24,
        "029 free144/source24 complete grid differs",
    )
    source_events = {}
    for i, (anchor, ref) in enumerate(zip(prepared["anchors"], actual["teacher_source_opens"], strict=True)):
        event = evidence.ref(ref)
        require(
            event["event_index"] == i + 1
            and event["free_complete_ref"] == actual["free_complete_ref"]
            and event["anchor_name"] == anchor["name"]
            and event["arrays_ref"] == dict(path=anchor["arrays_path"], sha256=anchor["arrays_sha256"])
            and event["denoise_calls_before"] == 1680 + i * 6
            and event["first_teacher_call_index"] == i * 6
            and event["future_source_opened_after_free_complete"] is True,
            "029 future-source open event precedes/bypasses complete48raw admission",
        )
        source_events[anchor["name"]] = ref
    sources = {}
    for planned, recorded in zip(prepared["teacher_calls"], actual["teacher_calls"], strict=True):
        require(all(recorded[k] == v for k, v in planned.items()), "029 teacher144 actual-time grid changed")
        metadata, arrays = OLD.bundle(recorded["bundle"], evidence)
        point_identity(metadata, planned, evidence, teacher=True)
        require(
            all(metadata[k] == v for k, v in planned.items())
            and metadata["free_complete_ref"] == actual["free_complete_ref"]
            and metadata["free_point_ref"] == free_points[(planned["free_generation_index"], planned["step_index"])],
            "029 teacher lost paired actual raw point identity",
        )
        tensor_contract(metadata, arrays, teacher=True)
        require(
            metadata["packed_network_times_exact"] is True
            and metadata["supervision"] == "epsilon-x0; distinct from free point pull velocity",
            "029 teacher mislabels its diagnostic supervision",
        )
        free_metadata, free_arrays = OLD.bundle(metadata["free_point_ref"], evidence)
        for key in (
            "time_action",
            "time_vision",
            "condition_action",
            "condition_vision",
            "projection_ids",
            "network_timestep",
        ):
            bits_equal(arrays[key], free_arrays[key], f"029 teacher/free actual packed {key}")
            require(
                metadata["original_dtypes"][key] == free_metadata["original_dtypes"][key],
                "029 paired actual packed dtype differs",
            )
        source_meta = evidence.ref(metadata["source_ref"])
        require(
            source_meta["anchor"] == anchors[planned["anchor_name"]]
            and source_meta["source_open_ref"] == source_events[planned["anchor_name"]]
            and source_meta["free_complete_ref"] == actual["free_complete_ref"],
            "029 teacher source lost causal/source admission",
        )
        condition = evidence.ref(metadata["conditioning_ref"])
        free_condition = evidence.ref(free_metadata["conditioning_ref"])
        require(
            condition["inference_only_current_image"] is False
            and condition["source_ref"] == metadata["source_ref"]
            and condition["caption"] == free_condition["caption"]
            and condition["context"] == free_condition["context"],
            "029 true-future teacher was mislabeled as online or changed task",
        )
        previous = sources.setdefault(planned["anchor_name"], metadata["source_ref"])
        require(previous == metadata["source_ref"], "029 same anchor used different source windows")
    return sources


def normalized_actions(physical, transform):
    result = np.asarray(physical, np.float32).copy()
    mask = np.asarray(transform["normalization_mask"], bool)
    scale, bias = (np.asarray(transform[k], np.float32) for k in ("scale", "bias"))
    require(mask.shape == scale.shape == bias.shape == (128,), "029 original normalization geometry differs")
    result[:, mask] = np.clip(result[:, mask] * scale[mask] + bias[mask], transform["clip_min"], transform["clip_max"])
    result[:, ~OLD.ACTIVE] = 0
    return result


def verified_labels(prepared, sources, transform, evidence):
    """Called only after all1824 protocol checks and actual release succeed."""
    labels = {}
    require(len(sources) == 24, "029 source labels attempted before complete24anchor admission")
    canvas_file = str(REPO / "src/s2a/libero_simulation.py")
    for episode in prepared["config"]["validation_episodes"]:
        anchors = [a for a in prepared["anchors"] if a["episode"] == episode]
        require(
            len(anchors) == 4 and all(a["split"] == "validation" for a in anchors),
            "029 attempted a training/fresh episode",
        )
        source = anchors[0]
        with np.load(evidence.check(source["arrays_path"], source["arrays_sha256"]), allow_pickle=False) as archive:
            # Slice each complete camera only once per EP. This audit never
            # imports the training loader or a producer's label mathematics.
            actions = numeric(archive["actions"], "original actions").copy()
            action_mask, action_valid = archive["action_mask"].copy(), archive["action_valid"].copy()
            cameras = {role: archive[role].copy() for role in ("auxiliary_0", "left_wrist")}
        for anchor in anchors:
            require(
                anchor["arrays_path"] == source["arrays_path"] and anchor["arrays_sha256"] == source["arrays_sha256"],
                "029 EP source changed between anchors",
            )
            frame = anchor["anchor_frame"]
            physical = actions[frame : frame + 40]
            normalized = normalized_actions(physical, transform)
            metadata, saved = OLD.bundle(sources[anchor["name"]], evidence)
            require(
                metadata["anchor"] == anchor and metadata["transform"] == transform,
                "029 actual teacher source/normalization differs",
            )
            OLD.context_check(metadata["context"])
            bits_equal(saved["physical_action"], physical, "029 original40 physical actions")
            bits_equal(saved["normalized_action"], normalized, "029 independently normalized true future")
            bits_equal(saved["action_mask"], action_mask[frame : frame + 40], "029 original action mask")
            bits_equal(saved["action_valid"], action_valid[frame : frame + 40], "029 original temporal validity")
            exact(saved["valid"], np.broadcast_to(OLD.ACTIVE, (40, 128)), "029 future active10 validity")
            require(
                np.all(saved["action_valid"]) and np.array_equal(saved["action_mask"], saved["valid"]),
                "029 source includes invalid future endpoints",
            )
            for role in cameras:
                bits_equal(
                    saved[f"raw_camera_{role}"], cameras[role][frame : frame + 41], "029 original true41 camera bytes"
                )
                require(
                    saved[f"raw_camera_{role}"].shape == (41, 256, 256, 3)
                    and saved[f"raw_camera_{role}"].dtype == np.uint8,
                    "029 true camera geometry differs",
                )
            require(
                saved["canvas"].shape == (41, 960, 640, 3)
                and saved["canvas"].dtype == np.uint8
                and hashlib.sha256(saved["canvas"].tobytes()).hexdigest() == metadata["true_future_rgb_sha256"]
                and metadata["canvas_source_sha256"] == prepared["source_sha256"][canvas_file],
                "029 shared frozen cover-crop canvas contract/source differs",
            )
            labels[anchor["name"]] = normalized
    return labels


def teacher_flow(arrays, normalized, sigma):
    require(arrays["valid"].dtype == np.bool_, "029 actual teacher valid mask dtype differs")
    exact(arrays["valid"], np.broadcast_to(OLD.ACTIVE, (40, 128)), "029 actual teacher full active mask")
    bits_equal(arrays["x0_normalized"], normalized, "029 teacher exact original normalized action")
    require(
        arrays["sigma_action"].shape == (40, 1) and arrays["sigma_action"].dtype == np.float32,
        "029 actual teacher sigma geometry/dtype differs",
    )
    bits_equal(arrays["sigma_action"], np.full((40, 1), sigma, np.float32), "029 teacher actual FP32 sigma")
    for suffix, clean_key, noise_key, sigma_key, target_key, packed_key in (
        ("action", "x0_normalized", "epsilon", "sigma_action", "target", "packed_action"),
        ("vision", "x0_vision", "epsilon_vision", "sigma_vision", "target_vision", "packed_vision"),
    ):
        clean, epsilon, actual_sigma = (arrays[k].astype(np.float32) for k in (clean_key, noise_key, sigma_key))
        require(clean.shape == epsilon.shape == arrays[packed_key].shape, "029 true/noise/packed flow geometry differs")
        if suffix == "vision":
            exact(
                actual_sigma,
                np.float32(sigma) * (1 - arrays["condition_vision"]),
                "029 vision actual conditioned sigma",
            )
        target = epsilon - clean
        xt = OLD.quantize_bfloat16(epsilon * actual_sigma + clean * (np.float32(1) - actual_sigma))
        if suffix == "action":
            target[:, 10:], xt[:, 10:] = 0, 0
        # Frozen024 flow-only tolerances are retained. Exact replay, timestamps,
        # actual scheduler conversion and precision-bit checks get no slack.
        for name, gold in ((target_key, target), (packed_key, xt)):
            require(
                np.allclose(arrays[name], gold, rtol=1e-6, atol=1e-7),
                f"029 independent epsilon-x0/{suffix} interpolation differs",
            )
    bits_equal(arrays["velocity_action"], arrays["pred_action"], "029 actual teacher unconditioned action velocity")


def aggregate_diagnostics(rows):
    require(len(rows) == 144 and len({r["episode"] for r in rows}) == 6, "029 complete six-EP diagnostic rows differ")
    keys = set(rows[0]["metrics"])
    require(all(set(r["metrics"]) == keys for r in rows), "029 diagnostic metric columns differ")

    def group(items):
        episodes = sorted({row["episode"] for row in items})
        by_episode = [[r for r in items if r["episode"] == ep] for ep in episodes]
        return dict(
            points=len(items),
            episodes=len(episodes),
            endpoints=len(items) * 40,
            aggregation="mean within each EP, then equal mean across EPs; point_max_abs denotes mean point maxima",
            metrics={
                key: float(np.mean([np.mean([r["metrics"][key] for r in members]) for members in by_episode]))
                for key in sorted(keys)
            },
        )

    return dict(
        pooled=group(rows),
        **{
            f"by_{key}": {
                str(value): group([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})
            }
            for key in ("sigma", "seed", "episode", "task")
        },
    )


def verify_collection(prepared, transform, precision_path, verified_path, path, evidence):
    actual = evidence.json(path)
    precision_ref, verified_ref = precision_admission(prepared, precision_path, verified_path, evidence)
    require(
        actual["precision_ref"] == precision_ref and actual["precision_verification_ref"] == verified_ref,
        "029 collection used different precision admission",
    )
    collection_contract(actual, prepared)
    identity = STATE.public_lifecycle(actual, prepared, "collection", evidence)
    sources = collection_records(prepared, actual, transform, evidence)
    # This is the only path that opens original future labels. All complete
    # protocol, replay, source and actual exit/resource gates precede it.
    labels = verified_labels(prepared, sources, transform, evidence)
    model = STATE.read_model(prepared["adapted_model_ref"], evidence)
    anchors = {a["name"]: a for a in prepared["anchors"]}
    rows = []
    for call in actual["teacher_calls"]:
        metadata, teacher = OLD.bundle(call["bundle"], evidence)
        free_meta, free = OLD.bundle(metadata["free_point_ref"], evidence)
        teacher_flow(teacher, labels[call["anchor_name"]], call["sigma_actual"])
        anchor = anchors[call["anchor_name"]]
        rows.append(
            dict(
                call_index=call["call_index"],
                anchor_name=anchor["name"],
                episode=anchor["episode"],
                task=anchor["task_index"],
                seed=call["sampling_seed"],
                sigma=call["sigma_actual"],
                step_index=call["step_index"],
                metrics=point_diagnostics(
                    free,
                    teacher,
                    labels[anchor["name"]],
                    model,
                    call["sigma_actual"],
                    free_meta["original_dtypes"]["velocity"],
                ),
            )
        )
    return dict(
        collection_verified=True,
        complete_free_replays=48,
        full128_raw_and_normalized_exact=True,
        free_saved_points=144,
        teacher_saved_points=144,
        actual_denoise_calls=1824,
        actual_owned_model_exit=identity,
        original028_state_gate=False,
        original027_eligible_for_sampling=False,
        diagnostics=aggregate_diagnostics(rows),
        diagnostic_rows=rows,
        shared_scope="Frozen025/026 pure independent normalization/domain/initial/schedule helpers and027 no-fit NumPy readout; frozen Torch canvas implementation is source-bound, not reimplemented",
        interpretation="Teacher epsilon-x0 and chosen-demonstration free pull are distinct targets. Offline027 uses the observed raw path; it is not an actual adapted trajectory, causal root-cause intervention or task-success evaluation.",
        task_success_proven=False,
    )


def phase_arguments(args):
    present = (args.precision is not None, args.precision_verification is not None, args.collection is not None)
    require(
        present
        == dict(prepare=(False, False, False), precision=(True, False, False), collection=(True, True, True))[
            args.phase
        ],
        "029 evidence arguments do not match explicit phase",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "precision", "collection"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--precision", type=Path)
    parser.add_argument("--precision-verification", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "029 verifier requires emptyCUDA/LD and seed42",
    )
    phase_arguments(args)
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "029 receipt must be new under outputs; refuse overwrite",
    )
    evidence = OLD.Evidence()
    receipt = dict(
        schema_version="s2a.free-trajectory-independent-verification.v1",
        phase=args.phase,
        passed=False,
        eligible_for_fit=False,
        physics_authorized=False,
        new_model_calls=0,
        new_candidate_fits=0,
        independent_candidate_reconstructions=0,
        new_physics_steps=0,
    )
    error = None
    try:
        prepared, transform = verify_preparation(args.prepared, evidence)
        receipt["prepared"] = reference(args.prepared)
        if args.phase == "prepare":
            receipt.update(
                preparation_verified=True, original028_state_gate=False, original027_eligible_for_sampling=False
            )
        elif args.phase == "precision":
            receipt.update(verify_precision(prepared, args.precision, evidence))
        else:
            receipt.update(
                verify_collection(
                    prepared, transform, args.precision, args.precision_verification, args.collection, evidence
                )
            )
        evidence.recheck()
        receipt["passed"] = True
    except Exception as caught:
        error = caught
        receipt["error"] = dict(type=type(caught).__name__, message=str(caught))
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=reference(output))))
    if error is not None:
        raise SystemExit(f"029 independent audit rejected: {error}")


def statistic(values):
    values = numeric(values, "diagnostic values").astype(np.float64)
    return dict(
        mse=float(np.mean(values**2)), mae=float(np.mean(np.abs(values))), point_max_abs=float(np.max(np.abs(values)))
    )


def point_diagnostics(free, teacher, x0, model, sigma, velocity_dtype="torch.bfloat16"):
    """No fit: actual raw path plus frozen027 offline pointwise counterfactual."""
    sigma = float(np.float32(sigma))
    require(sigma > 0 and x0.shape == (40, 128), "029 pointwise target geometry/positive actual sigma differs")
    f = {k: np.asarray(v).astype(np.float64) for k, v in free.items()}
    t = {k: np.asarray(v).astype(np.float64) for k, v in teacher.items()}
    metrics = {}

    def add(name, values):
        metrics.update({f"{name}.{key}": value for key, value in statistic(values).items()})

    for branch, arrays in (("free", f), ("teacher", t)):
        centered = (arrays["hidden"] - model["center"]) / float(model["scale"])
        metrics[f"hidden.{branch}.mean_row_norm"] = float(np.mean(np.linalg.norm(centered, axis=1)))
        metrics[f"hidden.{branch}.point_mean_shift_norm"] = float(np.linalg.norm(centered.mean(axis=0)))
    add("hidden.paired_scaled_difference", (f["hidden"] - t["hidden"]) / float(model["scale"]))
    for kind in ("vision", "action"):
        add(f"packed.{kind}.paired_difference", f[f"packed_{kind}"] - t[f"packed_{kind}"])
    add("packed.vision.current_condition_difference", f["packed_vision"][:, :, :1] - t["packed_vision"][:, :, :1])
    q = f["packed_action"] - f["sample_action"]
    v_pull = (f["sample_action"] - x0.astype(np.float64)) / sigma
    variants = {}
    for branch, arrays, target in (("free", free, v_pull), ("teacher", teacher, t["target"])):
        raw = arrays["velocity_action"] if branch == "free" else arrays["pred_action"]
        adapt = SCALED.quantized_velocity(
            arrays["hidden"], arrays["base"], sigma, model["weights"], model["center"], model["scale"]
        )
        adapt = adapt.copy()
        adapt[:, 10:] = 0  # Match the frozen post-head full-domain mask.
        variants[branch] = (raw.astype(np.float64), adapt.astype(np.float64), target)
    components = dict(active10=slice(0, 10), position=slice(0, 3), rotation6d=slice(3, 9), opening=slice(9, 10))
    components.update({f"slot{i}": slice(i, i + 1) for i in range(10)})
    for component, columns in components.items():
        add(f"quantization.{component}.q", q[:, columns])
        add(f"quantization.{component}.q_over_sigma", q[:, columns] / sigma)
        for branch, (raw, adapt, target) in variants.items():
            add(f"velocity.{branch}.{component}.raw_error", (raw - target)[:, columns])
            add(f"velocity.{branch}.{component}.offline027_error", (adapt - target)[:, columns])
        raw, adapt, _ = variants["free"]
        ideal_raw = f["sample_action"] - sigma * raw
        ideal_adapt = f["sample_action"] - sigma * adapt
        _, actual_adapt = mixed_convert(
            free["sample_action"], adapt.astype(np.float32), np.asarray(sigma, np.float32), velocity_dtype
        )
        actual_raw = f["converted_action"]
        add(f"clean.{component}.raw_actual_error", (actual_raw - x0)[:, columns])
        add(f"clean.{component}.offline027_actual_error", (actual_adapt.astype(np.float64) - x0)[:, columns])
        add(f"clean.{component}.raw_actual_minus_ideal", (actual_raw - ideal_raw)[:, columns])
        add(
            f"clean.{component}.offline027_actual_minus_ideal",
            (actual_adapt.astype(np.float64) - ideal_adapt)[:, columns],
        )
        for arithmetic, before, after in (
            ("ideal", ideal_raw, ideal_adapt),
            ("actual", actual_raw, actual_adapt.astype(np.float64)),
        ):
            error, delta = (before - x0)[:, columns], (after - before)[:, columns]
            cross, square = float(np.mean(2 * error * delta)), float(np.mean(delta**2))
            change = float(np.mean((error + delta) ** 2) - np.mean(error**2))
            require(
                np.isclose(change, cross + square, rtol=2e-12, atol=2e-12),
                "029 clean error decomposition identity failed",
            )
            prefix = f"decomposition.{component}.{arithmetic}"
            metrics.update(
                {f"{prefix}.two_e_delta": cross, f"{prefix}.delta_squared": square, f"{prefix}.mse_change": change}
            )
    return metrics


if __name__ == "__main__":
    main()
