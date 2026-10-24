"""Saved-online protocol checks, with no model loading or GPU execution."""

import copy
import hashlib
import importlib
import json

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("verify_wam_clean_bridge")


def head_parameters(residual):
    coefficients = np.zeros((1291, 10), np.float64)
    coefficients[-1] = residual
    return dict(
        schema_version="s2a.clean-residual-readout.v1",
        coefficients=coefficients.tolist(),
        center=np.zeros(1290).tolist(),
        scale=1.0,
        ridge=0.1,
    )


def point_fixture(audit, variant="nonzero", dtype="torch.bfloat16"):
    offset, sigma = 8, float(np.float32(0.5081728))
    sample = np.zeros((40, 128), np.float32)
    sample[:, :10] = 1.0003 + np.arange(40, dtype=np.float32)[:, None] * 0.0001
    raw = np.zeros((40, 128), np.float32)
    raw[:, :10] = audit.cast_deployed(np.full((40, 10), 0.001), dtype)
    raw = raw.astype(np.float32 if dtype != "torch.float16" else np.float16)
    flat_sample = np.r_[np.arange(offset, dtype=np.float32), sample.reshape(-1)]
    flat_raw = np.r_[np.arange(offset, dtype=raw.dtype), raw.reshape(-1)]
    c0 = sample.astype(np.float64) - sigma * raw.astype(np.float64)
    residual = np.full((40, 10), 1e-4 if variant == "nonzero" else 0, np.float64)
    delta = -residual / sigma
    deployed = raw.copy()
    if variant == "nonzero":
        deployed[:, :10] = audit.cast_deployed(raw[:, :10].astype(np.float64) + delta, dtype)
    flat_deployed = flat_raw.copy()
    flat_deployed[offset:] = deployed.reshape(-1)
    head = np.ones((40, 128), np.float32)
    head[:, :10] = raw[:, :10]
    arrays = dict(
        sample_fp32_flat=flat_sample,
        sample_full128_fp32=sample,
        raw_velocity_flat=flat_raw,
        raw_velocity_full128=raw,
        deployed_velocity_flat=flat_deployed,
        deployed_velocity_full128=deployed,
        c0_full128=c0,
        hidden=np.zeros((40, 1280), np.float32),
        projection_ids=np.zeros(40, np.int64),
        packed_action=audit.bf16(sample),
        packed_vision=np.ones((1, 1, 2, 2, 2), np.float32),
        time_action=np.full(40, 509, np.float32),
        time_vision=np.full(2, 509, np.float32),
        condition_action=np.zeros((40, 1), np.float32),
        condition_vision=np.zeros((2, 1, 1), np.float32),
        head_base_full128=head,
        clean_residual_active10=residual,
        velocity_delta_active10=delta,
    )
    dtypes = {
        key: {
            np.dtype("float32"): "torch.float32",
            np.dtype("float64"): "torch.float64",
            np.dtype("int64"): "torch.int64",
            np.dtype("float16"): "torch.float16",
        }[value.dtype]
        for key, value in arrays.items()
    }
    for key in ("raw_velocity_flat", "raw_velocity_full128", "deployed_velocity_flat", "deployed_velocity_full128"):
        dtypes[key] = dtype
    dtypes.update(packed_action="torch.bfloat16", packed_vision="torch.bfloat16", hidden="torch.bfloat16")
    metadata = dict(
        request_name=variant,
        noise_seed=42,
        purpose="evaluation",
        step_index=0,
        sigma_actual=sigma,
        network_timestep=509,
        action_offset=offset,
        action_dim_mask=audit.ACTIVE.tolist(),
        feature_terms=[],
        original_dtypes=dtypes,
        clean_semantics="ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0",
    )
    generation = dict(name=variant, variant=variant, sampling_seed=42, case_id="case0")
    step = dict(
        step_index=0, solver_sigma=sigma, solver_timestep=509, returned_original_velocity_object=variant != "nonzero"
    )
    parameters = head_parameters(residual[0])
    head = dict(coefficients=np.asarray(parameters["coefficients"]), center=np.asarray(parameters["center"]), scale=1.0)
    return metadata, arrays, generation, step, head


@pytest.mark.parametrize("variant", ["raw", "zero", "nonzero"])
@pytest.mark.parametrize("dtype", ["torch.float32", "torch.bfloat16", "torch.float16"])
def test_actual_fp32_latent_and_clean_velocity_arithmetic(audit, variant, dtype):
    metadata, arrays, generation, step, head = point_fixture(audit, variant, dtype)
    result = audit.verify_point(metadata, arrays, generation=generation, step=step, head=head)
    assert (
        result["nonzero_deployed_active_values"] > 0
        if variant == "nonzero"
        else result["nonzero_deployed_active_values"] == 0
    )
    # Conditioning may replace a vision token before packing. Its provenance is
    # observed and paired, rather than incorrectly inferred from the flat latent.
    assert not np.array_equal(arrays["packed_vision"].reshape(-1), arrays["sample_fp32_flat"][:8])
    assert not np.array_equal(arrays["sample_full128_fp32"], arrays["packed_action"])


@pytest.mark.parametrize(
    "error",
    [
        "packed_as_sample",
        "sigma_from_time",
        "future",
        "projection",
        "inactive",
        "vision",
        "c0",
        "delta_sign",
        "deployed",
        "unbound_action",
        "dtype",
        "missing",
        "identity",
    ],
)
def test_reject_unsupported_actual_sample_or_changed_deployment(audit, error):
    metadata, arrays, generation, step, head = point_fixture(audit)
    if error == "packed_as_sample":
        arrays["sample_full128_fp32"] = arrays["packed_action"].copy()
    elif error == "sigma_from_time":
        metadata["sigma_actual"] = metadata["network_timestep"] / 1000
    elif error == "future":
        metadata["purpose"] = "training"
    elif error == "projection":
        arrays["projection_ids"][0] = 1
    elif error == "inactive":
        arrays["deployed_velocity_full128"][0, 10] = 1
    elif error == "vision":
        arrays["deployed_velocity_flat"][0] = 1
    elif error == "c0":
        arrays["c0_full128"][0, 0] += 1e-10
    elif error == "delta_sign":
        arrays["velocity_delta_active10"] *= -1
    elif error == "deployed":
        arrays["deployed_velocity_full128"][0, 0] += 0.001
        arrays["deployed_velocity_flat"][8:] = arrays["deployed_velocity_full128"].reshape(-1)
    elif error == "unbound_action":
        arrays["raw_velocity_full128"] = arrays["raw_velocity_full128"].copy()
        arrays["raw_velocity_full128"][0, 0] += 0.001
    elif error == "dtype":
        arrays["sample_fp32_flat"] = arrays["sample_fp32_flat"].astype(np.float64)
    elif error == "missing":
        arrays.pop("time_vision")
    else:
        step["returned_original_velocity_object"] = True
    with pytest.raises(ValueError):
        audit.verify_point(metadata, arrays, generation=generation, step=step, head=head)


def test_bf16_rounding_avoids_double_rounding_and_preserves_signed_zero(audit):
    below = 1 + 2**-8 - 2**-30
    above = 1 + 2**-8 + 2**-30
    values = np.array([below, 1 + 2**-8, above, 0.0, -0.0], np.float64)
    result = audit.bf16(values)
    np.testing.assert_array_equal(result[:3], np.array([1, 1, 1 + 2**-7], np.float32))
    assert np.signbit(result[-1]) and not np.signbit(result[-2])
    assert audit.bf16(np.array(above, np.float32)).item() == 1


def test_qualification_constant_heads_cannot_hide_a_fit(audit, monkeypatch):
    residual = np.arange(1, 11) * 1e-5
    parameters = head_parameters(residual)
    monkeypatch.setattr(audit, "read", lambda _: parameters)
    assert audit.read_head("ref", residual=residual)["coefficients"].shape == (1291, 10)
    parameters["coefficients"][0][0] = 1e-12
    with pytest.raises(ValueError, match="constant-bias"):
        audit.read_head("ref", residual=residual)
    parameters["coefficients"][0][0] = 0
    parameters["center"][0] = 1
    with pytest.raises(ValueError, match="center"):
        audit.read_head("ref", residual=residual)


def trace_fixture():
    sigmas = np.r_[np.linspace(1, 0.1, 35, dtype=np.float32), np.float32(0)].tolist()
    timesteps = list(range(999, 964, -1))
    generation = dict(name="raw", sampling_seed=42)
    steps = [
        dict(
            step_index=i,
            solver_step_index_before=None if i == 0 else i,
            solver_sigma=sigmas[i],
            solver_timestep=timesteps[i],
            velocity_delegations=1,
            denoise_calls=1,
            head_calls=1,
            conversion_calls=1,
            conversion_sample_is_original_storage_view=True,
            conversion_velocity_is_deployed_storage_view=True,
            conversion_returned_original_object=True,
        )
        for i in range(35)
    ]
    trace = dict(
        complete=True,
        name="raw",
        seed=42,
        purpose="evaluation",
        sampler_calls=1,
        parameters_before={"version": 0},
        parameters_after={"version": 0},
        actual_sampler_arguments=dict(num_steps=35, shift=5.0, seed=[42]),
        solver_sigma_dtype="torch.float32",
        terminal_sigma=0,
        terminal_neural_calls=0,
        sigma_values=sigmas,
        scheduler_timesteps=timesteps,
        steps=steps,
    )
    return trace, generation


def test_all35_observed_solver_steps_are_required(audit):
    trace, generation = trace_fixture()
    assert len(audit.verify_trace(trace, generation)) == 35
    trace["steps"].pop(19)
    with pytest.raises(ValueError, match="denominator"):
        audit.verify_trace(trace, generation)


@pytest.mark.parametrize(
    "error", ["conversion", "storage", "repeated_head", "terminal", "timestep", "parameters", "sigma"]
)
def test_trace_binds_deployed_velocity_to_real_scheduler_conversion(audit, error):
    trace, generation = trace_fixture()
    if error == "conversion":
        trace["steps"][3]["conversion_calls"] = 0
    elif error == "storage":
        trace["steps"][3]["conversion_velocity_is_deployed_storage_view"] = False
    elif error == "repeated_head":
        trace["steps"][3]["head_calls"] = 2
    elif error == "terminal":
        trace["terminal_neural_calls"] = 1
    elif error == "timestep":
        trace["steps"][3]["solver_timestep"] += 1
    elif error == "parameters":
        trace["parameters_after"]["version"] = 1
    else:
        trace["steps"][3]["solver_sigma"] = trace["steps"][3]["solver_timestep"] / 1000
    with pytest.raises(ValueError):
        audit.verify_trace(trace, generation)


def response(value=0):
    array = np.zeros((40, 128), np.float64)
    array[:, :10] = value
    return dict(outputs=[dict(content=dict(action=array.tolist(), action128=dict(normalized_action=array.tolist())))])


def pair_fixture(audit):
    raw_points = [point_fixture(audit, "raw")[:2] for _ in range(35)]
    zero_points = copy.deepcopy(raw_points)
    raw = dict(entry=dict(case_id="case", sampling_seed=42), response=response(), points=raw_points)
    zero = dict(entry=dict(case_id="case", sampling_seed=42), response=response(), points=zero_points)
    return raw, zero


def test_pair_needs_all35_actual_tensors_and_both_output_representations(audit):
    raw, zero = pair_fixture(audit)
    assert audit.verify_pair(raw, zero)["online_points_exact"] == 35
    zero["points"][22][1]["hidden"][0, 0] = 1
    with pytest.raises(ValueError, match="hidden"):
        audit.verify_pair(raw, zero)
    raw, zero = pair_fixture(audit)
    zero["response"]["outputs"][0]["content"]["action128"]["normalized_action"][39][9] = 1e-15
    with pytest.raises(ValueError, match="final raw/normalized"):
        audit.verify_pair(raw, zero)


def test_historical_output_mismatch_is_a_false_qualification_not_a_fake_match(audit):
    assert audit.output_exact(response(), response()) is True
    assert audit.output_exact(response(), response(1e-15)) is False


def test_current_conditioning_uses_task_text_and_full_domain(audit):
    caption = json.dumps(dict(task_title="fixed task", command="fixed task"))
    value = dict(
        caption=caption,
        caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),
        context=dict(
            domain=audit.LIBERO_DOMAIN, normalization_id=31, projection_id=0, action_dim_mask=audit.ACTIVE.tolist()
        ),
        inference_only_current_image=True,
    )
    audit.verify_conditioning(value, dict(prompt="fixed task"))
    value["context"]["normalization_id"] = 30
    with pytest.raises(ValueError, match="full LIBERO"):
        audit.verify_conditioning(value, dict(prompt="fixed task"))


def test_request_can_only_relocate_identical_current_image(audit, monkeypatch, tmp_path):
    image = tmp_path / "current.png"
    image.write_bytes(b"unchanged-image")
    action = dict(
        action_dim_mask=audit.ACTIVE.tolist(),
        action_domain=audit.LIBERO_DOMAIN,
        action_fps=20,
        camera_layout=[],
        return_video=False,
        selected_roles=["auxiliary_0", "left_wrist"],
        task_title="fixed task",
    )
    historical = dict(
        model_mode="wam",
        action_chunk_size=40,
        num_frames=41,
        num_steps=35,
        guidance=1.0,
        shift=5.0,
        fps=20,
        seed=42,
        prompt="fixed task",
        vision_path="old/current.png",
        extra=dict(action128=action),
    )
    case = dict(
        request=dict(historical, vision_path=str(image)),
        historical_entry=dict(request=copy.deepcopy(historical)),
        image_ref=dict(path="old/current.png", sha256="same"),
    )
    monkeypatch.setattr(audit, "reference", lambda _: dict(path="old/current.png", sha256="same"))
    monkeypatch.setattr(audit, "resolve_path", lambda _: image)
    audit.verify_request(case)
    case["request"]["extra"]["action128"]["desired_future_state"] = [[1]]
    with pytest.raises(ValueError, match="beyond|privileged"):
        audit.verify_request(case)


def test_initial_capture_binds_actual_fp32_sample_without_future_actions(audit):
    metadata, points, generation, _, _ = point_fixture(audit)
    generation.update(index=0)
    caption = "actual-current-task"
    initial_meta = dict(
        name=generation["name"],
        sampling_seed=42,
        variant="nonzero",
        generation_index=0,
        inference_only_current_image=True,
        caption=caption,
    )
    arrays = dict(
        initial_noise=points["sample_fp32_flat"].copy(),
        condition_mask=np.zeros_like(points["sample_fp32_flat"]),
        first_condition_latent=np.zeros((1, 48, 1, 60, 40), np.float32),
    )
    audit.verify_initial(initial_meta, arrays, generation, dict(caption=caption), points)
    arrays["condition_mask"][-1] = 1
    with pytest.raises(ValueError, match="privileged"):
        audit.verify_initial(initial_meta, arrays, generation, dict(caption=caption), points)
    arrays["condition_mask"][-1] = 0
    arrays["initial_noise"][0] += 1
    with pytest.raises(ValueError, match="first actual sampler latent"):
        audit.verify_initial(initial_meta, arrays, generation, dict(caption=caption), points)


def test_all_actual_cosmos_imports_must_stay_in_frozen_source_root(audit, monkeypatch, tmp_path):
    root = tmp_path / "source/cosmos"
    path = root / "cosmos_framework/model.py"
    ref = dict(path=str(path), sha256="frozen")
    monkeypatch.setattr(audit, "reference", lambda _: ref)
    record = dict(file=str(path), sha256="frozen")
    sources = {"cosmos_framework.model": record}
    actual = dict(
        cosmos_imported_sources_after_load=copy.deepcopy(sources), cosmos_imported_sources_final=copy.deepcopy(sources)
    )
    prepared = dict(config=dict(cosmos_source_root=str(root)), source_sha256={str(path): "frozen"})
    audit.verify_imports(actual, prepared)
    outside = tmp_path / "borrowed/cosmos_framework/extra.py"
    actual["cosmos_imported_sources_final"]["cosmos_framework.extra"] = dict(file=str(outside), sha256="frozen")
    with pytest.raises(ValueError, match="escaped"):
        audit.verify_imports(actual, prepared)
