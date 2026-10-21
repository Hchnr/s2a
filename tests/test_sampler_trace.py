"""CPU contract tests exercise real observer boundaries, not model quality."""

import hashlib
import inspect
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s2a.sampler_trace import (  # noqa: E402
    SamplerTrace,
    file_sha,
    observation_guard,
    rng_identity,
    tensor_snapshot,
    write_new_json,
)


class FakeScheduler:
    def __init__(self, steps):
        self.sigmas = torch.linspace(1, 0, steps + 1)
        self.timesteps = torch.arange(steps, 0, -1, dtype=torch.int64) * 3
        self.config = dict(solver_order=2)
        self.step_index = None


class FakeSampler(torch.nn.Module):
    @torch.no_grad()
    def forward(self, velocity_fn, noise, num_steps=35, shift=5.0, seed=None):
        sample_scheduler = [FakeScheduler(num_steps)]
        latent = noise
        for index, timestep in enumerate(sample_scheduler[0].timesteps):
            velocity = velocity_fn(latent, timestep.reshape(1, 1))
            # New tensors, just like a solver: never mutate input/velocity.
            latent = [x - 0.001 * v for x, v in zip(latent, velocity, strict=True)]
            sample_scheduler[0].step_index = index + 1
        self.returned = latent
        return latent


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.5))
        self.sampler = FakeSampler()
        self.seen_noise = []
        self.generated_velocities = []

    def generate_samples_from_batch(self, steps=35, seed=42):
        # These remain caller locals rather than closure members, just as in the
        # actual generator. Their values qualify derived, not hooked, net NFE.
        guidance, velocity_postprocess, velocity_postprocess_builder = 1.0, None, None  # noqa: F841
        _dp_shard_group, _extra_num_steps, scheduler_type = None, 0, "unipc"  # noqa: F841
        reuse_pack_templates, reuse_text_kv = True, False  # noqa: F841
        gen_data_clean = types.SimpleNamespace(
            num_vision_items_per_sample=None,
            x0_tokens_vision=[torch.arange(24).reshape(1, 2, 3, 4).float()],
            x0_tokens_action=[torch.zeros(40, 128)],
            x0_tokens_sound=None,
        )
        condition_reference = [torch.cat((gen_data_clean.x0_tokens_vision[0].flatten(), torch.zeros(40 * 128)))]  # noqa: F841
        condition_mask = [torch.zeros(24 + 40 * 128)]  # noqa: F841
        cond_tokens = [[11, 22, 33]]
        initial_noise = [torch.linspace(-0.5, 0.5, 24 + 40 * 128)]

        def velocity_fn(noise_x, timestep):
            # Deliberately omit condition_mask/reference from the closure, as in
            # the actual serving generator. Fake tests must not invent freevars.
            assert gen_data_clean.x0_tokens_action[0].shape == (40, 128) and cond_tokens
            self.seen_noise.append(noise_x)
            v = [torch.ones_like(noise_x[0]) * 0.25]
            self.generated_velocities.append(v)
            return v

        sampler = self.sampler
        return sampler(velocity_fn, initial_noise, num_steps=steps, shift=5.0, seed=[seed])


def make_trace(model):
    path = Path(__file__).resolve()
    return SamplerTrace(model, forward_file=path, forward_sha256=file_sha(path), sources={str(path): file_sha(path)})


def begin(trace, steps=35, admit=lambda identity: None):
    trace.begin(
        name=f"sample{steps}",
        num_steps=steps,
        conditioning={"original_request_without_steps": {"seed": 42}},
        admit=admit,
    )


def test_actual_sampler_result_and_rng_exact_without_extra_velocity_calls():
    baseline = FakeModel()
    expected = baseline.generate_samples_from_batch()
    model = FakeModel()
    original_class = type(model.sampler)
    original_forward = inspect.unwrap(model.sampler.forward.__func__)
    trace = make_trace(model)
    initial_rng = rng_identity()
    begin(trace)
    actual = model.generate_samples_from_batch()
    result = trace.finish()
    assert actual is model.sampler.returned
    assert torch.equal(actual[0], expected[0])
    assert len(model.seen_noise) == len(model.generated_velocities) == result["velocity_evaluation_count"] == 35
    assert result["step_indexes"] == [None] + list(range(1, 35))
    assert result["timesteps"] == list(range(105, 0, -3))
    assert len(result["sigma_values"]) == 36
    assert rng_identity() == initial_rng
    assert type(model.sampler) is original_class
    trace.restore()
    assert "forward" not in model.sampler.__dict__
    assert inspect.unwrap(model.sampler.forward.__func__) is original_forward


def test_raw_bfloat16_bytes_and_noncontiguous_metadata():
    x = torch.arange(12, dtype=torch.float32).to(torch.bfloat16).reshape(3, 4).t()
    version, pointer = x._version, x.data_ptr()
    actual = tensor_snapshot(x)
    expected = hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert actual["value_bytes_sha256"] == expected
    assert actual["dtype"] == "torch.bfloat16" and actual["stride"] == list(x.stride())
    assert actual["value_bytes"] == 24
    assert (x._version, x.data_ptr()) == (version, pointer)


def test_actual_service_inference_mode_without_tensor_version_counters():
    model = FakeModel()
    trace = make_trace(model)
    with torch.inference_mode():
        begin(trace)
        model.generate_samples_from_batch()
        result = trace.finish()
    assert result["initial_state"]["action"]["inference_tensor"] is True
    assert result["initial_state"]["action"]["version_counter_available"] is False


def test_request_isolation_and_actual_initial_state_pairing():
    model, pairs = FakeModel(), []
    trace = make_trace(model)
    for steps in (35, 70, 140):
        begin(trace, steps, admit=pairs.append)
        model.generate_samples_from_batch(steps)
        record = trace.finish()
        assert record["velocity_evaluation_count"] == steps
    assert pairs[0] == pairs[1] == pairs[2]
    assert trace.current is None


def test_bad_initial_state_pair_fails_before_any_model_evaluation():
    model = FakeModel()
    trace = make_trace(model)

    def reject(identity):
        raise ValueError("initial state mismatch")

    begin(trace, admit=reject)
    with pytest.raises(ValueError, match="initial state mismatch"):
        model.generate_samples_from_batch()
    assert model.seen_noise == []
    assert trace.abort()["calls"] == 0


def test_wrong_caller_fails_closed_without_running_velocity():
    model = FakeModel()
    trace = make_trace(model)
    begin(trace)
    with pytest.raises(ValueError, match="generator caller frame is absent"):
        model.sampler(lambda x, t: x, [torch.zeros(5144)], num_steps=35, shift=5.0, seed=[42])
    assert model.seen_noise == []
    trace.abort()


def test_source_change_and_reentrant_request_are_rejected(tmp_path):
    model = FakeModel()
    trace = make_trace(model)
    source = tmp_path / "dependency.py"
    source.write_text("frozen")
    trace.sources[str(source)] = file_sha(source)
    begin(trace)
    with pytest.raises(ValueError, match="already active"):
        begin(trace)
    trace.abort()
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        begin(trace)


def test_observer_rng_and_parameter_mutation_are_detected():
    model = FakeModel()
    with pytest.raises(ValueError, match="observer changed"):
        with observation_guard(model):
            np.random.uniform()
    with pytest.raises(ValueError, match="observer changed"):
        with observation_guard(model):
            with torch.no_grad():
                model.parameter.add_(1)


def test_no_request_no_duplicate_calls_and_no_overwrite(tmp_path):
    model = FakeModel()
    trace = make_trace(model)
    with pytest.raises(ValueError, match="missing request"):
        model.generate_samples_from_batch()
    begin(trace)
    model.generate_samples_from_batch()
    with pytest.raises(ValueError, match="repeated sampler"):
        model.generate_samples_from_batch()
    trace.abort()
    path = tmp_path / "record.json"
    write_new_json(path, {"complete": True})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"complete": False})


def test_real_serving_generator_has_required_locals_without_assumed_closure():
    # Static compilation neither imports Cosmos nor constructs a model. Use the
    # config's actual serving tree, which is distinct from the weight-run tree.
    import json

    root = Path(__file__).resolve().parents[1]
    c = json.loads((root / "configs/wam_sampler_steps_35_20261021.json").read_text())
    path = Path(c["cosmos_repo"]) / "cosmos_framework/model/generator/omni_mot_model.py"
    code = compile(path.read_text(), str(path), "exec")

    def find(parent, name):
        for item in parent.co_consts:
            if isinstance(item, types.CodeType):
                if item.co_name == name:
                    return item
                if result := find(item, name):
                    return result
        return None

    generate = find(code, "generate_samples_from_batch")
    assert generate is not None
    required = {
        "initial_noise",
        "gen_data_clean",
        "condition_reference",
        "condition_mask",
        "cond_tokens",
        "sampler",
        "velocity_fn",
        "guidance",
        "velocity_postprocess",
        "velocity_postprocess_builder",
        "_extra_num_steps",
        "scheduler_type",
        "reuse_pack_templates",
        "reuse_text_kv",
    }
    assert required <= set(generate.co_varnames + generate.co_cellvars)
    # The sampler trace reads actual guarded generator locals, not these
    # implementation-dependent closure slots.
    velocity = find(generate, "velocity_fn")
    assert velocity is not None and "gen_data_clean" in velocity.co_freevars


def test_exact35_checks_normalized_and_inactive_slots_without_tolerance():
    import copy
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/serve_wam_sampler_trace.py"
    spec = importlib.util.spec_from_file_location("sampler_trace_service_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = dict(
        status="success",
        outputs=[
            dict(
                content=dict(
                    action=np.zeros((40, 128)).tolist(),
                    action128=dict(normalized_action=np.zeros((40, 128)).tolist()),
                )
            )
        ],
    )
    module.require_exact35(raw, copy.deepcopy(raw))
    different = copy.deepcopy(raw)
    different["outputs"][0]["content"]["action128"]["normalized_action"][39][127] = 1e-15
    with pytest.raises(ValueError, match="not exactly"):
        module.require_exact35(raw, different)


def test_consecutive_prepare_acceptance_records_use_unique_request_directories(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/serve_wam_sampler_trace.py"
    spec = importlib.util.spec_from_file_location("sampler_trace_service_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = []
    for name in ("generate_first", "generate_second"):
        directory = tmp_path / name
        directory.mkdir()
        item = types.SimpleNamespace(seed=42, sample_args={"output_dir": str(directory)})
        module.save_acceptance_input(item, {"ai_caption": [name]}, tmp_path)
        prepared.append(item)
        assert (directory / "acceptance_input.json").is_file()
    assert not (tmp_path / "acceptance_input.json").exists()
    with pytest.raises(FileExistsError):
        module.save_acceptance_input(prepared[0], {"ai_caption": ["replacement"]}, tmp_path)
    with pytest.raises(ValueError, match="separate request directory"):
        module.save_acceptance_input(
            types.SimpleNamespace(seed=42, sample_args={"output_dir": str(tmp_path)}), {}, tmp_path
        )


def actual_unipc_cpu_smoke(output):
    """Run only from the actual Cosmos interpreter; no model/checkpoint loaded."""
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler, UniPCSamplerConfig

    def model():
        instance = FakeModel()
        instance.sampler = UniPCSampler(UniPCSamplerConfig(), dict(device="cpu", dtype=torch.float32))
        return instance

    baseline = model().generate_samples_from_batch(35)
    instance = model()
    path = Path(inspect.getsourcefile(UniPCSampler)).resolve()
    sources = {str(p): file_sha(p) for p in (path, Path(__file__).resolve())}
    trace = SamplerTrace(instance, forward_file=path, forward_sha256=file_sha(path), sources=sources)
    records = []
    for steps in (35, 70, 140):
        begin(trace, steps)
        with torch.inference_mode():
            result = instance.generate_samples_from_batch(steps)
        record = trace.finish()
        assert len(instance.seen_noise) == sum((35, 70, 140)[: len(records) + 1])
        if steps == 35:
            assert torch.equal(result[0], baseline[0])
        if records:
            assert record["pair_identity"] == records[0]["pair_identity"]
        records.append(record)
    trace.restore()
    write_new_json(
        output,
        dict(
            scope="Actual serving-tree UniPC/scheduler CPU execution with a constant fake velocity; no network/checkpoint/physics",
            all_three_actual_schedules_observed=True,
            traced35_untraced35_exact=True,
            actual_initial_state_and_condition_equal=True,
            velocity_evaluations=[r["velocity_evaluation_count"] for r in records],
            source_sha256=sources,
            records=records,
            new_model_inference_requests=0,
            new_physics_steps=0,
        ),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "--actual-unipc-cpu":
        raise ValueError("CPU smoke requires --actual-unipc-cpu NEW_OUTPUT_JSON")
    actual_unipc_cpu_smoke(Path(sys.argv[2]))
