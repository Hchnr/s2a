"""Isolated, read-only UniPC observations for the registered sampler-step study.

This module does not import Cosmos, change a solver, or install network hooks.
It observes the actual conditioned/masked sampler input, not pure Gaussian RNG.
"""

import hashlib
import inspect
import json
import pickle
import random
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def check_sources(expected):
    for path, digest in expected.items():
        if file_sha(path) != digest:
            raise ValueError(f"frozen source/input changed: {path}")


def tensor_snapshot(value):
    """Hash canonical value bytes without casting, preserving original metadata."""
    import torch

    identity = (id(value), value.data_ptr(), tensor_version(value))
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    result = dict(
        dtype=str(value.dtype),
        shape=list(value.shape),
        stride=list(value.stride()),
        device=str(value.device),
        storage_offset=value.storage_offset(),
        value_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        value_bytes=len(raw),
        inference_tensor=value.is_inference(),
        version_counter_available=identity[-1] is not None,
    )
    if identity != (id(value), value.data_ptr(), tensor_version(value)):
        raise ValueError("observer changed tensor identity/storage/version")
    if identity[-1] is None:
        # Actual execute uses inference_mode: verify bytes when there is no
        # version counter instead of inventing a version witness.
        again = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        if raw != again:
            raise ValueError("observer changed inference tensor bytes")
    return result


def value_identity(snapshot):
    """Device and stride are logged, but pairing concerns numerical input bytes."""
    return {k: snapshot[k] for k in ("dtype", "shape", "value_bytes_sha256", "value_bytes")}


def parameters_identity(model):
    records = [
        (name, id(p), p.data_ptr(), tensor_version(p), str(p.dtype), list(p.shape))
        for name, p in model.named_parameters()
    ]
    return dict(count=len(records), metadata_sha256=json_sha(records))


def tensor_version(value):
    return None if value.is_inference() else value._version


def rng_identity():
    import torch

    states = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()]
    if torch.cuda.is_initialized():
        # Study processes have exactly one visible allowed physical GPU.
        if torch.cuda.device_count() != 1:
            raise ValueError("trace requires one visible CUDA device")
        states.append(torch.cuda.get_rng_state(0).cpu().numpy().tobytes())
    return hashlib.sha256(pickle.dumps(states, protocol=5)).hexdigest()


@contextmanager
def observation_guard(model):
    """Check the observer separately from any legitimate model computation."""
    before = (parameters_identity(model), rng_identity())
    try:
        yield
    finally:
        if before != (parameters_identity(model), rng_identity()):
            raise ValueError("observer changed parameters or RNG")


class SamplerTrace:
    """One request at a time; wrap an existing sampler instance's forward only.

    ``forward_file`` and its digest must come from the frozen launch manifest.
    The original decorated forward remains responsible for no_grad and sampling.
    Caller-frame access is restricted to the unwrapped original code identity.
    """

    def __init__(self, model, *, forward_file, forward_sha256, sources):
        self.model = model
        self.sampler = model.sampler
        self.original = self.sampler.forward
        self.body = inspect.unwrap(self.original.__func__)
        self.generate_body = inspect.unwrap(model.generate_samples_from_batch.__func__)
        self.forward_file = str(Path(forward_file).resolve())
        self.forward_sha256 = forward_sha256
        self.sources = dict(sources)
        self.current = None
        if str(Path(self.body.__code__.co_filename).resolve()) != self.forward_file:
            raise ValueError("unexpected sampler forward source")
        if file_sha(self.forward_file) != forward_sha256 or self.body.__name__ != "forward":
            raise ValueError("sampler source SHA/name differs")
        generate_file = str(Path(self.generate_body.__code__.co_filename).resolve())
        matching = [h for p, h in self.sources.items() if str(Path(p).resolve()) == generate_file]
        if matching != [file_sha(generate_file)]:
            raise ValueError("actual generator source is not frozen")
        if "forward" in self.sampler.__dict__:
            raise ValueError("sampler instance already has a forward override")
        if any(self.sampler._forward_hooks.values()) or any(self.sampler._forward_pre_hooks.values()):
            raise ValueError("sampler hooks are not permitted")
        owner = self

        def forward(sampler, velocity_fn, noise, *args, **kwargs):
            if sampler is not owner.sampler:
                raise ValueError("sampler identity changed")
            return owner._forward(velocity_fn, noise, *args, **kwargs)

        self.sampler.forward = types.MethodType(forward, self.sampler)

    def begin(self, *, name, num_steps, conditioning, admit):
        if self.current is not None:
            raise ValueError("trace request is already active")
        if num_steps not in (35, 70, 140):
            raise ValueError("unregistered sampling steps")
        check_sources(self.sources)
        self.current = dict(
            name=name,
            num_steps=num_steps,
            conditioning=conditioning,
            admit=admit,
            calls=0,
            sampler_calls=0,
            timesteps=[],
            step_indexes=[],
            parameters_before=parameters_identity(self.model),
        )

    def _initial(self, velocity_fn, noise):
        c = self.current
        # Actual serving and weight-run source trees can differ. Never assume a
        # closure contains locals seen in another tree/version of the generator.
        frame = inspect.currentframe().f_back
        try:
            while frame is not None and frame.f_code is not self.generate_body.__code__:
                frame = frame.f_back
            if frame is None:
                raise ValueError("frozen original generator caller frame is absent")
            local = frame.f_locals
            if (
                local.get("self") is not self.model
                or local.get("sampler") is not self.sampler
                or local.get("velocity_fn") is not velocity_fn
                or local.get("initial_noise") is not noise
            ):
                raise ValueError("original generator object identities differ")
            clean = local["gen_data_clean"]
            references, masks = local["condition_reference"], local["condition_mask"]
            cond_tokens = local["cond_tokens"]
            if (
                local.get("guidance") != 1.0
                or local.get("velocity_postprocess") is not None
                or local.get("velocity_postprocess_builder") is not None
                or local.get("_dp_shard_group") is not None
                or local.get("_extra_num_steps") != 0
                or local.get("scheduler_type") != "unipc"
            ):
                raise ValueError("source-derived single-network-forward NFE conditions are absent")
            c["actual_generation_branch"] = {
                k: local[k]
                for k in (
                    "guidance",
                    "velocity_postprocess",
                    "velocity_postprocess_builder",
                    "_extra_num_steps",
                    "scheduler_type",
                    "reuse_pack_templates",
                    "reuse_text_kv",
                )
            }
        finally:
            del frame
        if (
            not isinstance(noise, list)
            or len(noise) != 1
            or noise[0].ndim != 1
            or clean.num_vision_items_per_sample is not None
            or len(clean.x0_tokens_vision) != 1
            or len(clean.x0_tokens_action) != 1
            or clean.x0_tokens_action[0].shape != (40, 128)
            or getattr(clean, "x0_tokens_sound", None)
        ):
            raise ValueError("unregistered initial-state layout")
        vision_shape = clean.x0_tokens_vision[0].shape
        nvision = clean.x0_tokens_vision[0].numel()
        if noise[0].numel() != nvision + 40 * 128:
            raise ValueError("sampler initial-state offsets disagree")
        if (
            len(references) != 1
            or len(masks) != 1
            or references[0].shape != noise[0].shape
            or masks[0].shape != noise[0].shape
        ):
            raise ValueError("unexpected sampler conditioning layout")
        c["initial_state"] = {
            "flat": tensor_snapshot(noise[0]),
            "vision": tensor_snapshot(noise[0][:nvision].reshape(vision_shape)),
            "action": tensor_snapshot(noise[0][nvision:].reshape(40, 128)),
        }
        c["clean_condition_inputs"] = dict(
            vision=tensor_snapshot(clean.x0_tokens_vision[0]),
            action=tensor_snapshot(clean.x0_tokens_action[0]),
        )
        c["condition_reference"] = tensor_snapshot(references[0])
        c["condition_mask"] = tensor_snapshot(masks[0])
        c["conditional_token_ids_sha256"] = json_sha(cond_tokens)
        c["pair_identity"] = dict(
            initial_state={k: value_identity(v) for k, v in c["initial_state"].items()},
            clean_condition_inputs={k: value_identity(v) for k, v in c["clean_condition_inputs"].items()},
            condition_reference=value_identity(c["condition_reference"]),
            condition_mask=value_identity(c["condition_mask"]),
            conditional_token_ids_sha256=c["conditional_token_ids_sha256"],
            request_conditioning=c["conditioning"],
        )
        c["pair_identity_sha256"] = json_sha(c["pair_identity"])
        c["admit"](c["pair_identity"])

    def _observe_step(self, timestep, caller):
        c = self.current
        if caller.f_code is not self.body.__code__ or caller.f_locals.get("self") is not self.sampler:
            raise ValueError("velocity caller is not the frozen original sampler forward")
        schedulers = caller.f_locals["sample_scheduler"]
        if not isinstance(schedulers, list) or len(schedulers) != 1:
            raise ValueError("expected exactly one actual UniPC scheduler")
        scheduler = schedulers[0]
        if c["calls"] == 0:
            c["scheduler"] = scheduler
            c["sigmas"] = tensor_snapshot(scheduler.sigmas)
            c["sigma_values"] = scheduler.sigmas.detach().cpu().tolist()
            c["scheduler_timesteps"] = tensor_snapshot(scheduler.timesteps)
            c["scheduler_timestep_values"] = scheduler.timesteps.detach().cpu().tolist()
            c["scheduler_config"] = dict(scheduler.config)
            if len(c["sigma_values"]) != c["num_steps"] + 1 or len(scheduler.timesteps) != c["num_steps"]:
                raise ValueError("actual scheduler length differs")
        elif scheduler is not c["scheduler"]:
            raise ValueError("actual scheduler changed during request")
        if timestep.shape != (1, 1) or str(timestep.dtype) != "torch.int64":
            raise ValueError("actual timestep shape/dtype differs")
        value = timestep.item()
        index = c["calls"]
        if index >= c["num_steps"] or value != c["scheduler_timestep_values"][index]:
            raise ValueError("actual velocity timestep differs from actual scheduler")
        if scheduler.step_index != (None if index == 0 else index):
            raise ValueError("unexpected actual scheduler step index")
        c["timesteps"].append(value)
        c["step_indexes"].append(scheduler.step_index)
        c["calls"] += 1

    def _forward(self, velocity_fn, noise, *args, **kwargs):
        c = self.current
        if c is None or c["sampler_calls"]:
            raise ValueError("missing request or repeated sampler call")
        if args or kwargs.get("num_steps") != c["num_steps"] or kwargs.get("shift") != 5.0:
            raise ValueError("unregistered sampler arguments")
        if kwargs.get("seed") != [c["conditioning"]["original_request_without_steps"]["seed"]]:
            raise ValueError("actual sampler seed differs")
        c["actual_sampler_arguments"] = kwargs.copy()
        c["sampler_calls"] += 1
        with observation_guard(self.model):
            self._initial(velocity_fn, noise)

        def observed_velocity(actual_noise, timestep):
            caller = inspect.currentframe().f_back
            try:
                with observation_guard(self.model):
                    self._observe_step(timestep, caller)
            finally:
                del caller
            # Delegate exactly once with the original objects; return its object.
            return velocity_fn(actual_noise, timestep)

        result = self.original(observed_velocity, noise, **kwargs)
        with observation_guard(self.model):
            if c["calls"] != c["num_steps"] or c["scheduler"].step_index != c["num_steps"]:
                raise ValueError("incomplete actual sampler evaluations")
            if not isinstance(result, list) or len(result) != 1:
                raise ValueError("unexpected sampler result layout")
            c["final_sampler_state"] = tensor_snapshot(result[0])
        return result

    def finish(self):
        c = self.current
        if c is None or "final_sampler_state" not in c:
            raise ValueError("trace did not complete")
        check_sources(self.sources)
        after = parameters_identity(self.model)
        if after != c["parameters_before"]:
            raise ValueError("model parameters changed during request")
        result = {k: v for k, v in c.items() if k not in ("admit", "scheduler", "calls")}
        result.update(
            complete=True,
            initial_state_semantics="actual conditioned/masked sampler input; pure Gaussian RNG not observed",
            velocity_evaluation_count=c["calls"],
            derived_network_forward_count=c["calls"],
            network_forward_count_basis="frozen g1/single-rank/no-postprocess source; not a network hook count",
            parameters_after=after,
            observer_parameter_rng_identity_checks="passed around each observation, separately from computation",
            source_sha256=self.sources,
        )
        self.current = None
        return result

    def abort(self):
        c, self.current = self.current, None
        if c is None:
            return None
        return {k: v for k, v in c.items() if k not in ("admit", "scheduler")}

    def restore(self):
        if self.current is not None:
            raise ValueError("cannot remove trace during a request")
        del self.sampler.forward
