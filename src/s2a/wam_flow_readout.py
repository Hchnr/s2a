"""026 clean-action residuals, fitted and applied in actual flow-time units."""

import contextlib
import contextvars
import inspect
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2a.sampler_trace import check_sources, file_sha, observation_guard, parameters_identity
from s2a.wam_readout import (
    ACTIVE_SLOTS,
    HIDDEN_SIZE,
    RIDGE,
    DomainContext,
    _bfloat16_rne,
    _fit_inputs,
    _matrix,
    features,
)

_REQUEST = contextvars.ContextVar("s2a_flow_readout_request", default=None)
_STEP = contextvars.ContextVar("s2a_flow_readout_step", default=None)


def sigma_rows(sigma, count):
    """Expand exact stored FP32/solver values to FP64; never guess from time IDs."""
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full((count, 1), sigma, dtype=np.float64)
    elif sigma.shape == (count,):
        sigma = sigma[:, None]
    if sigma.shape != (count, 1) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("actual sigma must be finite and positive for every row")
    return sigma.copy()


def design_matrix(hidden, sigma):
    phi = features(hidden)
    return -phi / sigma_rows(sigma, len(phi))


@dataclass(frozen=True)
class FlowResidualReadout:
    weights: np.ndarray

    def __post_init__(self):
        weights = _matrix(self.weights, "flow readout weights", columns=10)
        if len(weights) < 2:
            raise ValueError("requires hidden features and a regularized bias")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    @property
    def hidden_size(self):
        return len(self.weights) - 1

    @property
    def is_zero(self):
        return not np.any(self.weights)

    @classmethod
    def zero(cls, hidden_size=HIDDEN_SIZE):
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden size must be a positive integer")
        return cls(np.zeros((hidden_size + 1, 10), dtype=np.float64))

    @classmethod
    def fit(cls, hidden, sigma_actual, residual_target, weights):
        phi, target, weights = _fit_inputs(hidden, residual_target, weights)
        design = -phi / sigma_rows(sigma_actual, len(phi))
        normal = design.T @ (weights[:, None] * design) + RIDGE * np.eye(design.shape[1])
        return cls(np.linalg.solve(normal, design.T @ (weights[:, None] * target)))

    @classmethod
    def fit_bias(cls, hidden, sigma_actual, residual_target, weights):
        phi, target, weights = _fit_inputs(hidden, residual_target, weights)
        design = -1 / sigma_rows(sigma_actual, len(phi))[:, 0]
        coefficients = np.zeros((phi.shape[1], 10), dtype=np.float64)
        coefficients[-1] = ((weights * design) @ target) / (np.dot(weights, design**2) + RIDGE)
        return cls(coefficients)

    def apply_torch(self, hidden, base, context, projection_ids, sigma_actual):
        import torch

        if not isinstance(context, DomainContext):
            raise ValueError("requires the full-domain request context")
        if not isinstance(base, torch.Tensor) or base.ndim != 2 or base.shape[1] != 128:
            raise ValueError("base must be N×128")
        if (
            not isinstance(projection_ids, torch.Tensor)
            or projection_ids.shape != (len(base),)
            or projection_ids.dtype != torch.long
            or not bool(torch.all(projection_ids == context.projection_id))
        ):
            raise ValueError("mixed or inconsistent projection IDs")
        if not context.is_libero:
            return base
        sigma = torch.tensor(sigma_rows(sigma_actual, len(base)), dtype=torch.float64, device=base.device)
        if self.is_zero:
            return base
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.shape != (len(base), self.hidden_size)
            or hidden.device != base.device
            or not hidden.is_floating_point()
            or not base.is_floating_point()
            or not bool(torch.isfinite(hidden).all())
            or not bool(torch.isfinite(base).all())
        ):
            raise ValueError("requires finite matching hidden/base tensors on one device")
        phi = torch.cat(
            (
                hidden.to(torch.float64) / np.sqrt(self.hidden_size),
                torch.ones((len(base), 1), dtype=torch.float64, device=base.device),
            ),
            dim=1,
        )
        # The operation order is part of026: divide the complete design before
        # multiplying W, rather than multiplying phi first or dividing W.
        design = -phi / sigma
        coefficients = torch.tensor(self.weights.copy(), dtype=torch.float64, device=base.device)
        summed = base[:, :ACTIVE_SLOTS].to(torch.float64) + design @ coefficients
        active = _bfloat16_rne(summed) if base.dtype == torch.bfloat16 else summed.to(base.dtype)
        if not bool(torch.isfinite(active).all()):
            raise ValueError("flow readout produced nonfinite deployment output")
        result = base.clone()
        result[:, :ACTIVE_SLOTS] = active
        return result


class FlowReadoutHook:
    """A homogeneous request and its actual velocity step must both be active."""

    def __init__(self, head, readout):
        if not isinstance(readout, FlowResidualReadout) or readout.hidden_size != HIDDEN_SIZE:
            raise ValueError("registered flow hook requires hidden1280")
        self.readout = readout
        self._handle = head.register_forward_hook(self._forward, with_kwargs=True)

    @contextlib.contextmanager
    def request(self, domain):
        if not isinstance(domain, DomainContext) or _REQUEST.get() is not None or _STEP.get() is not None:
            raise ValueError("one non-nested full-domain flow request is required")
        request = dict(hook=self, domain=domain)
        token = _REQUEST.set(request)
        try:
            yield
        finally:
            _REQUEST.reset(token)

    @contextlib.contextmanager
    def _velocity_step(self, sigma, record):
        request = _REQUEST.get()
        if request is None or request["hook"] is not self or _STEP.get() is not None:
            raise ValueError("actual flow step is outside its request")
        value = float(sigma_rows(sigma, 1)[0, 0])
        token = _STEP.set(dict(request=request, sigma=value, record=record))
        record["head_calls"] = 0
        try:
            yield
            if record["head_calls"] != 1:
                raise ValueError("single conditional velocity must invoke its head exactly once")
        finally:
            _STEP.reset(token)

    def _forward(self, module, args, kwargs, base):
        request, step = _REQUEST.get(), _STEP.get()
        if request is None or request["hook"] is not self or step is None or step["request"] is not request:
            raise ValueError("flow head lacks its current request/actual solver step")
        if kwargs or len(args) != 2 or step["record"]["head_calls"]:
            raise ValueError("unexpected or repeated original flow head invocation")
        step["record"]["head_calls"] += 1
        return self.readout.apply_torch(args[0], base, request["domain"], args[1], step["sigma"])

    def remove(self):
        self._handle.remove()


class SolverSigmaBridge:
    """Delegate once to the unchanged solver, exposing only its actual sigma."""

    def __init__(self, model, hook, *, forward_file, forward_sha256, sources):
        self.model, self.hook, self.sampler = model, hook, model.sampler
        self.original = self.sampler.forward
        self.body = inspect.unwrap(self.original.__func__)
        self.generate_body = inspect.unwrap(model.generate_samples_from_batch.__func__)
        self.sources = dict(sources)
        self.current, self.last = None, None
        if (
            str(Path(self.body.__code__.co_filename).resolve()) != str(Path(forward_file).resolve())
            or self.body.__name__ != "forward"
            or file_sha(forward_file) != forward_sha256
        ):
            raise ValueError("flow sampler source/code identity is not frozen")
        generate_file = str(Path(self.generate_body.__code__.co_filename).resolve())
        if (
            self.sources.get(generate_file) != file_sha(generate_file)
            or self.sources.get(str(Path(forward_file).resolve())) != forward_sha256
        ):
            raise ValueError("flow sampler/generator source SHA is absent")
        if "forward" in self.sampler.__dict__ or self.sampler._forward_hooks or self.sampler._forward_pre_hooks:
            raise ValueError("flow bridge requires the unchanged sampler instance")
        owner = self

        def forward(sampler, velocity_fn, noise, *args, **kwargs):
            if sampler is not owner.sampler:
                raise ValueError("flow sampler instance changed")
            return owner._forward(velocity_fn, noise, *args, **kwargs)

        self.sampler.forward = types.MethodType(forward, self.sampler)

    @contextlib.contextmanager
    def request(self, domain, *, name, seed):
        if self.current is not None or seed not in (42, 43):
            raise ValueError("one registered flow request is required")
        check_sources(self.sources)
        current = dict(
            name=name,
            seed=seed,
            sampler_calls=0,
            steps=[],
            complete=False,
            parameters_before=parameters_identity(self.model),
        )
        self.current = current
        try:
            with self.hook.request(domain):
                yield current
            if current["sampler_calls"] != 1 or len(current["steps"]) != 35 or current["scheduler"].step_index != 35:
                raise ValueError("flow request did not complete its single35-step solver")
            check_sources(self.sources)
            current["parameters_after"] = parameters_identity(self.model)
            if current["parameters_after"] != current["parameters_before"]:
                raise ValueError("original model parameter identity changed")
            current["source_sha256"] = self.sources.copy()
            current["complete"] = True
        finally:
            self.last = {k: v for k, v in current.items() if k != "scheduler"}
            self.current = None

    def _generator_guard(self, velocity_fn, noise):
        frame = inspect.currentframe().f_back
        try:
            while frame is not None and frame.f_code is not self.generate_body.__code__:
                frame = frame.f_back
            if frame is None:
                raise ValueError("frozen original generator caller is absent")
            local = frame.f_locals
            if (
                local.get("self") is not self.model
                or local.get("sampler") is not self.sampler
                or local.get("velocity_fn") is not velocity_fn
                or local.get("initial_noise") is not noise
            ):
                raise ValueError("actual generator object identities differ")
            if (
                local.get("guidance") != 1.0
                or local.get("velocity_postprocess") is not None
                or local.get("velocity_postprocess_builder") is not None
                or local.get("_dp_shard_group") is not None
                or local.get("_extra_num_steps") != 0
                or local.get("scheduler_type") != "unipc"
            ):
                raise ValueError("requires the actual single-conditional unmodified UniPC branch")
        finally:
            del frame

    def _observe(self, timestep, caller):
        current = self.current
        if caller.f_code is not self.body.__code__ or caller.f_locals.get("self") is not self.sampler:
            raise ValueError("velocity caller is not the frozen original sampler")
        schedulers = caller.f_locals.get("sample_scheduler")
        if not isinstance(schedulers, list) or len(schedulers) != 1:
            raise ValueError("requires one actual per-sample scheduler")
        scheduler, index = schedulers[0], len(current["steps"])
        if index == 0:
            current.update(
                scheduler=scheduler,
                sigma_values=scheduler.sigmas.detach().cpu().tolist(),
                scheduler_timesteps=scheduler.timesteps.detach().cpu().tolist(),
                solver_sigma_dtype=str(scheduler.sigmas.dtype),
                terminal_sigma=float(scheduler.sigmas[-1].item()),
                terminal_neural_calls=0,
            )
            if (
                len(current["sigma_values"]) != 36
                or len(current["scheduler_timesteps"]) != 35
                or current["solver_sigma_dtype"] != "torch.float32"
                or current["terminal_sigma"] != 0
            ):
                raise ValueError("actual solver grid is not35steps plus terminal sigma")
        if scheduler is not current["scheduler"] or scheduler.step_index != (None if index == 0 else index):
            raise ValueError("actual scheduler instance/step does not match current velocity")
        if (
            index >= 35
            or timestep.shape != (1, 1)
            or str(timestep.dtype) != "torch.int64"
            or timestep.item() != current["scheduler_timesteps"][index]
            or scheduler.timesteps[index].item() != timestep.item()
        ):
            raise ValueError("velocity timestep does not match actual solver step")
        sigma = float(scheduler.sigmas[index].item())
        if sigma != current["sigma_values"][index]:
            raise ValueError("actual scheduler sigma changed")
        sigma_rows(sigma, 1)  # Reject terminal/invalid sigma BEFORE velocity/NN.
        record = dict(
            step_index=index,
            solver_step_index_before=scheduler.step_index,
            solver_timestep=int(timestep.item()),
            solver_sigma=sigma,
            velocity_delegations=0,
        )
        current["steps"].append(record)
        return sigma, record

    def _forward(self, velocity_fn, noise, *args, **kwargs):
        current = self.current
        if current is None or current["sampler_calls"]:
            raise ValueError("missing or repeated flow sampling request")
        if (
            args
            or kwargs.get("num_steps") != 35
            or kwargs.get("shift") != 5.0
            or kwargs.get("seed") != [current["seed"]]
        ):
            raise ValueError("requires registered35/shift5/single-seed sampling")
        with observation_guard(self.model):
            self._generator_guard(velocity_fn, noise)
        current["actual_sampler_arguments"] = kwargs.copy()
        current["sampler_calls"] += 1

        def velocity(actual_noise, timestep):
            caller = inspect.currentframe().f_back
            try:
                with observation_guard(self.model):
                    sigma, record = self._observe(timestep, caller)
            finally:
                del caller
            with self.hook._velocity_step(sigma, record):
                record["velocity_delegations"] += 1
                return velocity_fn(actual_noise, timestep)

        return self.original(velocity, noise, **kwargs)

    def restore(self):
        if self.current is not None:
            raise ValueError("cannot remove the flow bridge during a request")
        del self.sampler.forward
