"""Bind a clean residual to the actual single-conditional sampler velocity.

This bridge delegates the original head and velocity exactly once. It observes
the FP32 sampler action state separately from the packed network input, then
optionally returns ``raw_velocity - residual / actual_sigma`` to the unchanged
solver. Captures are online inputs only; no future label is accepted. CPU hook
tests establish this protocol, not Cosmos compatibility or task performance.
"""

import contextlib
import dataclasses
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from s2a.sampler_trace import check_sources, file_sha, json_sha, observation_guard, parameters_identity
from s2a.wam_clean_readout import CleanResidualReadout, clean_estimate, clean_features, clean_residual_to_velocity
from s2a.wam_flow_readout import SolverSigmaBridge
from s2a.wam_readout import ACTIVE_MASK, HIDDEN_SIZE, DomainContext, _bfloat16_rne
from s2a.wam_trajectory_capture import capture_packed, tensor_array

FEATURE_TERMS = ("sigma_actual", "network_timestep")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seeds(values, name):
    values = tuple(values)
    _require(
        values
        and all(type(value) is int and 0 <= value < 2**63 for value in values)
        and len(set(values)) == len(values),
        f"{name} must be an explicit nonempty unique integer roster",
    )
    return values


@dataclass(frozen=True)
class NoiseRoster:
    """A caller-declared training roster disjoint from the frozen eval seeds.

    Register this object in the run manifest before opening training inputs.
    Constructing it does not establish that preregistration happened; the
    external worker remains responsible for manifest/source admission.
    """

    training_seeds: tuple
    evaluation_seeds: tuple = (42, 43)

    def __post_init__(self):
        training = _seeds(self.training_seeds, "training seeds")
        evaluation = _seeds(self.evaluation_seeds, "evaluation seeds")
        _require(evaluation == (42, 43), "the reserved evaluation roster must remain (42, 43)")
        _require(not set(training) & set(evaluation), "training noise must exclude evaluation seeds42/43")
        object.__setattr__(self, "training_seeds", training)
        object.__setattr__(self, "evaluation_seeds", evaluation)

    def admit(self, seed, purpose):
        _require(purpose in ("training", "evaluation"), "request purpose must explicitly be training or evaluation")
        allowed = self.training_seeds if purpose == "training" else self.evaluation_seeds
        _require(type(seed) is int and seed in allowed, "noise seed is outside the registered purpose roster")


@dataclass(frozen=True, eq=False)
class CleanCapturePoint:
    """Independent read-only snapshots from one actual sampler invocation.

    C0 is the ideal FP64 formula, not the mixed precision scheduler conversion.
    ``raw_velocity_full128`` is the masked actual velocity return, whereas
    ``head_base_full128`` is the unmasked projection output. They are distinct.
    """

    metadata: object
    arrays: object

    def __post_init__(self):
        arrays = {name: np.array(value, copy=True) for name, value in self.arrays.items()}
        for array in arrays.values():
            array.setflags(write=False)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def action(self, name, *, layout):
        _require(layout in ("active10", "full128"), "action layout must explicitly be active10 or full128")
        _require(
            name in ("sample_full128_fp32", "raw_velocity_full128", "deployed_velocity_full128", "c0_full128"),
            "unknown action snapshot",
        )
        value = self.arrays[name]
        return value[:, :10] if layout == "active10" else value


def point_features(point, *, feature_terms):
    """Use [hidden, C0_active10] and only explicitly selected raw time columns.

    No polynomial basis, sigma weighting, per-block scaling, or new statistics
    are inferred. CleanResidualReadout owns its frozen training center/scale.
    """
    terms = tuple(feature_terms)
    _require(
        len(set(terms)) == len(terms) and all(term in FEATURE_TERMS for term in terms), "unregistered feature terms"
    )
    _require(terms == point.metadata["feature_terms"], "feature terms differ from the captured request contract")
    extra = None
    if terms:
        extra = np.column_stack([np.full(40, point.metadata[term], dtype=np.float64) for term in terms])
    return clean_features(
        point.arrays["hidden"], point.action("c0_full128", layout="active10"), explicit_features=extra
    )


class CleanCaptureBridge(SolverSigmaBridge):
    """A restorable bridge for actual 35-node/shift5/guidance1 Cosmos requests.

    ``sink(point)`` receives each of the35 immutable online snapshots, including
    for adapted trajectories. It must not consume model RNG or mutate model
    parameters. ``readout=None`` captures raw trajectories. A zero coefficient
    readout returns the original velocity list and tensor objects unchanged.

    Sources must freeze the original sampler, generator, denoise and projection
    head implementations. Existing head hooks are rejected. An existing instance
    denoise wrapper requires an explicit exact-object ``denoise_delegate`` and a
    frozen implementation source, making mask instrumentation an admitted seam.
    """

    def __init__(
        self, model, *, noise_roster, readout=None, feature_terms=(), sink=None, denoise_delegate=None, **kwargs
    ):
        _require(isinstance(noise_roster, NoiseRoster), "requires a preregistered noise roster")
        _require(
            readout is None or isinstance(readout, CleanResidualReadout), "requires an explicit clean residual head"
        )
        self.noise_roster, self.readout, self.sink = noise_roster, readout, sink
        self.feature_terms = tuple(feature_terms)
        _require(
            len(set(self.feature_terms)) == len(self.feature_terms)
            and all(term in FEATURE_TERMS for term in self.feature_terms),
            "unregistered feature terms",
        )
        if readout is not None:
            _require(
                len(readout.center) == HIDDEN_SIZE + 10 + len(self.feature_terms), "readout feature contract differs"
            )
        self.head = model.net.llm2action
        self.prior_denoise_override = model.__dict__.get("denoise")
        _require(
            (self.prior_denoise_override is None or denoise_delegate is self.prior_denoise_override)
            and not self.head._forward_hooks
            and not self.head._forward_pre_hooks,
            "clean capture requires an unmodified head and explicit exact denoise delegate",
        )
        sources = kwargs.get("sources", {})
        for method in (model.denoise, self.head.forward):
            body = inspect.unwrap(method.__func__)
            path = str(Path(body.__code__.co_filename).resolve())
            _require(sources.get(path) == file_sha(path), "original denoise/head source is not frozen")
        self.original_denoise = model.denoise
        self._step = None
        self.total_head_calls = 0
        self.last_points = ()
        super().__init__(model, None, **kwargs)
        model.denoise = self._denoise
        self.head_handle = self.head.register_forward_hook(self._head, with_kwargs=True)

    @contextlib.contextmanager
    def request(self, domain, *, name, seed, purpose):
        _require(self.current is None and self._step is None, "one non-nested clean request is required")
        _require(isinstance(domain, DomainContext) and domain.is_libero, "requires the full LIBERO active10 domain")
        _require(isinstance(name, str) and bool(name.strip()), "request name must be nonempty")
        self.noise_roster.admit(seed, purpose)
        check_sources(self.sources)
        readout_before = self._readout_sha()
        current = dict(
            name=name,
            seed=seed,
            purpose=purpose,
            noise_roster=dataclasses.asdict(self.noise_roster),
            feature_terms=self.feature_terms,
            domain=domain,
            sampler_calls=0,
            steps=[],
            points=[],
            complete=False,
            parameters_before=parameters_identity(self.model),
        )
        self.current = current
        try:
            yield current
            _require(
                current["sampler_calls"] == 1
                and len(current["steps"]) == len(current["points"]) == 35
                and current["scheduler"].step_index == 35,
                "clean request did not complete its single35-node solver",
            )
            check_sources(self.sources)
            current["parameters_after"] = parameters_identity(self.model)
            _require(current["parameters_before"] == current["parameters_after"], "original model parameters changed")
            _require(readout_before == self._readout_sha(), "clean readout changed during request")
            current["source_sha256"] = self.sources.copy()
            current["complete"] = True
        finally:
            self.last_points = tuple(current["points"])
            self.last = {key: value for key, value in current.items() if key not in ("scheduler", "domain", "points")}
            self.current = self._step = None

    def _readout_sha(self):
        return None if self.readout is None else json_sha(self.readout.to_parameters())

    def _denoise(self, *args, **kwargs):
        step = self._step
        _require(self.current is not None and step is not None, "denoise lacks the actual sampler invocation")
        _require(step["record"]["denoise_calls"] == 0, "repeated denoise in one velocity delegation")
        packed = kwargs.get("data_batch_packed")
        _require(packed is not None, "denoise must expose its actual packed input")
        with observation_guard(self.model):
            arrays, dtypes = capture_packed(packed)
            offset = int(np.prod(arrays["packed_vision"].shape))
            _require(
                arrays["packed_action"].shape == (40, 128)
                and step["sample"].numel() == offset + 40 * 128
                and not np.any(arrays["packed_action"][:, 10:]),
                "actual packed vision/action offset or action layout differs",
            )
            sample_action = tensor_array(step["sample"][offset:].reshape(40, 128))
            _require(not np.any(sample_action[:, 10:]), "actual sampler inactive118 slots are not zero")
            arrays["sample_fp32_flat"] = tensor_array(step["sample"])
            dtypes["sample_fp32_flat"] = str(step["sample"].dtype)
            step.update(arrays=arrays, dtypes=dtypes, action_offset=offset, sample_action=sample_action)
            step["record"]["denoise_calls"] += 1
        return self.original_denoise(*args, **kwargs)

    def _head(self, module, args, kwargs, base):
        import torch

        step = self._step
        _require(
            self.current is not None and step is not None and step["record"]["denoise_calls"] == 1,
            "head lacks its current denoise and sampler step",
        )
        _require(
            not kwargs and len(args) == 2 and step["record"]["head_calls"] == 0, "unexpected or repeated head call"
        )
        hidden, ids = args
        _require(
            hidden.shape == (40, HIDDEN_SIZE)
            and base.shape == (40, 128)
            and ids.shape == (40,)
            and ids.dtype == torch.long
            and hidden.is_floating_point()
            and base.is_floating_point()
            and hidden.device == base.device == step["sample"].device
            and bool(torch.all(ids == self.current["domain"].projection_id))
            and bool(torch.isfinite(hidden).all())
            and bool(torch.isfinite(base).all()),
            "head hidden/base/projection contract differs",
        )
        with observation_guard(self.model):
            step["arrays"].update(
                hidden=tensor_array(hidden), projection_ids=tensor_array(ids), head_base_full128=tensor_array(base)
            )
            step["dtypes"].update(
                hidden=str(hidden.dtype), head_base_full128=str(base.dtype), projection_ids=str(ids.dtype)
            )
            step["record"]["head_calls"] += 1
            self.total_head_calls += 1
        # A forward observer returns None, preserving the original head object.

    def _finish_velocity(self, raw):
        import torch

        step, current = self._step, self.current
        record, sample = step["record"], step["sample"]
        _require(record["head_calls"] == record["denoise_calls"] == 1, "velocity requires one original denoise/head")
        _require(
            isinstance(raw, list)
            and len(raw) == 1
            and raw[0].ndim == 1
            and raw[0].shape == sample.shape
            and raw[0].device == sample.device
            and raw[0].is_floating_point()
            and bool(torch.isfinite(raw[0]).all()),
            "original returned velocity must match its actual flat sampler state",
        )
        offset, sigma = step["action_offset"], record["solver_sigma"]
        actual_action = tensor_array(sample[offset:].reshape(40, 128))
        _require(
            np.array_equal(actual_action, step["sample_action"]), "velocity modified its bound sampler action state"
        )
        _require(
            np.array_equal(tensor_array(sample), step["arrays"]["sample_fp32_flat"]),
            "velocity modified its bound full sampler state",
        )
        raw_action = tensor_array(raw[0][offset:].reshape(40, 128))
        c0 = clean_estimate(actual_action, raw_action, sigma, layout="full128")
        point = CleanCapturePoint(
            dict(
                request_name=current["name"],
                noise_seed=current["seed"],
                purpose=current["purpose"],
                step_index=record["step_index"],
                sigma_actual=sigma,
                network_timestep=record["solver_timestep"],
                action_offset=offset,
                action_dim_mask=ACTIVE_MASK,
                feature_terms=self.feature_terms,
                original_dtypes=tuple(sorted(dict(step["dtypes"], raw_velocity_full128=str(raw[0].dtype)).items())),
                clean_semantics="ideal FP64 sample_FP32 - actual_sigma * raw_velocity; not scheduler x0",
            ),
            dict(step["arrays"], sample_full128_fp32=actual_action, raw_velocity_full128=raw_action, c0_full128=c0),
        )
        result = raw
        residual = np.zeros((40, 10), dtype=np.float64)
        delta = np.zeros((40, 10), dtype=np.float64)
        if self.readout is not None and np.any(self.readout.coefficients):
            residual = self.readout.predict_clean_residual(point_features(point, feature_terms=self.feature_terms))
            delta = clean_residual_to_velocity(residual, sigma, layout="active10")
            base = raw[0][offset:].reshape(40, 128)
            summed = base[:, :10].to(torch.float64) + torch.as_tensor(delta, dtype=torch.float64, device=base.device)
            active = _bfloat16_rne(summed) if base.dtype == torch.bfloat16 else summed.to(base.dtype)
            _require(bool(torch.isfinite(active).all()), "adapted velocity became nonfinite after deployment rounding")
            adapted = raw[0].clone()
            adapted[offset:].reshape(40, 128)[:, :10] = active
            result = [adapted]
        record["returned_original_velocity_object"] = result is raw
        point = CleanCapturePoint(
            dict(point.metadata, returned_original_velocity_object=result is raw),
            dict(
                point.arrays,
                raw_velocity_flat=tensor_array(raw[0]),
                deployed_velocity_flat=tensor_array(result[0]),
                deployed_velocity_full128=tensor_array(result[0][offset:].reshape(40, 128)),
                clean_residual_active10=residual,
                velocity_delta_active10=delta,
            ),
        )
        dtypes = dict(point.metadata["original_dtypes"])
        dtypes.update(
            sample_full128_fp32=str(sample.dtype),
            raw_velocity_flat=str(raw[0].dtype),
            deployed_velocity_flat=str(result[0].dtype),
            deployed_velocity_full128=str(result[0].dtype),
            c0_full128="numpy.float64",
            clean_residual_active10="numpy.float64",
            velocity_delta_active10="numpy.float64",
        )
        point = CleanCapturePoint(dict(point.metadata, original_dtypes=tuple(sorted(dtypes.items()))), point.arrays)
        current["points"].append(point)
        if self.sink is not None:
            self.sink(point)
        return result

    def _forward(self, velocity_fn, noise, *args, **kwargs):
        import torch

        current = self.current
        _require(current is not None and not current["sampler_calls"], "missing or repeated clean sampling request")
        _require(
            not args
            and kwargs.get("num_steps") == 35
            and kwargs.get("shift") == 5.0
            and kwargs.get("seed") == [current["seed"]],
            "requires registered35/shift5/single-seed sampling",
        )
        with observation_guard(self.model):
            self._generator_guard(velocity_fn, noise)
        current["actual_sampler_arguments"] = kwargs.copy()
        current["sampler_calls"] += 1

        def velocity(actual_noise, timestep):
            _require(self._step is None, "overlapping sampler velocity delegations")
            caller = inspect.currentframe().f_back
            try:
                with observation_guard(self.model):
                    sigma, record = self._observe(timestep, caller)
                    _require(
                        isinstance(actual_noise, list)
                        and len(actual_noise) == 1
                        and caller.f_locals.get("latent") is actual_noise
                        and isinstance(actual_noise[0], torch.Tensor)
                        and actual_noise[0].dtype == torch.float32
                        and actual_noise[0].ndim == 1,
                        "requires the actual original FP32 flat sampler latent, never BF16 packed tokens",
                    )
                    _require(sigma > 0, "terminal sigma must never invoke a clean residual head")
                    record.update(head_calls=0, denoise_calls=0)
                    self._step = dict(sample=actual_noise[0], record=record)
            finally:
                del caller
            try:
                record["velocity_delegations"] += 1
                raw = velocity_fn(actual_noise, timestep)
                with observation_guard(self.model):
                    return self._finish_velocity(raw)
            finally:
                self._step = None

        return self.original(velocity, noise, **kwargs)

    def restore(self):
        _require(self.current is None and self._step is None, "cannot remove clean bridge during a request")
        self.head_handle.remove()
        if self.prior_denoise_override is None:
            del self.model.denoise
        else:
            self.model.denoise = self.prior_denoise_override
        super().restore()
