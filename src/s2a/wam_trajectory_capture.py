"""029 transparent observations of the original sampler's actual flow trajectory."""

import contextlib
import dataclasses
import inspect
import types
from pathlib import Path

import numpy as np

from s2a.sampler_trace import file_sha, observation_guard
from s2a.wam_flow_readout import SolverSigmaBridge


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_array(value):
    value = value.detach().cpu()
    return (value.float() if str(value.dtype) == "torch.bfloat16" else value).numpy().copy()


def same_sample_view(sample, original):
    """The scheduler unsqueezes the real latent; object identity would be wrong."""
    import torch

    return (
        isinstance(sample, torch.Tensor)
        and isinstance(original, torch.Tensor)
        and sample.dtype == original.dtype
        and sample.device == original.device
        and sample.numel() == original.numel()
        and sample.is_contiguous()
        and original.is_contiguous()
        and sample.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        and sample.storage_offset() == original.storage_offset()
        and torch.equal(sample.reshape(-1), original.reshape(-1))
    )


def capture_packed(packed):
    arrays, dtypes = {}, {}
    for kind in ("vision", "action"):
        modality = getattr(packed, kind)
        for key, value in (
            (f"packed_{kind}", modality.tokens[0]),
            (f"time_{kind}", modality.timesteps),
            (f"condition_{kind}", modality.condition_mask[0]),
        ):
            arrays[key], dtypes[key] = tensor_array(value), str(value.dtype)
    return arrays, dtypes


def exact_packed_times(packed, arrays, original_dtypes):
    """Compare the actual final network time, not an inferred scalar convention."""
    for kind in ("vision", "action"):
        key = f"time_{kind}"
        value = getattr(packed, kind).timesteps
        actual = tensor_array(value)
        require(
            str(value.dtype) == original_dtypes[key]
            and actual.dtype == arrays[key].dtype
            and actual.shape == arrays[key].shape
            and np.array_equal(actual, arrays[key]),
            f"029 teacher final packed {kind} time differs from its free point",
        )


def packing_config(config_type, options):
    """Keep frozen dataclass defaults absent from the model's input options."""
    return config_type(
        **{field.name: options[field.name] for field in dataclasses.fields(config_type) if field.name in options}
    )


class TrajectoryCaptureBridge(SolverSigmaBridge):
    """Capture three original velocity/conversion calls without another evaluation.

    ``sink(record, arrays, metadata)`` writes one point after the original
    scheduler's conversion. No labels or future source are available here.
    """

    def __init__(
        self,
        model,
        hook,
        *,
        sink,
        convert_file,
        convert_sha256,
        selected_steps=(19, 29, 33),
        rng_capture=None,
        **kwargs,
    ):
        require(tuple(selected_steps) == (19, 29, 33), "029 requires the three registered trajectory points")
        self.sink, self.convert_file, self.convert_sha = sink, str(Path(convert_file).resolve()), convert_sha256
        self.selected_steps = tuple(selected_steps)
        self.rng_capture = rng_capture
        self.point = self.actual_sample = self.scheduler_conversion = self.record = None
        self.points, self.last_points, self.total_head_calls = [], [], 0
        self.context = None
        super().__init__(model, hook, **kwargs)
        self.original_denoise = model.denoise
        model.denoise = self.denoise
        self.head_handle = model.net.llm2action.register_forward_hook(self.observe_head, with_kwargs=True)

    @contextlib.contextmanager
    def request(self, domain, *, name, seed):
        self.points = []
        self.context = dataclasses.asdict(domain)
        try:
            with super().request(domain, name=name, seed=seed) as current:
                yield current
                require(
                    len(self.points) == 3 and all(step.get("conversion_calls") == 1 for step in current["steps"]),
                    "029 actual conversions or three selected captures are incomplete",
                )
        finally:
            if self.scheduler_conversion is not None:
                scheduler, _ = self.scheduler_conversion
                del scheduler.convert_model_output
            self.last_points = list(self.points)
            self.scheduler_conversion = self.actual_sample = self.point = self.record = self.context = None

    def _install_conversion(self, scheduler):
        require("convert_model_output" not in scheduler.__dict__, "029 scheduler conversion already overridden")
        original = scheduler.convert_model_output
        body = inspect.unwrap(original.__func__)
        require(
            str(Path(body.__code__.co_filename).resolve()) == self.convert_file
            and file_sha(self.convert_file) == self.convert_sha
            and self.sources.get(self.convert_file) == self.convert_sha
            and scheduler.predict_x0 is True
            and scheduler.config.prediction_type == "flow_prediction"
            and scheduler.config.thresholding is False,
            "029 requires the frozen unmodified flow-to-x0 conversion",
        )
        self.scheduler_conversion = (scheduler, original)
        owner = self

        def convert(instance, model_output, *args, **kwargs):
            record = owner.record
            sample = kwargs.get("sample", args[1] if len(args) > 1 else None)
            require(
                owner.current is not None
                and instance is owner.current["scheduler"]
                and instance.step_index == record["step_index"]
                and record["conversion_calls"] == 0,
                "029 repeated, out-of-order or foreign scheduler conversion",
            )
            require(
                same_sample_view(sample, owner.actual_sample), "029 actual conversion sample is not its latent view"
            )
            record["conversion_calls"] += 1
            point = owner.point
            if point is not None:
                with observation_guard(owner.model):
                    require(
                        model_output.ndim == 1 and model_output.numel() == sample.numel(), "029 velocity layout differs"
                    )
                    owner.add("velocity", model_output)
                    owner.add("velocity_action", model_output[point["metadata"]["action_offset"] :].reshape(40, 128))
            result = original(model_output, *args, **kwargs)
            if point is not None:
                with observation_guard(owner.model):
                    require(result.shape == sample.shape, "029 converted x0 layout differs")
                    owner.add(
                        "converted_action", result.reshape(-1)[point["metadata"]["action_offset"] :].reshape(40, 128)
                    )
                    require(
                        point["head_calls"] == point["denoise_calls"] == 1,
                        "029 point lacks its single actual head/denoise",
                    )
                    point["metadata"].update(
                        conversion_calls=1,
                        velocity_delegations=record["velocity_delegations"],
                        head_calls=point["head_calls"],
                        denoise_calls=point["denoise_calls"],
                        conversion_sample_is_original_storage_view=True,
                        conversion_returned_original_object=True,
                        velocity_source="original single-conditional _get_velocity return passed to scheduler conversion",
                    )
                    if owner.rng_capture is not None:
                        point["metadata"]["rng_after"] = owner.rng_capture("after", record)
                    owner.points.append(owner.sink(record, point["arrays"], point["metadata"]))
                owner.point = None
            return result

        scheduler.convert_model_output = types.MethodType(convert, scheduler)

    def _observe(self, timestep, caller):
        require(self.point is None, "029 prior selected point was not converted")
        if self.record is not None:
            require(self.record["conversion_calls"] == 1, "029 previous step lacked its actual conversion")
        sigma, record = super()._observe(timestep, caller)
        latent = caller.f_locals.get("latent")
        velocity_frame = inspect.currentframe().f_back
        try:
            require(
                isinstance(latent, list)
                and len(latent) == 1
                and velocity_frame.f_locals.get("actual_noise") is latent
                and str(latent[0].dtype) == "torch.float32"
                and latent[0].ndim == 1,
                "029 original velocity did not receive the actual FP32 flat sampler latent",
            )
        finally:
            del velocity_frame
        self.actual_sample, self.record = latent[0], record
        record["conversion_calls"] = 0
        if self.scheduler_conversion is None:
            self._install_conversion(self.current["scheduler"])
        if record["step_index"] in self.selected_steps:
            self.point = dict(
                arrays={},
                metadata=dict(
                    step_index=record["step_index"],
                    sigma_actual=sigma,
                    network_timestep=record["solver_timestep"],
                    original_dtypes={},
                    context=self.context,
                ),
                head_calls=0,
                denoise_calls=0,
            )
            self.add("sample", self.actual_sample)
            self.add("network_timestep", timestep)
            if self.rng_capture is not None:
                self.point["metadata"]["rng_before"] = self.rng_capture("before", record)
        return sigma, record

    def add(self, name, tensor):
        self.point["arrays"][name] = tensor_array(tensor)
        self.point["metadata"]["original_dtypes"][name] = str(tensor.dtype)

    def denoise(self, *args, **kwargs):
        require(self.current is not None and self.record is not None, "029 denoise outside a free trajectory")
        if self.point is not None:
            with observation_guard(self.model):
                require(self.point["denoise_calls"] == 0, "029 repeated selected denoise")
                self.point["denoise_calls"] += 1
                arrays, dtypes = capture_packed(kwargs["data_batch_packed"])
                self.point["arrays"].update(arrays)
                self.point["metadata"]["original_dtypes"].update(dtypes)
                offset = int(np.prod(arrays["packed_vision"].shape))
                require(self.actual_sample.numel() == offset + 5120, "029 actual vision/action flat layout differs")
                self.point["metadata"].update(vision_shape=list(arrays["packed_vision"].shape), action_offset=offset)
                self.add("sample_action", self.actual_sample[offset:].reshape(40, 128))
        result = self.original_denoise(*args, **kwargs)
        if self.point is not None:
            with observation_guard(self.model):
                self.add("pred_action", result["preds_action"][0])
        return result

    def observe_head(self, module, values, kwargs, base):
        require(self.current is not None and not kwargs and len(values) == 2, "029 head outside free request")
        self.total_head_calls += 1
        if self.point is not None:
            with observation_guard(self.model):
                require(self.point["head_calls"] == 0, "029 repeated selected head")
                self.point["head_calls"] += 1
                self.add("hidden", values[0])
                self.add("projection_ids", values[1])
                self.add("base", base)

    def restore(self):
        require(self.current is None, "029 cannot remove observers during a request")
        self.head_handle.remove()
        self.model.denoise = self.original_denoise
        super().restore()
