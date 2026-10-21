"""025 phase2:96 paired current-observation samples after both teacher gates."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from audit_wam_current_numerics import bundle, check_hashes, check_weights, numpy_tensor, ref, write_new
from run_wam_readout_features import (
    BoundaryCounter,
    environment,
    generate,
    parameter_versions,
    read_prepared,
    setup_service,
)

from s2a.artifacts import sha256
from s2a.wam_readout import ReadoutHook, ResidualReadout


def checked_json(path, digest):
    if sha256(path) != digest:
        raise ValueError("025 sampling admission SHA differs")
    return json.loads(Path(path).read_text())


def sampling_admission(prepared_path, fit_path, fit_sha, verification_path, verification_sha):
    """Reject a missing, failed or unrelated teacher gate before loading WAM."""
    fit = checked_json(fit_path, fit_sha)
    verified = checked_json(verification_path, verification_sha)
    prepared_ref = ref(prepared_path)
    if not (
        fit.get("complete") is True
        and fit.get("passed") is True
        and fit.get("eligible_for_sampling") is True
        and fit.get("teacher_gate", {}).get("passed") is True
        and fit.get("producer_fit_groups") == fit.get("fit_attempts") == 2
        and fit.get("fitted_rows") == fit.get("validation_rows") == 2880
        and fit.get("prepared_ref") == prepared_ref
    ):
        raise ValueError("025 producer teacher gate did not admit sampling")
    if not (
        verified.get("phase") == "fit"
        and verified.get("passed") is True
        and verified.get("eligible_for_sampling") is True
        and verified.get("prepared") == prepared_ref
        and verified.get("input_sha256", {}).get(str(Path(fit_path).resolve())) == fit_sha
    ):
        raise ValueError("025 independent teacher gate did not admit this fit/preparation")
    check_hashes(verified["input_sha256"])
    model_ref = fit["models"]["adapt"]
    if sha256(model_ref["path"]) != model_ref["sha256"]:
        raise ValueError("025 adapted coefficients changed after the teacher gate")
    with np.load(model_ref["path"], allow_pickle=False) as archive:
        if archive.files != ["weights"] or archive["weights"].shape != (1281, 10):
            raise ValueError("025 adapted head has wrong coefficient roster/geometry")
        readout = ResidualReadout(archive["weights"])
    return fit, readout


class InitialCapture:
    """Observe the actual masked inference preparation once per current request."""

    def __init__(self, model):
        self.model, self.original = model, model._prepare_inference_data
        self.entry = self.directory = self.reference = None
        model._prepare_inference_data = self.forward

    def arm(self, entry, directory):
        if self.entry is not None:
            raise ValueError("025 initial capture is already armed")
        self.entry, self.directory, self.reference = entry, directory, None

    def finish(self):
        if self.entry is None or self.reference is None:
            raise ValueError("025 sampling did not observe one actual inference preparation")
        reference = self.reference
        self.entry = self.directory = self.reference = None
        return reference

    def forward(self, data_batch, seed, has_negative_prompt=False):
        if self.entry is None or self.reference is not None:
            raise ValueError("025 unexpected or repeated inference preparation")
        if seed != [self.entry["sampling_seed"]] or has_negative_prompt:
            raise ValueError("025 actual inference seed/negative-prompt branch differs")
        result = self.original(data_batch, seed, has_negative_prompt)
        if len(result) != 8 or len(result[4]) != 1 or len(result[6]) != 1 or result[7] is not True:
            raise ValueError("025 actual inference preparation is not one WAM sample")
        clean, tokens, initial, mask = result[1], result[2], result[4][0], result[6][0]
        if len(clean.x0_tokens_vision) != 1 or tuple(clean.x0_tokens_vision[0].shape) != (1, 48, 11, 60, 40):
            raise ValueError("025 actual vision packing geometry differs")
        arrays = dict(
            initial_noise=numpy_tensor(initial),
            condition_mask=numpy_tensor(mask),
            first_condition_latent=numpy_tensor(clean.x0_tokens_vision[0][:, :, :1]),
        )
        if any(array.dtype != np.float32 or not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("025 actual inference prefix/noise must remain finite FP32")
        self.reference = bundle(
            self.directory,
            "initial",
            arrays,
            dict(
                generation_index=self.entry["generation_index"],
                name=self.entry["name"],
                pair_id=self.entry["pair_id"],
                variant=self.entry["variant"],
                sampling_seed=seed[0],
                caption=data_batch["ai_caption"][0],
                text_tokens=tokens,
                inference_only_current_image=True,
            ),
        )
        return result

    def remove(self):
        self.model._prepare_inference_data = self.original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--fit-sha256", required=True)
    parser.add_argument("--fit-verification", type=Path, required=True)
    parser.add_argument("--fit-verification-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    environment()
    prepared = read_prepared(args.prepared, args.prepared_sha256)
    fit, adapted = sampling_admission(
        args.prepared, args.fit, args.fit_sha256, args.fit_verification, args.fit_verification_sha256
    )
    entries = prepared["validation_generations"]
    if len(entries) != 96 or [entry["variant"] for entry in entries] != ["raw", "adapt"] * 48:
        raise ValueError("025 fixed paired96 sampling roster differs")
    output = Path(prepared["output_root"]) / "sampling"
    output.mkdir(exist_ok=False)
    write_new(output / "worker_started.json", dict(pid=os.getpid(), ppid=os.getppid(), cuda="6", ld=""))
    result = dict(
        schema_version="s2a.frozen-readout-sampling.v1",
        complete=False,
        passed=False,
        prepared_ref=ref(args.prepared),
        fit_ref=ref(args.fit),
        fit_verification_ref=ref(args.fit_verification),
        adapted_model_ref=fit["models"]["adapt"],
        generations=[],
        denoise_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    counter = hook = capture = head_observer = None
    try:
        import torch

        service_dir = output / "service"
        service_dir.mkdir()
        service = setup_service(prepared, service_dir)
        model = service.model
        before_parameters = parameter_versions(model)
        zero = ResidualReadout.zero()
        hook = ReadoutHook(model.net.llm2action, zero)
        counter = BoundaryCounter(model, 3360)
        capture = InitialCapture(model)
        head_counts = dict(raw=0, adapt=0)
        active_variant = None

        def observe_head(module, values, kwargs, base):
            if active_variant is None or counter.current is None or kwargs:
                raise ValueError("025 sampling head called outside its registered scope")
            if tuple(values[0].shape) != (40, 1280) or tuple(base.shape) != (40, 128):
                raise ValueError("025 sampling head geometry differs")
            head_counts[active_variant] += 1

        head_observer = model.net.llm2action.register_forward_hook(observe_head, with_kwargs=True)
        with torch.inference_mode():
            for entry in entries:
                active_variant = entry["variant"]
                hook.readout = adapted if active_variant == "adapt" else zero
                directory = output / entry["name"]
                capture.arm(entry, directory)
                count_before = head_counts[active_variant]
                generate(service, entry["request"], directory, counter, hook, f"sampling:{entry['generation_index']}")
                initial = capture.finish()
                if head_counts[active_variant] - count_before != 35:
                    raise ValueError("025 actual readout did not observe every sampler forward")
                result["generations"].append(
                    dict(
                        entry,
                        response=ref(directory / "response.json"),
                        conditioning=ref(directory / "conditioning.json"),
                        initial=initial,
                    )
                )
                print(
                    json.dumps(
                        dict(
                            generations_completed=len(result["generations"]),
                            name=entry["name"],
                            denoise_calls=len(counter.calls),
                        )
                    ),
                    flush=True,
                )
                active_variant = None
        if len(counter.calls) != 3360 or head_counts != dict(raw=1680, adapt=1680):
            raise ValueError("025 sampling did not finish its registered96/3360 budget")
        if parameter_versions(model) != before_parameters:
            raise ValueError("025 original parameters changed during adapted sampling")
        for name in ("input_sha256", "source_sha256", "snapshot_sha256"):
            check_hashes(prepared[name])
        check_hashes(checked_json(args.fit_verification, args.fit_verification_sha256)["input_sha256"])
        check_weights(prepared["weight_files"])
        result.update(
            complete=True,
            passed=True,
            all_original_parameters_frozen=True,
            head_call_counts=head_counts,
            runtime=service.info,
        )
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if head_observer is not None:
            head_observer.remove()
        if capture is not None:
            capture.remove()
        if hook is not None:
            hook.remove()
        result["denoise_calls"] = 0 if counter is None else len(counter.calls)
        result["denoise_ledger"] = [] if counter is None else counter.calls
        write_new(output / "result.json", result)


if __name__ == "__main__":
    main()
