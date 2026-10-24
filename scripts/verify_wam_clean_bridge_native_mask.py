"""Read-only039 re-audit of037 captures with the native inactive-slot mask.

The frozen verifier still performs all arithmetic, source, budget, raw/zero,
historical-output and245-point checks. This wrapper replaces only its initial
conditioning check and routes its final receipt to a new audit root. Native
inactive118 mask ones clamp zero padding; active10 remain unconditioned.
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import verify_wam_clean_bridge as original

from s2a.artifacts import REPO

AUDIT_ROOT = REPO / "outputs/wam_stability_20261021/clean_bridge_native_mask_audit_20261023"
VISION_SHAPE = (1, 48, 11, 60, 40)
CURRENT_SHAPE = (1, 48, 1, 60, 40)
VISION_VALUES = 1 * 48 * 11 * 60 * 40
TOTAL_VALUES = VISION_VALUES + 40 * 128


def verify_initial(metadata, arrays, generation, conditioning, first_point):
    """Bind the full native mask and current frame, without admitting a future.

    Metadata, array roster, FP32 dtype and initial-sample binding requirements
    from the original checker are retained explicitly. Additional exact layout,
    current-frame and zero-padding checks apply to the actual saved arrays.
    """
    require, bits = original.require, original.bits
    require(
        set(arrays) == {"initial_noise", "condition_mask", "first_condition_latent"},
        "initial capture array roster differs",
    )
    require(
        all(isinstance(value, np.ndarray) and value.dtype == np.float32 for value in arrays.values()),
        "initial sample/condition must retain actual FP32",
    )
    require(
        {"sample_fp32_flat", "sample_full128_fp32"} <= set(first_point),
        "first-point actual sampler arrays missing",
    )
    require(
        metadata["name"] == generation["name"]
        and metadata["sampling_seed"] == generation["sampling_seed"]
        and metadata["variant"] == generation["variant"]
        and metadata["generation_index"] == generation["index"]
        and metadata["inference_only_current_image"] is True
        and metadata["caption"] == conditioning["caption"],
        "actual initial current-image request binding differs",
    )
    require(
        arrays["condition_mask"].shape
        == arrays["initial_noise"].shape
        == first_point["sample_fp32_flat"].shape
        == (TOTAL_VALUES,),
        "initial mask/noise layout differs from the frozen native vision/action geometry",
    )
    require(
        first_point["sample_fp32_flat"].dtype == np.float32
        and first_point["sample_full128_fp32"].dtype == np.float32
        and first_point["sample_full128_fp32"].shape == (40, 128),
        "first actual sampler action must retain FP32/full40x128 layout",
    )
    bits(
        arrays["initial_noise"],
        first_point["sample_fp32_flat"],
        "first actual sampler latent differs from saved initial FP32 sample",
    )
    require(arrays["first_condition_latent"].shape == CURRENT_SHAPE, "current-frame condition latent geometry differs")
    vision_mask = np.zeros(VISION_SHAPE, dtype=np.float32)
    vision_mask[:, :, :1] = 1
    action_mask = np.ones((40, 128), dtype=np.float32)
    action_mask[:, :10] = 0
    expected = np.concatenate((vision_mask.reshape(-1), action_mask.reshape(-1)))
    bits(
        arrays["condition_mask"],
        expected,
        "native mask requires current-frame vision only, active10 zero and inactive118 one; no privileged future",
    )
    initial_action = arrays["initial_noise"][VISION_VALUES:].reshape(40, 128)
    first_action = first_point["sample_full128_fp32"]
    bits(initial_action, first_action, "first-point action slice is not bound to the actual initial flat sample")
    zeros = np.zeros((40, 118), dtype=np.float32)
    bits(initial_action[:, 10:], zeros, "initial inactive118 is not actually zero-clamped")
    bits(first_action[:, 10:], zeros, "first-point inactive118 is not actually zero-clamped")
    current = arrays["initial_noise"][:VISION_VALUES].reshape(VISION_SHAPE)[:, :, :1]
    bits(
        arrays["first_condition_latent"], current, "saved current-image latent differs from the conditioned first frame"
    )


def verify(args):
    """Run the unmodified verifier with scoped initial-check/output routing.

    Every read first calls the original full manifest/source/SHA/root verifier;
    only a deep-copied root field is remapped for its final output guard. Config,
    input references, qualification identity and on-disk manifests stay intact.
    All patched module globals are restored even on a verification error.
    """
    require = original.require
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "native-mask re-audit requires seed42 and empty CUDA/LD",
    )
    root = AUDIT_ROOT.resolve()
    output = Path(args.output).absolute()
    require(
        root.is_relative_to((REPO / "outputs").resolve())
        and output == root / "review/verified.json"
        and output.resolve() == output,
        "native-mask audit output escaped the registered new root",
    )
    require(not output.parent.exists(), "refuses existing review output or retry")
    originals = {name: getattr(original, name) for name in ("verify_initial", "read_prepared", "publish")}
    require(originals["verify_initial"] is not verify_initial, "native-mask verifier is not reentrant")
    source_root = None

    def routed_prepared(*values, **kwargs):
        nonlocal source_root
        actual = originals["read_prepared"](*values, **kwargs)
        actual_root = Path(actual["root"]).resolve()
        require(actual_root != root, "cannot route an audit into the original prepared root")
        require(source_root is None or actual_root == source_root, "original prepared root changed between reads")
        source_root = actual_root
        view = copy.deepcopy(actual)
        view["root"] = str(root)
        return view

    def routed_publish(path, receipt):
        require(Path(path).absolute() == output and source_root is not None, "unexpected original verifier write")
        value = dict(receipt)
        value["native_initial_mask_audit"] = dict(
            contract="current_vision_frame_only_active10_unconditioned_inactive118_zero_clamped",
            original_verifier_ref=original.reference(Path(original.__file__).resolve()),
            wrapper_ref=original.reference(Path(__file__).resolve()),
            original_prepared_root=str(source_root),
            audit_root=str(root),
            output_only_in_memory_root_routing=True,
            new_model_initializations=0,
            new_denoise_calls=0,
            new_fit_groups=0,
            new_physics_steps=0,
            new_fresh_cases=0,
        )
        return originals["publish"](path, value)

    try:
        original.verify_initial = verify_initial
        original.read_prepared = routed_prepared
        original.publish = routed_publish
        return original.verify(args)
    finally:
        for name, value in originals.items():
            setattr(original, name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
