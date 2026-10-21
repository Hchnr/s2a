"""Independent CPU verification of024's recorded training/inference tensors.

NumPy implements the recorded external rectified-flow equations and explicit
valid-mask loss branch. No producer math, Cosmos modules, torch or dataloader is
imported. Tensor recording/forward execution belongs to the separate producer.
"""

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), f"{name}: requires finite numeric data")
    return array


def exact(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(actual.shape == expected.shape and np.array_equal(actual, expected), f"{name}: elementwise exact mismatch")


def cast_model_precision(value, dtype):
    """Round float32 to recorded model precision without importing torch.

    BF16 uses round-to-nearest, ties-to-even on the retained upper16 bits.
    Values are re-expanded losslessly to float32 for ordinary NPZ storage.
    """
    value = numeric(value, "precision input").astype(np.float32)
    if dtype in ("torch.float32", "float32"):
        return value.copy()
    if dtype in ("torch.float16", "float16"):
        result = value.astype(np.float16).astype(np.float32)
    elif dtype in ("torch.bfloat16", "bfloat16"):
        bits = value.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
        result = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    else:
        raise ValueError(f"unsupported recorded model dtype: {dtype}")
    require(np.isfinite(result).all(), "precision cast overflowed")
    return result


def expanded_mask(mask, shape, name):
    mask = numeric(mask, name)
    require(np.all((mask == 0) | (mask == 1)), f"{name}: requires binary mask")
    try:
        return np.broadcast_to(mask, shape)
    except ValueError as error:
        raise ValueError(f"{name}: cannot broadcast {mask.shape} to {shape}") from error


def flow_equations(x0, epsilon, condition, valid, sigma, *, modality, model_dtype):
    """epsilon-x0 target; xt=sigma_eff*epsilon+(1-sigma_eff)*x0.

    This follows rectified_flow.get_interpolation and the caller's subsequent
    dtype cast/Action128 valid-mask zeroing. Conditioning vision targets remain
    epsilon-x0; conditioning is excluded by the loss, not by changing targets.
    """
    x0, epsilon = numeric(x0, "x0").astype(np.float32), numeric(epsilon, "epsilon").astype(np.float32)
    require(x0.shape == epsilon.shape and x0.size > 0, "flow: x0/epsilon shape mismatch")
    require(modality in ("vision", "action"), "flow: unsupported modality")
    require(x0.ndim == (4 if modality == "vision" else 2), "flow: wrong modality rank")
    sigma = np.float32(sigma)
    require(np.isfinite(sigma) and 0 < sigma < 1, "flow: invalid fixed sigma")
    condition = numeric(condition, "condition").astype(np.float32)
    expanded_mask(condition, x0.shape, "condition")
    effective_sigma = sigma * (np.float32(1) - condition)
    target = epsilon - x0
    xt = cast_model_precision(epsilon * effective_sigma + x0 * (np.float32(1) - effective_sigma), model_dtype)
    if modality == "action":
        require(valid is not None, "flow: training action needs explicit valid mask")
        active = expanded_mask(valid, x0.shape, "valid").astype(bool)
        xt = np.where(active, xt, np.float32(0))
        target = np.where(active, target, np.float32(0))
    return dict(xt=xt, target=target, effective_sigma=effective_sigma)


def explicit_mask_loss(prediction, target, condition, valid, timesteps, *, modality, time_weight_method):
    """Rebuild the explicit loss_valid_mask branch, including its denominator.

    The current external implementation supports only uniform RF time weights
    (legacy 'reweighting' aliases uniform). The valid-mask branch always divides
    by active broadcast elements, independent of normalize_loss_by_active.
    """
    prediction, target = numeric(prediction, "prediction"), numeric(target, "target")
    require(prediction.shape == target.shape and prediction.size > 0, "loss: prediction/target shape mismatch")
    require(modality in ("vision", "action"), "loss: unsupported modality")
    require(prediction.ndim == (4 if modality == "vision" else 2), "loss: wrong modality rank")
    require(valid is not None, "loss: explicit valid mask is required")
    condition = numeric(condition, "condition").astype(np.float32)
    condition_broadcast = expanded_mask(condition, prediction.shape, "condition")
    valid_broadcast = expanded_mask(valid, prediction.shape, "valid")
    active = ((1 - condition_broadcast) * valid_broadcast).astype(bool)
    denominator = max(int(np.count_nonzero(active)), 1)
    # Subtraction/square execute in float32 in the source (BF16 pred minus
    # float32 target). Accumulate independently in float64 within declared
    # float32-reduction tolerances; never use this tolerance for B/C equality.
    difference = prediction.astype(np.float32) - target.astype(np.float32)
    squared = np.square(np.where(active, difference, np.float32(0)), dtype=np.float32)
    require(time_weight_method in ("uniform", "reweighting"), "loss: unregistered time weight")
    times = numeric(timesteps, "timesteps").reshape(-1)
    temporal_count = prediction.shape[1 if modality == "vision" else 0]
    require(len(times) in (1, temporal_count), "loss: incompatible timestep count")
    weights = np.ones(times.shape, dtype=np.float32)
    time_shape = (1, len(times), 1, 1) if modality == "vision" else (len(times), 1)
    weighted = squared * weights.reshape(time_shape)
    return dict(
        denominator=denominator,
        active_elements=int(np.count_nonzero(active)),
        loss=float(np.sum(weighted, dtype=np.float64) / denominator),
        per_instance_unweighted=float(np.sum(squared, dtype=np.float64) / denominator),
        time_weight=weights,
    )


def registered_call_grid(records):
    """Require the registered nine triplets/27 denoise calls without duplicates."""
    expected = {(task, sigma, branch) for task in range(3) for sigma in (0.2, 0.5, 0.8) for branch in ("A", "B", "C")}
    actual = [(record["task_index"], record["sigma"], record["branch"]) for record in records]
    require(len(actual) == 27 and len(set(actual)) == 27 and set(actual) == expected, "wrong registered 27-call grid")
