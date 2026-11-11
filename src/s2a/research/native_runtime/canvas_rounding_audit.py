"""Inspect integer-valued bilinear interpolation ties for the fixed WAM canvas."""

import numpy as np


def bilinear_half_integer_mask(images):
    """Mark exact rational half-integers before rounding, excluding the seam.

    The registered canvas cover-resizes each 256-square image to 640-square
    with align_corners=False and takes the central 480 rows. Input images have
    already received the original two-axis camera flip.
    """
    if len(images) != 2:
        raise ValueError("Two camera images are required")
    x, y = np.arange(640), np.arange(80, 560)
    x_num, y_num = 4 * x - 3, 4 * y - 3
    xi, yi = np.floor_divide(x_num, 10), np.floor_divide(y_num, 10)
    xf, yf = x_num % 10, y_num % 10
    x0, x1, y0, y1 = np.clip(xi, 0, 255), np.clip(xi + 1, 0, 255), np.clip(yi, 0, 255), np.clip(yi + 1, 0, 255)
    masks = []
    for image in images:
        image = np.asarray(image)
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError("Each original flipped camera must be 256x256 RGB uint8")
        z = image.astype(np.int64)
        numerator = (
            z[y0[:, None], x0[None, :]] * (10 - yf)[:, None, None] * (10 - xf)[None, :, None]
            + z[y0[:, None], x1[None, :]] * (10 - yf)[:, None, None] * xf[None, :, None]
            + z[y1[:, None], x0[None, :]] * yf[:, None, None] * (10 - xf)[None, :, None]
            + z[y1[:, None], x1[None, :]] * yf[:, None, None] * xf[None, :, None]
        )
        masks.append(numerator % 100 == 50)
    mask = np.concatenate(masks, axis=0)
    mask[478:482] = False
    return mask


def compare_canvases(recorded, saved, one_thread, two_threads, tie_mask):
    """Compare exact pixels and count where one-thread changes lie."""
    arrays = [np.asarray(value) for value in (recorded, saved, one_thread, two_threads)]
    if any(value.shape != (960, 640, 3) or value.dtype != np.uint8 for value in arrays):
        raise ValueError("Four 960x640 RGB uint8 canvases are required")
    ties = np.asarray(tie_mask)
    if ties.shape != (960, 640, 3) or ties.dtype != np.bool_:
        raise ValueError("A matching boolean tie mask is required")
    recorded, saved, one, two = arrays
    delta = np.abs(recorded.astype(np.int16) - one.astype(np.int16))
    changed = delta != 0
    return {
        "one_matches_saved": bool(np.array_equal(one, saved)),
        "two_matches_recorded": bool(np.array_equal(two, recorded)),
        "different_channel_values": int(changed.sum()),
        "different_pixels": int(np.any(changed, axis=2).sum()),
        "max_channel_difference": int(delta.max()),
        "different_values_on_exact_ties": int(np.count_nonzero(changed & ties)),
    }
