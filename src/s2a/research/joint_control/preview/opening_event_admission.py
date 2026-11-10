"""Causal reference alignment and recorded-state admission for release feedback."""

from xml.etree import ElementTree

import numpy as np

from s2a.geometry import validate_transform

JOINT_NAMES = tuple(f"robot0_joint{index}" for index in range(1, 8))


def recorded_joint_ranges(xml):
    """Read the seven original robot joint ranges without creating a model."""
    root = ElementTree.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Original XML must contain a worldbody")
    joints = [item for item in world.iter() if item.tag in ("joint", "freejoint")]
    if tuple(item.get("name") for item in joints[:7]) != JOINT_NAMES:
        raise ValueError("The first seven world joints must be the recorded robot joints")
    for name in JOINT_NAMES:
        if sum(item.get("name") == name for item in joints) != 1:
            raise ValueError("Each recorded robot joint must occur exactly once")
    try:
        limits = np.asarray([[float(value) for value in joint.attrib["range"].split()] for joint in joints[:7]])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each robot joint requires a finite range") from error
    if limits.shape != (7, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Each robot joint requires an ordered finite range")
    if any(item.tag != "joint" or item.get("type", "hinge") not in ("hinge", "slide") for item in joints[:7]):
        raise ValueError("Robot joint positions must each occupy one qpos slot")
    return limits


def admit_recorded_state(q, limits, *, margin_rad):
    """Apply the same strict current-state buffer used before action dispatch."""
    q, limits = np.asarray(q, dtype=float), np.asarray(limits, dtype=float)
    if (
        q.shape != (7,)
        or limits.shape != (7, 2)
        or not np.isfinite(q).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or isinstance(margin_rad, bool)
        or not np.isfinite(margin_rad)
        or margin_rad <= 0
    ):
        raise ValueError("Finite seven-joint state, limits and positive buffer required")
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    return {
        "joint_names": list(JOINT_NAMES),
        "joint_positions_rad": q.copy(),
        "joint_limits_rad": limits.copy(),
        "joint_margins_rad": margins,
        "minimum_margin_rad": float(np.min(margins)),
        "buffer_rad": float(margin_rad),
        "recorded_endpoint_admitted": bool(np.min(margins) >= margin_rad),
    }


def aligned_opening_references(spec, old, latest, shadow, checked, measured_tcp, measured_opening):
    """Bind paired references to a previously verified first opening intervention."""
    n, source = spec["prefix_steps"], spec["source_step"]
    event, horizon = spec["event_endpoint"], 8
    if (
        type(n) is not int
        or type(source) is not int
        or type(event) is not int
        or n <= source
        or source % horizon
        or n % horizon
        or not n < event <= n + horizon
        or old["step"] != source
        or latest["step"] != n
        or checked["first_intervention_step"] != n
        or checked["case"]["case_index"] != spec["case_index"]
        or checked["case"]["case_id"] != spec["case_id"]
    ):
        raise ValueError("Original case, first intervention and absolute event timing required")
    offset = n - source
    indices = list(range(offset, offset + horizon))
    if indices[-1] >= 40:
        raise ValueError("Remembered reference must cover all selected endpoints")
    if (
        shadow["step"] != n
        or shadow["source"] != "remembered"
        or shadow["source_step"] != source
        or shadow["source_indices"] != indices
        or shadow["absolute_endpoints"] != list(range(n + 1, n + horizon + 1))
        or shadow["event"]["direction"] != "opening"
        or shadow["event"]["absolute_endpoint"] != event
        or shadow["baseline_observation_label"] != "first_intervention_on_baseline"
    ):
        raise ValueError("A verified first opening intervention with aligned endpoints is required")
    measured = validate_transform(measured_tcp)
    opening = float(measured_opening)
    if measured.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening < 0.35:
        raise ValueError("Current TCP and closed measured opening are required")
    references = {}
    for name, chunk, cursor in (("remembered", old, offset), ("latest", latest, 0)):
        poses = validate_transform(chunk["effective_targets_tcp"])
        openings = np.asarray(chunk["openings"], dtype=float)
        anchor = validate_transform(chunk["anchor_tcp"])
        if (
            poses.shape != (40, 4, 4)
            or openings.shape != (40,)
            or anchor.shape != (4, 4)
            or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 1))
        ):
            raise ValueError("Complete finite paired generated references are required")
        references[name] = {
            "poses": poses.copy(),
            "openings": openings.copy(),
            "anchor": anchor.copy(),
            "cursor": cursor,
        }
    np.testing.assert_allclose(references["latest"]["anchor"], measured, atol=1e-10, rtol=0)
    np.testing.assert_allclose(latest["anchor_opening"], opening, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(shadow["poses"], references["remembered"]["poses"][indices])
    np.testing.assert_array_equal(shadow["openings"], references["remembered"]["openings"][indices])
    if references["remembered"]["openings"][event - source - 1] <= 0.8:
        raise ValueError("The remembered reference must express the registered opening")
    return references
