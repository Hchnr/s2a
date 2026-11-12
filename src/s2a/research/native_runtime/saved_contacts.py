"""Offline inspection of target contacts at one reconstructed MuJoCo state."""

import math


def target_contacts(model, data, *, mujoco, body_id, target_names, left_pad, right_pad):
    """Return native contact pairs, keeping pad sides and other objects separate.

    The contact-list semantics deliberately match robosuite's check_contact:
    every listed pair counts regardless of the signed distance or contact force.
    """
    names = tuple(target_names)
    if not names or len(set(names)) != len(names) or left_pad == right_pad or {left_pad, right_pad} & set(names):
        raise ValueError("Distinct target geoms and two different pads are required")
    kind = mujoco.mjtObj.mjOBJ_GEOM
    ids = {name: int(mujoco.mj_name2id(model, kind, name)) for name in (*names, left_pad, right_pad)}
    if any(value < 0 for value in ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError("Every registered target geom and pad must resolve uniquely")
    target_ids = {ids[name] for name in names}
    actual = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == body_id and int(model.geom_group[i]) == 0}
    if actual != target_ids or any(int(model.geom_group[ids[pad]]) != 0 for pad in (left_pad, right_pad)):
        raise ValueError("Compiled target and pad groups must match the registered contact geometry")
    groups = {"left_pad": [], "right_pad": [], "other": []}
    pad_group = {ids[left_pad]: "left_pad", ids[right_pad]: "right_pad"}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        one, two = int(contact.geom1), int(contact.geom2)
        if (one in target_ids) == (two in target_ids):
            continue
        target, other = (one, two) if one in target_ids else (two, one)
        distance = float(contact.dist)
        if not math.isfinite(distance):
            raise ValueError("Contact distances must be finite")
        other_name = mujoco.mj_id2name(model, kind, other) if other >= 0 else None
        groups[pad_group.get(other, "other")].append(
            dict(
                target_geom=mujoco.mj_id2name(model, kind, target),
                other_geom=other_name if other_name is not None else f"unnamed_geom_{other}",
                distance_m=distance,
            )
        )
    return dict(**groups, bilateral=bool(groups["left_pad"] and groups["right_pad"]))


def summarize_contacts(points):
    """Summarize saved endpoint pairs without asserting dynamic grasp stability."""
    mismatches = []
    counts = dict(left_pad=0, right_pad=0, bilateral=0, other=0)
    for point in points:
        contacts = point["contacts"]
        for key in ("left_pad", "right_pad", "other"):
            counts[key] += bool(contacts[key])
        counts["bilateral"] += contacts["bilateral"]
        if contacts["bilateral"] != point["target_grasp"]:
            mismatches.append(point["endpoint_step"])
    return dict(endpoint_counts=counts, grasp_label_mismatch_steps=mismatches)
