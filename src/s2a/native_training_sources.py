"""030 native source selection; no filesystem, simulator or training dependencies."""

import copy
import hashlib
import re

import numpy as np


def initial_state_hash(value):
    """Use the existing physical-fixture float64 state identity convention."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("native initial state must be a nonempty finite vector")
    return hashlib.sha256(array.tobytes()).hexdigest()


def numeric_demo_order(names):
    values = list(names)
    if len(set(values)) != len(values) or any(
        not isinstance(name, str) or re.fullmatch(r"demo_(0|[1-9][0-9]*)", name) is None for name in values
    ):
        raise ValueError("requires unique canonical native demo IDs")
    return sorted(values, key=lambda name: int(name.removeprefix("demo_")))


def training_anchor_frames(native_steps):
    """Four complete native windows and the independent terminal-hold window."""
    if type(native_steps) is not int or not 43 <= native_steps <= 600:
        raise ValueError("selected native length must be43..600; replacement is forbidden")
    result = [j * (native_steps - 40) // 3 for j in range(4)] + [native_steps]
    if len(set(result)) != 5 or any(frame + 40 > native_steps + 40 for frame in result):
        raise ValueError("five distinct fullH40 anchors required")
    return result


def select_training_sources(candidates_by_task, excluded_by_task, forbidden_initial_hashes, *, count=10, seed=42):
    """Permute the numeric ID roster, then skip only physical-identity aliases.

    Bounds/quality/action values do not participate in selection. In particular,
    an invalid shape in the selected roster must fail later, never be replaced.
    """
    if seed != 42 or type(count) is not int or count < 1 or len(candidates_by_task) != len(excluded_by_task):
        raise ValueError("requires the registered seed and aligned task source lists")
    rng = np.random.default_rng(seed)
    forbidden = set(forbidden_initial_hashes)
    selected, traces = [], []
    for task_index, (candidates, excluded) in enumerate(zip(candidates_by_task, excluded_by_task, strict=True)):
        excluded = set(excluded)
        available = [item for item in candidates if item["demo"] not in excluded]
        names = numeric_demo_order(item["demo"] for item in available)
        by_name = {item["demo"]: item for item in available}
        order = [names[int(index)] for index in rng.permutation(len(names))]
        chosen, skipped = [], []
        for name in order:
            item = by_name[name]
            identity = item["initial_state_sha256"]
            if identity in forbidden:
                skipped.append(name)
                continue
            chosen.append(copy.deepcopy(item))
            forbidden.add(identity)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("insufficient identity-isolated sources; no smaller training roster authorized")
        selected.append(chosen)
        traces.append(
            dict(
                task_index=task_index,
                excluded_demo_ids=numeric_demo_order(excluded),
                numeric_demo_order=names,
                permutation_demo_order=order,
                skipped_identity_aliases=skipped,
                selected_demo_ids=[item["demo"] for item in chosen],
                candidates=[
                    {
                        key: by_name[name][key]
                        for key in ("demo", "initial_state_sha256", "model_xml_sha256", "actions_shape")
                    }
                    for name in names
                ],
            )
        )
    return selected, dict(seed=seed, tasks=traces)


def validate_selected_shapes(selected):
    """Validate the entire fixed roster, after all task permutations are done."""
    for task in selected:
        for record in task:
            shape = record["actions_shape"]
            if len(shape) != 2 or type(shape[0]) is not int or shape[1] != 7:
                raise ValueError("selected native actions must be N×7; replacement is forbidden")
            training_anchor_frames(shape[0])
