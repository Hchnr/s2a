"""Fixed056 aggregate validation matrix; current-only, no model initialization."""

import copy

import numpy as np

from s2a.clean_training_runtime import require, validate_current_request
from s2a.synchronous_state_sampling import CURRENT_KEYS


def paired_entries(current_inputs, template, zero_cases, qualification):
    """Return exactly3 historical zero regressions then48 complete raw/probe pairs."""
    require(
        len(current_inputs) == 24 and len({a["name"] for a in current_inputs}) == 24,
        "All24 unique current anchors required",
    )
    require(len(zero_cases) == 3, "All3 exact historical zero regressions required")
    entries = []
    gold = [(i, g) for i, g in enumerate(qualification["generations"]) if g["variant"] == "raw"]
    require(len(gold) == 3 and [i for i, _ in gold] == [0, 1, 2], "Original3 qualified raw cases required")
    for case, (index, original) in zip(zero_cases, gold, strict=True):
        require(
            case["name"] == original["case_id"] and original["historical_output_exact"] is True, "Zero identity differs"
        )
        request = copy.deepcopy(case["request"])
        validate_current_request(
            request,
            name=request["name"],
            seed=42,
            image_ref=case["image_ref"],
            task=request["prompt"],
            template=case["request"],
        )
        entries.append(
            {
                "index": len(entries),
                "name": "zero_" + case["name"],
                "variant": "zero",
                "validation_index": None,
                "anchor_name": None,
                "zero_case_name": case["name"],
                "pair_id": case["name"],
                "sampling_seed": 42,
                "image_ref": case["image_ref"],
                "request": request,
                "qualified_generation_index": index,
                "expected_response_ref": case["expected_response_ref"],
                "qualified_response_ref": original["response_ref"],
            }
        )
    for anchor_index, anchor in enumerate(current_inputs):
        require(
            set(anchor) == CURRENT_KEYS and anchor["initial_sync_certified"] is True,
            "Current-only synchronized schema differs",
        )
        state, mask = np.asarray(anchor["measured_state128"]), np.asarray(anchor["state_mask128"])
        require(
            state.shape == (128,)
            and np.isfinite(state).all()
            and mask.dtype == np.bool_
            and np.array_equal(mask, np.arange(128) < 10)
            and np.isfinite(anchor["current_time_s"]),
            "Current measured state/mask differs",
        )
        for seed in (42, 43):
            raw_index = len(entries)
            for variant in ("raw", "terminal058"):
                name = f"{anchor['name']}_{variant}_s{seed}"
                request = copy.deepcopy(template)
                request.update(name=name, seed=seed, vision_path=anchor["image_ref"]["path"], prompt=anchor["task"])
                request["extra"]["action128"]["task_title"] = anchor["task"]
                validate_current_request(
                    request, name=name, seed=seed, image_ref=anchor["image_ref"], task=anchor["task"], template=template
                )
                entries.append(
                    {
                        "index": len(entries),
                        "validation_index": len(entries) - 3,
                        "anchor_index": anchor_index,
                        "name": name,
                        "variant": variant,
                        "anchor_name": anchor["name"],
                        "zero_case_name": None,
                        "pair_id": f"{anchor['name']}_s{seed}",
                        "sampling_seed": seed,
                        "image_ref": anchor["image_ref"],
                        "request": request,
                        "paired_raw_generation_index": raw_index if variant == "terminal058" else None,
                    }
                )
    require(len(entries) == len({e["name"] for e in entries}) == 99, "056 aggregate generation identity collision")
    return entries
