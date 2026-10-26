"""Bind generated references to the fixed development case and current request.

This is a CPU transport check, not physical qualification or model execution
admission. The caller must separately verify the service, its sources, GPU UUID,
and the completed 18-case control gate before issuing any model request.
"""

from s2a.stability_recovery_simulation import validate_wam_response
from s2a.terminal_development_policy import case_order

TERMINAL_SHA = "30210a2bda69962bbd24f8e86849f58c4516d2209bfb3d8a5903fc45ae543b4f"
REQUEST_KEYS = frozenset(
    (
        "name",
        "model_mode",
        "prompt",
        "vision_path",
        "domain_name",
        "image_size",
        "action_chunk_size",
        "fps",
        "num_frames",
        "seed",
        "num_steps",
        "guidance",
        "shift",
        "extra",
    )
)


def validate_preview_wam_response(response, request, *, case, request_index, checkpoint_path, readout_ref):
    """Reject stale/cross-case responses before a generated chunk reaches control.

    ``request`` must be the actual current-image request saved by the caller,
    and ``case`` must come from the independently registered fixed case ledger.
    No label, expert trajectory, or next physical observation is read here.
    """
    index = case.get("case_index")
    if type(index) is not int or not 0 <= index < 12 or case != case_order()[index]:
        raise ValueError("Fixed development case identity required")
    if type(request_index) is not int or not 0 <= request_index < 75:
        raise ValueError("Fixed per-case request index required")
    if set(request) != REQUEST_KEYS or any(response["args"].get(k) != v for k, v in request.items()):
        raise ValueError("Response differs from the actual current request")
    if (
        request["seed"] != case["sampling_seed"]
        or request["name"] != f"s2a_{case['record_name']}_s{case['sampling_seed']}_{request_index}"
        or request["action_chunk_size"] != 40
        or request["fps"] != 20
    ):
        raise ValueError("Current request differs from fixed case/chunk contract")
    if (
        not isinstance(readout_ref, dict)
        or set(readout_ref) != {"path", "sha256"}
        or readout_ref["sha256"] != TERMINAL_SHA
    ):
        raise ValueError("Fixed terminal058 parameter identity required")
    expected = dict(
        case_index=index,
        case_id=case["case_id"],
        request_index=request_index,
        global_request_index=index * 75 + request_index,
        variant=case["variant"],
        sampling_seed=case["sampling_seed"],
        readout_ref=readout_ref if case["variant"] == "terminal058" else None,
    )
    provenance = response.get("clean_training")
    if provenance != expected or any(
        type(provenance[k]) is not int for k in ("case_index", "request_index", "global_request_index", "sampling_seed")
    ):
        raise ValueError("Response case/variant/readout/request identity differs")
    actions, identity = validate_wam_response(
        response,
        checkpoint_path=checkpoint_path,
        seed=case["sampling_seed"],
        domain=request["extra"]["action128"]["action_domain"],
        fps=20,
        horizon=40,
    )
    return actions, dict(identity, development_response=expected)
