"""Serve previously audited references without claiming new model inference."""

import copy
import math
import re

from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KEYS = {
    "step",
    "sim_time_s",
    "state_sha256",
    "image_sha256",
    "response_sha256",
    "historical_denoise_calls",
    "action",
    "normalized_action",
}


def _sha(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _matrix(value):
    return (
        type(value) is list
        and len(value) == 40
        and all(
            type(row) is list
            and len(row) == 128
            and all(_finite(item) for item in row)
            and all(item == 0 for item in row[10:])
            for row in value
        )
    )


class RecordedReferenceRelay:
    """Match a current request to exactly one source-verified saved response.

    The caller must independently verify each recorded response SHA, historical
    model identity and denoise ledger before constructing this relay. Metadata
    and denoise counts in the response describe that historical inference; this
    class makes no new model calls and never claims new denoise work.
    """

    def __init__(self, admission, records):
        if (
            type(admission) is not CausalModelAdmission
            or admission.phase != "ready"
            or admission.admitted_requests != 0
            or type(records) is not list
            or len(records) != admission.max_requests
        ):
            raise ValueError("One fresh service admission and its full registered response roster are required")
        for index, record in enumerate(records):
            if (
                type(record) is not dict
                or set(record) != _KEYS
                or type(record["step"]) is not int
                or record["step"] != admission.initial["step"] + index * 8
                or not _finite(record["sim_time_s"])
                or not math.isclose(
                    record["sim_time_s"], admission.initial["sim_time_s"] + index * 0.4, abs_tol=1e-9, rel_tol=0
                )
                or not all(_sha(record[key]) for key in ("state_sha256", "image_sha256", "response_sha256"))
                or type(record["historical_denoise_calls"]) is not int
                or record["historical_denoise_calls"] != admission.denoise_steps
                or not _matrix(record["action"])
                or not _matrix(record["normalized_action"])
            ):
                raise ValueError("Every saved response must match the registered causal roster")
        self.admission = admission
        self.records = copy.deepcopy(records)
        self.used = []
        self.failed = False

    def respond(self, request):
        try:
            index = len(self.used)
            if self.failed or index >= len(self.records):
                raise ValueError("No saved reference remains in this session")
            record = self.records[index]
            if (
                type(request) is not dict
                or request.get("request_id") != index
                or request.get("step") != record["step"]
                or request.get("sim_time_s") != record["sim_time_s"]
                or request.get("state_sha256") != record["state_sha256"]
                or type(request.get("image")) is not dict
                or request["image"].get("sha256") != record["image_sha256"]
            ):
                raise ValueError("The current observation differs from the saved reference binding")
            inputs = self.admission.admit(request)
            identity = {
                key: request[key]
                for key in (
                    "version",
                    "session",
                    "request_id",
                    "step",
                    "sim_time_s",
                    "state_sha256",
                    "seed",
                    "model_sha256",
                )
            }
            response = {
                **identity,
                "image_sha256": record["image_sha256"],
                "denoise_calls": record["historical_denoise_calls"],
                "action_dim_mask": [True] * 10 + [False] * 118,
                "action": copy.deepcopy(record["action"]),
                "normalized_action": copy.deepcopy(record["normalized_action"]),
            }
            self.admission.complete_recorded(
                request_id=index, historical_denoise_calls=record["historical_denoise_calls"]
            )
            self.used.append(
                {
                    "request_id": index,
                    "step": record["step"],
                    "historical_response_sha256": record["response_sha256"],
                    "historical_denoise_calls": record["historical_denoise_calls"],
                    "new_model_requests": 0,
                    "new_denoise_calls": 0,
                }
            )
            return response, inputs
        except BaseException:
            self.failed = True
            raise
