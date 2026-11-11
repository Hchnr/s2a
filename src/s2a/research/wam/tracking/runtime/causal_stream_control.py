"""Validate a minimal producer stop without exposing task-success labels."""

from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalModelAdmission


def make_stop(*, session, next_step):
    """Build a stop frame; the service validates its position independently."""
    if type(session) is not str or not 0 < len(session) <= 128 or type(next_step) is not int or next_step < 0:
        raise ValueError("A registered session and nonnegative stop step are required")
    return {"version": 1, "kind": "stop", "session": session, "next_step": next_step}


def check_stop(service, frame):
    """Accept only a stop while the service is waiting for its next request.

    The service knows only how many responses it sent. A stop can occur at any
    completed step within the most recently served window, including its end.
    The actual cause and completed actions remain in the producer receipt.
    """
    if type(service) is not CausalModelAdmission or type(frame) is not dict:
        raise ValueError("A model service and explicit stop frame are required")
    count = service.admitted_requests
    first = service.initial["step"]
    lower = first if count == 0 else first + 8 * (count - 1)
    upper = first + 8 * count
    if (
        service.phase != "ready"
        or count >= service.max_requests
        or set(frame) != {"version", "kind", "session", "next_step"}
        or type(frame["version"]) is not int
        or frame["version"] != 1
        or frame["kind"] != "stop"
        or frame["session"] != service.session
        or type(frame["next_step"]) is not int
        or not lower <= frame["next_step"] <= upper
    ):
        raise ValueError("Stop frame differs from the waiting registered stream")
    return dict(frame)
