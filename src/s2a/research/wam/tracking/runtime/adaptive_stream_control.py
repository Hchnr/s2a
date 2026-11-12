"""Close a completed adaptive producer without inventing a pending action."""

from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.research.wam.tracking.runtime.adaptive_stream_protocol import AdaptiveStreamLedger
from s2a.research.wam.tracking.runtime.causal_stream_control import make_stop


def close_adaptive_model_stream(channel, ledger, stream, receipt):
    """Send a bounded stop only when the model is waiting for another request.

    An interrupted in-flight action, incomplete feedback or transport failure
    remains the owner's failure to close and reap; it cannot be declared a
    clean pre-dispatch stop here.
    """
    if (
        type(channel) is not FramedJSONChannel
        or type(ledger) is not AdaptiveStreamLedger
        or type(stream) is not dict
        or type(receipt) is not dict
        or "adaptive_stop_frame" in receipt
    ):
        raise ValueError("A registered adaptive stream and fresh stop receipt are required")
    stop = stream.get("stop")
    current_step = ledger.current["step"]
    if type(stop) is not dict or stop.get("next_action_dispatched") is not False:
        raise ValueError("A recorded pre-dispatch or post-period stop is required")
    at_step, after_step = stop.get("at_step"), stop.get("after_step")
    boundary = (type(at_step) is int and after_step is None and at_step == current_step) or (
        type(after_step) is int and at_step is None and after_step + 1 == current_step
    )
    if (
        not boundary
        or ledger.inflight is not None
        or ledger.phase not in ("done", "failed")
        or stream.get("requests_issued") != ledger.issued_requests
        or stream.get("responses_accepted") != ledger.accepted_requests
        or ledger.issued_requests != ledger.accepted_requests
        or stream.get("control_steps_attempted") != ledger.completed_periods
        or stream.get("control_steps_completed") != ledger.completed_periods
        or stream.get("feedback_periods_committed") != ledger.completed_periods
        or any(row.get("inflight") is not None for row in stream.get("segments", []))
    ):
        raise ValueError("The stop must follow all completed dispatches and responses")
    if ledger.issued_requests == ledger.max_requests:
        receipt["adaptive_stop_frame"] = {"required": False, "reason": "model_request_budget_exhausted"}
        return
    frame = make_stop(session=ledger.session, next_step=current_step)
    receipt["adaptive_stop_frame"] = {"required": True, "frame": frame, "send_attempted": True, "sent": False}
    channel.send(frame)
    receipt["adaptive_stop_frame"]["sent"] = True
