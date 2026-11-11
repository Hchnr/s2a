"""Bind a live image, model response and feedback policy at each stream boundary."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.geometry import validate_transform
from s2a.infrastructure.framed_channel import FramedJSONChannel
from s2a.infrastructure.repository_paths import repo_path, repository_root
from s2a.references.action_reference import decode_action_targets
from s2a.research.joint_control.preview.streaming_feedback_boundary import advance_feedback
from s2a.research.native_runtime.streaming_continuation import StreamingSegment, measured_snapshot
from s2a.research.wam.tracking.runtime.causal_stream_protocol import CausalStreamLedger


def _write_json_once(path, value):
    body = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(body)
    return {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


class CausalStreamingReference:
    """Use current-only inputs and preserve the previous feedback policy state.

    The owner must separately register the actual runtime, device, source and
    physical budgets. A required response gate can enforce previously audited
    reference identities before any action is dispatched.
    """

    def __init__(
        self,
        *,
        env,
        observers,
        ledger,
        channel,
        runtime_output,
        tool,
        receipt,
        first_policy,
        response_gate,
        advance_policy=advance_feedback,
    ):
        if (
            type(ledger) is not CausalStreamLedger
            or type(channel) is not FramedJSONChannel
            or type(receipt) is not dict
            or not callable(first_policy)
            or not callable(response_gate)
            or not callable(advance_policy)
            or not callable(getattr(observers, "render_anchor", None))
        ):
            raise ValueError("A registered ledger, local channel, renderer and policy gates are required")
        root = repository_root()
        capture_root = repo_path(ledger.image_root, root=root)
        output = repo_path(runtime_output, root=root)
        if not output.is_relative_to(capture_root) or not output.is_dir():
            raise ValueError("The existing runtime output must be inside the registered capture directory")
        tool = validate_transform(np.asarray(tool, dtype=float)).copy()
        if tool.shape != (4, 4):
            raise ValueError("The fixed source-to-TCP tool must be one transform")
        if "streaming_model_exchanges" in receipt or "streaming_captures" in receipt:
            raise ValueError("A fresh per-run capture and exchange receipt is required")
        self.env, self.observers, self.ledger, self.channel = env, observers, ledger, channel
        self.capture_root, self.output, self.tool, self.receipt = capture_root, output, tool, receipt
        self.first_policy, self.response_gate, self.advance_policy = first_policy, response_gate, advance_policy
        self.policy, self.owner, self.previous = None, None, None
        self.receipt["streaming_captures"] = []
        self.receipt["streaming_model_exchanges"] = []

    def __call__(self, observation, previous):
        try:
            if (
                previous is not self.previous
                or type(observation) is not dict
                or set(observation) != {"snapshot", "source_pose", "opening"}
            ):
                raise ValueError("The last completed segment and one current observation are required")
            snapshot = dict(observation["snapshot"])
            step = snapshot["step"]
            if snapshot != self.ledger.current or measured_snapshot(self.env, step) != snapshot:
                raise ValueError("A current measured state is required before image capture")
            source_pose = validate_transform(np.asarray(observation["source_pose"], dtype=float)).copy()
            opening = float(observation["opening"])
            if source_pose.shape != (4, 4) or not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError("One current measured pose and opening are required")
            sample = {"source_pose": source_pose.copy(), "opening": opening}
            endpoint = {
                "endpoint_step": step,
                "sim_time": snapshot["sim_time_s"],
                "sim_state_sha256": snapshot["state_sha256"],
            }
            capture = self.observers.render_anchor(self.env, step, sample, endpoint, self.output)
            self.receipt["streaming_captures"].append(capture)
            binding = {"sim_time": snapshot["sim_time_s"], "sim_state_sha256": snapshot["state_sha256"]}
            if (
                type(capture) is not dict
                or capture.get("frame_index") != step
                or capture.get("current_endpoint") != endpoint
                or capture.get("render_before") != binding
                or capture.get("render_after") != binding
                or capture.get("future_rgb_frames") != 0
                or capture.get("camera_keys") != ["agentview_image", "robot0_eye_in_hand_image"]
                or capture.get("canvas_shape") != [960, 640, 3]
                or capture.get("canvas_dtype") != "uint8"
                or not np.array_equal(capture.get("current_source_pose"), source_pose)
                or capture.get("current_opening") != opening
                or measured_snapshot(self.env, step) != snapshot
            ):
                raise ValueError("Rendered cameras must bind the unchanged current state and clock")
            png = capture.get("canvas_png")
            if type(png) is not dict or set(png) != {"path", "sha256"} or type(png["path"]) is not str:
                raise ValueError("A recorded current image path and digest are required")
            image_path = Path(png["path"])
            if not image_path.is_absolute() or not image_path.is_relative_to(self.capture_root):
                raise ValueError("The current image must be under the registered capture directory")
            image = {"path": str(image_path.relative_to(self.capture_root)), "sha256": png["sha256"]}
            request = self.ledger.issue(before_render=snapshot, after_render=snapshot, image=image)
            exchange = {
                "request_id": request["request_id"],
                "step": step,
                "request_send_attempted": False,
                "request_sent": False,
                "response_receive_attempted": False,
                "response_received": False,
                "response_accepted": False,
                "policy_ready": False,
            }
            self.receipt["streaming_model_exchanges"].append(exchange)
            exchange["request_ref"] = _write_json_once(
                self.output / f"request_{request['request_id']:03d}.json", request
            )
            exchange["request_send_attempted"] = True
            self.channel.send(request)
            exchange["request_sent"] = True
            exchange["response_receive_attempted"] = True
            response = self.channel.receive()
            exchange["response_received"] = True
            exchange["response_ref"] = _write_json_once(
                self.output / f"response_{request['request_id']:03d}.json", response
            )
            after_wait = measured_snapshot(self.env, step)
            actions = self.ledger.accept(response, after_wait=after_wait)
            exchange["response_accepted"] = True
            if self.response_gate(copy.deepcopy(request), copy.deepcopy(response)) is not True:
                raise ValueError("The registered response identity gate rejected this reference")
            anchor = validate_transform(source_pose @ self.tool)
            poses, openings = decode_action_targets(anchor, actions, horizon=40)
            if self.policy is None:
                policy, owner = self.first_policy(poses.copy(), openings.copy(), anchor.copy(), step)
                if owner is None:
                    raise ValueError("A persistent feedback history owner is required")
                self.policy, self.owner = policy, owner
            else:
                segment_receipt = {}
                self.policy = self.advance_policy(
                    self.policy,
                    self.owner,
                    poses,
                    openings,
                    anchor,
                    generated_step=step,
                    receipt=segment_receipt,
                )
                self.receipt.setdefault("segment_receipts", []).append(segment_receipt)
            policy = self.policy
            if not callable(getattr(policy, "select", None)) or not callable(getattr(policy, "complete", None)):
                raise ValueError("A staged feedback policy with completion is required")

            def complete(env, command, offset, actual_inner):
                return policy.complete(env, command, offset, [item["q"] for item in actual_inner])

            segment = StreamingSegment(poses[:8].copy(), policy.select, complete)
            self.previous = segment
            exchange["policy_ready"] = True
            return segment
        except BaseException:
            self.ledger.abort()
            raise
