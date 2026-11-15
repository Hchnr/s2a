"""Capture only completed physical endpoints for training-recovery qualification."""

import copy
from pathlib import Path

import numpy as np

from s2a.geometry import validate_transform
from s2a.research.wam.training.native.native_training import (
    gripper_cache_receipt,
    pack_gripper_caches,
    state_sha256,
    synchronous_labels,
)

ANCHORS = (0, 16, 56)


class TrainingRecoveryCapture:
    """Collect a reset and consecutively completed observations, never targets."""

    def __init__(self, tool, directory, render_anchor):
        self.tool = validate_transform(np.asarray(tool, dtype=float)).copy()
        if self.tool.shape != (4, 4) or not callable(render_anchor):
            raise ValueError("A fixed TCP and current-endpoint image callback are required")
        self.directory = Path(directory)
        self.render_anchor = render_anchor
        self.samples, self.receipts, self.images = [], [], []

    def append(self, env, sample, receipt):
        """Record an actual endpoint and optionally render at that exact state."""
        index = len(self.samples)
        if not isinstance(sample, dict) or not isinstance(receipt, dict):
            raise ValueError("Measured sample and receipt dictionaries are required")
        state = np.asarray(sample["sim_states"])
        motor = gripper_cache_receipt(sample["gripper_current_action"])
        if (
            int(sample["endpoint_step"]) != index
            or receipt.get("endpoint_step") != index
            or receipt.get("sim_state_sha256") != state_sha256(state)
            or receipt.get("sim_time") != float(sample["sim_time"])
            or receipt.get("native_success") != bool(sample["native_success"])
            or any(receipt.get(key) != value for key, value in motor.items())
        ):
            raise ValueError("Consecutive actual endpoint, clock, success and motor receipts must agree")
        if self.samples and not np.isclose(
            float(sample["sim_time"]) - float(self.samples[-1]["sim_time"]), 0.05, atol=1e-9, rtol=0
        ):
            raise ValueError("Measured endpoints must advance by exactly one control period")
        if index in ANCHORS:
            image = self.render_anchor(env, index, sample, receipt, self.directory)
            if image.get("frame_index") != index or image.get("current_endpoint") != receipt:
                raise ValueError("Each image must bind to the actual endpoint")
            self.images.append(copy.deepcopy(image))
        self.samples.append(copy.deepcopy(sample))
        self.receipts.append(copy.deepcopy(receipt))

    def wrap_endpoint(self, endpoint):
        if not callable(endpoint) or len(self.samples) != 1:
            raise ValueError("Capture the reset before wrapping the real endpoint reader")

        def observed(env, step):
            sample, receipt = endpoint(env, step)
            self.append(env, sample, receipt)
            return sample, receipt

        return observed

    def arrays(self):
        """Return captured values; Action128 opening is next measured aperture.

        That observed state is distinct from the seven-dimensional command
        recorded by the execution owner. A reset-only stop has no action label.
        """
        if not self.samples:
            raise ValueError("The actual reset endpoint is required")
        keys = set(self.samples[0])
        if any(set(sample) != keys for sample in self.samples):
            raise ValueError("Endpoint fields must remain stable within a trajectory")
        arrays = {
            key: np.stack([sample[key] for sample in self.samples])
            for key in self.samples[0]
            if key != "gripper_current_action"
        }
        arrays.update(pack_gripper_caches([sample["gripper_current_action"] for sample in self.samples]))
        if len(self.samples) > 1:
            arrays.update(synchronous_labels(arrays["source_pose"], arrays["opening"], self.tool, arrays["sim_time"]))
        return arrays

    def anchor_windows(self):
        """Report only reached images with 40 subsequently completed actions."""
        steps = len(self.samples) - 1
        reached = {image["frame_index"] for image in self.images}
        return [
            dict(step=step, image_reached=step in reached, full_h40=step in reached and steps >= step + 40)
            for step in ANCHORS
        ]
