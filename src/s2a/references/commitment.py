"""Experimental one-replan commitment to an already generated opening phase.

Uses only generated base-frame TCP references and current measured TCP state.
It has no task predicate, object state, expert command, or future observation.
This candidate changes the executed reference and has no performance qualification.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.geometry import validate_transform


@dataclass(frozen=True)
class CommitmentConfig:
    position_agreement_m: float
    rotation_agreement_rad: float
    measured_position_gate_m: float
    measured_rotation_gate_rad: float
    opening_level: float

    def __post_init__(self):
        values = np.asarray(
            [
                self.position_agreement_m,
                self.rotation_agreement_rad,
                self.measured_position_gate_m,
                self.measured_rotation_gate_rad,
                self.opening_level,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values <= 0) or not self.opening_level < 1:
            raise ValueError("Require explicit positive finite gates and opening_level in (0,1)")


def _angles(a, b):
    return Rotation.from_matrix(a @ b.swapaxes(-1, -2)).magnitude()


def select_committed_prefix(
    previous_poses,
    previous_openings,
    latest_poses,
    latest_openings,
    *,
    previous_step,
    current_step,
    measured_tcp,
    prefix_count,
    config,
):
    """Return a prefix plus provenance; retain at most the immediately prior plan.

    Whole-prefix max errors gate the old pose/opening pair together. No opening
    value is invented or taken from a different endpoint. Caller must isolate
    previous plans per episode and provide integer 20Hz physical step indices.
    """
    if not isinstance(config, CommitmentConfig):
        raise ValueError("Explicit CommitmentConfig required")
    for value in (previous_step, current_step, prefix_count):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Step indices and prefix_count must be nonnegative integers")
    if prefix_count < 1:
        raise ValueError("prefix_count must be positive")
    old, new = validate_transform(previous_poses), validate_transform(latest_poses)
    measured = validate_transform(measured_tcp)
    old_o, new_o = np.asarray(previous_openings, dtype=float), np.asarray(latest_openings, dtype=float)
    if old.ndim != 3 or new.ndim != 3 or measured.shape != (4, 4):
        raise ValueError("Expected trajectory batches and one measured TCP pose")
    for poses, openings in [(old, old_o), (new, new_o)]:
        if (
            openings.shape != (len(poses),)
            or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 1))
        ):
            raise ValueError("Expected one normalized opening per generated pose")
    if len(new) < prefix_count:
        raise ValueError("Latest reference is too short")
    shift = current_step - previous_step
    result = {
        "source": "latest",
        "reason": "no_one_replan_overlap",
        "previous_step": previous_step,
        "current_step": current_step,
        "source_indices": list(range(prefix_count)),
    }
    poses, openings = new[:prefix_count], new_o[:prefix_count]
    if shift == prefix_count and shift + prefix_count <= len(old):
        old_prefix, old_openings = old[shift : shift + prefix_count], old_o[shift : shift + prefix_count]
        position = float(np.max(np.linalg.norm(old_prefix[:, :3, 3] - poses[:, :3, 3], axis=1)))
        rotation = float(np.max(_angles(old_prefix[:, :3, :3], poses[:, :3, :3])))
        measured_position = float(np.linalg.norm(old_prefix[0, :3, 3] - measured[:3, 3]))
        measured_rotation = float(_angles(old_prefix[0, :3, :3], measured[:3, :3]))
        phase_withdrawn = bool(np.max(old_openings) >= config.opening_level > np.max(openings))
        gates = {
            "opening_phase_withdrawn": phase_withdrawn,
            "position_agreement": position <= config.position_agreement_m,
            "rotation_agreement": rotation <= config.rotation_agreement_rad,
            "measured_position": measured_position <= config.measured_position_gate_m,
            "measured_rotation": measured_rotation <= config.measured_rotation_gate_rad,
        }
        result.update(
            gates=gates,
            position_revision_max_m=position,
            rotation_revision_max_rad=rotation,
            measured_position_error_m=measured_position,
            measured_rotation_error_rad=measured_rotation,
            reason="commitment_gate_rejected",
        )
        if all(gates.values()):
            poses, openings = old_prefix, old_openings
            result.update(
                source="previous",
                reason="retain_agreeing_generated_opening_phase",
                source_indices=list(range(shift, shift + prefix_count)),
            )
    return poses.copy(), openings.copy(), result
