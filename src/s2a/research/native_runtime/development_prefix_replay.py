"""Replay saved development commands through the fixed native controller."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2a.infrastructure.snapshots.xml_asset_snapshot import _encoded_path, _file_attributes, _replace

ATOL = 1e-10
SAMPLE_FIELDS = (
    "sim_states",
    "sim_time",
    "source_position",
    "source_quaternion_xyzw",
    "source_pose",
    "actual_source_site_pose",
    "actual_eef_body_pose",
    "physical_source_pose",
    "controller_site_pose",
    "opening",
    "gripper_qpos",
    "controller_ee_pos",
    "controller_ee_ori_mat",
    "controller_goal_pos",
    "controller_goal_ori",
    "controller_joint_pos",
    "controller_joint_vel",
)


def map_record_xml(original, template_original, template_mapped):
    """Use the previously verified ordered mapping and prove byte reversibility."""
    attributes = _file_attributes(original)
    old = _file_attributes(template_original)
    mapped = _file_attributes(template_mapped)
    if len(attributes) != 99 or len(old) != 99 or len(mapped) != 99:
        raise ValueError("Exactly 99 original and mapped asset declarations required")
    old_names = [value for _, _, value in old]
    if [value for _, _, value in attributes] != old_names or len(set(old_names)) != 96:
        raise ValueError("Record asset declaration sequence differs from verified template")
    mapping = {}
    for (_, _, source), (_, _, target) in zip(old, mapped, strict=True):
        if source in mapping and mapping[source] != target:
            raise ValueError("Conflicting asset mapping")
        mapping[source] = target
    if any(not Path(value).is_absolute() or value.startswith("/share/project/") for value in mapping.values()):
        raise ValueError("Mapped paths must be absolute and never use the retired root")
    migrated = _replace(original, attributes, [_encoded_path(mapping[value]) for _, _, value in attributes])
    restored = _replace(migrated, _file_attributes(migrated), [_encoded_path(value) for value in old_names])
    if restored != original:
        raise ValueError("Record XML migration changed non-path bytes")
    return migrated


@dataclass(frozen=True)
class ReplayObservers:
    endpoint: object
    current: object
    render_anchor: object
    render_cameras: object


def saved_prefix(archive, steps):
    """Expose exactly the registered past and terminal endpoint to the replay."""
    if not isinstance(steps, int) or not 0 < steps <= 600:
        raise ValueError("A bounded prefix length is required")
    saved = {name: np.asarray(archive[name][: steps + 1]).copy() for name in (*SAMPLE_FIELDS, "native_success")}
    if any(len(value) != steps + 1 for value in saved.values()):
        raise ValueError("Incomplete saved endpoints")
    offsets = np.asarray(archive["gripper_current_action_offsets"][: steps + 2])
    flat = np.asarray(archive["gripper_current_action_flat"])
    if offsets.shape != (steps + 2,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] > len(flat):
        raise ValueError("Invalid packed gripper cache")
    saved["gripper_current_action"] = [flat[offsets[i] : offsets[i + 1]].copy() for i in range(steps + 1)]
    return saved


def replay_saved_prefix(env, commands, saved, receipts, output, observers, receipt, *, initialize=None, progress=None):
    """Compare each endpoint before dispatching the next previously saved command."""
    commands = np.asarray(commands, dtype=float)
    count = len(commands)
    if commands.shape != (count, 7) or count == 0 or not np.isfinite(commands).all():
        raise ValueError("Finite saved seven-dimensional commands required")
    if np.max(abs(commands)) > 1 + 1e-12:
        raise ValueError("Saved command box violation")
    if np.max(abs(np.diff(np.vstack((np.zeros(7), commands)), axis=0)[:, :6])) > 0.25 + 1e-12:
        raise ValueError("Saved arm command rate violation")
    if len(receipts) != count + 1 or any(len(saved[name]) != count + 1 for name in (*SAMPLE_FIELDS, "native_success")):
        raise ValueError("Exactly the registered comparison prefix required")
    if len(saved["gripper_current_action"]) != count + 1:
        raise ValueError("Exactly one motor cache per endpoint required")
    receipt.update(
        checked_endpoints=0,
        exact_state_endpoints=0,
        max_state_difference=0.0,
        max_cached_field_difference=0.0,
        control_steps_attempted=0,
        control_steps_completed=0,
    )
    observers.current(env)
    if initialize is not None:
        initialize(env)
    for index in range(count + 1):
        actual, actual_receipt = observers.endpoint(env, index)
        expected_receipt = receipts[index]
        if actual_receipt["endpoint_step"] != index or expected_receipt["endpoint_step"] != index:
            raise ValueError("Endpoint identity differs")
        for name in SAMPLE_FIELDS:
            value = np.asarray(actual[name])
            expected = np.asarray(saved[name][index])
            if value.shape != expected.shape or not np.isfinite(value).all() or not np.isfinite(expected).all():
                raise ValueError(f"Invalid endpoint field {name}")
            difference = float(np.max(np.abs(value - expected)))
            if name == "sim_states":
                receipt["max_state_difference"] = max(receipt["max_state_difference"], difference)
            elif name.startswith("controller_"):
                receipt["max_cached_field_difference"] = max(receipt["max_cached_field_difference"], difference)
            np.testing.assert_allclose(value, expected, atol=ATOL, rtol=0)
        np.testing.assert_allclose(
            actual["gripper_current_action"], saved["gripper_current_action"][index], atol=ATOL, rtol=0
        )
        if bool(actual["native_success"]) != bool(saved["native_success"][index]):
            raise ValueError("Native task label differs")
        if actual_receipt["sim_state_sha256"] == expected_receipt["sim_state_sha256"]:
            receipt["exact_state_endpoints"] += 1
        receipt["checked_endpoints"] += 1
        receipt["last_checked_endpoint"] = index
        if index in (0, 16, 56):
            observers.render_anchor(env, index, actual, actual_receipt, output)
        if progress is not None:
            progress(receipt)
        if index == count:
            break
        if index % 8 == 0:
            observers.render_cameras(env, 256)
        receipt["control_steps_attempted"] += 1
        env.step(commands[index].copy())
        receipt["control_steps_completed"] += 1
        observers.current(env)
    return receipt
