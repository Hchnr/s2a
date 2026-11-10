"""The replay preserves source identity, causal order and failure accounting."""

from pathlib import Path

import numpy as np
import pytest

from s2a.research.native_runtime.development_prefix_replay import (
    SAMPLE_FIELDS,
    ReplayObservers,
    map_record_xml,
    replay_saved_prefix,
    saved_prefix,
)


def xml(paths, *, gravity="0 0 -9.81"):
    elements = "".join(f'<mesh name="m{i}" file="{path}"/>' for i, path in enumerate(paths))
    return f'<mujoco><option gravity="{gravity}"/><asset>{elements}</asset></mujoco>'.encode()


def test_record_mapping_is_reversible_and_rejects_changed_asset_order():
    old = [f"/historical/assets/{i}.stl" for i in range(96)]
    old += old[:3]
    new = [path.replace("/historical/", "/current/") for path in old]
    template, mapped = xml(old), xml(new)
    other_record = xml(old, gravity="0 0 -9.8")
    assert map_record_xml(other_record, template, mapped) == xml(new, gravity="0 0 -9.8")
    changed = old.copy()
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="sequence"):
        map_record_xml(xml(changed), template, mapped)


def test_record_mapping_rejects_conflicting_duplicate_asset():
    old = [f"/historical/assets/{i}.stl" for i in range(96)]
    old += old[:3]
    new = [path.replace("/historical/", "/current/") for path in old]
    new[-1] = "/current/assets/wrong.stl"
    with pytest.raises(ValueError, match="Conflicting"):
        map_record_xml(xml(old), xml(old), xml(new))


def archive(length=4):
    values = {name: np.arange(length, dtype=float)[:, None] for name in SAMPLE_FIELDS}
    values["native_success"] = np.zeros(length, dtype=bool)
    values["gripper_current_action_flat"] = np.arange(2 * length, dtype=float)
    values["gripper_current_action_offsets"] = np.arange(0, 2 * length + 1, 2)
    return values


def test_saved_prefix_excludes_later_states_and_rejects_broken_cache():
    source = archive()
    saved = saved_prefix(source, 2)
    assert len(saved["sim_states"]) == 3
    np.testing.assert_array_equal(saved["gripper_current_action"][2], [4, 5])
    source["gripper_current_action_offsets"][2] = -1
    with pytest.raises(ValueError, match="cache"):
        saved_prefix(source, 2)


def fixture(*, mismatch=None, fail_step=False):
    saved = saved_prefix(archive(), 2)
    events = []

    class Env:
        index = 0

        def step(self, command):
            events.append(("step", self.index))
            if fail_step:
                raise RuntimeError("native step failed")
            self.index += 1

    env = Env()

    def endpoint(_env, index):
        events.append(("endpoint", index))
        sample = {name: saved[name][index].copy() for name in SAMPLE_FIELDS}
        if mismatch == index:
            sample["sim_states"] += 1e-4
        sample["native_success"] = bool(saved["native_success"][index])
        sample["gripper_current_action"] = saved["gripper_current_action"][index].copy()
        return sample, {"endpoint_step": index, "sim_state_sha256": f"sha{index}"}

    observers = ReplayObservers(
        endpoint=endpoint,
        current=lambda _env: events.append(("current", env.index)),
        render_anchor=lambda _env, index, *_args: events.append(("anchor", index)),
        render_cameras=lambda _env, _resolution: events.append(("camera", env.index)),
    )
    receipts = [{"endpoint_step": i, "sim_state_sha256": f"sha{i}"} for i in range(3)]
    return env, saved, receipts, observers, events


def test_replay_preserves_observation_schedule_and_exact_prefix():
    env, saved, receipts, observers, events = fixture()
    result = replay_saved_prefix(env, np.zeros((2, 7)), saved, receipts, Path("."), observers, {})
    assert events == [
        ("current", 0),
        ("endpoint", 0),
        ("anchor", 0),
        ("camera", 0),
        ("step", 0),
        ("current", 1),
        ("endpoint", 1),
        ("step", 1),
        ("current", 2),
        ("endpoint", 2),
    ]
    assert result["control_steps_attempted"] == result["control_steps_completed"] == 2
    assert result["checked_endpoints"] == result["exact_state_endpoints"] == 3


def test_motor_initialization_follows_first_observation_before_endpoint():
    env, saved, receipts, observers, events = fixture()
    replay_saved_prefix(
        env,
        np.zeros((2, 7)),
        saved,
        receipts,
        Path("."),
        observers,
        {},
        initialize=lambda _env: events.append(("initialize_motor", env.index)),
    )
    assert events[:4] == [("current", 0), ("initialize_motor", 0), ("endpoint", 0), ("anchor", 0)]


def test_mismatch_stops_before_the_next_dispatch():
    env, saved, receipts, observers, events = fixture(mismatch=1)
    receipt = {}
    with pytest.raises(AssertionError):
        replay_saved_prefix(env, np.zeros((2, 7)), saved, receipts, Path("."), observers, receipt)
    assert [event for event in events if event[0] == "step"] == [("step", 0)]
    assert receipt["control_steps_attempted"] == receipt["control_steps_completed"] == 1
    assert receipt["checked_endpoints"] == 1


def test_failed_dispatch_is_counted_as_attempted():
    env, saved, receipts, observers, _ = fixture(fail_step=True)
    receipt = {}
    with pytest.raises(RuntimeError, match="native step"):
        replay_saved_prefix(env, np.zeros((2, 7)), saved, receipts, Path("."), observers, receipt)
    assert receipt["control_steps_attempted"] == 1
    assert receipt["control_steps_completed"] == 0
