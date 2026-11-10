"""Exercise original buffer and causally aligned reference admission."""

from copy import deepcopy

import numpy as np
import pytest

from s2a.research.joint_control.preview.opening_event_admission import (
    admit_recorded_state,
    aligned_opening_references,
    recorded_joint_ranges,
)


def xml():
    joints = "".join(f'<joint name="robot0_joint{i}" range="-1 1"/>' for i in range(1, 8))
    return f"<mujoco><worldbody><body>{joints}</body></worldbody></mujoco>"


def case():
    poses = np.repeat(np.eye(4)[None], 40, axis=0)
    openings = np.full(40, 0.2)
    openings[13] = 0.9
    old = dict(step=120, effective_targets_tcp=poses.copy(), openings=openings, anchor_tcp=np.eye(4))
    latest = dict(
        step=128,
        effective_targets_tcp=poses.copy(),
        openings=np.full(40, 0.2),
        anchor_tcp=np.eye(4),
        anchor_opening=0.2,
    )
    spec = dict(case_index=3, case_id="case", prefix_steps=128, source_step=120, event_endpoint=134)
    shadow = dict(
        step=128,
        source="remembered",
        source_step=120,
        source_indices=list(range(8, 16)),
        absolute_endpoints=list(range(129, 137)),
        event=dict(direction="opening", absolute_endpoint=134),
        baseline_observation_label="first_intervention_on_baseline",
        poses=poses[8:16].copy(),
        openings=openings[8:16].copy(),
    )
    checked = dict(first_intervention_step=128, case=dict(case_index=3, case_id="case"))
    return spec, old, latest, shadow, checked


def test_joint_ranges_and_buffer_boundary():
    limits = recorded_joint_ranges(xml())
    q = np.zeros(7)
    q[2] = 0.95
    result = admit_recorded_state(q, limits, margin_rad=0.05)
    assert result["recorded_endpoint_admitted"]
    q[2] = 0.950001
    assert not admit_recorded_state(q, limits, margin_rad=0.05)["recorded_endpoint_admitted"]
    q[2] = 1.01
    assert admit_recorded_state(q, limits, margin_rad=0.05)["minimum_margin_rad"] < 0


@pytest.mark.parametrize("mutate", ["duplicate", "order", "nonfinite", "freejoint"])
def test_joint_range_mapping_rejects_invalid_xml(mutate):
    text = xml()
    if mutate == "duplicate":
        text = text.replace("</body>", '<joint name="robot0_joint1" range="-1 1"/></body>')
    elif mutate == "order":
        text = text.replace("robot0_joint1", "other", 1)
    elif mutate == "nonfinite":
        text = text.replace('range="-1 1"', 'range="nan 1"', 1)
    else:
        text = text.replace('<joint name="robot0_joint1"', '<freejoint name="robot0_joint1"', 1)
    with pytest.raises(ValueError):
        recorded_joint_ranges(text)


def test_paired_reference_uses_only_the_aligned_previously_generated_window():
    spec, old, latest, shadow, checked = case()
    result = aligned_opening_references(spec, old, latest, shadow, checked, np.eye(4), 0.2)
    assert result["remembered"]["cursor"] == 8
    assert result["latest"]["cursor"] == 0
    result["remembered"]["openings"][13] = 0.0
    assert old["openings"][13] == 0.9


@pytest.mark.parametrize(
    "fault", ["future", "wrong_endpoint", "wrong_direction", "wrong_first", "anchor", "shadow", "closed_event"]
)
def test_rejects_noncausal_or_unverified_opening_reference(fault):
    spec, old, latest, shadow, checked = deepcopy(case())
    if fault == "future":
        old["step"] = 136
    elif fault == "wrong_endpoint":
        shadow["absolute_endpoints"][0] += 1
    elif fault == "wrong_direction":
        shadow["event"]["direction"] = "closure"
    elif fault == "wrong_first":
        checked["first_intervention_step"] = 120
    elif fault == "anchor":
        latest["anchor_tcp"][0, 3] = 0.1
    elif fault == "shadow":
        shadow["openings"][0] = 0.7
    else:
        old["openings"][13] = 0.2
        shadow["openings"][5] = 0.2
    with pytest.raises((ValueError, AssertionError)):
        aligned_opening_references(spec, old, latest, shadow, checked, np.eye(4), 0.2)
