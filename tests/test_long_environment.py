"""The017 duration boundary must preserve clocks and reject short references."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2a.long_environment import episode_contract, episode_snapshot, validate_episode_settings
from s2a.long_simulation import _read_oracle, validate_long_config


def clock(step=0, time=1.25, done=False):
    return dict(
        effective_horizon=1300,
        ignore_done=False,
        internal_timestep=step,
        internal_done=done,
        control_freq=20.0,
        control_timestep_s=0.05,
        model_timestep_s=0.002,
        sim_time_s=time,
    )


def test_actual_policy_counter_separates_scoring_budget_from_horizon():
    result = episode_contract(clock(), clock(1200, 61.25), 1200)
    assert result["clock_consistent"]
    assert result["scored_physical_step_budget"] == result["observed_policy_steps"] == 1200
    assert result["effective_horizon"] == 1300
    assert result["initial_timestep"] == result["initialization_policy_steps_added"] == 0
    assert result["initial_sim_time_s"] == 1.25


@pytest.mark.parametrize(
    ("initial", "final"),
    [
        (clock(5), clock(1205, 61.25)),
        (clock(), clock(1201, 61.25)),
        (clock(), clock(1200, 61.2)),
        (clock(), clock(1200, 61.25, done=True)),
    ],
)
def test_contract_records_inconsistent_clocks_without_rewriting_them(initial, final):
    before = deepcopy((initial, final))
    result = episode_contract(initial, final, 1200)
    assert not result["clock_consistent"]
    assert (initial, final) == before
    assert result["initial_timestep"] == initial["internal_timestep"]
    assert result["final_timestep"] == final["internal_timestep"]


def test_snapshot_only_reads_environment_without_adding_preroll():
    core = SimpleNamespace(
        horizon=1300,
        ignore_done=False,
        timestep=0,
        done=False,
        control_freq=20,
        control_timestep=0.05,
        model_timestep=0.002,
    )
    env = SimpleNamespace(env=core, sim=SimpleNamespace(data=SimpleNamespace(time=1.25)))
    assert episode_snapshot(env) == clock()
    assert core.timestep == 0 and env.sim.data.time == 1.25
    core.horizon = 1000
    with pytest.raises(ValueError, match="settings differ"):
        episode_snapshot(env)


@pytest.mark.parametrize(
    "settings",
    [
        dict(episode_horizon=1000, ignore_done=False),
        dict(episode_horizon=1300.0, ignore_done=False),
        dict(episode_horizon=1300, ignore_done=True),
        dict(episode_horizon=1300, ignore_done=0),
    ],
)
def test_horizon_and_done_policy_are_explicit(settings):
    with pytest.raises(ValueError):
        validate_episode_settings(settings)


def test_only_registered1200_raw8_contract_is_admitted():
    path = Path(__file__).resolve().parents[1] / "configs/long_raw_qualification_20261021.json"
    config = json.loads(path.read_text())
    for source in ("native", "oracle", "wam"):
        validate_long_config(config, source=source)
    for field, value in (
        ("online_max_steps", 600),
        ("max_observation_interval_steps", 40),
        ("guidance", True),
        ("ignore_done", True),
    ):
        bad = dict(config, **{field: value})
        with pytest.raises(ValueError):
            validate_long_config(bad, source="oracle")


def test_oracle_cannot_pad_a600_step_native_or_use_legacy_ambiguous_source(tmp_path):
    folder = tmp_path / "native/task0_init0_s42"
    folder.mkdir(parents=True)
    record = dict(name="task0_init0", initial_state_sha256="a" * 64)
    manifest = dict(record=record, source="native", schema_version="s2a.long-native-execution.v1")
    result = dict(
        record=record["name"],
        source="native",
        sampling_seed=42,
        complete=True,
        execution_inputs_unchanged=True,
        steps=600,
    )
    contract = dict(clock_consistent=True, effective_horizon=1300, ignore_done=False, physical_steps=1200)
    for name, value in (("manifest.json", manifest), ("result.json", result), ("episode_contract.json", contract)):
        (folder / name).write_text(json.dumps(value))
    # Even a1200-row array cannot replace missing real native endpoints.
    np.savez(folder / "reference.npz", poses=np.repeat(np.eye(4)[None], 1200, axis=0), openings=np.ones(1200))
    config = dict(oracle_reference_root=str(folder.parent), online_max_steps=1200, stable_success_last_steps=40)
    with pytest.raises(ValueError, match="actual matching native1200"):
        _read_oracle(config, record, 42)
    legacy = folder.parent / record["name"]
    legacy.mkdir()
    (legacy / "reference.npz").write_bytes(b"not consulted")
    with pytest.raises(ValueError, match="unambiguous"):
        _read_oracle(config, record, 42)
