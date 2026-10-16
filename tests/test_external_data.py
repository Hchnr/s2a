import pytest

from s2a.analytic import analytic
from s2a.artifacts import load_config
from s2a.data import extract


@pytest.mark.integration
def test_selected_episode_contracts_and_geometry(request, tmp_path):
    path = request.config.getoption("--data-config")
    if path is None:
        pytest.fail("integration test requires --data-config (no silent missing-data pass)")
    config, _ = load_config(path)
    arrays, audit = extract(config, tmp_path)
    metrics = analytic(config, arrays, tmp_path)
    assert audit["selected_transitions"] == 300
    assert audit["episode_rows"] == 471
    assert not audit["terminal_pose_valid"] and audit["terminal_action_valid"]
    assert metrics["geometry_passed"]
