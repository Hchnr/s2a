"""054 state protection, future-read ordering, and worker command identity."""

import copy

import pytest

from s2a.synchronous_state_metrics import diagnosis


def metric_rows():
    rows = []
    for episode in range(6):
        for anchor in range(4):
            for seed in (42, 43):
                for variant, error in [("raw", 0.01), ("probe052", 0.008)]:
                    rows.append(
                        {
                            "name": f"{episode}_{anchor}_{seed}_{variant}",
                            "episode": str(episode),
                            "task_index": episode // 2,
                            "sampling_seed": seed,
                            "variant": variant,
                            "errors": {
                                method: {k: [value] * 40 for k in ["position", "rotation", "opening"]}
                                for method, value in [("wam", error), ("hold", 0.02), ("constant_velocity", 0.03)]
                            },
                        }
                    )
    return rows


def test_all96_rows_protected_with_both_seeds():
    groups, gate = diagnosis(metric_rows())
    assert gate["passed"] and len(groups["by_episode"]) == 6
    assert groups["pooled"]["8"]["probe052"]["valid_endpoints"] == 384


@pytest.mark.parametrize("fault", ["opening", "seed", "episode", "missing"])
def test_pooled_gain_cannot_hide_component_seed_or_demo_failure(fault):
    rows = metric_rows()
    if fault == "missing":
        rows.pop()
        with pytest.raises(ValueError):
            diagnosis(rows)
        return
    for row in rows:
        if row["variant"] == "probe052":
            if fault == "opening":
                row["errors"]["wam"]["opening"] = [0.04] * 40
            if fault == "seed" and row["sampling_seed"] == 43:
                row["errors"]["wam"] = {k: [0.03] * 40 for k in ["position", "rotation", "opening"]}
            if fault == "episode" and row["episode"] == "0":
                row["errors"]["wam"] = {k: [0.05] * 40 for k in ["position", "rotation", "opening"]}
    assert not diagnosis(rows)[1]["passed"]


@pytest.mark.parametrize("field", ["passed", "complete", "numerical_contract_passed", "lifecycle_passed"])
def test_future_scorer_stops_before_prepared_or_archive_read(monkeypatch, tmp_path, field):
    import score_wam_synchronous_state as scorer

    value = {
        "passed": True,
        "complete": True,
        "numerical_contract_passed": True,
        "lifecycle_passed": True,
        "prepared_ref": "prepared",
    }
    value[field] = False
    calls = []

    def read(ref):
        calls.append(ref)
        assert ref == "numerical"
        return value

    monkeypatch.setattr(scorer, "read", read)
    with pytest.raises(ValueError):
        scorer.score("prepared", "numerical", tmp_path / "never.json")
    assert calls == ["numerical"] and not (tmp_path / "never.json").exists()


def test_owner_binds_new_cli_candidate_and_uuid():
    from run_wam_synchronous_state import worker_command

    from s2a.synchronous_state_sampling import GPU

    ref = {"path": "/repo/f.json", "sha256": "a" * 64}
    prepared = {
        "model_config": {"model_python": "/model/python"},
        "candidate_fit_ref": ref,
        "candidate_verification_ref": ref,
    }
    command = worker_command(prepared, ref, ref)
    assert command[:2] == ["/model/python", "scripts/sample_wam_synchronous_state.py"]
    assert command[command.index("--gpu-uuid") + 1] == GPU
    assert "--candidate-fit" in command and "--w2-fit" not in command


def test_scores_inherit_unchanged_numeric_thresholds():
    import verify_wam_clean_training_state as previous

    rows = metric_rows()
    oldrows = copy.deepcopy(rows)
    for row in oldrows:
        if row["variant"] == "probe052":
            row["variant"] = "w2"
    current = diagnosis(rows)[0]["pooled"]
    historical = previous.pool(oldrows)
    for horizon in ["8", "16", "40"]:
        assert current[horizon]["probe052"] == historical[horizon]["w2"]
