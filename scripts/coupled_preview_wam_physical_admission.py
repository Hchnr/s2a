"""Independent JSON-only audit of the completed physical prerequisite for WAM.

Does not import the producer control gate or any controller/model implementation.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "outputs/coupled_preview_control_20261026"
OWNER = REPO / "outputs/coupled_preview_continuation_20261026"
REPAIR = REPO / "outputs/coupled_preview_verification_repair_20261026"
CONFIG = REPO / "configs/coupled_preview_control_registered_20261026.json"


def audit_physical_control(ev, closure_ref):
    closure = ev.json(closure_ref)
    assert Path(closure_ref["path"]) == RUN / "closure_verified.json"
    assert all(
        closure[k] is True
        for k in ("complete", "owner_normal_wait_verified", "mechanical_qualification", "performance_qualified")
    )
    assert closure["verified_cases"] == 18 and closure["physics_steps"] == 4907
    sources = closure["source_sha256"]
    ev.mapping(sources)

    def document(path):
        path = str(path)
        assert path in sources
        return ev.json(dict(path=path, sha256=sources[path]))

    def normal(directory):
        launch, exited, release = (document(directory / f"{name}.json") for name in ("launch", "exit", "release"))
        assert exited["actual_wait"] is True and exited["returncode"] == 0 and exited["automatic_retry"] is False
        assert release["actual_wait"] is release["reaped"] is release["only_owned_child_signalled"] is True
        assert release["signal_sent"] is release["kill_sent"] is False and release["original_error"] is None
        assert release["returncode"] == 0
        assert all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
        assert (
            launch["selected_environment"]["CUDA_VISIBLE_DEVICES"]
            == launch["selected_environment"]["LD_LIBRARY_PATH"]
            == ""
        )
        child = launch["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]

    assert Path(closure["summary_ref"]["path"]) == RUN / "regression_summary.json"
    summary = ev.json(closure["summary_ref"])
    assert summary == document(RUN / "regression_summary.json")
    config = document(CONFIG)
    assert (
        summary["complete"]
        is summary["execution_and_independent_checks_passed"]
        is summary["performance_qualified"]
        is True
    )
    assert (
        summary["candidate_trials"]
        == summary["candidate_stable"]
        == len(summary["cases"])
        == len(config["profiles"])
        == 18
    )
    assert sum(c["physics_steps"] for c in summary["cases"]) == summary["physics_steps"] == 4907
    for i, (case, profile) in enumerate(zip(summary["cases"], config["profiles"], strict=True)):
        proof = document(RUN / f"case{i:02d}/independent_verified.json")
        assert proof["complete"] is proof["mechanical_labels_images_passed"] is proof["performance_qualified"] is True
        ctrl = proof["control"]
        assert case["case"] == i and case["record"] == profile["record"]["name"]
        assert ctrl["steps"] == case["physics_steps"] == profile["physics_steps"]
        assert ctrl["independent_coupled_log_rows"] == ctrl["steps"]
        assert ctrl["coupled_max_stationarity_residual"] < 2e-6
        assert case["outcome"] == ctrl["outcome"] and ctrl["outcome"]["stable"] is True
        assert case["tracking"] == ctrl["tracking"]["admitted"]
        assert 0 <= case["tracking"]["position_rmse_m"] <= 0.01
        assert 0 <= case["tracking"]["so3_rmse_rad"] <= 0.05
        normal(RUN / "orchestration" / f"case{i:02d}")
        normal(REPAIR / "case00" if i == 0 else RUN / "verification" / f"case{i:02d}")
    normal(OWNER / "owner_wait")
    repair = document(REPAIR / "verified.json")
    assert repair["complete"] is repair["actual_owned_wait_recorded"] is repair["reuses_original_collection"] is True
    return dict(
        closure_ref=closure_ref,
        summary_ref=closure["summary_ref"],
        control_config_ref=dict(path=str(CONFIG), sha256=sources[str(CONFIG)]),
        physical_control_qualified=True,
        wam_execution_authorized=False,
        fresh_authorized=False,
    )
