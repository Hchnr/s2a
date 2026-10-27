"""242 publish the fixed18 physical registration only after both complete record gates."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import REPO, sha256
from s2a.owned_receipt_contract import normal_receipts


def main():
    e = Evidence()
    config = e.json("configs/current_coupled_control_registered_20261026.json")
    for key in (
        "coupled_chain_ref",
        "coupled_chain_supervision_ref",
        "coupled_log_ref",
        "coupled_log_supervision_ref",
        "arm_translation_objective",
        "arm_translation_policy",
        "arm_translation_algorithm",
        "arm_translation_solver",
        "arm_rotation_inverse_weight",
    ):
        del config[key]
    refs = {
        "full_pose_chain_ref": "outputs/full_coupled_all_records_checked_20261026/independent_verified.json",
        "full_pose_chain_supervision_ref": "outputs/full_coupled_all_records_startup_20261026/execution_result.json",
        "full_pose_log_ref": "outputs/full_coupled_control_log_20261026/independent_verified.json",
        "full_pose_log_supervision_ref": "outputs/full_coupled_log_supervision_20261026/execution_result.json",
    }
    for key, name in refs.items():
        path = REPO / name
        proof = e.json(path)
        assert proof["complete"], key
        for source, signature in proof.get("source_sha256", {}).items():
            e.check(source, signature)
        config[key] = dict(path=str(path), sha256=sha256(path))
    chain = e.json(config["full_pose_chain_ref"]["path"])
    assert chain["independent_full_pose_inverse_and_native_history"] and chain["cases"] == 18 and chain["rows"] == 4907
    assert chain["max_independent_stationarity"] < 2e-6
    log = e.json(config["full_pose_log_ref"]["path"])
    assert (
        log["independent_full_pose_control_log"]
        and log["rows"] == log["json_roundtrip_rows"] == log["source_commit_cycles"] == 164
    )
    assert log["reconstructed_controller_goals"] and len(log["corruptions_rejected"]) == 20
    assert log["max_stationarity_residual"] < 2e-6
    for name, stage in (
        ("full_coupled_all_records_startup_20261026", "admission"),
        ("full_coupled_log_supervision_20261026", "verification"),
    ):
        directory = REPO / "outputs" / name / stage
        child = normal_receipts(
            e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
        )["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    original = e.json("configs/pose_velocity_loaded_control_20261025.json")
    assert config["profiles"] == original["profiles"] and len(config["profiles"]) == 18
    assert config["seed"] == 42 and config["physics_steps"] == 4907
    config.update(
        output_dir="outputs/full_coupled_control_20261026",
        arm_pose_policy="239 full pose;actual previous +/-0.25;reset/latch",
        arm_pose_objective="full_pose_geodesic_plus_dt_world_twist_squared",
        arm_pose_algorithm="239 goal-dependent coupled pose/twist inverse;native reset/latch",
        arm_pose_solver="bounded nonlinear least squares;local stationarity and mode comparison;no global claim",
        arm_physical_feedback_policy="239 SO3-composed completed SITE innovation;150 independent nominal history;173 actual SOURCE commit",
        arm_residual_policy="completed native coupled SITE innovation;continuous proposal separate;150 baseline separate;173 actual SOURCE commit",
        source_provisional_inverse_weight=0.5,
    )
    e.recheck()
    path = REPO / "configs/full_coupled_control_registered_20261026.json"
    with path.open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(path), sha256=sha256(path), physics_started=False)))


if __name__ == "__main__":
    main()
