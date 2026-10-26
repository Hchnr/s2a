"""232 bind the qualified recorded-log admission to the original18 physical profiles."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import REPO, sha256
from s2a.owned_receipt_contract import normal_receipts


def main():
    e = Evidence()
    config = e.json("configs/preview_control_registered_20261026.json")
    for key in ("preview_runtime_ref", "preview_qp_ref", "preview_log_binding_ref"):
        del config[key]
    refs = {
        "coupled_chain_ref": "outputs/current_coupled_recorded_chain_checked_20261026/independent_verified.json",
        "coupled_reader_ref": "outputs/current_coupled_reader_admission_20261026/verified.json",
        "coupled_chain_supervision_ref": "outputs/coupled_recorded_chain_supervision_checked_20261026/execution_result.json",
        "coupled_log_ref": "outputs/current_coupled_control_log_serialized_20261026/independent_verified.json",
        "coupled_log_supervision_ref": "outputs/current_coupled_log_supervision_serialized_20261026/execution_result.json",
    }
    for key, name in refs.items():
        path = REPO / name
        proof = e.json(path)
        assert proof["complete"], key
        for source, signature in proof.get("source_sha256", {}).items():
            e.check(source, signature)
        config[key] = dict(path=str(path), sha256=sha256(path))
    log = e.json(config["coupled_log_ref"]["path"])
    assert log["independent_full_control_log"] and log["rows"] == log["json_roundtrip_rows"] == 164
    assert log["source_commit_cycles"] == 164 and len(log["corruptions_rejected"]) == 16
    assert log["max_stationarity_residual"] < 2e-6
    for name in (
        "coupled_recorded_chain_supervision_checked_20261026",
        "current_coupled_log_supervision_serialized_20261026",
    ):
        directory = REPO / "outputs" / name / "verification"
        child = normal_receipts(
            e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
        )["child"]
        stat = Path(f"/proc/{child['pid']}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    original = e.json("configs/pose_velocity_loaded_control_20261025.json")
    assert config["profiles"] == original["profiles"] and len(config["profiles"]) == 18
    assert config["seed"] == 42 and config["physics_steps"] == 4907
    config.update(
        output_dir="outputs/current_coupled_control_20261026",
        arm_translation_algorithm="228 current inertial-coupled single-step inverse",
        arm_translation_solver="strict convex QP;actual interface bounds and actual previous +/-0.25",
        arm_physical_feedback_policy="228 completed coupled SITE innovation;150 independent nominal history;173 actual SOURCE commit",
        arm_translation_policy="228 current coupled H1;actual previous +/-0.25",
        arm_translation_objective="current_coupled_single_step_position_plus_dt_velocity_squared",
        arm_residual_policy="coupled completed SITE innovation;150 baseline separate;173 actual SOURCE commit",
        arm_rotation_inverse_weight=0.5,
    )
    e.recheck()
    path = REPO / "configs/current_coupled_control_registered_20261026.json"
    with path.open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(path), sha256=sha256(path), physics_started=False)))


if __name__ == "__main__":
    main()
