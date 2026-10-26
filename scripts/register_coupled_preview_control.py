"""Register the unchanged18-case control gate for the qualified four-step preview."""

import json
from pathlib import Path

from audit_wam_position import Evidence

from s2a.artifacts import REPO, sha256
from s2a.owned_receipt_contract import normal_receipts


def main():
    e = Evidence()
    e.read(__file__)
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
        "pose_preview_chain_ref": "outputs/coupled_pose_preview_continuous_chain_20261026/verified.json",
        "pose_preview_log_ref": "outputs/coupled_preview_control_log_bound_20261026/independent_verified.json",
        "pose_preview_qualification_ref": "outputs/coupled_preview_qualification_20261026/verified.json",
        "pose_preview_supervision_ref": "outputs/coupled_preview_qualification_20261026/execution_result.json",
    }
    for key, name in refs.items():
        path = REPO / name
        proof = e.json(path)
        assert proof["complete"]
        for source, digest in proof.get("source_sha256", {}).items():
            e.check(source, digest)
        config[key] = dict(path=str(path), sha256=sha256(path))
    qualification = e.json(config["pose_preview_qualification_ref"]["path"])
    assert qualification["record_cases"] == 18 and qualification["record_periods"] == 4907
    assert qualification["full_log_periods"] == 164 and qualification["corruptions_rejected"] == 22
    assert e.json(config["pose_preview_supervision_ref"]["path"])["actual_owned_wait_recorded"]
    directory = REPO / "outputs/coupled_preview_qualification_20261026/qualification"
    child = normal_receipts(
        e, directory / "launch.json", directory / "exit.json", directory / "release.json", format="durable"
    )["child"]
    stat = Path(f"/proc/{child['pid']}/stat")
    assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != child["start_ticks"]
    original = e.json("configs/pose_velocity_loaded_control_20261025.json")
    assert config["profiles"] == original["profiles"] and len(config["profiles"]) == 18
    assert config["seed"] == 42 and config["physics_steps"] == 4907
    config.update(
        output_dir="outputs/coupled_preview_control_20261026",
        arm_pose_policy="247 four-step pose preview;actual previous +/-0.25;explicit reset",
        arm_pose_objective="four_step_world_tangent_pose_plus_dt_world_twist_squared",
        arm_pose_algorithm="247 frozen current M/J four-step preview;strict feasible reset command",
        arm_pose_solver="active-set QP;post-projection KKT;no nonlinear optimality claim",
        arm_physical_feedback_policy="completed SITE innovation;150 independent history;173 final SOURCE commit",
        arm_residual_policy="completed native coupled SITE innovation;preview and continuous forecast separate;173 final SOURCE commit",
        source_provisional_inverse_weight=0.5,
    )
    e.recheck()
    path = REPO / "configs/coupled_preview_control_registered_20261026.json"
    with path.open("x") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(path=str(path), sha256=sha256(path), physics_started=False)))


if __name__ == "__main__":
    main()
