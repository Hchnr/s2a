"""220 bind the online wrapper to independently certified real-state preview QPs."""

import json

import numpy as np
from audit_wam_position import Evidence
from guarded_loading_acceptance_checker import independent_reference_velocity
from rate_preview_independent import verify_plan

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.preview_translation_runtime import plan_current_translation


def main():
    root = REPO / "outputs/preview_runtime_binding_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/preview_translation_runtime.py",
        "src/s2a/reference_linear_velocity.py",
        "tests/test_preview_translation_runtime.py",
        "plans/v0.0.1/220_preview_runtime_binding_20261026.md",
    ):
        e.read(path)
    prior = e.json("outputs/rate_preview_active_set_admission_20261026/independent_verified.json")
    assert prior["complete"] and prior["rows"] == 164 and prior["max_stationarity_residual"] < 2e-6
    for path, signature in prior["source_sha256"].items():
        e.check(path, signature)
    run = REPO / "outputs/late_clipped_inverse_control_20261026"
    case = next((run / "case00/simulation").glob("*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    chunks, controller = e.json(case / "chunks.json"), e.json(case / "controller.json")
    config = e.json(run / "case00/execution_config.json")
    template = e.json(config["policy_template"])
    tool = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    mapping = np.linalg.inv(tool) @ np.asarray(controller["source_from_site"])
    contract = controller["contract"]
    lo, hi, outlo, outhi = (
        np.asarray(contract[key])[:3] for key in ("input_min", "input_max", "output_min", "output_max")
    )
    results = []
    for index, (row, expected) in enumerate(zip(rows, prior["results"], strict=True)):
        assert expected["step"] == index
        chunk = chunks[row["chunk_index"]]
        cursor = index - chunk["step"]
        generated = np.asarray(chunk["effective_targets_tcp"]) @ mapping
        anchor = np.asarray(chunk["anchor_tcp"]) @ mapping
        data = {key: np.asarray(value) for key, value in row["qualified_forward_input"].items()}
        plan = plan_current_translation(data, generated, cursor, anchor, contract)
        np.testing.assert_allclose(plan["planned_commands"], expected["planned_commands"], atol=1e-10, rtol=0)
        horizon = min(6, len(generated) - cursor)
        assert plan["generated_indices"] == list(range(cursor, cursor + horizon))
        refs = generated[cursor : cursor + horizon, :3, 3]
        velocities = np.array(
            [independent_reference_velocity(generated, cursor + j, anchor, 0.05)["velocity"] for j in range(horizon)]
        )
        innovation = np.vstack([data["x_response_residual"][0, :3], data["x_response_residual"][0, 6:9] / 0.05])
        np.testing.assert_allclose(plan["reference_positions"], refs, atol=1e-12, rtol=0)
        np.testing.assert_allclose(plan["reference_velocities"], velocities, atol=1e-12, rtol=0)
        np.testing.assert_allclose(plan["completed_innovation"], innovation, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            plan["physical_delta"], outlo + (plan["command"] - lo) * (outhi - outlo) / (hi - lo), atol=1e-12, rtol=0
        )
        certificate = verify_plan(
            plan,
            data["x_site_pose"][0, :3, 3],
            data["x_site_twist"][0, :3],
            refs,
            velocities,
            data["x_previous_command"][0, :3],
            contract,
            innovation,
        )
        results.append(dict(step=index, horizon=horizon, **certificate))
    e.recheck()
    root.mkdir()
    publish(
        root / "independent_verified.json",
        dict(
            complete=True,
            rows=164,
            online_input_and_reference_binding=True,
            results=results,
            max_stationarity_residual=max(row["max_stationarity_residual"] for row in results),
            source_sha256=e.hashes,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
            seed=42,
            control_qualified=False,
        ),
    )
    print("220 current input/generated reference/actual command and independent KKT verified164rows;no physics")


if __name__ == "__main__":
    main()
