"""Reconstruct the stoppedcase04/71 problem and diagnose bounded optimizer stationarity."""

import json
import os
import sys
import time

import numpy as np
import scipy
from audit_wam_position import Evidence
from full_coupled_pose_independent import objective as independent_objective
from scipy.optimize import minimize

from s2a.artifacts import REPO
from s2a.durable_stage_supervisor import publish
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, projected_gradient, task_objective
from s2a.generated_reference_twist import reference_world_twist
from s2a.osc import OSCInterface


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
    assert all(os.environ[name] == "8" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
    root = REPO / "outputs/full_pose_stationarity_failure_20261026"
    assert not root.exists()
    e = Evidence()
    for path in (
        __file__,
        "src/s2a/full_coupled_pose_inverse.py",
        "scripts/full_coupled_pose_independent.py",
        "src/s2a/generated_reference_twist.py",
    ):
        e.read(path)
    failed = REPO / "outputs/full_coupled_all_records_checked_20261026/case04"
    failure = e.json(failed / "failure.json")
    assert failure["checked_rows"] == 71 and failure["error_type"] == "RuntimeError"
    prefix = [json.loads(line) for line in e.read(failed / "reconstructed_steps.jsonl").splitlines()]
    assert len(prefix) == 71 and prefix[-1]["step"] == 70
    physical = REPO / "outputs/current_coupled_control_20261026/case04"
    case = next(physical.glob("simulation/*/*"))
    rows = [json.loads(line) for line in e.read(case / "steps.jsonl").splitlines()]
    row, previous_row = rows[71], prefix[-1]
    controller, chunks = e.json(case / "controller.json"), e.json(case / "chunks.json")
    config = e.json(physical / "execution_config.json")
    template = e.json(config["policy_template"])
    h = np.asarray(e.json(template["tool_contract"])["source_from_tcp"])
    tcp_from_site = np.linalg.inv(h) @ np.asarray(controller["source_from_site"])
    chunk = chunks[row["chunk_index"]]
    reference = reference_world_twist(
        np.asarray(chunk["effective_targets_tcp"]) @ tcp_from_site,
        71 - chunk["step"],
        np.asarray(chunk["anchor_tcp"]) @ tcp_from_site,
        0.05,
    )
    fixture = dict(
        case_index=4,
        step=71,
        contract=controller["contract"],
        mass=row["coupled_dynamics"]["mass"],
        jacobian=row["coupled_dynamics"]["jacobian"],
        measured=row["qualified_forward_input"]["x_site_pose"][0],
        desired=row["desired_site"],
        twist=row["qualified_forward_input"]["x_site_twist"][0],
        reference_twist=reference["twist"].tolist(),
        previous=previous_row["solution"]["command"],
        previous_goal=previous_row["actual_goal"],
        innovation=previous_row["innovation_after"],
    )
    contract = fixture["contract"]
    interface = OSCInterface(*(contract[key] for key in ("input_min", "input_max", "output_min", "output_max")))
    model = FullCoupledPoseModel(fixture["mass"], fixture["jacobian"], contract["kp"], contract["kd"])
    previous = np.asarray(fixture["previous"])
    lower, upper = np.maximum(interface.input_min, previous - 0.25), np.minimum(interface.input_max, previous + 0.25)

    def evaluated(command):
        return task_objective(
            model,
            command,
            interface,
            fixture["measured"],
            fixture["desired"],
            fixture["twist"],
            fixture["reference_twist"],
            fixture["innovation"],
        )

    def cost(command):
        value = evaluated(command)
        return value["objective"], value["gradient"]

    results = []
    for method, options in (
        ("L-BFGS-B", dict(ftol=1e-15, gtol=1e-9, maxiter=200, maxls=40)),
        ("SLSQP", dict(ftol=1e-13, maxiter=500)),
    ):
        started = time.monotonic()
        result = minimize(
            cost,
            np.clip(previous, lower, upper),
            jac=True,
            method=method,
            bounds=list(zip(lower, upper, strict=True)),
            options=options,
        )
        command = np.clip(result.x, lower, upper)
        value = evaluated(command)
        numeric = []
        arguments = (
            fixture["mass"],
            fixture["jacobian"],
            contract,
            fixture["measured"],
            fixture["desired"],
            fixture["twist"],
            fixture["reference_twist"],
            fixture["innovation"],
            fixture["previous_goal"],
            "reset",
        )
        for axis in range(6):
            step = np.eye(6)[axis] * 2e-5
            numeric.append(
                (
                    independent_objective(command + step, *arguments)["objective"]
                    - independent_objective(command - step, *arguments)["objective"]
                )
                / 4e-5
            )
        numeric = np.asarray(numeric)
        results.append(
            dict(
                method=method,
                options=options,
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                iterations=int(result.nit),
                evaluations=int(result.nfev),
                seconds=time.monotonic() - started,
                command=command.tolist(),
                objective=value["objective"],
                analytic_gradient=value["gradient"].tolist(),
                independent_gradient=numeric.tolist(),
                gradient_max_difference=float(np.max(np.abs(numeric - value["gradient"]))),
                projected_gradient_max=float(
                    np.max(np.abs(projected_gradient(value["gradient"], command, lower, upper)))
                ),
                independent_projected_gradient_max=float(
                    np.max(np.abs(projected_gradient(numeric, command, lower, upper)))
                ),
            )
        )
    e.recheck()
    root.mkdir()
    publish(root / "fixture.json", fixture)
    publish(
        root / "diagnosis.json",
        dict(
            complete=True,
            failure_ref=dict(path=str(failed / "failure.json"), sha256=e.hashes[str(failed / "failure.json")]),
            source_sha256=e.hashes,
            results=results,
            interpreter=sys.executable,
            python_version=sys.version,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            stationarity_threshold_unchanged=2e-6,
            physics_steps=0,
            new_fits=0,
            model_calls=0,
        ),
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
