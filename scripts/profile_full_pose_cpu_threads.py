"""Read-only timing of the same admitted pose problem; no controller qualification."""

import argparse
import json
import os
import time

import numpy as np
from full_coupled_pose_independent import verify_pose_solution

from s2a.artifacts import REPO, sha256
from s2a.durable_stage_supervisor import publish
from s2a.full_coupled_pose_inverse import FullCoupledPoseModel, solve_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1, 8), required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["LD_LIBRARY_PATH"] == ""
    assert all(
        os.environ[name] == str(args.threads) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    # This is the completed232 physical run, not the not-yet-started239 run.
    directory = REPO / "outputs/current_coupled_control_20261026/case00"
    case = next(directory.glob("simulation/*/*"))
    log = REPO / "outputs/full_coupled_control_log_20261026/reconstructed_steps.jsonl"
    proof_path = log.with_name("independent_verified.json")
    proof = json.loads(proof_path.read_text())
    assert proof["complete"] and proof["rows"] == 164
    with log.open() as stream:
        row = json.loads(stream.readline())
    controller_path = case / "controller.json"
    controller = json.loads(controller_path.read_text())
    contract = controller["contract"]
    current = np.asarray(row["measured_source"]) @ np.asarray(controller["source_from_site"])
    desired = np.asarray(row["desired_site"])
    packet, dynamics = row["qualified_forward_input"], row["coupled_dynamics"]
    twist = np.asarray(packet["x_site_twist"])[0]
    previous = np.asarray(packet["x_previous_command"])[0, :6]
    previous_goal = np.asarray(packet["x_previous_goal_ori"])[0]
    target_twist = np.asarray(row["site_twist_reference"]["twist"])
    bias = np.asarray(row["coupled_response_history"]["innovation_before"])
    timings, iterations, certificates = [], [], []
    for _ in range(5):
        start = time.perf_counter()
        model = FullCoupledPoseModel(dynamics["mass"], dynamics["jacobian"], contract["kp"], contract["kd"])
        solved = solve_pose(
            model, current, desired, twist, target_twist, previous, previous_goal, contract, innovation=bias
        )
        solve_seconds = time.perf_counter() - start
        wire = json.loads(json.dumps(solved, default=lambda value: value.tolist(), allow_nan=False))
        start = time.perf_counter()
        certificate = verify_pose_solution(
            wire,
            dynamics["mass"],
            dynamics["jacobian"],
            contract,
            current,
            desired,
            twist,
            target_twist,
            previous,
            previous_goal,
            bias,
        )
        verify_seconds = time.perf_counter() - start
        np.testing.assert_allclose(wire["command"], row["full_pose_inverse"]["command"], atol=1e-7, rtol=0)
        timings.append(dict(solve_seconds=solve_seconds, independent_verify_seconds=verify_seconds))
        iterations.append(wire["solver_iterations"])
        certificates.append(certificate)
    root = REPO / "outputs/full_pose_cpu_threads_20261026"
    root.mkdir(exist_ok=True)
    paths = [
        __file__,
        log,
        proof_path,
        controller_path,
        REPO / "src/s2a/full_coupled_pose_inverse.py",
        REPO / "scripts/full_coupled_pose_independent.py",
    ]
    result = dict(
        complete=True,
        threads=args.threads,
        seed=42,
        repeats=5,
        timings=timings,
        median_solve_seconds=float(np.median([value["solve_seconds"] for value in timings])),
        median_verify_seconds=float(np.median([value["independent_verify_seconds"] for value in timings])),
        solver_iterations=iterations,
        certificates=certificates,
        source_sha256={str(path): sha256(path) for path in paths},
        physics_steps=0,
        new_fits=0,
        model_calls=0,
        scope="single admitted same problem;runtime diagnostic only;no supervision policy or frozen worker changed",
        global_performance_or_control_qualification=False,
    )
    publish(root / f"threads{args.threads}.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("threads", "median_solve_seconds", "median_verify_seconds", "solver_iterations")
            }
        )
    )


if __name__ == "__main__":
    main()
