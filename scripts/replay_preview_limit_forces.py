"""Instrument original deterministic case00 branch without changing its dynamics."""

from pathlib import Path

import numpy as np

from s2a.artifacts import REPO
from s2a.clean_training_runtime import publish, read, reference, require
from s2a.native_environment import strict_native_environment


def main():
    strict_native_environment()
    root = REPO / "outputs/preview_limit_force_replay_20261027"
    root.mkdir(exist_ok=False)
    registration = dict(
        source_ref=reference(Path(__file__)),
        worker_ref=reference(REPO / "scripts/run_preview_chunk_branch.py"),
        plan_ref=reference(REPO / "plans/v0.0.1/preview_limit_force_replay_20261027.md"),
        original_rows_ref=reference(REPO / "outputs/preview_chunk_branches_20261027/case00_h40/steps.json"),
        physics_budget=128,
        new_model_calls=0,
        fresh_cases=0,
        automatic_retry=False,
    )
    registration_ref = publish(root / "registration.json", registration)
    import run_preview_chunk_branch as branch

    import s2a.libero_simulation as simulation

    original_make = simulation.make_environment
    inner_rows = []

    def instrumented_make(*args, **kwargs):
        env, obs = original_make(*args, **kwargs)
        import mujoco

        original_step = env.sim.step
        model, data = env.sim.model._model, env.sim.data._data
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot0_joint7")
        dof, qid = int(model.jnt_dofadr[joint]), int(model.jnt_qposadr[joint])

        def step():
            value = original_step()
            # Read-only: these are solver caches from this actual inner step.
            before = env.sim.get_state().flatten().copy()
            forces = [
                float(data.efc_force[k])
                for k in range(data.nefc)
                if int(data.efc_type[k]) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
                and int(data.efc_id[k]) == joint
            ]
            inner_rows.append(
                dict(
                    inner_step=len(inner_rows),
                    time_s=float(data.time),
                    q7_rad=float(data.qpos[qid]),
                    q7_constraint_torque_nm=float(data.qfrc_constraint[dof]),
                    joint7_limit_forces=forces,
                )
            )
            np.testing.assert_array_equal(before, env.sim.get_state().flatten())
            return value

        env.sim.step = step
        return env, obs

    simulation.make_environment = instrumented_make
    branch.ROOT = root / "branches"
    try:
        branch.execute(0, 40)
        actual_ref = reference(branch.ROOT / "case00_h40/steps.json")
        require(read(actual_ref) == read(registration["original_rows_ref"]), "Instrumented branch differs")
        require(len(inner_rows) == 128 * 25, "Actual fixed20Hz/2ms inner-step budget differs")
        inner_ref = publish(root / "inner_steps.json", inner_rows)
        publish(
            root / "verified.json",
            dict(
                complete=True,
                passed=True,
                registration_ref=registration_ref,
                actual_ref=actual_ref,
                inner_ref=inner_ref,
                physical_steps=128,
                inner_steps=len(inner_rows),
                new_model_calls=0,
                fresh_cases=0,
                all40_branch_rows_exact=True,
                branch_joint7_limit_active_inner_steps=sum(
                    bool(r["joint7_limit_forces"]) for r in inner_rows[88 * 25 :]
                ),
                branch_max_abs_joint7_constraint_torque_nm=max(
                    abs(r["q7_constraint_torque_nm"]) for r in inner_rows[88 * 25 :]
                ),
                scope="exact original branch with read-only post-inner-step solver cache observation",
            ),
        )
    finally:
        simulation.make_environment = original_make


if __name__ == "__main__":
    main()
