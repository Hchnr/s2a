"""Replay saved actions with read-only controller torque and contact observations."""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from s2a.infrastructure.owned_processes import run_owned_workers
from s2a.infrastructure.publication import publish_result as publish
from scripts.maintenance.run_cpu_tests import snapshot
from scripts.simulation.runtime import prepare_early_causal_innovation_physical as admission
from scripts.simulation.runtime import prepare_quality_rate_repair_recorded_feedback as recorded
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import run_causal_stream_live_cached as cached
from scripts.simulation.runtime import run_causal_stream_recorded_physical as physical
from scripts.simulation.runtime import run_causal_wam_two_window_feedback as historical
from scripts.simulation.runtime import run_ordered_correct_stream as prior
from scripts.simulation.runtime import run_quality_rate_repair_recorded_feedback as old
from scripts.wam.tracking.preview import run_causal_stream_live_48 as live_owner

ROOT = prefix.REPO
CONFIG = ROOT / "configs/tracking/response_models/replayed_contact_torque_20261111.json"
CONFIG_SHA = "6428139dfa8d0706c683a7710e0f8335ae62445540b308bb00fefcb31b94a7fa"
PLAN = ROOT / "plans/v0.0.1/tracking/response_models/replayed_contact_torque_20261111.md"
OUTPUT = ROOT / "outputs/replayed_contact_torque_20261111"
DRY = ROOT / "outputs/replayed_contact_torque_dry_20261111/result.json"
QA = ROOT / "outputs/qa/replayed_contact_torque_20261111/result.json"
MODULE = "scripts.tracking.response_models.replay_contact_torque"
FORBIDDEN = old.FORBIDDEN + ("s2a.research.trajectory_search.innovation_capped_quality_solver",)


def source_refs():
    return old.source_refs() + [
        prefix.reference(path)
        for path in (
            CONFIG,
            PLAN,
            Path(__file__).resolve(),
            ROOT / "scripts/simulation/runtime/prepare_early_causal_innovation_physical.py",
            ROOT / "tests/research/response_models/test_replay_contact_torque.py",
        )
    ]


def isolation():
    prior.checker_isolation()
    prefix.require(all(name not in sys.modules for name in FORBIDDEN), "Replay must not load candidate solvers")


def load_inputs():
    prefix.environment()
    inputs = admission.load_inputs()
    consumed = dict(inputs["source_sha256"])
    config = prefix.read({"path": str(CONFIG), "sha256": CONFIG_SHA}, consumed)
    prefix.require(
        config["schema"] == "s2a.replayed-contact-torque.v1"
        and config["output"] == str(OUTPUT.relative_to(ROOT))
        and [row["name"] for row in config["runs"]] == ["historical", "innovation"]
        and [row["continuation_periods"] for row in config["runs"]] == [46, 43]
        and [row["max_native_steps"] for row in config["runs"]] == [3150, 3075]
        and config["max_native_models"] == config["max_software_contexts"] == 6
        and config["new_model_initializations"] == config["new_model_requests"] == config["new_denoise_calls"] == 0
        and config["fresh_cases"] == 0
        and config["automatic_retry"] is False,
        "The exact replay identities and physical budgets are required",
    )
    branches = {}
    for run in config["runs"]:
        refs = {
            key: {"path": str(ROOT / run[key]), "sha256": run[key + "_sha256"]}
            for key in ("producer", "owner", "audit")
        }
        producer, owner, audit = (prefix.read(refs[key], consumed) for key in ("producer", "owner", "audit"))
        prefix.require(
            producer["complete"] is producer["passed"] is owner["complete"] is owner["passed"] is True
            and owner["sources_unchanged"] is audit["complete"] is audit["passed"] is True
            and audit["checker_policy_isolated"] is True
            and audit["control_qualified"] is False
            and owner["branch_ref"] == refs["producer"]
            and audit["continuation_periods"] == len(recorded.periods(producer)) == run["continuation_periods"]
            and run["max_control_periods"] == 80 + run["continuation_periods"]
            and run["max_native_steps"] == 25 * run["max_control_periods"],
            "A source-pinned, independently replayed producer and its exact physical budget are required",
        )
        branches[run["name"]] = {"config": run, "refs": refs, "producer": producer}
    inputs["replayed_contact_torque"] = {"config": config, "branches": branches, "source_sha256": consumed}
    return inputs


def input_receipt(inputs):
    item = inputs["replayed_contact_torque"]
    return {
        "prior": admission.input_receipt(inputs),
        "config_ref": {"path": str(CONFIG), "sha256": CONFIG_SHA},
        "branches": {key: value["refs"] for key, value in item["branches"].items()},
        "source_sha256": item["source_sha256"],
    }


def contact_reader(sim):
    """Read the solver cache for contacts involving the robot's body subtree."""
    import mujoco

    model, data = sim.model._model, sim.data._data
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
    prefix.require(base >= 0, "The original robot base body must exist")

    def robot_geom(geom):
        body = int(model.geom_bodyid[geom])
        while body:
            if body == base:
                return True
            body = int(model.body_parentid[body])
        return False

    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        on_first, on_second = robot_geom(first), robot_geom(second)
        if not (on_first or on_second):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        prefix.require(np.isfinite(force).all() and np.isfinite(contact.dist), "Finite actual contact data required")
        contacts.append(
            {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, first) or f"geom_{first}",
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, second) or f"geom_{second}",
                "kind": "robot_self" if on_first and on_second else "robot_environment",
                "distance_m": float(contact.dist),
                "wrench_contact_frame": force,
                "active_constraint": int(contact.efc_address) >= 0,
            }
        )
    return contacts


class ReplayRecorder:
    """Wrap native inner steps without changing inputs, state, or step count."""

    def __init__(self, env, periods, *, contacts=contact_reader):
        self.env, self.periods, self.contacts = env, periods, contacts
        self.samples = []
        self.original = env.sim.step
        robot = env.robots[0]
        low, high = robot.torque_limits
        self.lower, self.upper = np.asarray(low, dtype=float).copy(), np.asarray(high, dtype=float).copy()
        prefix.require(
            type(periods) is int
            and 0 < periods <= 48
            and self.lower.shape == self.upper.shape == (7,)
            and np.isfinite(self.lower).all()
            and np.isfinite(self.upper).all()
            and np.all(self.lower < self.upper),
            "One bounded replay and seven finite original torque limits are required",
        )

    def step(self, *args, **kwargs):
        prefix.require(len(self.samples) < self.periods * 25, "The registered inner-step budget is exhausted")
        sim, robot = self.env.sim, self.env.robots[0]
        data = sim.data
        before = sim.get_state().flatten().copy()
        raw = np.asarray(robot.controller.torques, dtype=float).copy()
        clipped = np.asarray(robot.torques, dtype=float).copy()
        control = np.asarray(data.ctrl)[robot._ref_joint_actuator_indexes].copy()
        prefix.require(
            raw.shape == clipped.shape == control.shape == (7,) and np.isfinite(raw).all(), "Arm torque shape"
        )
        np.testing.assert_allclose(clipped, np.clip(raw, self.lower, self.upper), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(control, clipped)
        np.testing.assert_array_equal(before, sim.get_state().flatten())
        time_before = float(data.time)
        value = self.original(*args, **kwargs)
        after = sim.get_state().flatten().copy()
        dofs = robot._ref_joint_vel_indexes
        constraint = np.asarray(data.qfrc_constraint)[dofs].copy()
        actuator = np.asarray(data.qfrc_actuator)[dofs].copy()
        bias = np.asarray(data.qfrc_bias)[dofs].copy()
        contacts = self.contacts(sim)
        np.testing.assert_array_equal(after, sim.get_state().flatten())
        prefix.require(
            all(np.isfinite(item).all() for item in (clipped, control, constraint, actuator, bias))
            and np.isclose(float(data.time) - time_before, 0.002, atol=1e-12, rtol=0),
            "Each observed integration must retain finite forces and the original 2 ms interval",
        )
        count = len(self.samples)
        self.samples.append(
            {
                "step": 80 + count // 25,
                "inner_index": count % 25,
                "time_before": time_before,
                "time_after": float(data.time),
                "controller_raw_torque_nm": raw,
                "commanded_torque_nm": clipped,
                "actuator_control": control,
                "joint_constraint_force_nm": constraint,
                "joint_actuator_force_nm": actuator,
                "joint_bias_force_nm": bias,
                "contacts": contacts,
            }
        )
        return value


def instrumented_continuation(env, terminal, observers, receipt, callback, periods, *, progress=None):
    recorder = ReplayRecorder(env, periods)
    receipt["replay_dynamics"] = {
        "samples": recorder.samples,
        "torque_lower_nm": recorder.lower,
        "torque_upper_nm": recorder.upper,
        "controller_sample_phase": "after_control_before_integration",
        "constraint_sample_phase": "last_inner_step_solver_cache_after_integration",
        "actual_policy_inputs_changed": False,
    }
    env.sim.step = recorder.step
    try:
        result = callback(env, terminal, observers, receipt, progress=progress)
        prefix.require(len(recorder.samples) == periods * 25, "Every completed replay inner step must be observed")
        return result
    finally:
        env.sim.step = recorder.original


def bounded_prefix(old_inputs, directory, continuation, *, periods):
    def fixed_budget(values, output, callback):
        from s2a.research.native_runtime.guarded_stream_prefix import GuardedStreamPrefixRuntime

        runtime = GuardedStreamPrefixRuntime()
        return runtime._run(
            values["runtime_config"],
            values["record"],
            values["record_xml"],
            values["commands"],
            values["saved"],
            values["receipts"],
            output / "runtime",
            values["asset_refs"],
            values["controller"],
            continuation=callback,
            continuation_budget=periods,
            continuation_limit=48,
            registered_seed=42,
            capture_final_prefix_inner=True,
        )

    def observed(env, terminal, observers, receipt, *, progress=None):
        return instrumented_continuation(env, terminal, observers, receipt, continuation, periods, progress=progress)

    adapted = old.clone(cached._run_cached_prefix, {"frozen": SimpleNamespace(_run_prefix=fixed_budget)})
    return adapted(old_inputs, directory, observed)


def worker(name):
    directory = OUTPUT / name / "checker/simulation"
    os.environ.update(TMPDIR=str(directory / "tmp"), XDG_CACHE_HOME=str(directory / "cache"))
    tempfile.tempdir = None
    inputs = load_inputs()
    registration = prefix.read(prefix.reference(OUTPUT / "registration.json"))
    prefix.require(
        registration["inputs"] == input_receipt(inputs) and registration["source_refs"] == source_refs(),
        "Worker inputs and source identities must match the owner registration",
    )
    branch = inputs["replayed_contact_torque"]["branches"][name]
    isolation()
    original_root = Path(branch["refs"]["producer"]["path"]).parents[2] / "producer/simulation/runtime"

    def decode(values, producer, *, image_root, max_windows):
        return cached.references.decode_actual_references(
            values, producer, image_root=original_root, max_windows=max_windows
        )

    def guarded(old_inputs, output, callback):
        return bounded_prefix(old_inputs, output, callback, periods=branch["config"]["continuation_periods"])

    runner = old.clone(
        cached.checker_worker,
        {"references": SimpleNamespace(decode_actual_references=decode), "_run_cached_prefix": guarded},
    )
    runner(output=OUTPUT / name, loader=lambda _phase: (inputs, {"producer_branch_ref": branch["refs"]["producer"]}))
    isolation()


def summarize(branch, producer, config):
    counts = live_owner.summarize_branch(branch, phase="checker", producer=producer)
    runtime = branch["runtime"]
    rows, saved = recorded.periods(branch), recorded.periods(producer)
    prefix.require(
        counts["control_steps"] == config["max_control_periods"]
        and counts["native_steps"] == runtime["limits"]["mj_step"] == config["max_native_steps"]
        and len(rows) == len(saved) == config["continuation_periods"],
        "All original commands and exactly the registered native steps must complete",
    )
    for actual, expected in zip(rows, saved, strict=True):
        prefix.require(actual["step"] == expected["step"], "Replay period identity changed")
        historical.compare_first_period(actual, expected)
    observed = runtime["replay_dynamics"]
    samples = observed["samples"]
    prefix.require(
        len(samples) == 25 * len(rows) and observed["actual_policy_inputs_changed"] is False,
        "Complete observations required",
    )
    periods = []
    for index, row in enumerate(rows):
        chunk = samples[index * 25 : (index + 1) * 25]
        prefix.require(
            [item["step"] for item in chunk] == [row["step"]] * 25
            and [item["inner_index"] for item in chunk] == list(range(25)),
            "Recorded forces must align with the original 25 inner samples",
        )
        clipped = [
            np.max(np.abs(np.asarray(item["controller_raw_torque_nm"]) - item["commanded_torque_nm"])) for item in chunk
        ]
        env = [[c for c in item["contacts"] if c["kind"] == "robot_environment"] for item in chunk]
        active = [[c for c in contacts if c["active_constraint"]] for contacts in env]
        periods.append(
            {
                "step": row["step"],
                "torque_clipped_samples": sum(value > 1e-9 for value in clipped),
                "max_torque_clip_nm": max(clipped),
                "robot_environment_contact_samples": sum(bool(value) for value in env),
                "active_robot_environment_contact_samples": sum(bool(value) for value in active),
                "max_joint_constraint_force_nm": max(
                    float(np.max(np.abs(item["joint_constraint_force_nm"]))) for item in chunk
                ),
                "max_robot_environment_contact_force_n": max(
                    (float(np.linalg.norm(c["wrench_contact_frame"][:3])) for contacts in active for c in contacts),
                    default=0.0,
                ),
            }
        )
    return {**counts, "periods": periods, "replayed_inner_samples": len(samples), "exact_original_replay": True}


def dry_run():
    inputs = load_inputs()
    print(
        publish(
            DRY,
            {
                "complete": True,
                "passed": True,
                "inputs": input_receipt(inputs),
                "source_refs": source_refs(),
                "native_models": 0,
                "native_steps": 0,
                "world_model_calls": 0,
                "fresh_cases": 0,
            },
        ),
        flush=True,
    )


def execute():
    inputs = load_inputs()
    sources, before = source_refs(), snapshot(ROOT)["sources"]
    qa, dry = prefix.read(prefix.reference(QA)), prefix.read(prefix.reference(DRY))
    prefix.require(
        qa["complete"] is qa["passed"] is qa["all_tests_passed"] is qa["source_unchanged"] is True
        and qa["actual_wait"] is qa["reaped"] is True
        and qa["signal_sent"] is False
        and qa["configured_parallelism"] == 64
        and qa["skipped_count"] == qa["xfail_count"] == qa["xpass_count"] == 0
        and not qa["collection"]["collection_errors"]
        and prefix.read(qa["source_before_ref"])["sources"] == before
        and dry["complete"] is dry["passed"] is True
        and dry["inputs"] == input_receipt(inputs)
        and dry["source_refs"] == sources
        and dry["native_models"] == dry["native_steps"] == 0,
        "Source-matched 64-worker QA and zero-work dry are required",
    )
    OUTPUT.mkdir(exist_ok=False)
    registration = publish(OUTPUT / "registration.json", {"inputs": input_receipt(inputs), "source_refs": sources})
    result = {
        "complete": False,
        "passed": False,
        "registration_ref": registration,
        "world_model_calls": 0,
        "fresh_cases": 0,
        "automatic_retry": False,
        "control_qualified": False,
    }
    try:
        specs = []
        for name in inputs["replayed_contact_torque"]["branches"]:
            directory = OUTPUT / name / "checker/simulation"
            directory.mkdir(parents=True)
            physical._cpu_environment(directory)
            specs.append(
                {
                    "key": name,
                    "argv": [sys.executable, "-m", MODULE, "--worker", name],
                    "log": OUTPUT / name / "worker.log",
                }
            )
        lifecycle = run_owned_workers(
            specs, cwd=ROOT, timeout_seconds=inputs["replayed_contact_torque"]["config"]["worker_timeout_seconds"]
        )
        result["workers_ref"] = publish(OUTPUT / "workers.json", lifecycle)
        prefix.require(
            lifecycle["launch_error"] is None
            and len(lifecycle["workers"]) == 2
            and all(
                item["actual_wait"] is item["reaped"] is True
                and item["returncode"] == 0
                and item["signal_sent"] is False
                for item in lifecycle["workers"]
            ),
            "Both isolated replay workers must finish and be reaped",
        )
        summaries = {}
        for name, source in inputs["replayed_contact_torque"]["branches"].items():
            ref = prefix.reference(OUTPUT / name / "checker/simulation/result.json")
            branch = prefix.read(ref)
            summaries[name] = {"branch_ref": ref, "counts": summarize(branch, source["producer"], source["config"])}
        prefix.require(
            source_refs() == sources and snapshot(ROOT)["sources"] == before, "Replay source identity changed"
        )
        for path, digest in inputs["replayed_contact_torque"]["source_sha256"].items():
            prefix.checked({"path": path, "sha256": digest})
        result.update(complete=True, passed=True, sources_unchanged=True, summaries=summaries)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        print(publish(OUTPUT / "result.json", result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--worker", choices=("historical", "innovation"))
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.execute:
        execute()
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
