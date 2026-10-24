#!/usr/bin/env python3
"""Inspect fixed saved native endpoints with globally guarded static MuJoCo."""

import argparse
import ctypes
import functools
import importlib
import inspect
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from collect_wam_native_training import reference, require, write_once  # noqa: E402

from s2a.native_environment import (  # noqa: E402
    EnvironmentAdmissionError,
    environment_snapshot,
    restore_cv2_environment,
    strict_native_environment,
)
from s2a.native_training import state_sha256  # noqa: E402


class StaticCallGuard:
    """Count at entry, persist violations, and reject before any delegation."""

    FORBIDDEN = {
        "env_step_calls",
        "sim_step_calls",
        "mj_step_calls",
        "mj_step2_calls",
        "render_calls",
        "source_hdf5_reads",
        "controller_command_calls",
    }
    KINDS = FORBIDDEN | {
        "mj_step1_calls",
        "forward_calls",
        "official_success_calls",
        "additional_atomic_predicate_calls",
        "single_reset_calls",
        "internal_On_calls",
        "internal_under_calls",
        "internal_check_ontop_calls",
        "internal_check_contact_calls",
        "environment_initializations",
        "installed_saved_endpoints",
    }

    def __init__(self, ledger_path):
        self.stream = Path(ledger_path).open("x")
        self.counts = Counter({key: 0 for key in self.KINDS})
        self.phase_counts = defaultdict(Counter)
        self.phase, self.case_name, self.frame_index = "bootstrap", None, None
        self.ledger_count = 0
        self.violation = None
        self.patches = []
        self.hooks = []

    def event(self, value):
        value = dict(
            event_index=self.ledger_count,
            phase=self.phase,
            case_name=self.case_name,
            frame_index=self.frame_index,
            **value,
        )
        self.stream.write(json.dumps(value, allow_nan=False) + "\n")
        self.stream.flush()
        self.ledger_count += 1

    def time_of(self, args):
        if len(args) >= 2 and hasattr(args[1], "time"):
            return float(args[1].time)
        if args and hasattr(args[0], "sim"):
            return float(args[0].sim.data.time)
        return None

    def invoke(self, kind, name, original, args, kwargs):
        self.counts[kind] += 1
        self.phase_counts[self.phase][kind] += 1
        before = self.time_of(args)
        blocked = self.violation is not None or kind in self.FORBIDDEN
        if kind == "mj_step1_calls":
            blocked |= self.phase != "setup" or self.counts[kind] > 64
        if kind == "official_success_calls":
            blocked |= self.phase not in ("setup", "static") or self.counts[kind] > 32
            blocked |= self.phase_counts[self.phase][kind] > 16
        if kind == "additional_atomic_predicate_calls":
            blocked |= self.phase != "static" or self.counts[kind] > 64
        if kind == "single_reset_calls":
            blocked |= self.phase != "setup"
        if blocked:
            self.violation = self.violation or dict(
                api=name, kind=kind, phase=self.phase, case_name=self.case_name, frame_index=self.frame_index
            )
            self.event(dict(api=name, kind=kind, event="rejected_before_delegate", time_before=before, delegated=False))
            raise RuntimeError(f"static diagnostic rejected {name} before delegation: {self.violation}")
        self.event(dict(api=name, kind=kind, event="before_delegate", time_before=before, delegated=False))
        self.counts[kind + "_delegated"] += 1
        self.phase_counts[self.phase][kind + "_delegated"] += 1
        try:
            value = original(*args, **kwargs)
        except BaseException as error:
            self.event(
                dict(
                    api=name,
                    kind=kind,
                    event="delegate_raised",
                    time_before=before,
                    time_after=self.time_of(args),
                    delegated=True,
                    error=dict(type=type(error).__name__, message=str(error)),
                )
            )
            raise
        after = self.time_of(args)
        self.event(
            dict(api=name, kind=kind, event="after_delegate", time_before=before, time_after=after, delegated=True)
        )
        if kind in ("mj_step1_calls", "forward_calls") and before is not None:
            require(before == after, f"non-integrating {name} changed simulation time")
        return value

    def patch(self, owner, attribute, kind, name=None, replacement=None):
        original = getattr(owner, attribute)
        delegated = original if replacement is None else replacement
        label = name or f"{getattr(owner, '__name__', type(owner).__name__)}.{attribute}"

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            return self.invoke(kind, label, delegated, args, kwargs)

        self.patches.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)
        self.hooks.append(
            dict(
                api=label,
                kind=kind,
                original_module=getattr(original, "__module__", None),
                original_qualname=getattr(original, "__qualname__", None),
                replacement="single_delegate_no_retry_process_local" if replacement else None,
            )
        )

    @contextmanager
    def context(self, phase, case_name=None, frame_index=None):
        previous = self.phase, self.case_name, self.frame_index
        self.phase, self.case_name, self.frame_index = phase, case_name, frame_index
        try:
            yield
        finally:
            self.phase, self.case_name, self.frame_index = previous

    def extra(self, name, function, *args):
        return self.invoke("additional_atomic_predicate_calls", name, function, args, {})

    def close(self):
        for owner, attribute, original in reversed(self.patches):
            setattr(owner, attribute, original)
        self.stream.close()


def single_reset(instance):
    """One native reset delegation; never enter ControlEnv's swallowing loop."""
    return instance.env.reset()


def install_low_level_guards(guard, mujoco):
    for name, kind in (
        ("mj_step", "mj_step_calls"),
        ("mj_step2", "mj_step2_calls"),
        ("mj_step1", "mj_step1_calls"),
        ("mj_forward", "forward_calls"),
        ("mjr_render", "render_calls"),
    ):
        guard.patch(mujoco, name, kind, "mujoco." + name)
    h5py = importlib.import_module("h5py")
    guard.patch(h5py, "File", "source_hdf5_reads", "h5py.File")
    guard.patch(h5py.h5f, "open", "source_hdf5_reads", "h5py.h5f.open")
    guard.patch(h5py.h5f, "create", "source_hdf5_reads", "h5py.h5f.create")


def registered_tabletop_class(mapping, base):
    actual = mapping.get("libero_tabletop_manipulation")
    require(
        isinstance(actual, type)
        and issubclass(actual, base)
        and actual.__name__ == "Libero_Tabletop_Manipulation"
        and actual.__module__ == "libero.libero.envs.problems.libero_tabletop_manipulation"
        and callable(getattr(actual, "_check_success", None)),
        "actual registered tabletop class identity differs",
    )
    return actual


def install_runtime_guards(guard):
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, BDDLBaseDomain
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.object_states.base_object_states import SiteObjectState
    from libero.libero.envs.objects.site_object import SiteObject
    from libero.libero.envs.predicates.base_predicates import On
    from robosuite.environments.base import MujocoEnv
    from robosuite.robots.single_arm import SingleArm
    from robosuite.utils.binding_utils import MjRenderContext, MjSim

    actual_tabletop = registered_tabletop_class(TASK_MAPPING, BDDLBaseDomain)
    guard.patch(ControlEnv, "reset", "single_reset_calls", replacement=single_reset)
    for owner in (ControlEnv, BDDLBaseDomain, MujocoEnv):
        guard.patch(owner, "step", "env_step_calls")
    guard.patch(MjSim, "step", "sim_step_calls")
    guard.patch(MjSim, "render", "render_calls")
    guard.patch(MjRenderContext, "render", "render_calls")
    guard.patch(MujocoEnv, "render", "render_calls")
    guard.patch(SingleArm, "control", "controller_command_calls")
    guard.patch(actual_tabletop, "_check_success", "official_success_calls")
    guard.patch(On, "__call__", "internal_On_calls")
    guard.patch(SiteObjectState, "check_ontop", "internal_check_ontop_calls")
    guard.patch(SiteObject, "under", "internal_under_calls")
    guard.patch(MujocoEnv, "check_contact", "internal_check_contact_calls")


def bootstrap_static_runtime(policy, destination, guard):
    """Select and preload OSMesa before MuJoCo imports its GL backend."""
    os.environ.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
    ctypes.CDLL(policy["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
    mujoco = importlib.import_module("mujoco")
    install_low_level_guards(guard, mujoco)
    from s2a.libero_simulation import configure_simulator

    configure_simulator(policy, destination / "runtime_bootstrap")
    install_runtime_guards(guard)
    strict_native_environment()


def raw_state_record(env):
    state = np.asarray(env.sim.get_state().flatten()).copy()
    return dict(values=state.tolist(), dtype=str(state.dtype), shape=list(state.shape), sha256=state_sha256(state))


def object_record(env, name):
    model = env.get_object(name)
    require(model is not None, f"missing actual object {name}")
    sim = env.sim
    body_id = int(env.obj_body_id[name])
    geoms = []
    for geom_name in model.contact_geoms:
        geom_id = int(sim.model.geom_name2id(geom_name))
        geoms.append(
            dict(
                name=geom_name,
                id=geom_id,
                body_id=int(sim.model.geom_bodyid[geom_id]),
                size=np.asarray(sim.model.geom_size[geom_id]).tolist(),
                type=int(sim.model.geom_type[geom_id]),
                world_pos=np.asarray(sim.data.geom_xpos[geom_id]).tolist(),
                world_mat=np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3).tolist(),
            )
        )
    return dict(
        name=name,
        actual_class=type(model).__name__,
        root_body=model.root_body,
        body_id=body_id,
        body_name=sim.model.body_id2name(body_id),
        world_pos=np.asarray(sim.data.body_xpos[body_id]).tolist(),
        world_mat=np.asarray(sim.data.body_xmat[body_id]).reshape(3, 3).tolist(),
        world_quaternion_wxyz=np.asarray(sim.data.body_xquat[body_id]).tolist(),
        contact_geoms=geoms,
    )


def contact_records(env):
    result = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        result.append(
            dict(
                index=index,
                geom1_id=int(contact.geom1),
                geom2_id=int(contact.geom2),
                geom1_name=env.sim.model.geom_id2name(contact.geom1),
                geom2_name=env.sim.model.geom_id2name(contact.geom2),
                dist=float(contact.dist),
                pos=np.asarray(contact.pos).tolist(),
                frame=np.asarray(contact.frame).reshape(3, 3).tolist(),
            )
        )
    return result


def under_margins(site_pos, site_mat, python_size, wine_pos, other_height):
    """Literal SiteObject.under convention; do not substitute its transpose."""
    size = np.asarray(python_size, dtype=np.float64)
    delta = np.asarray(site_mat, dtype=np.float64) @ (
        np.asarray(wine_pos, dtype=np.float64) - np.asarray(site_pos, dtype=np.float64)
    )
    return dict(
        delta_position=delta.tolist(),
        xy_lower=(size[:2] - np.abs(delta[:2])).tolist(),
        z_lower=float(delta[2] - (size[2] - 0.005)),
        z_upper=float(size[2] + other_height - delta[2]),
    )


def capture_endpoint(env, spec, frame, saved_state, expected_source, original_success, receipt, guard):
    from s2a.libero_simulation import current_observation, source_pose

    raw_before_install = raw_state_record(env)
    guard.counts["installed_saved_endpoints"] += 1
    guard.phase_counts[guard.phase]["installed_saved_endpoints"] += 1
    env.sim.set_state_from_flattened(saved_state)
    installed = raw_state_record(env)
    expected_hash = state_sha256(saved_state)
    require(installed["sha256"] == expected_hash == receipt["sim_state_sha256"], "saved state installation differs")
    guard.event(
        dict(
            event="installed_saved_state",
            api="MjSim.set_state_from_flattened",
            time_before=raw_before_install["values"][0],
            time_after=installed["values"][0],
            source_state_sha256=expected_hash,
            integration=False,
        )
    )
    env.sim.forward()
    obs = current_observation(env)
    actual_source = source_pose(obs)
    require(np.allclose(actual_source, expected_source, rtol=0, atol=1e-10), "restored source observation differs")
    before = raw_state_record(env)
    require(before["sha256"] == expected_hash, "forward changed installed state")
    goal = env.env.parsed_problem["goal_state"]
    require(
        len(goal) == 1
        and len(goal[0]) == 3
        and str(goal[0][0]).lower() == "on"
        and goal[0][1:] == ["wine_bottle_1", "wine_rack_1_top_region"],
        "unexpected actual task goal",
    )
    official = bool(env.check_success())
    wine_name, site_name = goal[0][1:]
    region_state = env.env.object_states_dict[site_name]
    region = env.env.object_sites_dict[site_name]
    require(region_state.object_state_type == "site" and hasattr(region, "under"), "unexpected On object branch")
    parent_name = region_state.parent_name
    require(parent_name == region.parent_name == "wine_rack_1", "actual region parent differs")
    wine, rack = object_record(env.env, wine_name), object_record(env.env, parent_name)
    site_id = int(env.sim.model.site_name2id(site_name))
    site_pos = np.asarray(env.sim.data.get_site_xpos(site_name)).copy()
    site_mat = np.asarray(env.sim.data.get_site_xmat(site_name)).reshape(3, 3).copy()
    size = np.asarray(region.size).copy()
    require(size.shape == (3,) and np.isfinite(size).all(), "actual Python region size invalid")
    other_height = inspect.signature(region.under).parameters["other_height"].default
    require(other_height == 0.10, "actual region under default height differs")
    under = bool(
        guard.extra("SiteObject.under.explicit", region.under, site_pos, site_mat, np.asarray(wine["world_pos"]))
    )
    contact = bool(
        guard.extra(
            "MujocoEnv.check_contact.explicit",
            env.env.check_contact,
            env.env.get_object(parent_name),
            env.env.get_object(wine_name),
        )
    )
    ontop = bool(guard.extra("Libero_Tabletop_Manipulation._eval_predicate.explicit", env.env._eval_predicate, goal[0]))
    after = raw_state_record(env)
    require(after["sha256"] == before["sha256"], "predicate observation changed installed state")
    return dict(
        schema_version="s2a.native-terminal-predicate-endpoint.v1",
        name=spec["name"],
        role=spec["role"],
        frame_index=frame,
        source_state_ref=dict(arrays=spec["arrays_ref"], key="sim_states", frame_index=frame),
        source_receipt_ref=spec["endpoint_receipts_ref"],
        original_success=bool(original_success),
        official_goal=goal,
        official_success=official,
        static_success_reproduced=official == bool(original_success),
        actual_raw_state_before=before,
        actual_raw_state_after=after,
        sim_time_before=float(before["values"][0]),
        sim_time_after=float(after["values"][0]),
        actual_source_pose=actual_source.tolist(),
        atoms=dict(under=under, contact=contact, ontop=ontop),
        under_inputs=dict(
            site_name=site_name,
            site_id=site_id,
            site_body_id=int(env.sim.model.site_bodyid[site_id]),
            site_world_pos=site_pos.tolist(),
            site_world_mat=site_mat.tolist(),
            python_size=size.tolist(),
            python_size_dtype=str(size.dtype),
            model_site_size=np.asarray(env.sim.model.site_size[site_id]).tolist(),
            wine_body_id=wine["body_id"],
            wine_body_name=wine["body_name"],
            wine_world_pos=wine["world_pos"],
            parent_name=parent_name,
            region_state_parent_name=region_state.parent_name,
            python_region_class=type(region).__name__,
            other_height=other_height,
            has_under=True,
            parent_exists=True,
        ),
        geometry_margins=under_margins(site_pos, site_mat, size, wine["world_pos"], other_height),
        objects=dict(wine=wine, rack=rack),
        contacts=contact_records(env),
        ncon=int(env.sim.data.ncon),
    )


def checked_case_arrays(spec):
    require(reference(spec["arrays_ref"]["path"]) == spec["arrays_ref"], "source arrays SHA changed")
    require(
        reference(spec["endpoint_receipts_ref"]["path"]) == spec["endpoint_receipts_ref"],
        "source endpoint receipts SHA changed",
    )
    receipts = [json.loads(line) for line in Path(spec["endpoint_receipts_ref"]["path"]).read_text().splitlines()]
    with np.load(spec["arrays_ref"]["path"], allow_pickle=False) as archive:
        state = archive["sim_states"]
        success = archive["native_success"]
        source = archive["source_pose"]
        selected = []
        for index, frame in enumerate(spec["frame_indices"]):
            require(success[frame] == spec["expected_native_success"][index], "registered endpoint truth changed")
            require(
                receipts[frame]["native_success"] == bool(success[frame]) and receipts[frame]["endpoint_step"] == frame,
                "original endpoint truth binding differs",
            )
            selected.append((frame, state[frame].copy(), source[frame].copy(), bool(success[frame]), receipts[frame]))
    return selected


def validate_counts(guard, endpoints):
    counts, phases = guard.counts, guard.phase_counts
    require(guard.violation is None, "a forbidden operation was attempted")
    require(
        counts["environment_initializations"] == 4 and counts["installed_saved_endpoints"] == len(endpoints) == 16,
        "fixed4/16 denominator incomplete",
    )
    require(
        phases["static"]["official_success_calls"] == 16
        and phases["setup"]["official_success_calls"] <= 16
        and counts["official_success_calls"] <= 32,
        "official success phase budget differs",
    )
    require(
        phases["setup"]["mj_step1_calls"] <= 64 and phases["static"]["mj_step1_calls"] == 0,
        "mj_step1 phase budget differs",
    )
    require(counts["additional_atomic_predicate_calls"] <= 64, "explicit extra atomic budget exceeded")
    require(all(counts[key] == 0 for key in StaticCallGuard.FORBIDDEN), "forbidden operation count is nonzero")


def finalize_collection(result, guard):
    """Full evidence proceeds to independent audit even when truth differs."""
    validate_counts(guard, result["endpoints"])
    points = []
    for item in result["endpoints"]:
        ref = item["bundle"]
        require(reference(ref["path"]) == ref, "captured endpoint changed before finalization")
        points.append(json.loads(Path(ref["path"]).read_text()))
    result["all_saved_states_reproduced"] = all(point["static_success_reproduced"] for point in points)
    result["complete"] = True
    result["passed"] = result["all_saved_states_reproduced"]
    if not result["passed"]:
        result["error"] = dict(
            type="StaticPredicateReproductionMismatch",
            message="Original native truth differs; refuse causal attribution",
        )
    return 0  # Independent audit receives all16, including negative evidence.


def run(args):
    strict_native_environment()
    require(args.seed == 42, "fixed seed42 required")
    from run_native_terminal_predicate_diagnostic import read_prepared

    prepared_path = Path(args.prepared).resolve()
    prepared = read_prepared(prepared_path, args.prepared_sha256)
    require(
        prepared["schema_version"] == "s2a.native-terminal-predicate-preparation.v1",
        "static preparation schema differs",
    )
    destination = Path(prepared["output_root"]) / "collection"
    destination.mkdir(exist_ok=False)
    result = dict(
        schema_version="s2a.native-terminal-predicate-collection.v1",
        complete=False,
        passed=False,
        prepared_ref=reference(prepared_path),
        cases=[],
        endpoints=[],
        original_native_quality_passed=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        automatic_retry=False,
    )
    guard = None
    exit_code = 1
    try:
        try:
            readiness = restore_cv2_environment(prepared["cv2_probe_ref"], prepared["cv2_runtime_source_refs"])
        except EnvironmentAdmissionError as error:
            result["environment_readiness_ref"] = write_once(destination / "environment_readiness.json", error.receipt)
            raise
        readiness["prepared_ref"] = result["prepared_ref"]
        result["environment_readiness_ref"] = write_once(destination / "environment_readiness.json", readiness)
        write_once(
            destination / "worker_started.json",
            dict(
                pid=os.getpid(),
                ppid=os.getppid(),
                argv=list(sys.argv),
                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                ld=os.environ.get("LD_LIBRARY_PATH"),
                prepared_ref=result["prepared_ref"],
                environment_readiness_ref=result["environment_readiness_ref"],
            ),
        )
        guard = StaticCallGuard(destination / "call_ledger.jsonl")
        bootstrap_static_runtime(prepared["policy_config"], destination, guard)
        from s2a.libero_simulation import make_environment

        for spec in prepared["cases"]:
            selected = checked_case_arrays(spec)
            case_dir = destination / spec["name"]
            case_dir.mkdir()
            policy = dict(
                prepared["policy_config"],
                task_id=spec["record"]["task_id"],
                seed=42,
                camera_observations=False,
                refresh_controller_after_restore=True,
            )
            env = None
            try:
                with guard.context("setup", spec["name"]):
                    guard.counts["environment_initializations"] += 1
                    guard.phase_counts["setup"]["environment_initializations"] += 1
                    guard.event(dict(event="environment_initialization", api="make_environment", integration=False))
                    env, _ = make_environment(policy, case_dir, spec["record"], trial=0)
                strict_native_environment()
                references = []
                for frame, state, source, truth, receipt in selected:
                    with guard.context("static", spec["name"], frame):
                        point = capture_endpoint(env, spec, frame, state, source, truth, receipt, guard)
                    bundle = write_once(case_dir / f"endpoint{frame:04d}.json", point)
                    item = dict(name=spec["name"], role=spec["role"], frame_index=frame, bundle=bundle)
                    references.append(item)
                    result["endpoints"].append(item)
                result["cases"].append(
                    dict(
                        name=spec["name"],
                        role=spec["role"],
                        frame_indices=spec["frame_indices"],
                        endpoints=references,
                        case_result_ref=spec["case_result_ref"],
                    )
                )
            finally:
                if env is not None:
                    env.close()
            strict_native_environment()
            read_prepared(prepared_path, args.prepared_sha256)
        exit_code = finalize_collection(result, guard)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        if guard is not None:
            result["counts"] = dict(guard.counts)
            result["phase_counts"] = {phase: dict(counts) for phase, counts in guard.phase_counts.items()}
            result["installed_hooks"] = guard.hooks
            result["violation"] = guard.violation
            result["ledger_events"] = guard.ledger_count
            guard.close()
            result["call_ledger_ref"] = reference(destination / "call_ledger.jsonl")
        result["final_environment"] = environment_snapshot()
        write_once(destination / "result.json", result)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
