"""Independent NumPy audit of16 saved-state native terminal predicates.

No simulator or producer predicate is imported. Contact validation reconstructs
membership of exported actual active contacts; it is not a collision solver.
The original native28/30 qualification and all training permissions stay False.
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/native_terminal_predicate_diagnostic_20261021.json"
CONFIG_SHA = "0dc500ff423a7ade96b537a30d0945a2fc62b36ce228259637739189e68c7992"
PLAN = REPO / "plans/v0.0.1/030_native_terminal_predicates_20261021.md"
PLAN_SHA = "e733e9b5c78c394c6d2bfbc989c1b23288a4e5893cd7f347aae6a085bc5c7c27"
NEW_SOURCES = (
    "plans/v0.0.1/030_native_terminal_predicates_20261021.md",
    "configs/native_terminal_predicate_diagnostic_20261021.json",
    "scripts/run_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_runner.py",
    "scripts/collect_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_collection.py",
    "scripts/verify_native_terminal_predicate_diagnostic.py",
    "tests/test_native_terminal_predicate_verification.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, shape, name):
    array = np.asarray(value)
    require(array.shape == shape and array.dtype.kind in "iuf" and np.isfinite(array).all(), f"invalid {name}")
    return array.astype(np.float64)


def close(actual, expected, name, *, atol=1e-12):
    expected = np.asarray(expected)
    actual = numeric(actual, expected.shape, name)
    require(
        np.isfinite(expected).all() and np.allclose(actual, expected, rtol=0, atol=atol), f"{name}: numerical mismatch"
    )


def rotation(value, name):
    matrix = numeric(value, (3, 3), name)
    close(matrix.T @ matrix, np.eye(3), name + " orthogonality", atol=1e-10)
    require(abs(np.linalg.det(matrix) - 1) <= 1e-10, name + " is not SO(3)")
    return matrix


def under_geometry(site_position, site_matrix, size, wine_position, *, other_height=0.10):
    """Reconstruct SiteObject.under's actual R@d and strict point inequalities."""
    center = numeric(site_position, (3,), "site world position")
    matrix = rotation(site_matrix, "actual site world matrix")
    size = numeric(size, (3,), "Python SiteObject size")
    wine = numeric(wine_position, (3,), "wine root BODY world position")
    require(np.all(size >= 0) and other_height == 0.10, "unexpected site extent or height convention")
    delta = matrix @ (wine - center)
    xy = size[:2] - np.abs(delta[:2])
    lower = delta[2] - (size[2] - 0.005)
    upper = (size[2] + other_height) - delta[2]
    inside = bool(np.all(xy > 0) and lower > 0 and upper > 0)
    return dict(
        under=inside, delta_position=delta.tolist(), xy_lower=xy.tolist(), z_lower=float(lower), z_upper=float(upper)
    )


def contact_membership(contacts, wine_geoms, rack_geoms):
    """The actual contact predicate uses both orientations and no distance cut."""
    require(
        isinstance(wine_geoms, list)
        and isinstance(rack_geoms, list)
        and wine_geoms
        and rack_geoms
        and all(isinstance(name, str) and name for name in wine_geoms + rack_geoms)
        and len(set(wine_geoms)) == len(wine_geoms)
        and len(set(rack_geoms)) == len(rack_geoms),
        "complete named contact-geom groups required",
    )
    wine, rack = set(wine_geoms), set(rack_geoms)
    matches, identities = [], {}
    for index, item in enumerate(contacts):
        require(item["index"] == index, "actual complete active-contact ordering differs")
        for endpoint in ("geom1", "geom2"):
            geom_id, name = item[endpoint + "_id"], item[endpoint + "_name"]
            require(
                type(geom_id) is int and geom_id >= 0 and (name is None or isinstance(name, str)),
                "invalid actual contact identity",
            )
            require(geom_id not in identities or identities[geom_id] == name, "contact geom ID/name changed")
            identities[geom_id] = name
        require(np.isfinite(item["dist"]), "nonfinite actual contact distance")
        one, two = item["geom1_name"], item["geom2_name"]
        if (one in wine and two in rack) or (one in rack and two in wine):
            matches.append(index)
    return dict(contact=bool(matches), matching_contact_indices=matches)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        require(
            path.suffix not in (".hdf5", ".h5", ".png", ".npy", ".pt", ".pth", ".safetensors"),
            "forbidden HDF/image/expert/weight content read",
        )
        digest = sha256(path)
        require(expected is None or digest == expected, f"diagnostic source SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, f"diagnostic input changed: {path}")
        self.hashes[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def ref(self, ref):
        require(set(ref) == {"path", "sha256"}, "invalid diagnostic evidence ref")
        return self.json(ref["path"], ref["sha256"])

    def recheck(self):
        for path, digest in self.hashes.copy().items():
            self.check(path, digest)


def raw_state(record, name):
    require(record["dtype"] == "float64", name + " dtype differs")
    values = numeric(record["values"], tuple(record["shape"]), name)
    require(values.ndim == 1 and values.size > 0, name + " must be a flattened state")
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    require(record["sha256"] == digest, name + " raw bytes differ")
    return values


def source_closure(prepared, native, evidence):
    require(
        len(native["source_sha256"]) == len(native["snapshot_sha256"]) == 423, "original423 source closure required"
    )
    sources = dict(native["source_sha256"])
    for relative in NEW_SOURCES:
        path = str((REPO / relative).resolve())
        require(path not in sources, "diagnostic source overlaps frozen423")
        sources[path] = sha256(path)
    require(prepared["source_sha256"] == sources and len(sources) == 431, "exact423 plus registered8 sources required")
    snapshots = dict(native["snapshot_sha256"])
    for index, relative in enumerate(NEW_SOURCES):
        path = Path(prepared["output_root"]) / "preparation/snapshots" / (f"{index:02d}_" + relative.replace("/", "__"))
        snapshots[str(path)] = sources[str((REPO / relative).resolve())]
    require(prepared["snapshot_sha256"] == snapshots and len(snapshots) == 431, "431 snapshot/source mapping differs")
    for mapping in (sources, snapshots, native["first_continuation_snapshot_sha256"]):
        for path, digest in mapping.items():
            evidence.check(path, digest)


def runtime_inventory(config, expected, evidence):
    root = Path(config["predicate_runtime_root"]).resolve()
    require(
        config["predicate_runtime_extensions"] == [".py", ".bddl"] and root.is_dir(),
        "fixed predicate source inventory required",
    )
    paths = sorted(path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix in (".py", ".bddl"))
    require(
        paths and all(path.is_relative_to(root) for path in paths), "predicate inventory escaped its registered root"
    )
    actual = {str(path): sha256(path) for path in paths}
    require(actual == expected, "actual full predicate source inventory changed")
    for path, digest in actual.items():
        evidence.check(path, digest)


def verify_preparation(path, expected_sha, evidence):
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    prepared = evidence.json(path, expected_sha)
    root = (REPO / config["output_dir"]).resolve()
    require(
        prepared["schema_version"] == "s2a.native-terminal-predicate-preparation.v1"
        and prepared["seed"] == config["seed"] == 42
        and prepared["config"] == config
        and prepared["config_ref"] == dict(path=str(CONFIG), sha256=CONFIG_SHA)
        and prepared["registration_ref"] == dict(path=str(PLAN), sha256=PLAN_SHA)
        and root == REPO / "outputs/wam_stability_20261021/native_terminal_predicate_diagnostic_registry_v2"
        and prepared["output_root"] == str(root)
        and Path(path).resolve() == root / "preparation/prepared.json"
        and prepared["budget"] == config["budget"],
        "static predicate preparation/registration/budget/root differs",
    )
    prior = {
        key: evidence.ref(value) for key, value in config.items() if key.startswith("native_") and key.endswith("_ref")
    }
    require(len(prior) == 6, "fixed native metadata evidence roster required")
    native, audit = prior["native_prepared_ref"], prior["native_verification_ref"]
    require(
        prepared["native_prepared"] == native
        and prepared["policy_config"] == native["policy_config"]
        and audit["passed"] is audit["collection_verified"] is True
        and audit["native_quality"]
        == dict(
            total_cases=30,
            stable_native_cases=28,
            stable_native_by_task={"0": 10, "1": 8, "2": 10},
            failed_native_cases=["task1_native_demo_1", "task1_native_demo_35"],
            eligible_for_model_training_registration=False,
        )
        and audit["prepared"] == config["native_prepared_ref"]
        and prior["native_owner_ref"]["verification_ref"] == config["native_verification_ref"]
        and prior["native_owner_ref"]["producer_result"] == config["native_collection_ref"]
        and prior["native_owner_ref"]["passed"] is prior["native_owner_ref"]["complete"] is True
        and prior["native_tool_exit_ref"]["actual_tool_terminal_exit_code"] == 0,
        "original complete30/28-success evidence must remain qualified False",
    )
    for item in (config, native, audit, prior["native_owner_ref"], prior["native_collection_ref"]):
        require(
            item["model_training_authorized"] is item["fresh_performance_authorized"] is False,
            "diagnostic cannot authorize training/fresh",
        )
    source_closure(prepared, native, evidence)
    runtime_inventory(config, prepared["predicate_runtime_sha256"], evidence)
    require(
        prepared["runtime_source_refs"] == native["runtime_source_refs"]
        and len(prepared["runtime_source_refs"]) == 12
        and prepared["cv2_probe_ref"] == native["cv2_probe_ref"]
        and prepared["cv2_runtime_source_refs"] == native["cv2_runtime_source_refs"]
        and prepared["extra_runtime_source_refs"]
        == [reference(REPO / relative) for relative in config["extra_runtime_source_paths"]]
        and len(prepared["extra_runtime_source_refs"]) == 12,
        "actual observation/cv2/model/contact/binding runtime source roster differs",
    )
    expected_inputs = {str(CONFIG): CONFIG_SHA, str(PLAN): PLAN_SHA}
    for key, value in config.items():
        if key.startswith("native_") and key.endswith("_ref"):
            expected_inputs[value["path"]] = value["sha256"]
    for key in ("source_sha256", "snapshot_sha256", "first_continuation_snapshot_sha256"):
        expected_inputs.update(native[key])
    require(len(prior["native_owner_ref"]["owned_commands"]) == 4, "original four owned child exits required")
    for lifecycle in prior["native_owner_ref"]["owned_commands"]:
        owned_worker(lifecycle, evidence)
        expected_inputs.update({ref["path"]: ref["sha256"] for ref in lifecycle.values()})
    evidence.ref(prepared["cv2_probe_ref"])
    expected_inputs[prepared["cv2_probe_ref"]["path"]] = prepared["cv2_probe_ref"]["sha256"]
    task_ref = next(
        item["result"] for item in prior["native_collection_ref"]["task_results"] if item["task_index"] == 1
    )
    task = evidence.ref(task_ref)
    require(
        task["complete"] is task["passed"] is True and task["task_index"] == 1, "original wine task cache incomplete"
    )
    expected_inputs[task_ref["path"]] = task_ref["sha256"]
    rows = {row["name"]: row for row in audit["case_verification"]}
    records = {record["name"]: record for record in native["records"]}
    require(len(prepared["cases"]) == len(config["cases"]) == 4, "fixed four-case static denominator required")
    for case, planned in zip(prepared["cases"], config["cases"], strict=True):
        name = planned["name"]
        record, row = records[name], rows[name]
        require(
            all(case[key] == planned[key] for key in ("name", "role", "frame_indices"))
            and case["record"] == record
            and case["native_steps"] == record["native_steps"]
            and case["expected_native_success"]
            == [row["complete_native_success_sequence"][i] for i in planned["frame_indices"]]
            and case["case_result_ref"] in task["cases"]
            and case["case_result_ref"]["path"]
            == str(Path(native["output_root"]) / "collection/task1" / name / "result.json"),
            "selected static case/frames/old source booleans changed",
        )
        actual = evidence.ref(case["case_result_ref"])
        require(
            actual["complete"] is actual["passed"] is True
            and actual["record"] == record
            and actual["arrays"] == case["arrays_ref"]
            and actual["endpoint_receipts"] == case["endpoint_receipts_ref"]
            and audit["input_sha256"].get(case["case_result_ref"]["path"]) == case["case_result_ref"]["sha256"]
            and all(
                audit["input_sha256"].get(case[key]["path"]) == case[key]["sha256"]
                for key in ("arrays_ref", "endpoint_receipts_ref")
            ),
            "selected native source manifests/tensor/endpoint receipts lost original audit binding",
        )
        for key in ("case_result_ref", "arrays_ref", "endpoint_receipts_ref"):
            ref = case[key]
            expected_inputs[ref["path"]] = ref["sha256"]
            evidence.check(ref["path"], ref["sha256"])
    for ref in prepared["runtime_source_refs"] + prepared["extra_runtime_source_refs"]:
        evidence.check(ref["path"], ref["sha256"])
        expected_inputs[ref["path"]] = ref["sha256"]
    expected_inputs.update(prepared["predicate_runtime_sha256"])
    expected_inputs.update(
        {str(REPO / relative): prepared["source_sha256"][str(REPO / relative)] for relative in NEW_SOURCES}
    )
    require(
        prepared["input_sha256"] == expected_inputs,
        "static diagnostic input roster changed or expanded to excluded data",
    )
    return prepared


def owned_worker(lifecycle, evidence):
    require(set(lifecycle) == {"launch", "exit", "release"}, "full actual owned lifecycle required")
    launch, ended, release = (evidence.ref(lifecycle[key]) for key in ("launch", "exit", "release"))
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is False
        and ended == dict(actual_wait=True, automatic_retry=False, returncode=0)
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["only_owned_child_signalled"] is True
        and release["original_error"] is None
        and all(release[key] == launch["child"][key] for key in ("pid", "start_ticks")),
        "actual CPU worker wait0/reap/noSignal/source identity required before numerical audit",
    )
    return launch


def selected_source(spec, evidence):
    path = evidence.check(spec["arrays_ref"]["path"], spec["arrays_ref"]["sha256"])
    k = spec["native_steps"] + 41
    with np.load(path, allow_pickle=False) as archive:
        # Deliberately do not open native_commands, labels or any image array.
        states, truth, poses, times = (
            archive[key] for key in ("sim_states", "native_success", "source_pose", "sim_time")
        )
        require(
            states.dtype == poses.dtype == times.dtype == np.float64 and truth.dtype == np.bool_,
            "original selected raw state dtype differs",
        )
        require(
            states.ndim == 2
            and states.shape[0] == k
            and truth.shape == times.shape == (k,)
            and poses.shape == (k, 4, 4),
            "original selected raw state/source shape differs",
        )
        selected = [
            (frame, states[frame].copy(), bool(truth[frame]), poses[frame].copy(), float(times[frame]))
            for frame in spec["frame_indices"]
        ]
    receipts = [
        json.loads(line)
        for line in evidence.check(spec["endpoint_receipts_ref"]["path"], spec["endpoint_receipts_ref"]["sha256"])
        .read_text()
        .splitlines()
    ]
    require(len(receipts) == k, "original physical endpoint receipt count differs")
    result = {}
    for index, (frame, state, truth, pose, time) in enumerate(selected):
        require(
            np.isfinite(state).all() and np.isfinite(pose).all() and state[0] == time,
            "selected source raw state/time is invalid",
        )
        digest = hashlib.sha256(state.tobytes()).hexdigest()
        receipt = receipts[frame]
        require(
            receipt["endpoint_step"] == frame
            and receipt["sim_time"] == time
            and receipt["sim_state_sha256"] == digest
            and receipt["sim_state_dtype"] == "float64"
            and receipt["sim_state_shape"] == list(state.shape)
            and receipt["native_success"] is truth
            and truth is spec["expected_native_success"][index],
            "original selected source truth/endpoint/hash differs",
        )
        result[frame] = dict(state=state, state_sha256=digest, truth=truth, source_pose=pose, time=time)
    return result


def object_geometry(actual, expected_name, expected_class):
    require(
        actual["name"] == expected_name
        and actual["actual_class"] == expected_class
        and type(actual["body_id"]) is int
        and actual["body_id"] >= 0
        and actual["body_name"] == actual["root_body"]
        and isinstance(actual["body_name"], str),
        "actual object root BODY identity differs",
    )
    position = numeric(actual["world_pos"], (3,), expected_name + " body position")
    matrix = rotation(actual["world_mat"], expected_name + " body matrix")
    q = numeric(actual["world_quaternion_wxyz"], (4,), expected_name + " body quaternion")
    require(abs(float(q @ q) - 1) <= 1e-10, "actual object quaternion norm differs")
    w, x, y, z = q
    expected = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    close(matrix, expected, expected_name + " actual BODY orientation", atol=1e-10)
    geoms = actual["contact_geoms"]
    require(
        isinstance(geoms, list)
        and geoms
        and len({g["name"] for g in geoms}) == len(geoms)
        and len({g["id"] for g in geoms}) == len(geoms),
        "actual contact geom roster is empty/duplicated",
    )
    for geom in geoms:
        require(
            isinstance(geom["name"], str)
            and geom["name"]
            and type(geom["id"]) is int
            and geom["id"] >= 0
            and type(geom["body_id"]) is int
            and geom["body_id"] >= 0
            and type(geom["type"]) is int,
            "actual contact geom/body identity invalid",
        )
        require(np.all(numeric(geom["size"], (3,), "actual geom size") >= 0), "invalid actual geom size")
        numeric(geom["world_pos"], (3,), "actual geom position")
        rotation(geom["world_mat"], "actual geom matrix")
    return position, {geom["name"]: geom["id"] for geom in geoms}


def verify_endpoint(actual, spec, frame, source):
    require(
        actual["schema_version"] == "s2a.native-terminal-predicate-endpoint.v1"
        and actual["name"] == spec["name"]
        and actual["role"] == spec["role"]
        and actual["frame_index"] == frame
        and actual["source_state_ref"] == dict(arrays=spec["arrays_ref"], key="sim_states", frame_index=frame)
        and actual["source_receipt_ref"] == spec["endpoint_receipts_ref"]
        and actual["original_success"] is source["truth"],
        "static endpoint has wrong fixed original source",
    )
    before = raw_state(actual["actual_raw_state_before"], "installed state before predicates")
    after = raw_state(actual["actual_raw_state_after"], "installed state after predicates")
    require(
        before.shape == after.shape == source["state"].shape
        and before.tobytes() == after.tobytes() == source["state"].tobytes()
        and actual["actual_raw_state_before"]["sha256"] == source["state_sha256"]
        and actual["sim_time_before"] == actual["sim_time_after"] == source["time"] == before[0],
        "static saved-state installation/time did not preserve raw source bytes",
    )
    close(actual["actual_source_pose"], source["source_pose"], "restored actual source observation", atol=1e-10)
    goal = actual["official_goal"]
    require(
        len(goal) == 1
        and len(goal[0]) == 3
        and str(goal[0][0]).lower() == "on"
        and goal[0][1:] == ["wine_bottle_1", "wine_rack_1_top_region"],
        "unsupported actual goal; refuses inference from an unmodelled predicate",
    )
    inputs = actual["under_inputs"]
    require(
        inputs["site_name"] == "wine_rack_1_top_region"
        and type(inputs["site_id"]) is int
        and inputs["site_id"] >= 0
        and type(inputs["site_body_id"]) is int
        and inputs["site_body_id"] >= 0
        and inputs["parent_name"] == inputs["region_state_parent_name"] == "wine_rack_1"
        and inputs["python_region_class"] == "SiteObject"
        and inputs["has_under"] is inputs["parent_exists"] is True
        and inputs["python_size_dtype"] == "float64",
        "unsupported actual On site/parent/class branch",
    )
    wine, wine_geoms = object_geometry(actual["objects"]["wine"], "wine_bottle_1", "WineBottle")
    _, rack_geoms = object_geometry(actual["objects"]["rack"], "wine_rack_1", "WineRack")
    require(
        not (set(wine_geoms) & set(rack_geoms)) and not (set(wine_geoms.values()) & set(rack_geoms.values())),
        "wine/rack contact identities overlap",
    )
    require(
        inputs["wine_body_id"] == actual["objects"]["wine"]["body_id"]
        and inputs["wine_body_name"] == actual["objects"]["wine"]["body_name"],
        "On point is not actual wine root BODY",
    )
    close(inputs["wine_world_pos"], wine, "actual wine root BODY point")
    close(inputs["model_site_size"], inputs["python_size"], "Python/model region size")
    geometry = under_geometry(
        inputs["site_world_pos"],
        inputs["site_world_mat"],
        inputs["python_size"],
        wine,
        other_height=inputs["other_height"],
    )
    for key in ("delta_position", "xy_lower", "z_lower", "z_upper"):
        close(actual["geometry_margins"][key], geometry[key], "independent " + key)
    contacts = actual["contacts"]
    require(
        type(actual["ncon"]) is int and actual["ncon"] == len(contacts),
        "complete actual active-contact denominator differs",
    )
    named = {**wine_geoms, **rack_geoms}
    reverse = {value: key for key, value in named.items()}
    for item in contacts:
        for endpoint in ("geom1", "geom2"):
            name, geom_id = item[endpoint + "_name"], item[endpoint + "_id"]
            require(
                (name not in named or named[name] == geom_id) and (geom_id not in reverse or reverse[geom_id] == name),
                "active contact does not use exported actual geom identities",
            )
        numeric(item["pos"], (3,), "actual contact position")
        rotation(item["frame"], "actual contact frame")
    membership = contact_membership(contacts, list(wine_geoms), list(rack_geoms))
    on = geometry["under"] and membership["contact"]
    require(
        actual["atoms"] == dict(under=geometry["under"], contact=membership["contact"], ontop=on)
        and type(actual["official_success"]) is bool
        and actual["official_success"] is on,
        "independent region/contact/On composition differs from official or exported atoms",
    )
    reproduced = on is source["truth"]
    require(actual["static_success_reproduced"] is reproduced, "saved-state reproduction flag differs")
    return dict(
        name=spec["name"],
        role=spec["role"],
        frame_index=frame,
        original_success=source["truth"],
        official_success=on,
        static_success_reproduced=reproduced,
        under=geometry["under"],
        contact=membership["contact"],
        matching_contact_indices=membership["matching_contact_indices"],
        geometry_margins={key: value for key, value in geometry.items() if key != "under"},
        ncon=len(contacts),
        installed_source_state_sha256=source["state_sha256"],
        sim_time=source["time"],
    )


FORBIDDEN = {
    "env_step_calls",
    "sim_step_calls",
    "mj_step_calls",
    "mj_step2_calls",
    "render_calls",
    "source_hdf5_reads",
    "controller_command_calls",
}
REQUIRED_HOOKS = {
    "mujoco.mj_step",
    "mujoco.mj_step2",
    "mujoco.mj_step1",
    "mujoco.mj_forward",
    "mujoco.mjr_render",
    "h5py.File",
    "h5py.h5f.open",
    "h5py.h5f.create",
    "ControlEnv.reset",
    "ControlEnv.step",
    "BDDLBaseDomain.step",
    "MujocoEnv.step",
    "MjSim.step",
    "MjSim.render",
    "MjRenderContext.render",
    "MujocoEnv.render",
    "SingleArm.control",
    "Libero_Tabletop_Manipulation._check_success",
    "On.__call__",
    "SiteObjectState.check_ontop",
    "SiteObject.under",
    "MujocoEnv.check_contact",
}


def verify_ledger(actual, prepared, rows, evidence):
    require(actual["violation"] is None, "worker recorded a forbidden attempted operation")
    hooks = {item["api"]: item for item in actual["installed_hooks"]}
    require(
        len(hooks) == len(actual["installed_hooks"]) and REQUIRED_HOOKS <= set(hooks),
        "actual global/phase call guards incomplete",
    )
    require(
        hooks["ControlEnv.reset"]["replacement"] == "single_delegate_no_retry_process_local",
        "process-local reset delegation contract missing",
    )
    ledger = [
        json.loads(line)
        for line in evidence.check(actual["call_ledger_ref"]["path"], actual["call_ledger_ref"]["sha256"])
        .read_text()
        .splitlines()
    ]
    require(len(ledger) == actual["ledger_events"], "actual call ledger partial/extra")
    points = {(row["name"], row["frame_index"]): row for row in rows}
    names = {case["name"] for case in prepared["cases"]}
    counts, phases, stack, installations, initializations = Counter(), defaultdict(Counter), [], [], []
    for index, event in enumerate(ledger):
        require(
            event["event_index"] == index and event["phase"] in ("bootstrap", "setup", "static"),
            "actual call order/phase differs",
        )
        phase, key, kind = event["phase"], (event["case_name"], event["frame_index"]), event.get("kind")
        if phase == "static":
            require(key in points, "call used an unregistered static endpoint")
        elif phase == "setup":
            require(event["case_name"] in names and event["frame_index"] is None, "setup case identity differs")
        if event["event"] == "environment_initialization":
            require(
                phase == "setup" and event["integration"] is False,
                "initialization is not a non-integrating setup event",
            )
            counts["environment_initializations"] += 1
            phases[phase]["environment_initializations"] += 1
            initializations.append(event["case_name"])
        elif event["event"] == "installed_saved_state":
            require(
                phase == "static"
                and event["integration"] is False
                and event["source_state_sha256"] == points[key]["installed_source_state_sha256"]
                and event["time_after"] == points[key]["sim_time"],
                "clock change is not a registered saved-state installation",
            )
            counts["installed_saved_endpoints"] += 1
            phases[phase]["installed_saved_endpoints"] += 1
            installations.append(key)
        elif event["event"] == "before_delegate":
            require(
                kind not in FORBIDDEN and event["delegated"] is False,
                "forbidden integration/render/HDF/control delegation attempted",
            )
            if kind == "mj_step1_calls":
                require(phase == "setup", "mj_step1 attempted during static evaluation")
            if kind == "single_reset_calls":
                require(phase == "setup", "reset occurred during static evaluation")
            if kind == "additional_atomic_predicate_calls":
                require(phase == "static", "explicit predicate call outside static phase")
            counts[kind] += 1
            phases[phase][kind] += 1
            stack.append(event)
        elif event["event"] == "after_delegate":
            require(stack and event["delegated"] is True, "actual delegate completion has no matching entry")
            before = stack.pop()
            require(
                all(
                    event[field] == before[field]
                    for field in ("api", "kind", "phase", "case_name", "frame_index", "time_before")
                ),
                "actual nested delegate identity differs",
            )
            if kind in ("mj_step1_calls", "forward_calls") and event["time_before"] is not None:
                require(
                    event["time_before"] == event["time_after"], "non-integrating delegate advanced simulation time"
                )
            counts[kind + "_delegated"] += 1
            phases[phase][kind + "_delegated"] += 1
        else:
            raise ValueError("rejected/raised/unknown actual call event; refuses attribution")
    require(
        not stack and initializations == [case["name"] for case in prepared["cases"]] and installations == list(points),
        "complete four-environment/sixteen-state event grid differs",
    )
    for mapping, expected in [
        (actual["counts"], counts),
        *((actual["phase_counts"].get(phase, {}), values) for phase, values in phases.items()),
    ]:
        require(
            all(type(value) is int and value >= 0 and value == expected.get(key, 0) for key, value in mapping.items())
            and all(mapping.get(key, 0) == value for key, value in expected.items()),
            "reported call counts differ from actual ledger",
        )
    require(
        all(
            value == 0
            for phase, mapping in actual["phase_counts"].items()
            if phase not in phases
            for value in mapping.values()
        ),
        "unlogged phase count exists",
    )
    require(all(counts[kind] == 0 for kind in FORBIDDEN), "integrated/render/HDF/controller calls nonzero")
    require(
        phases["static"]["official_success_calls"] == 16
        and phases["setup"]["official_success_calls"] <= 16
        and counts["official_success_calls"] <= 32
        and phases["setup"]["mj_step1_calls"] <= 64
        and phases["static"]["mj_step1_calls"] == 0
        and counts["additional_atomic_predicate_calls"] == 48 <= 64,
        "fixed static/setup/explicit predicate call budget exceeded",
    )
    require(
        phases["static"]["internal_On_calls"] == phases["static"]["internal_check_ontop_calls"] == 32
        and phases["static"]["internal_under_calls"] == 48
        and phases["static"]["internal_check_contact_calls"] == 16 + 2 * sum(row["under"] for row in rows),
        "official/explicit internal short-circuit delegation counts differ",
    )
    return dict(
        counts=dict(counts),
        phase_counts={key: dict(value) for key, value in phases.items()},
        events=len(ledger),
        installed_saved_endpoints=16,
        environment_initializations=4,
        integrated_physics_steps=0,
    )


def summarize_endpoints(rows):
    require(
        len(rows) == 16 and len({(row["name"], row["frame_index"]) for row in rows}) == 16,
        "all16 fixed endpoints are required; no quality filtering",
    )
    mismatches = [
        dict(
            name=row["name"],
            frame_index=row["frame_index"],
            original_success=row["original_success"],
            official_success=row["official_success"],
        )
        for row in rows
        if not row["static_success_reproduced"]
    ]
    return dict(
        endpoints_verified=16,
        diagnostic_consistency_passed=not mismatches,
        attribution_supported=not mismatches,
        reproduction_mismatches=mismatches,
        endpoint_verification=rows,
    )


def verify_collection(path, lifecycle_path, prepared, evidence):
    actual = evidence.json(path)
    prepared_ref = reference(Path(prepared["output_root"]) / "preparation/prepared.json")
    lifecycle = evidence.json(lifecycle_path)
    require(
        lifecycle["prepared_ref"] == prepared_ref and lifecycle["collection_ref"] == reference(path),
        "actual worker lifecycle is not bound to this collection",
    )
    launch = owned_worker(lifecycle["worker_lifecycle"], evidence)
    require(
        actual["schema_version"] == "s2a.native-terminal-predicate-collection.v1"
        and actual["complete"] is True
        and actual["prepared_ref"] == prepared_ref
        and len(actual["cases"]) == 4
        and len(actual["endpoints"]) == 16,
        "static diagnostic worker is partial or has wrong denominator/source",
    )
    for flag in (
        "original_native_quality_passed",
        "eligible_for_model_training_registration",
        "model_training_authorized",
        "fresh_performance_authorized",
        "automatic_retry",
    ):
        require(actual[flag] is False, "static diagnostic promoted original native quality or permissions")
    strict = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO))
    require(actual["final_environment"] == strict, "actual worker final CPU environment differs")
    readiness = evidence.ref(actual["environment_readiness_ref"])
    probe = evidence.ref(prepared["cv2_probe_ref"])
    require(
        readiness["complete"] is readiness["passed"] is readiness["restore_performed"] is True
        and readiness["prepared_ref"] == prepared_ref
        and readiness["before"] == readiness["after_restore"] == strict
        and readiness["after_import_cv2"] == probe["after_import_cv2"]
        and readiness["cv2_probe_ref"] == prepared["cv2_probe_ref"]
        and readiness["cv2_runtime_source_refs"] == prepared["cv2_runtime_source_refs"]
        and readiness["actual_cv2_loader_ref"] == probe["cv2_loader_ref"]
        and readiness["source_before_sha256"]
        == readiness["source_after_sha256"]
        == {r["path"]: r["sha256"] for r in prepared["cv2_runtime_source_refs"]}
        and readiness["bytecode_before"]
        == readiness["bytecode_after_import"]
        == readiness["bytecode_after_restore"]
        == "1"
        and readiness["pid"] == launch["child"]["pid"]
        and readiness["argv"] == launch["command"][1:],
        "actual pre-simulator cv2 import/restore/source/process contract differs",
    )
    started = evidence.json(Path(prepared["output_root"]) / "collection/worker_started.json")
    require(
        started["pid"] == readiness["pid"]
        and started["argv"] == readiness["argv"]
        and started["prepared_ref"] == prepared_ref
        and started["environment_readiness_ref"] == actual["environment_readiness_ref"]
        and started["cuda"] == started["ld"] == "",
        "actual startup was not after successful environment restoration",
    )
    rows, flat = [], []
    for spec, case in zip(prepared["cases"], actual["cases"], strict=True):
        require(
            all(case[key] == spec[key] for key in ("name", "role", "frame_indices", "case_result_ref")),
            "static case grid/source differs",
        )
        source = selected_source(spec, evidence)
        require(len(case["endpoints"]) == 4, "a static case lost a selected endpoint")
        for frame, item in zip(spec["frame_indices"], case["endpoints"], strict=True):
            require(
                item["name"] == spec["name"]
                and item["role"] == spec["role"]
                and item["frame_index"] == frame
                and item["bundle"]["path"]
                == str(Path(prepared["output_root"]) / "collection" / spec["name"] / f"endpoint{frame:04d}.json"),
                "static endpoint grid/path changed",
            )
            rows.append(verify_endpoint(evidence.ref(item["bundle"]), spec, frame, source[frame]))
            flat.append(item)
    require(flat == actual["endpoints"], "flat and per-case complete16 endpoint grids disagree")
    ledger = verify_ledger(actual, prepared, rows, evidence)
    summary = summarize_endpoints(rows)
    require(
        actual["all_saved_states_reproduced"] is actual["passed"] is summary["diagnostic_consistency_passed"],
        "producer reproduction aggregate disagrees",
    )
    if summary["diagnostic_consistency_passed"]:
        require("error" not in actual, "passing static collection contains an error")
    else:
        require(
            actual["error"]["type"] == "StaticPredicateReproductionMismatch",
            "unexpected worker error; refuses attribution",
        )
    return dict(
        summary,
        complete=True,
        ledger_verified=ledger,
        actual_worker=dict(
            pid=launch["child"]["pid"],
            start_ticks=launch["child"]["start_ticks"],
            actual_wait_returncode=0,
            reaped=True,
            signal_sent=False,
        ),
        contact_membership_independently_reconstructed=True,
        collision_detection_independently_reimplemented=False,
        interpretation="Exact saved-state predicate reproduction permits descriptions of observed region/contact components only. It does not identify a unique control, rendering or gripper cause, change native28/30, or demonstrate WAM task success.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--worker-lifecycle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "independent diagnostic requires S2A cwd and empty CUDA/LD",
    )
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "new output under outputs required; refuses overwrite",
    )
    evidence = Evidence()
    receipt = dict(
        schema_version="s2a.native-terminal-predicate-independent-verification.v1",
        complete=False,
        passed=False,
        diagnostic_consistency_passed=False,
        attribution_supported=False,
        original_native_quality_passed=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        new_environment_initializations=0,
        new_physics_steps=0,
        new_model_calls=0,
        new_fits=0,
        hdf_content_reads=0,
        image_reads=0,
    )
    error = None
    try:
        prepared = verify_preparation(args.prepared, args.prepared_sha256, evidence)
        receipt["prepared"] = reference(args.prepared)
        receipt.update(verify_collection(args.collection, args.worker_lifecycle, prepared, evidence))
        evidence.recheck()
        receipt["passed"] = receipt["diagnostic_consistency_passed"]
        if not receipt["passed"]:
            error = ValueError("saved original and static predicate truth differ; refuses attribution")
    except Exception as caught:
        error = caught
        receipt.update(passed=False, diagnostic_consistency_passed=False, attribution_supported=False)
    if error is not None:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=reference(output))))
    if error is not None:
        raise SystemExit(f"static predicate independent audit rejected: {error}")


if __name__ == "__main__":
    main()
