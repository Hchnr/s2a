"""030 independent CPU audit of source-isolated synchronous native data.

No producer selection/label geometry, simulator, Torch, WAM or training reader
is imported. Preparation inspects candidate initial/XML/shape metadata and only
the fixed selected actions. Native quality and evidence integrity are separate;
this audit never authorizes model training or fresh evaluation.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import REPO, sha256

CONFIG = REPO / "configs/wam_synchronous_native_training_20261021.json"
CONFIG_SHA = "966c33348ce0a78f09e310f9d9efcf6d22201502babc8834cc8a2c4aa756115a"
PLAN = REPO / "plans/v0.0.1/030_synchronous_native_training_20261021.md"
PLAN_SHA = "fb4363797be72702da54710c6a910338d8a8769bca12d0de133b7633032db155"
NEW_SOURCES = (
    "src/s2a/native_training_sources.py",
    "scripts/prepare_wam_native_training.py",
    "tests/test_native_training_sources.py",
    "src/s2a/native_training.py",
    "scripts/collect_wam_native_training.py",
    "tests/test_native_training.py",
    "scripts/run_wam_native_training.py",
    "tests/test_native_training_runner.py",
    "scripts/verify_wam_native_training.py",
    "tests/test_native_training_verification.py",
)
ACTIVE = np.asarray([True] * 10 + [False] * 118)
RUNTIME_SOURCES = (
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/robots/single_arm.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/binding_utils.py",
    "outputs/libero_env311/lib/python3.11/site-packages/robosuite/utils/transform_utils.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, name):
    array = np.asarray(value)
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), f"{name}: nonfinite/nonnumeric")
    return array


def close(actual, expected, name, *, atol=1e-10):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(
        actual.shape == expected.shape and np.allclose(actual, expected, rtol=0, atol=atol),
        f"{name}: numerical mismatch",
    )


def exact(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(actual.shape == expected.shape and np.array_equal(actual, expected), f"{name}: exact mismatch")


def bits(actual, expected, name):
    actual, expected = numeric(actual, name), numeric(expected, name)
    require(
        actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
        f"{name}: dtype/shape/bytes mismatch",
    )


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f"source SHA changed: {path}")
        require(
            str(path) not in self.hashes or self.hashes[str(path)] == digest, f"source changed during audit: {path}"
        )
        self.hashes[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def ref(self, ref):
        require(set(ref) == {"path", "sha256"}, "invalid evidence reference")
        return self.json(ref["path"], ref["sha256"])

    def recheck(self):
        for path, digest in self.hashes.copy().items():
            self.check(path, digest)


def state_hash(initial):
    value = numeric(initial, "initial state").astype(np.float64)
    require(value.ndim == 1 and value.size > 0, "initial state must be a nonempty vector")
    return hashlib.sha256(value.tobytes()).hexdigest()


def demo_order(names):
    names = list(names)
    require(
        len(set(names)) == len(names) and all(re.fullmatch(r"demo_(0|[1-9][0-9]*)", value) for value in names),
        "native demo IDs are not unique canonical numeric names",
    )
    return sorted(names, key=lambda name: int(name[5:]))


def anchors_for(native_steps):
    require(
        type(native_steps) is int and 43 <= native_steps <= 600,
        "selected native length is outside43..600; replacement is forbidden",
    )
    result = [int(np.floor(j * (native_steps - 40) / 3)) for j in range(4)] + [native_steps]
    require(
        len(set(result)) == 5 and all(i + 40 <= native_steps + 40 for i in result),
        "five complete distinct anchor windows required",
    )
    return result


def independent_selection(candidates_by_task, excluded_by_task, forbidden):
    """Single RNG, task order, numeric ID permutation, then identity skipping."""
    require(len(candidates_by_task) == len(excluded_by_task) == 3, "three native task sources required")
    rng, seen, chosen, traces = np.random.default_rng(42), set(forbidden), [], []
    for task_index, (candidates, excluded) in enumerate(zip(candidates_by_task, excluded_by_task, strict=True)):
        allowed = {item["demo"]: item for item in candidates if item["demo"] not in excluded}
        require(
            len(allowed) == len([c for c in candidates if c["demo"] not in excluded]), "duplicate candidate metadata"
        )
        names = demo_order(allowed)
        order = [names[int(i)] for i in rng.permutation(len(names))]
        selected, skipped = [], []
        for name in order:
            digest = allowed[name]["initial_state_sha256"]
            if digest in seen:
                skipped.append(name)
                continue
            selected.append(allowed[name])
            seen.add(digest)
            if len(selected) == 10:
                break
        require(len(selected) == 10, "insufficient isolated identities; no smaller or replacement roster authorized")
        chosen.append(selected)
        traces.append(
            dict(
                task_index=task_index,
                excluded_demo_ids=demo_order(excluded),
                numeric_demo_order=names,
                permutation_demo_order=order,
                skipped_identity_aliases=skipped,
                selected_demo_ids=[r["demo"] for r in selected],
                candidates=[
                    {
                        key: allowed[name][key]
                        for key in ("demo", "initial_state_sha256", "model_xml_sha256", "actions_shape")
                    }
                    for name in names
                ],
            )
        )
    return chosen, dict(seed=42, tasks=traces)


def read_candidate_metadata(path, excluded, *, opener=None):
    """No excluded group is opened; no candidate action values or obs are read."""
    if opener is None:
        import h5py

        opener = h5py.File
    result = []
    with opener(path, "r") as handle:
        root = handle["data"]
        for name in demo_order(root.keys()):
            if name in excluded:
                continue
            group = root[name]
            initial = numeric(group["states"][0], "candidate initial").astype(np.float64)
            xml = group.attrs["model_file"]
            xml = xml.decode() if isinstance(xml, bytes) else xml
            require(isinstance(xml, str) and xml.strip(), "native candidate XML missing")
            result.append(
                dict(
                    demo=name,
                    initial_state=initial.tolist(),
                    initial_state_sha256=state_hash(initial),
                    model_xml=xml,
                    model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    actions_shape=[int(v) for v in group["actions"].shape],
                )
            )
    return result


def se3(value, name):
    value = numeric(value, name).astype(np.float64)
    require(value.ndim >= 2 and value.shape[-2:] == (4, 4), f"{name}: SE3 shape mismatch")
    close(value[..., 3, :], np.broadcast_to([0.0, 0.0, 0.0, 1.0], value[..., 3, :].shape), f"{name}: bottom row")
    r = value[..., :3, :3]
    close(np.swapaxes(r, -1, -2) @ r, np.broadcast_to(np.eye(3), r.shape), f"{name}: orthonormal rotation")
    close(np.linalg.det(r), np.ones(r.shape[:-2]), f"{name}: proper rotation")
    return value


def pose(positions, rotations):
    positions, rotations = np.asarray(positions, np.float64), np.asarray(rotations, np.float64)
    result = np.broadcast_to(np.eye(4), (*positions.shape[:-1], 4, 4)).copy()
    result[..., :3, :3], result[..., :3, 3] = rotations, positions
    return result


def rot6d(rotation):
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def independent_labels(tcp, opening):
    tcp = se3(tcp, "actual TCP")
    opening = numeric(opening, "actual opening").astype(np.float64)
    require(
        opening.shape == (len(tcp),) and np.all((opening >= 0) & (opening <= 1)),
        "actual opening geometry/range differs",
    )
    relative_rotation = np.swapaxes(tcp[:-1, :3, :3], -1, -2) @ tcp[1:, :3, :3]
    relative_translation = np.einsum(
        "nij,nj->ni", np.swapaxes(tcp[:-1, :3, :3], -1, -2), tcp[1:, :3, 3] - tcp[:-1, :3, 3]
    )
    delta = pose(relative_translation, relative_rotation)
    state = np.zeros((len(tcp), 128), np.float64)
    action = np.zeros_like(state)
    state[:, :3], state[:, 3:9], state[:, 9] = tcp[:, :3, 3], rot6d(tcp[:, :3, :3]), opening
    action[:-1, :3], action[:-1, 3:9], action[:-1, 9] = relative_translation, rot6d(relative_rotation), opening[1:]
    state_mask = np.broadcast_to(ACTIVE, state.shape).copy()
    action_mask = state_mask.copy()
    action_mask[-1] = False
    valid = np.ones(len(tcp), bool)
    valid[-1] = False
    return dict(
        state128=state,
        action128=action,
        state_mask=state_mask,
        action_mask=action_mask,
        action_valid=valid,
        delta_pose=delta,
    )


def native_quality(success):
    success = np.asarray(success)
    require(
        success.dtype == np.bool_ and success.ndim == 1 and len(success) >= 84,
        "native success must cover every initial/post-step endpoint",
    )
    return bool(np.all(success[-40:]))


def reservation(prepared, evidence):
    config = prepared["config"]
    expected_fresh = dict(path=str(REPO / config["fresh_config"]), sha256=config["fresh_config_sha256"])
    expected_fixture = dict(path=str(REPO / config["fresh_fixtures"]), sha256=config["fresh_fixtures_sha256"])
    require(
        prepared["fresh_config_ref"] == expected_fresh and prepared["fresh_fixtures_ref"] == expected_fixture,
        "030 original fresh refs changed",
    )
    fresh, fixture = evidence.ref(expected_fresh), evidence.ref(expected_fixture)
    require(
        fixture["config"] == fresh and config["tasks"] == fresh["tasks"] and len(fixture["records"]) == 12,
        "030 original fresh reservation changed",
    )
    forbidden, excluded, sources = set(), [], []
    for task in range(3):
        members = [r for r in fixture["records"] if r["task_index"] == task]
        require(len(members) == len({r["demo"] for r in members}) == 4, "030 fresh four-case identities differ")
        path = str((REPO / fresh["simulation_sources"][task]).resolve())
        require(
            len({r["source_hdf5_sha256"] for r in members}) == len({r["task_id"] for r in members}) == 1,
            "030 fresh task source is inconsistent",
        )
        for row in members:
            digest = state_hash(row["initial_state"])
            require(
                digest == row["initial_state_sha256"] and digest not in forbidden,
                "030 fresh initial hash/uniqueness differs",
            )
            require(
                str(Path(row["source_hdf5"]).resolve()) == path and row["task"] == config["tasks"][task],
                "030 fresh task/source differs",
            )
            forbidden.add(digest)
        excluded.append(
            set(fresh["excluded_demos_by_task"][task])
            | set(config["exclude_validation_demos_by_task"][task])
            | {r["demo"] for r in members}
        )
        sources.append(dict(path=path, sha256=members[0]["source_hdf5_sha256"], task_id=members[0]["task_id"]))
    frozen = {str((REPO / name).resolve()): digest for name, digest in fixture["exclusions"].items()}
    expected_history = [
        dict(path=str((REPO / name).resolve()), sha256=frozen[str((REPO / name).resolve())])
        for name in fresh["previous_simulation_fixtures"]
    ]
    require(
        prepared["historical_fixture_refs"] == expected_history,
        "030 historical fixtures changed after fresh reservation",
    )
    for ref in expected_history:
        historical = evidence.ref(ref)
        for row in historical.get("records", historical.get("episodes", [])):
            digest = state_hash(row["initial_state"])
            require(digest == row["initial_state_sha256"], "030 historical initial identity changed")
            forbidden.add(digest)
    require(prepared["forbidden_initial_state_sha256"] == sorted(forbidden), "030 initial alias exclusion set differs")
    return sources, excluded, forbidden


def source_closure(prepared, prior, path, *, tracked=None):
    if tracked is None:
        tracked = subprocess.check_output(
            ["git", "ls-files", "src", "scripts", "tests"], cwd=REPO, text=True
        ).splitlines()
    expected = set(prior["source_sha256"]) | {str((REPO / name).resolve()) for name in tracked if name.endswith(".py")}
    expected |= {str(REPO / name) for name in NEW_SOURCES} | {str(CONFIG), str(PLAN)}
    require(
        len(prior["source_sha256"]) == 111 and set(prepared["source_sha256"]) == expected,
        "030 full tracked/new source closure is incomplete",
    )
    require(
        prepared["inherited_source_sha256"] == prior["source_sha256"]
        and all(prepared["source_sha256"].get(k) == v for k, v in prior["source_sha256"].items()),
        "030 altered original111 frozen sources",
    )
    require(
        prepared["source_sha256"][str(CONFIG)] == CONFIG_SHA and prepared["source_sha256"][str(PLAN)] == PLAN_SHA,
        "030 source plan/config pin changed",
    )
    snapshots = {
        str(Path(path).resolve().parent / "source_snapshot" / f"{i:04d}_{Path(name).name}"): prepared["source_sha256"][
            name
        ]
        for i, name in enumerate(sorted(expected))
    }
    require(prepared["snapshot_sha256"] == snapshots, "030 sorted snapshots are not exactly the frozen sources")


def source_stats(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def selected_actions(records, excluded, evidence, *, opener=None):
    """Only after the whole fixed roster is validated may its actions be read."""
    if opener is None:
        import h5py

        opener = h5py.File
    result = {}
    for record in records:
        require(record["demo"] not in excluded[record["task_index"]], "attempted excluded/fresh native actions")
        with opener(record["source_hdf5"], "r") as handle:
            actual = numeric(handle["data"][record["demo"]]["actions"][:], "selected native actions").copy()
        saved = np.load(evidence.check(record["expert_path"], record["expert_sha256"]), allow_pickle=False)
        require(
            actual.shape == (record["native_steps"], 7)
            and actual.dtype.kind == "f"
            and str(actual.dtype) == record["expert_dtype"],
            "030 selected native action geometry/dtype differs",
        )
        bits(saved, actual, "030 original selected action bytes")
        result[record["name"]] = saved
    return result


def verify_preparation(path, expected_sha, evidence, *, opener=None):
    prepared = evidence.json(path, expected_sha)
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    root = REPO / config["output_dir"]
    require(
        prepared["schema_version"] == "s2a.native-training-preparation.v1"
        and prepared["seed"] == 42
        and prepared["config"] == config
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration_ref"] == reference(PLAN)
        and prepared["output_root"] == str(root)
        and Path(path).resolve() == root / "preparation/prepared.json",
        "030 prepared registration/seed/output differs",
    )
    require(
        root.is_relative_to(REPO / "outputs") and prepared["budget"] == config["budget"], "030 output/budget changed"
    )
    expected_prior = {
        "prior_diagnostic_verification_ref": dict(
            path=str(REPO / config["prior_diagnostic_verification"]),
            sha256=config["prior_diagnostic_verification_sha256"],
        ),
        "prior_diagnostic_final_peer_ref": dict(
            path=str(REPO / config["prior_diagnostic_final_peer"]), sha256=config["prior_diagnostic_final_peer_sha256"]
        ),
    }
    require(all(prepared[key] == ref for key, ref in expected_prior.items()), "030 changed completed029 refs")
    audit, final = (
        evidence.ref(expected_prior[key])
        for key in ("prior_diagnostic_verification_ref", "prior_diagnostic_final_peer_ref")
    )
    require(
        audit["passed"] is audit["collection_verified"] is True
        and audit["actual_denoise_calls"] == 1824
        and audit["complete_free_replays"] == 48
        and audit["original028_state_gate"] is audit["eligible_for_fit"] is audit["physics_authorized"] is False,
        "030 lacks completed029 diagnostic preserving oldFalse",
    )
    require(
        final["complete"] is final["passed"] is True
        and final["verification"] == expected_prior["prior_diagnostic_verification_ref"]
        and final["prepared"] == audit["prepared"] == prepared["prior_prepared_ref"]
        and final["frozen_sources_verified"] == final["source_snapshots_verified"] == 111
        and final["task_success_proven"] is False,
        "030 final029 evidence mismatch",
    )
    prior = evidence.ref(prepared["prior_prepared_ref"])
    source_closure(prepared, prior, path)
    for field in ("source_sha256", "snapshot_sha256"):
        for name, digest in prepared[field].items():
            evidence.check(name, digest)
    sources, excluded, forbidden = reservation(prepared, evidence)
    policy = evidence.ref(prepared["policy_template_ref"])
    require(
        prepared["policy_template_ref"]["path"] == str(REPO / config["policy_template"])
        and policy == prepared["policy_config"],
        "030 policy source changed",
    )
    require(
        prepared["tool_contract_ref"]["path"] == str(Path(policy["tool_contract"]).resolve()),
        "030 canonical tool source changed",
    )
    tool = evidence.ref(prepared["tool_contract_ref"])
    close(
        se3(prepared["tool_source_from_tcp"], "canonical tool"),
        se3(tool["source_from_tcp"], "published tool"),
        "030 actual H",
    )
    require(
        len(prepared["runtime_source_refs"]) == 3
        and {r["path"] for r in prepared["runtime_source_refs"]} == {str(REPO / name) for name in RUNTIME_SOURCES},
        "030 actual F64 observation source path changed",
    )
    for ref in prepared["runtime_source_refs"]:
        evidence.check(ref["path"], ref["sha256"])
    require(set(prepared["source_stats"]) == {s["path"] for s in sources}, "030 HDF source stat roster differs")
    candidates = []
    for task, source in enumerate(sources):
        require(
            source_stats(source["path"]) == prepared["source_stats"][source["path"]], "030 native source stat changed"
        )
        # Raw whole-file identity hashing does not access any HDF dataset API.
        evidence.check(source["path"], source["sha256"])
        candidates.append(read_candidate_metadata(source["path"], excluded[task], opener=opener))
    chosen, selection = independent_selection(candidates, excluded, forbidden)
    records = prepared["records"]
    require(
        len(records) == 30
        and [r["task_index"] for r in records] == [i for i in range(3) for _ in range(10)]
        and len({r["name"] for r in records}) == 30,
        "030 fixed30 three-task roster differs",
    )
    for task, (items, source) in enumerate(zip(chosen, sources, strict=True)):
        selection["tasks"][task]["source_hdf5"] = source["path"]
        for candidate, record in zip(items, records[task * 10 : (task + 1) * 10], strict=True):
            require(
                all(record[key] == value for key, value in candidate.items()), "030 selected candidate metadata changed"
            )
            n = candidate["actions_shape"][0]
            require(
                candidate["actions_shape"] == [n, 7] and record["native_steps"] == n,
                "030 selected shape invalid; replacement forbidden",
            )
            require(
                record["anchor_frames"] == anchors_for(n) and record["name"] == f"task{task}_native_{record['demo']}",
                "030 fixed five-anchor/name mapping differs",
            )
            require(
                record["task"] == config["tasks"][task]
                and record["task_id"] == source["task_id"]
                and record["source_hdf5"] == source["path"]
                and record["source_hdf5_sha256"] == source["sha256"]
                and record["source_stat"] == prepared["source_stats"][source["path"]],
                "030 task/HDF origin mapping changed",
            )
            require(
                record["expert_path"] == str(root / "sources" / record["name"] / "expert.npy"),
                "030 selected actions escaped frozen source root",
            )
    require(selection == prepared["selection"], "030 independent single-RNG selection differs")
    actions = selected_actions(records, excluded, evidence, opener=opener)
    require(
        prepared["metadata_admission"] is True
        and prepared["native_success_verified"]
        is prepared["eligible_for_model_training_registration"]
        is prepared["model_training_authorized"]
        is prepared["fresh_performance_authorized"]
        is False
        and prepared["actual_new_physics_steps"] == prepared["actual_wam_calls"] == 0,
        "030 preparation was promoted to performance/training",
    )
    expected_inputs = {str(CONFIG): CONFIG_SHA}
    for reference_ in (
        *expected_prior.values(),
        prepared["prior_prepared_ref"],
        prepared["fresh_config_ref"],
        prepared["fresh_fixtures_ref"],
        prepared["policy_template_ref"],
        prepared["tool_contract_ref"],
        *prepared["historical_fixture_refs"],
        *prepared["runtime_source_refs"],
        *sources,
    ):
        expected_inputs[reference_["path"]] = reference_["sha256"]
    expected_inputs.update({record["expert_path"]: record["expert_sha256"] for record in records})
    require(prepared["input_sha256"] == expected_inputs, "030 actual input roster changed/expanded to observations")
    return prepared, actions


def owned_exit(record, evidence):
    launch, ended, release = (evidence.ref(record[key]) for key in ("launch", "exit", "release"))
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
        "030 actual owned child was not CPU-only",
    )
    require(
        ended == dict(actual_wait=True, automatic_retry=False, returncode=0), "030 actual normal child wait missing"
    )
    require(
        all(release[key] == launch["child"][key] for key in ("pid", "start_ticks"))
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True,
        "030 actual owned child normal wait/reap identity differs",
    )
    return dict(pid=release["pid"], start_ticks=release["start_ticks"], reaped=True, signal_sent=False)


def load_arrays(actual, evidence):
    path = evidence.check(actual["arrays"]["path"], actual["arrays"]["sha256"])
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: numeric(source[key], key).copy() for key in source.files}
    require(
        set(arrays) == set(actual["array_index"]) == set(actual["original_dtypes"]),
        "030 saved array index omitted/added tensors",
    )
    for key, value in arrays.items():
        require(
            actual["array_index"][key] == dict(shape=list(value.shape), dtype=str(value.dtype))
            and actual["original_dtypes"][key] == str(value.dtype),
            f"030 actual saved {key} index/dtype differs",
        )
    return arrays


def verify_endpoint_geometry(arrays, record, h, native):
    n, k = record["native_steps"], record["native_steps"] + 41
    shapes = dict(
        endpoint_step=(k,),
        sim_time=(k,),
        native_success=(k,),
        source_position=(k, 3),
        source_quaternion_xyzw=(k, 4),
        gripper_qpos=(k, 2),
        source_pose=(k, 4, 4),
        actual_source_site_pose=(k, 4, 4),
        actual_eef_body_pose=(k, 4, 4),
        physical_source_pose=(k, 4, 4),
        controller_site_pose=(k, 4, 4),
        tcp_pose=(k, 4, 4),
        opening=(k,),
        native_commands=(n, 7),
        sent_commands=(n + 40, 7),
        controller_ee_pos=(k, 3),
        controller_ee_ori_mat=(k, 3, 3),
        controller_goal_pos=(k, 3),
        controller_goal_ori=(k, 3, 3),
        controller_joint_pos=(k, 7),
        controller_joint_vel=(k, 7),
    )
    for key, shape in shapes.items():
        require(arrays[key].shape == shape, f"030 endpoint {key} shape differs")
        if key not in ("endpoint_step", "native_success", "native_commands"):
            require(arrays[key].dtype == np.float64, f"030 actual {key} lost F64 endpoint precision")
    require(
        arrays["endpoint_step"].dtype == np.int64 and arrays["native_success"].dtype == np.bool_,
        "030 step/success dtype differs",
    )
    flat, offsets = arrays["gripper_current_action_flat"], arrays["gripper_current_action_offsets"]
    require(
        flat.ndim == 1
        and flat.dtype == np.float64
        and offsets.shape == (k + 1,)
        and offsets.dtype == np.int64
        and offsets[0] == 0
        and offsets[-1] == len(flat)
        and np.all(np.isin(np.diff(offsets), (1, 2))),
        "030 actual ragged gripper cache storage/offsets differs",
    )
    exact(arrays["endpoint_step"], np.arange(k), "030 every physical endpoint step")
    close(np.diff(arrays["sim_time"]), np.full(k - 1, 0.05), "030 synchronous control period", atol=1e-9)
    require(
        arrays["sim_states"].dtype == np.float64 and arrays["sim_states"].shape == (k, len(record["initial_state"])),
        "030 raw full simulation state layout differs",
    )
    bits(
        arrays["sim_states"][0],
        np.asarray(record["initial_state"], np.float64),
        "030 exact restored initial state without pre-roll",
    )
    bits(arrays["sim_time"], arrays["sim_states"][:, 0], "030 raw state/time endpoint binding")
    bits(arrays["native_commands"], native, "030 actual selected native commands")
    expected_sent = np.zeros((n + 40, 7), np.float64)
    expected_sent[:n] = native
    expected_sent[n:, -1] = native[-1, -1]
    bits(arrays["sent_commands"], expected_sent, "030 all native commands then40 zero-arm/final-gripper hold")
    observed = se3(arrays["source_pose"], "source observed SE3")
    site = se3(arrays["actual_source_site_pose"], "physical grip site")
    body = se3(arrays["actual_eef_body_pose"], "physical EEF BODY")
    controller = se3(arrays["controller_site_pose"], "independent physical controller site")
    close(np.linalg.norm(arrays["source_quaternion_xyzw"], axis=1), np.ones(k), "030 actual F64 body quaternion norm")
    obs_rotation = Rotation.from_quat(arrays["source_quaternion_xyzw"]).as_matrix()
    close(observed, pose(arrays["source_position"], obs_rotation), "030 independent observed source SE3")
    close(observed[:, :3, 3], site[:, :3, 3], "030 observed source position vs actual grip site")
    close(observed[:, :3, :3], body[:, :3, :3], "030 observed source orientation vs actual F64 BODY")
    physical = body.copy()
    physical[:, :3, 3] = site[:, :3, 3]
    close(arrays["physical_source_pose"], physical, "030 physical source origin/body-frame construction")
    close(arrays["controller_ee_pos"][0], controller[0, :3, 3], "030 initial refreshed controller position")
    close(arrays["controller_ee_ori_mat"][0], controller[0, :3, :3], "030 initial refreshed controller rotation")
    # Later controller caches are actual within-step cached fields, deliberately
    # kept separate from the newly forced physical endpoint observations.
    expected_opening = np.clip((arrays["gripper_qpos"][:, 0] - arrays["gripper_qpos"][:, 1] - 0.001) / 0.079, 0, 1)
    close(arrays["opening"], expected_opening, "030 independently measured opening")
    tcp = observed @ se3(h, "published source_from_tcp")
    close(arrays["tcp_pose"], tcp, "030 canonical TCP via source_from_tcp")
    labels = independent_labels(tcp, expected_opening)
    for key, expected in labels.items():
        require(arrays[key].dtype == expected.dtype, f"030 {key} saved dtype differs")
        if expected.dtype == np.bool_:
            bits(arrays[key], expected, f"030 {key} valid mask")
        else:
            close(arrays[key], expected, f"030 independently reconstructed {key}")
    maxima = dict(position=0.0, rotation=0.0)
    for frame in anchors_for(n):
        reconstructed = tcp[frame].copy()
        for step in range(frame, frame + 40):
            encoded = arrays["action128"][step]
            r0, r1 = encoded[3:6], encoded[6:9]
            r2 = np.cross(r0, r1)
            rotation = np.column_stack((r0, r1, r2))
            reconstructed = reconstructed @ pose(encoded[:3], rotation)
            target = tcp[step + 1]
            maxima["position"] = max(maxima["position"], float(np.max(np.abs(reconstructed[:3, 3] - target[:3, 3]))))
            maxima["rotation"] = max(maxima["rotation"], float(np.max(np.abs(reconstructed[:3, :3] - target[:3, :3]))))
            close(reconstructed, target, "030 local40-step anchor reconstruction")
            close(np.asarray(encoded[9]), np.asarray(expected_opening[step + 1]), "030 next-opening label timing")
        require(np.all(arrays["action_valid"][frame : frame + 40]), "030 anchor contains invalid transition")
    return maxima


def numpy_canvas(images):
    """Independent ideal bilinear half-pixel cover-crop, final uint8 rounding.

    The frozen Torch implementation can round a channel one unit differently
    near a floating point boundary. Roles, flip, geometry and seam are exact.
    """
    require(len(images) == 2, "030 exactly two current cameras required")
    coords = (np.arange(640, dtype=np.float64) + 0.5) * (256 / 640) - 0.5
    lo = np.floor(coords).astype(int)
    weight = coords - lo
    left, right = np.clip(lo, 0, 255), np.clip(lo + 1, 0, 255)
    result = np.zeros((960, 640, 3), np.uint8)
    for i, image in enumerate(images):
        require(image.shape == (256, 256, 3) and image.dtype == np.uint8, "030 current RGB geometry/dtype differs")
        pixels = image.astype(np.float64)
        horizontal = pixels[:, left] * (1 - weight)[None, :, None] + pixels[:, right] * weight[None, :, None]
        resized = horizontal[left] * (1 - weight)[:, None, None] + horizontal[right] * weight[:, None, None]
        result[i * 480 : (i + 1) * 480] = np.clip(np.rint(resized[80:560]), 0, 255).astype(np.uint8)
    result[478:482] = 0
    return result


def camera_contract(anchor, endpoint, arrays, frame, evidence):
    from PIL import Image

    require(
        anchor["frame_index"] == frame and anchor["current_endpoint"] == endpoint,
        "030 current render endpoint binding differs",
    )
    binding = dict(sim_time=endpoint["sim_time"], sim_state_sha256=endpoint["sim_state_sha256"])
    require(
        anchor["render_before"] == anchor["render_after"] == binding, "030 rendering advanced/restored physical state"
    )
    close(anchor["current_source_pose"], arrays["source_pose"][frame], "030 current image measured source pose")
    close(np.asarray(anchor["current_opening"]), arrays["opening"][frame], "030 current image measured opening")
    require(
        anchor["camera_keys"] == ["agentview_image", "robot0_eye_in_hand_image"]
        and anchor["camera_roles"] == ["auxiliary_0", "left_wrist"]
        and anchor["raw_camera_shape"] == [256, 256, 3]
        and anchor["raw_camera_dtype"] == "uint8"
        and anchor["live_flip_axes"] == [0, 1]
        and anchor["future_rgb_frames"] == 0,
        "030 current camera role/flip/future scope differs",
    )
    with np.load(
        evidence.check(anchor["raw_cameras"]["path"], anchor["raw_cameras"]["sha256"]), allow_pickle=False
    ) as source:
        require(
            set(source.files) == {"agentview_image", "robot0_eye_in_hand_image", "live_auxiliary_0", "live_left_wrist"},
            "030 camera archive added/missed roles or future frames",
        )
        raw = [source[key].copy() for key in ("agentview_image", "robot0_eye_in_hand_image")]
        live = [source[key].copy() for key in ("live_auxiliary_0", "live_left_wrist")]
    for original, flipped in zip(raw, live, strict=True):
        require(
            original.shape == (256, 256, 3) and original.dtype == np.uint8, "030 actual raw camera shape/dtype differs"
        )
        bits(flipped, original[::-1, ::-1].copy(), "030 actual live two-axis camera flip")
    with Image.open(evidence.check(anchor["canvas_png"]["path"], anchor["canvas_png"]["sha256"])) as image:
        require(image.mode == "RGB" and image.size == (640, 960), "030 current canvas is not actual960x640 RGB")
        pixels = np.asarray(image).copy()
    require(
        anchor["canvas_shape"] == [960, 640, 3]
        and anchor["canvas_dtype"] == "uint8"
        and hashlib.sha256(pixels.tobytes()).hexdigest() == anchor["canvas_rgb_sha256"],
        "030 canvas RGB identity differs",
    )
    expected = numpy_canvas(live)
    require(not np.any(pixels[478:482]), "030 historical four-pixel seam moved")
    difference = int(np.max(np.abs(pixels.astype(np.int16) - expected.astype(np.int16))))
    require(difference <= 1, "030 independent camera role/cover-crop canvas exceeds one uint8 rounding unit")
    return difference


def case_contract(actual, record, prepared, native, evidence):
    n, k = record["native_steps"], record["native_steps"] + 41
    require(
        actual["schema_version"] == "s2a.native-training-case.v1"
        and actual["complete"] is actual["passed"] is True
        and "error" not in actual
        and actual["record"] == record
        and actual["prepared_ref"] == reference(Path(prepared["output_root"]) / "preparation/prepared.json"),
        "030 partial/wrong native case origin",
    )
    require(
        actual["native_steps"] == n
        and actual["physics_steps"] == n + 40
        and actual["endpoint_count"] == k
        and actual["environment_initializations"] == 1
        and actual["current_canvas_images"] == 5
        and actual["future_rgb_frames"] == actual["source_hdf5_reads"] == 0
        and actual["stable_success_last_steps"] == 40
        and actual["native_command_ref"] == dict(path=record["expert_path"], sha256=record["expert_sha256"]),
        "030 fixed case budget/native source differs",
    )
    observation = actual["observation_contract"]
    require(
        observation["source_position"] == "robot0_eef_pos = actual grip-site position"
        and observation["source_rotation"] == "robot0_eef_quat xyzw = actual robot EEF BODY orientation"
        and observation["controller_state"] == "cached actual fields; no forced endpoint updates",
        "030 actual source/controller frame convention differs",
    )
    require(
        type(observation["actual_source_site_id"]) is int
        and observation["actual_source_site_id"] >= 0
        and type(observation["actual_eef_body_id"]) is int
        and observation["actual_eef_body_id"] >= 0
        and all(
            isinstance(observation[key], str) and observation[key]
            for key in ("actual_source_site_name", "actual_eef_body_name", "controller_site_name")
        ),
        "030 actual physical frame identities missing",
    )
    require(
        observation["tool_contract_ref"] == prepared["tool_contract_ref"]
        and observation["delta_convention"] == "inverse(T_tcp[k]) @ T_tcp[k+1], local right multiply"
        and observation["units"] == dict(position="m", time="s", rotation="column rot6d")
        and observation["labels_normalized"] is observation["camera_observables"] is False
        and observation["initial_seed"] == 42
        and observation["initial_trial"] == observation["pre_rollout_steps"] == 0
        and observation["initial_controller_refreshed"] is True,
        "030 task/control/normalization/pre-roll contract differs",
    )
    close(observation["source_from_tcp"], prepared["tool_source_from_tcp"], "030 actual canonical tool transform")
    contract = actual["controller_contract"]
    require(
        contract["control_freq"] == 20
        and contract["robot_model"] == "Panda"
        and contract["gripper_model"] == "PandaGripper"
        and contract["controller_site"] == observation["controller_site_name"]
        and contract["control_frame"] == "MuJoCo world"
        and 0 < contract["model_timestep"] <= 0.05,
        "030 native robot/controller/site/clock differs",
    )
    for key in ("input_min", "input_max", "output_min", "output_max", "kp", "kd"):
        require(numeric(contract[key], key).shape == (6,), "030 actual controller dimension differs")
    arrays = load_arrays(actual, evidence)
    maxima = verify_endpoint_geometry(arrays, record, prepared["tool_source_from_tcp"], native)
    receipts = [
        json.loads(line)
        for line in evidence.check(actual["endpoint_receipts"]["path"], actual["endpoint_receipts"]["sha256"])
        .read_text()
        .splitlines()
    ]
    require(len(receipts) == k, "030 endpoint receipts are partial/extra")
    for index, receipt in enumerate(receipts):
        offsets = arrays["gripper_current_action_offsets"]
        cache = arrays["gripper_current_action_flat"][offsets[index] : offsets[index + 1]]
        expected = dict(
            endpoint_step=index,
            sim_time=float(arrays["sim_time"][index]),
            sim_state_sha256=state_hash(arrays["sim_states"][index]),
            sim_state_dtype="float64",
            sim_state_shape=list(arrays["sim_states"][index].shape),
            native_success=bool(arrays["native_success"][index]),
            gripper_current_action_shape=list(cache.shape),
            gripper_current_action_dtype="float64",
            gripper_current_action_sha256=hashlib.sha256(cache.tobytes()).hexdigest(),
        )
        require(receipt == expected, "030 physical endpoint time/state/success receipt differs")
    require(
        [anchor["frame_index"] for anchor in actual["anchors"]] == anchors_for(n),
        "030 five fixed current anchors changed",
    )
    pixel_difference = max(
        camera_contract(anchor, receipts[anchor["frame_index"]], arrays, anchor["frame_index"], evidence)
        for anchor in actual["anchors"]
    )
    quality = native_quality(arrays["native_success"])
    require(actual["native_quality_passed"] is quality, "030 native quality flag disagrees with full endpoint success")
    return dict(
        name=record["name"],
        task_index=record["task_index"],
        native_steps=n,
        physics_steps=n + 40,
        endpoints=k,
        stable_native_success=quality,
        stable_success_last_steps=40,
        complete_native_success_sequence=arrays["native_success"].tolist(),
        maximum_anchor_reconstruction_error=maxima,
        maximum_canvas_channel_difference=pixel_difference,
    )


def quality_summary(cases):
    require(
        len(cases) == 30
        and [item["task_index"] for item in cases] == [i for i in range(3) for _ in range(10)]
        and len({item["name"] for item in cases}) == 30,
        "030 native quality requires all30 fixed cases/full denominator",
    )
    stable = sum(item["stable_native_success"] for item in cases)
    return dict(
        total_cases=30,
        stable_native_cases=stable,
        stable_native_by_task={
            str(task): sum(item["stable_native_success"] for item in cases if item["task_index"] == task)
            for task in range(3)
        },
        failed_native_cases=[item["name"] for item in cases if not item["stable_native_success"]],
        eligible_for_model_training_registration=stable == 30,
    )


def verify_collection(path, prepared, actions, evidence):
    actual = evidence.json(path)
    require(
        actual["schema_version"] == "s2a.native-training-collection.v1"
        and actual["complete"] is actual["passed"] is True
        and "error" not in actual
        and actual["prepared_ref"] == reference(Path(prepared["output_root"]) / "preparation/prepared.json"),
        "030 incomplete/wrong published native collection",
    )
    require(
        actual["environment_initializations"] == 30
        and actual["current_canvas_images"] == 150
        and actual["automatic_retry"]
        is actual["model_training_authorized"]
        is actual["fresh_performance_authorized"]
        is False
        and all(actual[key] == 0 for key in ("wam_initializations", "denoise_calls", "fits", "fresh_cases")),
        "030 collection exceeded finite CPU/no-model/fresh budget",
    )
    require(
        len(actual["task_results"]) == 3 and [item["task_index"] for item in actual["task_results"]] == [0, 1, 2],
        "030 incomplete/repeated task worker grid",
    )
    # Verify every actual worker's normal owned wait/reap before case arrays or
    # success are scored. The current fourth independent child is not finished.
    children = [owned_exit(item["lifecycle"], evidence) for item in actual["task_results"]]
    rows, total_steps = [], 0
    for index, item in enumerate(actual["task_results"]):
        task = evidence.ref(item["result"])
        require(
            task["schema_version"] == "s2a.native-training-task.v1"
            and task["complete"] is task["passed"] is True
            and "error" not in task
            and task["task_index"] == index
            and task["seed"] == 42
            and task["prepared_ref"] == actual["prepared_ref"],
            "030 task execution partial or wrong origin",
        )
        require(
            task["actual_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1")
            and task["environment_initializations"] == len(task["cases"]) == 10
            and task["current_canvas_images"] == 50
            and all(
                task[key] == 0
                for key in ("wam_initializations", "denoise_calls", "fits", "fresh_cases", "future_rgb_frames")
            ),
            "030 actual CPU/task budget differs",
        )
        records = prepared["records"][index * 10 : (index + 1) * 10]
        members = []
        for record, ref in zip(records, task["cases"], strict=True):
            expected = Path(prepared["output_root"]) / "collection" / f"task{index}" / record["name"] / "result.json"
            require(ref["path"] == str(expected), "030 case result escaped fixed task/source mapping")
            members.append(case_contract(evidence.ref(ref), record, prepared, actions[record["name"]], evidence))
        steps = sum(member["physics_steps"] for member in members)
        require(
            steps == task["physics_steps"]
            and 830 <= steps <= 6400
            and task["native_quality_passed"] is all(m["stable_native_success"] for m in members),
            "030 task step/native-quality aggregate differs",
        )
        total_steps += steps
        rows.extend(members)
    require(
        actual["physics_steps"] == total_steps <= 19200
        and total_steps == sum(r["native_steps"] + 40 for r in prepared["records"]),
        "030 complete30 physics budget differs",
    )
    summary = quality_summary(rows)
    return dict(
        collection_verified=True,
        evidence_integrity_passed=True,
        native_quality=summary,
        eligible_for_model_training_registration=summary["eligible_for_model_training_registration"],
        actual_owned_workers=children,
        physics_steps=total_steps,
        environment_initializations=30,
        current_canvas_images=150,
        future_rgb_frames=0,
        case_verification=rows,
        interpretation="Native task success qualifies these fixed training sources only; it is not WAM, S2A or reserved-fresh success. A failed native quality gate preserves all30 cases and does not invalidate otherwise complete protocol evidence.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "collection"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    args = parser.parse_args()
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "030 independent audit requires S2A cwd and emptyCUDA/LD",
    )
    require(
        (args.collection is not None) == (args.phase == "collection"),
        "030 collection evidence does not match explicit phase",
    )
    output = args.output.resolve()
    require(
        output.is_relative_to(REPO / "outputs") and not output.exists(),
        "030 audit must be new under outputs; refuses overwrite",
    )
    evidence = Evidence()
    receipt = dict(
        schema_version="s2a.native-training-independent-verification.v1",
        phase=args.phase,
        passed=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        new_physics_steps=0,
        new_wam_calls=0,
        new_fits=0,
        new_parameter_reconstructions=0,
        fresh_action_datasets_read=0,
        hdf_observation_datasets_read=0,
        future_rgb_frames_collected=0,
    )
    error = None
    try:
        prepared, actions = verify_preparation(args.prepared, args.prepared_sha256, evidence)
        receipt["prepared"] = reference(args.prepared)
        if args.phase == "prepare":
            receipt.update(
                preparation_verified=True,
                selected_native_action_datasets_verified=30,
                source_isolated_cases=30,
                original029_diagnostic_only=True,
            )
        else:
            receipt.update(verify_collection(args.collection, prepared, actions, evidence))
        evidence.recheck()
        receipt["passed"] = True
    except Exception as caught:
        error = caught
        receipt["error"] = dict(type=type(caught).__name__, message=str(caught))
    receipt["input_sha256"] = evidence.hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(passed=receipt["passed"], output=reference(output))))
    if error is not None:
        raise SystemExit(f"030 independent audit rejected: {error}")


if __name__ == "__main__":
    main()
