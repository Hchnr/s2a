"""030 synthetic source isolation, measured geometry, timing and quality checks."""

import copy
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_wam_native_training.py"
    spec = importlib.util.spec_from_file_location("independent_native_training_checker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory_fixture(checker, n=43):
    k = n + 41
    index = np.arange(k, dtype=np.float64)
    body_rotation = Rotation.from_euler(
        "xyz", np.column_stack((0.003 * index, -0.004 * index, 0.006 * index))
    ).as_matrix()
    position = np.column_stack((0.4 + 0.001 * index, -0.2 + 0.0001 * index**2, 0.8 - 0.0003 * index))
    source = checker.pose(position, body_rotation)
    site_rotation = body_rotation @ Rotation.from_euler("x", 0.7).as_matrix()
    site = checker.pose(position, site_rotation)
    body = checker.pose(position + np.asarray([0.01, 0.02, -0.03]), body_rotation)
    h = checker.pose(np.asarray([0.04, -0.03, 0.12]), Rotation.from_euler("yz", [0.2, -0.4]).as_matrix())
    tcp = source @ h
    opening = np.linspace(0.2, 0.8, k)
    q = (0.001 + 0.079 * opening) / 2
    native = np.tile(np.asarray([0.2, -0.3, 0.1, 0.4, 0.2, -0.1, -1.0], np.float32), (n, 1))
    sent = np.concatenate((native.astype(np.float64), np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (40, 1))))
    sim = np.zeros((k, 12), np.float64)
    sim[:, 0] = 0.25 + index * 0.05
    sim[:, 1:4] = position
    state = np.zeros((k, 128), np.float64)
    action = np.zeros_like(state)
    state[:, :3] = tcp[:, :3, 3]
    state[:, 3:9] = np.concatenate((tcp[:, :3, 0], tcp[:, :3, 1]), axis=1)
    state[:, 9] = opening
    # This fixture uses a general inverse, independent of the checker's
    # analytic transpose/translation construction.
    delta = np.stack([np.linalg.inv(tcp[i]) @ tcp[i + 1] for i in range(k - 1)])
    action[:-1, :3] = delta[:, :3, 3]
    action[:-1, 3:9] = np.concatenate((delta[:, :3, 0], delta[:, :3, 1]), axis=1)
    action[:-1, 9] = opening[1:]
    mask = np.zeros((k, 128), bool)
    mask[:, :10] = True
    amask = mask.copy()
    amask[-1] = False
    valid = np.ones(k, bool)
    valid[-1] = False
    cached = site[np.maximum(np.arange(k) - 1, 0)]
    arrays = dict(
        endpoint_step=np.arange(k, dtype=np.int64),
        sim_time=sim[:, 0].copy(),
        native_success=np.ones(k, bool),
        source_position=position,
        source_quaternion_xyzw=Rotation.from_matrix(body_rotation).as_quat(),
        gripper_qpos=np.column_stack((q, -q)),
        source_pose=source,
        actual_source_site_pose=site,
        actual_eef_body_pose=body,
        physical_source_pose=source.copy(),
        controller_site_pose=site.copy(),
        tcp_pose=tcp,
        opening=opening,
        native_commands=native,
        sent_commands=sent,
        sim_states=sim,
        controller_ee_pos=cached[:, :3, 3].copy(),
        controller_ee_ori_mat=cached[:, :3, :3].copy(),
        controller_goal_pos=position.copy(),
        controller_goal_ori=body_rotation.copy(),
        controller_joint_pos=np.zeros((k, 7)),
        controller_joint_vel=np.zeros((k, 7)),
        gripper_current_action_flat=np.r_[0.0, np.column_stack((q[1:], -q[1:])).reshape(-1)],
        gripper_current_action_offsets=np.r_[0, 1 + 2 * np.arange(k)].astype(np.int64),
        state128=state,
        action128=action,
        state_mask=mask,
        action_mask=amask,
        action_valid=valid,
        delta_pose=delta,
    )
    return dict(native_steps=n, initial_state=sim[0].tolist()), h, native, arrays


def test_independent_tcp_lever_noncommuting_rotation_and_all_five_windows(checker):
    record, h, native, arrays = trajectory_fixture(checker)
    error = checker.verify_endpoint_geometry(arrays, record, h, native)
    assert error["position"] < 1e-12 and error["rotation"] < 1e-12
    assert checker.anchors_for(43) == [0, 1, 2, 3, 43]
    assert checker.anchors_for(600) == [0, 186, 373, 560, 600]
    assert not np.allclose(arrays["source_pose"][:, :3, :3], arrays["controller_site_pose"][:, :3, :3])
    assert not np.allclose(arrays["source_pose"][:, :3, 3], arrays["tcp_pose"][:, :3, 3])
    wrong = copy.deepcopy(arrays)
    wrong["tcp_pose"] = h @ arrays["source_pose"]
    with pytest.raises(ValueError, match="canonical TCP"):
        checker.verify_endpoint_geometry(wrong, record, h, native)


def test_next_opening_final_mask_clock_command_hold_and_initial_are_distinct_contracts(checker):
    record, h, native, arrays = trajectory_fixture(checker)
    cases = [
        ("action128", lambda x: x.__setitem__((0, 9), arrays["opening"][0]), "action128"),
        ("action_mask", lambda x: x.__setitem__((-1, 0), True), "valid mask"),
        ("sent_commands", lambda x: x.__setitem__((-1, 0), 0.01), "native commands then40"),
        ("sim_time", lambda x: x.__setitem__(20, x[20] + 0.001), "control period"),
        ("sim_states", lambda x: x.__setitem__((0, 5), 0.01), "restored initial"),
    ]
    for key, mutate, message in cases:
        wrong = copy.deepcopy(arrays)
        mutate(wrong[key])
        with pytest.raises(ValueError, match=message):
            checker.verify_endpoint_geometry(wrong, record, h, native)


def test_actual_body_frame_errors_are_tight_but_controller_cache_can_lag(checker):
    record, h, native, arrays = trajectory_fixture(checker)
    checker.verify_endpoint_geometry(arrays, record, h, native)
    wrong = copy.deepcopy(arrays)
    wrong["actual_eef_body_pose"][10, :3, :3] = (
        Rotation.from_rotvec([1e-8, 0, 0]).as_matrix() @ wrong["actual_eef_body_pose"][10, :3, :3]
    )
    with pytest.raises(ValueError, match="actual F64 BODY"):
        checker.verify_endpoint_geometry(wrong, record, h, native)
    assert np.diff(arrays["gripper_current_action_offsets"]).tolist() == [1] + [2] * (len(arrays["opening"]) - 1)
    wrong = copy.deepcopy(arrays)
    wrong["gripper_current_action_offsets"][1] = 0
    with pytest.raises(ValueError, match="ragged gripper cache"):
        checker.verify_endpoint_geometry(wrong, record, h, native)
    wrong = copy.deepcopy(arrays)
    wrong["source_quaternion_xyzw"] = wrong["source_quaternion_xyzw"].astype(np.float32)
    with pytest.raises(ValueError, match="lost F64"):
        checker.verify_endpoint_geometry(wrong, record, h, native)


def test_selection_uses_single_rng_and_does_not_reselect_bad_length(checker):
    pools = [
        [
            dict(
                demo=f"demo_{i}",
                initial_state_sha256=f"t{task}-{i}",
                model_xml_sha256=f"x{task}-{i}",
                actions_shape=[43, 7],
            )
            for i in range(14)
        ]
        for task in range(3)
    ]
    pools[1][0]["initial_state_sha256"] = pools[0][0]["initial_state_sha256"]
    excluded = [{"demo_1", "demo_13"}] * 3
    selected, trace = checker.independent_selection(pools, excluded, {"t0-3"})
    assert all(len(x) == 10 for x in selected)
    assert len({item["initial_state_sha256"] for task in selected for item in task}) == 30
    assert "demo_3" not in trace["tasks"][0]["selected_demo_ids"]
    assert all("demo_1" not in task["numeric_demo_order"] for task in trace["tasks"])
    first = trace["tasks"][0]["selected_demo_ids"][0]
    next(row for row in pools[0] if row["demo"] == first)["actions_shape"] = [42, 7]
    again, after = checker.independent_selection(pools, excluded, {"t0-3"})
    assert [task["selected_demo_ids"] for task in after["tasks"]] == [
        task["selected_demo_ids"] for task in trace["tasks"]
    ]
    assert again[0][0]["actions_shape"] == [42, 7]
    with pytest.raises(ValueError, match="replacement is forbidden"):
        checker.anchors_for(again[0][0]["actions_shape"][0])
    with pytest.raises(ValueError, match="canonical numeric"):
        checker.demo_order(["demo_01", "demo_2"])


class MetadataDataset:
    def __init__(self, values, *, shape_only=False):
        self.values, self.shape, self.shape_only = values, values.shape, shape_only

    def __getitem__(self, item):
        if self.shape_only:
            raise AssertionError("candidate action content opened during selection")
        assert item == 0
        return self.values[0]


class MetadataFile:
    def __init__(self):
        self.names = ("demo_0", "demo_1")
        self.group = SimpleNamespace(attrs={"model_file": "<mujoco/>"})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def keys(self):
        return self.names

    def __getitem__(self, name):
        if name == "data":
            return self
        if name == "demo_0":
            raise AssertionError("fresh/excluded group opened")
        if name == "demo_1":
            return self
        if name == "states":
            return MetadataDataset(np.asarray([[0.25, 1.0, 2.0]]))
        if name == "actions":
            return MetadataDataset(np.zeros((43, 7), np.float32), shape_only=True)
        raise AssertionError("obs/images/unregistered dataset opened")

    @property
    def attrs(self):
        return self.group.attrs


def test_metadata_reader_never_opens_excluded_group_obs_or_action_values(checker):
    rows = checker.read_candidate_metadata("synthetic.hdf5", {"demo_0"}, opener=lambda *args: MetadataFile())
    assert [row["demo"] for row in rows] == ["demo_1"]
    assert rows[0]["actions_shape"] == [43, 7]
    assert (
        rows[0]["initial_state_sha256"]
        == hashlib.sha256(np.asarray([0.25, 1.0, 2.0], np.float64).tobytes()).hexdigest()
    )
    assert rows[0]["model_xml_sha256"] == hashlib.sha256(b"<mujoco/>").hexdigest()
    with pytest.raises(ValueError, match="excluded/fresh"):
        checker.selected_actions(
            [dict(demo="demo_0", task_index=0)],
            [{"demo_0"}],
            checker.Evidence(),
            opener=lambda *args: pytest.fail("excluded action source opened"),
        )


def camera_fixture(checker, tmp_path):
    yy, xx = np.indices((256, 256))
    raw = [
        np.stack((xx // 2, 10 + yy // 3, np.full_like(xx, 30)), axis=-1).astype(np.uint8),
        np.stack((200 - yy // 4, 20 + xx // 4, np.full_like(xx, 250)), axis=-1).astype(np.uint8),
    ]
    live = [image[::-1, ::-1].copy() for image in raw]
    npz = tmp_path / "cameras.npz"
    np.savez(
        npz, agentview_image=raw[0], robot0_eye_in_hand_image=raw[1], live_auxiliary_0=live[0], live_left_wrist=live[1]
    )
    png = tmp_path / "current.png"
    pixels = checker.numpy_canvas(live)
    Image.fromarray(pixels).save(png)
    receipt = dict(
        endpoint_step=0,
        sim_time=0.25,
        sim_state_sha256="a" * 64,
        sim_state_dtype="float64",
        sim_state_shape=[12],
        native_success=False,
    )
    pose = np.eye(4)
    arrays = dict(source_pose=pose[None], opening=np.asarray([0.3]))
    anchor = dict(
        frame_index=0,
        current_endpoint=receipt,
        current_source_pose=pose.tolist(),
        current_opening=0.3,
        render_before=dict(sim_time=0.25, sim_state_sha256="a" * 64),
        render_after=dict(sim_time=0.25, sim_state_sha256="a" * 64),
        camera_keys=["agentview_image", "robot0_eye_in_hand_image"],
        camera_roles=["auxiliary_0", "left_wrist"],
        raw_camera_shape=[256, 256, 3],
        raw_camera_dtype="uint8",
        live_flip_axes=[0, 1],
        future_rgb_frames=0,
        raw_cameras=checker.reference(npz),
        canvas_png=checker.reference(png),
        canvas_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        canvas_shape=[960, 640, 3],
        canvas_dtype="uint8",
    )
    return anchor, receipt, arrays, raw, live, pixels, png, npz


def test_asymmetric_current_camera_roles_flip_seam_and_no_step_binding(checker, tmp_path):
    anchor, endpoint, arrays, raw, live, pixels, png, npz = camera_fixture(checker, tmp_path)
    assert checker.camera_contract(anchor, endpoint, arrays, 0, checker.Evidence()) == 0
    wrong = copy.deepcopy(anchor)
    wrong["render_after"]["sim_time"] += 0.05
    with pytest.raises(ValueError, match="advanced/restored"):
        checker.camera_contract(wrong, endpoint, arrays, 0, checker.Evidence())
    for changed, message in (
        (checker.numpy_canvas(live[::-1]), "role/cover-crop"),
        (np.where(np.indices(pixels.shape)[0] < 482, 7, pixels).astype(np.uint8), "seam moved"),
    ):
        Image.fromarray(changed).save(png)
        wrong = dict(
            anchor, canvas_png=checker.reference(png), canvas_rgb_sha256=hashlib.sha256(changed.tobytes()).hexdigest()
        )
        with pytest.raises(ValueError, match=message):
            checker.camera_contract(wrong, endpoint, arrays, 0, checker.Evidence())
    Image.fromarray(pixels).save(png)
    np.savez(
        npz, agentview_image=raw[0], robot0_eye_in_hand_image=raw[1], live_auxiliary_0=raw[0], live_left_wrist=live[1]
    )
    wrong = dict(anchor, raw_cameras=checker.reference(npz), canvas_png=checker.reference(png))
    with pytest.raises(ValueError, match="camera flip"):
        checker.camera_contract(wrong, endpoint, arrays, 0, checker.Evidence())


def test_native_quality_uses_final40_poststep_endpoints_and_never_shrinks_denominator(checker):
    success = np.ones(84, bool)
    success[-41] = False
    assert checker.native_quality(success) is True
    success[-40] = False
    assert checker.native_quality(success) is False
    cases = [
        dict(name=f"task{i}_case{j}", task_index=i, stable_native_success=not (i == 1 and j == 2))
        for i in range(3)
        for j in range(10)
    ]
    summary = checker.quality_summary(cases)
    assert summary["total_cases"] == 30 and summary["stable_native_cases"] == 29
    assert summary["eligible_for_model_training_registration"] is False
    assert summary["stable_native_by_task"] == {"0": 10, "1": 9, "2": 10}
    with pytest.raises(ValueError, match="full denominator"):
        checker.quality_summary([item for item in cases if item["stable_native_success"]])


def test_owned_worker_requires_real_wait_matching_birth_and_cpu_environment(checker, tmp_path):
    paths = {key: tmp_path / f"{key}.json" for key in ("launch", "exit", "release")}
    values = dict(
        launch=dict(
            selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""), child=dict(pid=123, start_ticks=456)
        ),
        exit=dict(actual_wait=True, automatic_retry=False, returncode=0),
        release=dict(
            pid=123,
            start_ticks=456,
            returncode=0,
            reaped=True,
            signal_sent=False,
            original_error=None,
            only_owned_child_signalled=True,
        ),
    )

    def saved(changed):
        for key, path in paths.items():
            path.write_text(json.dumps(changed[key]))
        return {key: checker.reference(path) for key, path in paths.items()}

    assert checker.owned_exit(saved(values), checker.Evidence())["reaped"] is True
    for key, mutation, message in (
        ("exit", dict(actual_wait=False), "normal child wait"),
        ("release", dict(start_ticks=457), "wait/reap identity"),
        ("launch", dict(selected_environment=dict(CUDA_VISIBLE_DEVICES="6", LD_LIBRARY_PATH="")), "CPU-only"),
    ):
        wrong = copy.deepcopy(values)
        wrong[key].update(mutation)
        with pytest.raises(ValueError, match=message):
            checker.owned_exit(saved(wrong), checker.Evidence())


def test_all_workers_exit_before_case_audit_and_native_quality_false_is_a_complete_result(
    checker, tmp_path, monkeypatch
):
    prepared_path = tmp_path / "preparation/prepared.json"
    prepared_path.parent.mkdir()
    prepared_path.write_text("{}")
    records = [dict(name=f"task{i}_case{j}", task_index=i, native_steps=43) for i in range(3) for j in range(10)]
    prepared = dict(output_root=str(tmp_path), records=records)
    prep_ref = checker.reference(prepared_path)
    actual = dict(
        schema_version="s2a.native-training-collection.v1",
        complete=True,
        passed=True,
        prepared_ref=prep_ref,
        environment_initializations=30,
        current_canvas_images=150,
        automatic_retry=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        wam_initializations=0,
        denoise_calls=0,
        fits=0,
        fresh_cases=0,
        physics_steps=2490,
        task_results=[],
    )
    documents = {}
    for index in range(3):
        path = str(tmp_path / "collection" / f"task{index}/result.json")
        task = dict(
            schema_version="s2a.native-training-task.v1",
            complete=True,
            passed=True,
            task_index=index,
            seed=42,
            prepared_ref=prep_ref,
            actual_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", PYTHONDONTWRITEBYTECODE="1"),
            environment_initializations=10,
            current_canvas_images=50,
            physics_steps=830,
            wam_initializations=0,
            denoise_calls=0,
            fits=0,
            fresh_cases=0,
            future_rgb_frames=0,
            native_quality_passed=index != 1,
            cases=[],
        )
        for record in records[index * 10 : (index + 1) * 10]:
            case_path = str(tmp_path / "collection" / f"task{index}" / record["name"] / "result.json")
            task["cases"].append(dict(path=case_path, sha256="a" * 64))
            documents[case_path] = record
        documents[path] = task
        actual["task_results"].append(dict(task_index=index, result=dict(path=path, sha256="a" * 64), lifecycle=index))
    order = []
    monkeypatch.setattr(
        checker, "owned_exit", lambda record, evidence: order.append(f"exit{record}") or dict(reaped=True)
    )

    def case(actual_case, record, *args):
        assert order[:3] == ["exit0", "exit1", "exit2"]
        order.append(record["name"])
        return dict(
            name=record["name"],
            task_index=record["task_index"],
            physics_steps=83,
            stable_native_success=record["name"] != "task1_case2",
        )

    monkeypatch.setattr(checker, "case_contract", case)
    evidence = SimpleNamespace(json=lambda path: actual, ref=lambda ref: documents[ref["path"]])
    result = checker.verify_collection("collection.json", prepared, {r["name"]: None for r in records}, evidence)
    assert result["collection_verified"] is result["evidence_integrity_passed"] is True
    assert result["eligible_for_model_training_registration"] is False
    assert result["native_quality"]["stable_native_cases"] == 29
    assert len(result["case_verification"]) == 30 and len(order) == 33


def test_source_closure_preserves111_and_all_tracked_new_python_after_sorted_json(checker, tmp_path):
    inherited = {str(tmp_path / f"old{i}.py"): f"{i:064x}" for i in range(111)}
    tracked = ["src/current_math.py", "tests/current_test.py", "scripts/not_python.txt"]
    sources = {
        **inherited,
        **{str(checker.REPO / p): "a" * 64 for p in (*checker.NEW_SOURCES, *tracked[:2])},
        str(checker.CONFIG): checker.CONFIG_SHA,
        str(checker.PLAN): checker.PLAN_SHA,
    }
    path = tmp_path / "preparation/prepared.json"
    snapshots = {
        str(path.parent / "source_snapshot" / f"{i:04d}_{Path(p).name}"): sources[p]
        for i, p in enumerate(sorted(sources))
    }
    prepared = json.loads(
        json.dumps(
            dict(source_sha256=sources, inherited_source_sha256=inherited, snapshot_sha256=snapshots), sort_keys=True
        )
    )
    checker.source_closure(prepared, dict(source_sha256=inherited), path, tracked=tracked)
    for key in (
        str(checker.REPO / "scripts/collect_wam_native_training.py"),
        str(checker.REPO / "tests/current_test.py"),
    ):
        wrong = copy.deepcopy(prepared)
        wrong["source_sha256"].pop(key)
        with pytest.raises(ValueError, match="closure is incomplete"):
            checker.source_closure(wrong, dict(source_sha256=inherited), path, tracked=tracked)
    wrong = copy.deepcopy(prepared)
    wrong["source_sha256"][next(iter(inherited))] = "f" * 64
    with pytest.raises(ValueError, match="original111"):
        checker.source_closure(wrong, dict(source_sha256=inherited), path, tracked=tracked)


def test_real_script_main_entry_has_every_phase_helper_bound_before_any_io(checker):
    class MainEntryObserved(Exception):
        pass

    path = Path(checker.__file__).resolve()

    def trace(frame, event, argument):
        if event == "call" and frame.f_code.co_filename == str(path) and frame.f_code.co_name == "main":
            assert all(
                callable(frame.f_globals.get(key))
                for key in (
                    "verify_preparation",
                    "verify_collection",
                    "camera_contract",
                    "verify_endpoint_geometry",
                    "native_quality",
                )
            )
            raise MainEntryObserved
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(MainEntryObserved):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.settrace(previous)
