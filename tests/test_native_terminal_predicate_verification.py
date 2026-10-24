"""Pure fixtures for static predicate semantics, provenance, counters and CLI."""

import copy
import hashlib
import importlib.util
import json
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_native_terminal_predicate_diagnostic.py"
    spec = importlib.util.spec_from_file_location("independent_terminal_predicate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(checker, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return checker.reference(path)


def state_record(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(
        values=value.tolist(),
        dtype="float64",
        shape=list(value.shape),
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def object_record(name, actual_class, body_id, geom_id, position):
    return dict(
        name=name,
        actual_class=actual_class,
        root_body=name + "_root",
        body_name=name + "_root",
        body_id=body_id,
        world_pos=list(position),
        world_mat=np.eye(3).tolist(),
        world_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        contact_geoms=[
            dict(
                name=name + "_collision",
                id=geom_id,
                body_id=body_id,
                size=[0.1, 0.1, 0.1],
                type=6,
                world_pos=list(position),
                world_mat=np.eye(3).tolist(),
            )
        ],
    )


def endpoint_fixture(checker, spec, frame, state, truth=True):
    source = dict(
        state=state.copy(),
        state_sha256=state_record(state)["sha256"],
        truth=truth,
        source_pose=np.eye(4),
        time=float(state[0]),
    )
    point = dict(
        schema_version="s2a.native-terminal-predicate-endpoint.v1",
        name=spec["name"],
        role=spec["role"],
        frame_index=frame,
        source_state_ref=dict(arrays=spec["arrays_ref"], key="sim_states", frame_index=frame),
        source_receipt_ref=spec["endpoint_receipts_ref"],
        original_success=truth,
        actual_raw_state_before=state_record(state),
        actual_raw_state_after=state_record(state),
        sim_time_before=float(state[0]),
        sim_time_after=float(state[0]),
        actual_source_pose=np.eye(4).tolist(),
        official_goal=[["on", "wine_bottle_1", "wine_rack_1_top_region"]],
        official_success=True,
        static_success_reproduced=truth,
        under_inputs=dict(
            site_name="wine_rack_1_top_region",
            site_id=5,
            site_body_id=20,
            parent_name="wine_rack_1",
            region_state_parent_name="wine_rack_1",
            python_region_class="SiteObject",
            has_under=True,
            parent_exists=True,
            python_size_dtype="float64",
            wine_body_id=10,
            wine_body_name="wine_bottle_1_root",
            wine_world_pos=[0.0, 0.0, 0.05],
            site_world_pos=[0.0, 0.0, 0.0],
            site_world_mat=np.eye(3).tolist(),
            python_size=[0.1, 0.1, 0.01],
            model_site_size=[0.1, 0.1, 0.01],
            other_height=0.10,
        ),
        geometry_margins=dict(delta_position=[0.0, 0.0, 0.05], xy_lower=[0.1, 0.1], z_lower=0.045, z_upper=0.06),
        objects=dict(
            wine=object_record("wine_bottle_1", "WineBottle", 10, 1, [0.0, 0.0, 0.05]),
            rack=object_record("wine_rack_1", "WineRack", 20, 2, [0.0, 0.0, 0.0]),
        ),
        atoms=dict(under=True, contact=True, ontop=True),
        ncon=1,
        contacts=[
            dict(
                index=0,
                geom1_id=2,
                geom2_id=1,
                geom1_name="wine_rack_1_collision",
                geom2_name="wine_bottle_1_collision",
                dist=0.001,
                pos=[0.0, 0.0, 0.0],
                frame=np.eye(3).tolist(),
            )
        ],
    )
    return point, source


def test_actual_R_not_transpose_and_strict_region_boundaries(checker):
    matrix = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert checker.under_geometry([0, 0, 0], matrix, [0.02, 0.02, 0.01], [0, 0.04, -0.01])["under"] is True
    assert checker.under_geometry([0, 0, 0], matrix.T, [0.02, 0.02, 0.01], [0, 0.04, -0.01])["under"] is False
    for position in ([0.1, 0, 0.05], [-0.1, 0, 0.05], [0, 0.1, 0.05], [0, 0, 0.01 - 0.005], [0, 0, 0.01 + 0.10]):
        assert checker.under_geometry([0, 0, 0], np.eye(3), [0.1, 0.1, 0.01], position)["under"] is False
    inside = np.nextafter(0.01 - 0.005, np.inf)
    assert checker.under_geometry([0, 0, 0], np.eye(3), [0.1, 0.1, 0.01], [0, 0, inside])["under"] is True


def test_contact_both_orientations_positive_distance_and_complete_identity(checker):
    contact = dict(index=0, geom1_id=2, geom2_id=1, geom1_name="rack", geom2_name="wine", dist=0.04)
    assert checker.contact_membership([contact], ["wine"], ["rack"])["contact"] is True
    assert checker.contact_membership([dict(contact, geom1_name="other")], ["wine"], ["rack"])["contact"] is False
    with pytest.raises(ValueError, match="ordering"):
        checker.contact_membership([dict(contact, index=1)], ["wine"], ["rack"])


def test_endpoint_source_state_atoms_and_negative_reproduction_are_distinct(checker):
    spec = dict(
        name="fixture",
        role="failure",
        arrays_ref=dict(path="source.npz", sha256="a" * 64),
        endpoint_receipts_ref=dict(path="source.jsonl", sha256="b" * 64),
    )
    point, source = endpoint_fixture(checker, spec, 43, np.asarray([2.15, 0, 0], np.float64), truth=False)
    row = checker.verify_endpoint(point, spec, 43, source)
    assert row["official_success"] is True and row["static_success_reproduced"] is False
    changed = copy.deepcopy(point)
    changed["actual_raw_state_after"] = state_record([2.2, 0, 0])
    with pytest.raises(ValueError, match="installation/time"):
        checker.verify_endpoint(changed, spec, 43, source)
    changed = copy.deepcopy(point)
    changed["atoms"]["contact"] = False
    with pytest.raises(ValueError, match="composition"):
        checker.verify_endpoint(changed, spec, 43, source)
    changed = copy.deepcopy(point)
    changed["contacts"][0]["geom1_id"] = 9
    with pytest.raises(ValueError, match="actual geom identities"):
        checker.verify_endpoint(changed, spec, 43, source)


def lifecycle(checker, root, returncode=0):
    command = ["python", "scripts/collect_native_terminal_predicate_diagnostic.py", "--seed", "42"]
    data = dict(
        launch=dict(
            automatic_retry=False,
            selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            child=dict(pid=123, start_ticks=456),
            command=command,
        ),
        exit=dict(actual_wait=True, automatic_retry=False, returncode=returncode),
        release=dict(
            pid=123,
            start_ticks=456,
            returncode=returncode,
            reaped=True,
            signal_sent=False,
            only_owned_child_signalled=True,
            original_error=None,
        ),
    )
    return {key: save(checker, root / (key + ".json"), value) for key, value in data.items()}


def test_owned_normal_wait_identity_required_before_any_scoring(checker, tmp_path):
    life = lifecycle(checker, tmp_path)
    assert checker.owned_worker(life, checker.Evidence())["child"]["start_ticks"] == 456
    bad = lifecycle(checker, tmp_path, returncode=1)
    with pytest.raises(ValueError, match="wait0"):
        checker.owned_worker(bad, checker.Evidence())


def diagnostic_fixture(checker, tmp_path, monkeypatch, *, mismatch=False):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    root = tmp_path / "outputs/diagnostic"
    prepared = dict(output_root=str(root), cases=[])
    strict = dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(tmp_path))
    cv2sources = [save(checker, tmp_path / f"cv2_{i}.json", dict(source=i)) for i in range(3)]
    probe = dict(after_import_cv2=dict(strict, LD_LIBRARY_PATH="registered:/lib:"), cv2_loader_ref=cv2sources[0])
    prepared.update(cv2_probe_ref=save(checker, tmp_path / "probe.json", probe), cv2_runtime_source_refs=cv2sources)
    all_items = []
    points = []
    for case_index in range(4):
        directory = root / "collection" / f"case{case_index}"
        directory.mkdir(parents=True)
        states = np.zeros((84, 5), np.float64)
        states[:, 0] = np.arange(84) * 0.05
        truth = np.ones(84, bool)
        if mismatch and case_index == 0:
            truth[43] = False
        poses = np.tile(np.eye(4), (84, 1, 1))
        arrays = directory / "source.npz"
        np.savez(
            arrays,
            sim_states=states,
            native_success=truth,
            source_pose=poses,
            sim_time=states[:, 0],
            native_commands=np.asarray([dict(unused="must not be decoded")], dtype=object),
        )
        receipts = [
            dict(
                endpoint_step=i,
                sim_time=float(s[0]),
                sim_state_sha256=state_record(s)["sha256"],
                sim_state_dtype="float64",
                sim_state_shape=[5],
                native_success=bool(truth[i]),
            )
            for i, s in enumerate(states)
        ]
        path = directory / "source.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in receipts))
        spec = dict(
            name=f"case{case_index}",
            role="failure" if case_index < 2 else "matched_positive",
            frame_indices=[43, 44, 45, 46],
            native_steps=43,
            arrays_ref=checker.reference(arrays),
            endpoint_receipts_ref=checker.reference(path),
            case_result_ref=save(checker, directory / "original_case.json", dict(source=True)),
            expected_native_success=truth[[43, 44, 45, 46]].tolist(),
        )
        items = []
        for frame in spec["frame_indices"]:
            point, _ = endpoint_fixture(checker, spec, frame, states[frame], truth=bool(truth[frame]))
            points.append(point)
            items.append(
                dict(
                    name=spec["name"],
                    role=spec["role"],
                    frame_index=frame,
                    bundle=save(checker, directory / f"endpoint{frame:04d}.json", point),
                )
            )
        prepared["cases"].append(spec)
        all_items.extend(items)
    prep_ref = save(checker, root / "preparation/prepared.json", prepared)
    life = lifecycle(checker, root / "owned")
    readiness = dict(
        complete=True,
        passed=True,
        restore_performed=True,
        prepared_ref=prep_ref,
        before=strict,
        after_restore=strict,
        after_import_cv2=probe["after_import_cv2"],
        cv2_probe_ref=prepared["cv2_probe_ref"],
        cv2_runtime_source_refs=cv2sources,
        actual_cv2_loader_ref=probe["cv2_loader_ref"],
        source_before_sha256={x["path"]: x["sha256"] for x in cv2sources},
        source_after_sha256={x["path"]: x["sha256"] for x in cv2sources},
        bytecode_before="1",
        bytecode_after_import="1",
        bytecode_after_restore="1",
        pid=123,
        argv=["scripts/collect_native_terminal_predicate_diagnostic.py", "--seed", "42"],
    )
    ready_ref = save(checker, root / "collection/environment_readiness.json", readiness)
    save(
        checker,
        root / "collection/worker_started.json",
        dict(
            pid=123, argv=readiness["argv"], prepared_ref=prep_ref, environment_readiness_ref=ready_ref, cuda="", ld=""
        ),
    )
    ledger = []
    counts = Counter()
    phases = defaultdict(Counter)

    def event(phase, name, frame, **value):
        ledger.append(dict(event_index=len(ledger), phase=phase, case_name=name, frame_index=frame, **value))

    def call(phase, name, frame, kind, api, children=()):
        counts[kind] += 1
        phases[phase][kind] += 1
        event(phase, name, frame, kind=kind, api=api, event="before_delegate", delegated=False, time_before=None)
        for child in children:
            child()
        counts[kind + "_delegated"] += 1
        phases[phase][kind + "_delegated"] += 1
        event(
            phase,
            name,
            frame,
            kind=kind,
            api=api,
            event="after_delegate",
            delegated=True,
            time_before=None,
            time_after=None,
        )

    for spec in prepared["cases"]:
        name = spec["name"]
        counts["environment_initializations"] += 1
        phases["setup"]["environment_initializations"] += 1
        event("setup", name, None, event="environment_initialization", api="make_environment", integration=False)
        call("setup", name, None, "single_reset_calls", "ControlEnv.reset")
        call("setup", name, None, "official_success_calls", "Libero_Tabletop_Manipulation._check_success")
        for frame in spec["frame_indices"]:
            point = next(p for p in points if p["name"] == name and p["frame_index"] == frame)
            counts["installed_saved_endpoints"] += 1
            phases["static"]["installed_saved_endpoints"] += 1
            event(
                "static",
                name,
                frame,
                event="installed_saved_state",
                api="MjSim.set_state_from_flattened",
                integration=False,
                time_before=0.0,
                time_after=point["sim_time_before"],
                source_state_sha256=point["actual_raw_state_before"]["sha256"],
            )

            def leaf(kind, api):
                return lambda: call("static", name, frame, kind, api)

            under = leaf("internal_under_calls", "SiteObject.under")
            contact = leaf("internal_check_contact_calls", "MujocoEnv.check_contact")

            def ontop():
                call(
                    "static",
                    name,
                    frame,
                    "internal_On_calls",
                    "On.__call__",
                    [
                        lambda: call(
                            "static",
                            name,
                            frame,
                            "internal_check_ontop_calls",
                            "SiteObjectState.check_ontop",
                            [under, contact],
                        )
                    ],
                )

            call("static", name, frame, "forward_calls", "mujoco.mj_forward")
            call(
                "static", name, frame, "official_success_calls", "Libero_Tabletop_Manipulation._check_success", [ontop]
            )
            call("static", name, frame, "additional_atomic_predicate_calls", "SiteObject.under.explicit", [under])
            call(
                "static",
                name,
                frame,
                "additional_atomic_predicate_calls",
                "MujocoEnv.check_contact.explicit",
                [contact],
            )
            call(
                "static",
                name,
                frame,
                "additional_atomic_predicate_calls",
                "Libero_Tabletop_Manipulation._eval_predicate.explicit",
                [ontop],
            )
    path = root / "collection/call_ledger.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in ledger))
    hooks = [
        dict(api=api, replacement="single_delegate_no_retry_process_local" if api == "ControlEnv.reset" else None)
        for api in sorted(checker.REQUIRED_HOOKS)
    ]
    actual = dict(
        schema_version="s2a.native-terminal-predicate-collection.v1",
        complete=True,
        passed=not mismatch,
        prepared_ref=prep_ref,
        cases=[
            dict(
                name=s["name"],
                role=s["role"],
                frame_indices=s["frame_indices"],
                case_result_ref=s["case_result_ref"],
                endpoints=[i for i in all_items if i["name"] == s["name"]],
            )
            for s in prepared["cases"]
        ],
        endpoints=all_items,
        original_native_quality_passed=False,
        eligible_for_model_training_registration=False,
        model_training_authorized=False,
        fresh_performance_authorized=False,
        automatic_retry=False,
        final_environment=strict,
        environment_readiness_ref=ready_ref,
        counts=dict(counts),
        phase_counts={k: dict(v) for k, v in phases.items()},
        installed_hooks=hooks,
        violation=None,
        call_ledger_ref=checker.reference(path),
        ledger_events=len(ledger),
        all_saved_states_reproduced=not mismatch,
    )
    if mismatch:
        actual["error"] = dict(type="StaticPredicateReproductionMismatch")
    collection_path = root / "collection/result.json"
    collection_ref = save(checker, collection_path, actual)
    wrapper = root / "worker_lifecycle.json"
    save(checker, wrapper, dict(prepared_ref=prep_ref, collection_ref=collection_ref, worker_lifecycle=life))
    return prepared, actual, collection_path, wrapper


@pytest.mark.parametrize("mismatch", [False, True])
def test_complete_four_case_sixteen_endpoint_cli_contract_keeps_negative_truth(
    checker, tmp_path, monkeypatch, mismatch
):
    prepared, _, collection, life = diagnostic_fixture(checker, tmp_path, monkeypatch, mismatch=mismatch)
    result = checker.verify_collection(collection, life, prepared, checker.Evidence())
    assert result["endpoints_verified"] == 16 and len(result["endpoint_verification"]) == 16
    assert result["diagnostic_consistency_passed"] is (not mismatch)
    assert len(result["reproduction_mismatches"]) == int(mismatch)
    assert result["ledger_verified"]["integrated_physics_steps"] == 0
    assert result["collision_detection_independently_reimplemented"] is False
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    prepared_path = Path(prepared["output_root"]) / "preparation/prepared.json"

    def prepared_fixture(path, expected, evidence):
        return evidence.json(path, expected)

    monkeypatch.setattr(checker, "verify_preparation", prepared_fixture)
    output = Path(prepared["output_root"]) / "review/verified.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--prepared",
            str(prepared_path),
            "--prepared-sha256",
            checker.sha256(prepared_path),
            "--collection",
            str(collection),
            "--worker-lifecycle",
            str(life),
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    if mismatch:
        with pytest.raises(SystemExit, match="saved original and static predicate truth differ"):
            checker.main()
    else:
        checker.main()
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is (not mismatch) and receipt["endpoints_verified"] == 16
    assert (
        receipt["original_native_quality_passed"]
        is receipt["eligible_for_model_training_registration"]
        is receipt["model_training_authorized"]
        is False
    )
    assert receipt["input_sha256"][str(collection)] == checker.sha256(collection)


def test_ledger_catches_delegated_integration_or_hidden_summary_count(checker, tmp_path, monkeypatch):
    prepared, actual, path, life = diagnostic_fixture(checker, tmp_path, monkeypatch)
    result = checker.verify_collection(path, life, prepared, checker.Evidence())
    rows = result["endpoint_verification"]
    altered = copy.deepcopy(actual)
    altered["counts"]["mj_step_calls"] = 1
    with pytest.raises(ValueError, match="counts"):
        checker.verify_ledger(altered, prepared, rows, checker.Evidence())
    altered = copy.deepcopy(actual)
    altered["violation"] = dict(api="mj_step")
    with pytest.raises(ValueError, match="forbidden attempted"):
        checker.verify_ledger(altered, prepared, rows, checker.Evidence())
    with pytest.raises(ValueError, match="all16"):
        checker.summarize_endpoints(rows[:-1])


def test_runtime_inventory_and_forbidden_data_file_reads_are_closed(checker, tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    for name in ("predicate.py", "task.bddl", "must_not_open.hdf5", "must_not_open.png"):
        (root / name).write_text("fixture")
    expected = {str(root / name): checker.sha256(root / name) for name in ("predicate.py", "task.bddl")}
    checker.runtime_inventory(
        dict(predicate_runtime_root=str(root), predicate_runtime_extensions=[".py", ".bddl"]),
        expected,
        checker.Evidence(),
    )
    with pytest.raises(ValueError, match="forbidden"):
        checker.Evidence().check(root / "must_not_open.hdf5")


def test_source_snapshot_mapping_survives_sorted_json_and_rejects_missing_source(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    sources = {}
    snapshots = {}
    for i in range(423):
        source = tmp_path / "old" / f"{i}.py"
        source.parent.mkdir(exist_ok=True)
        source.write_text(f"# frozen{i}")
        snap = tmp_path / "old_snap" / f"{i}.py"
        snap.parent.mkdir(exist_ok=True)
        snap.write_bytes(source.read_bytes())
        sources[str(source)] = checker.sha256(source)
        snapshots[str(snap)] = checker.sha256(snap)
    native = dict(
        source_sha256=sources.copy(),
        snapshot_sha256=snapshots.copy(),
        first_continuation_snapshot_sha256=dict(list(snapshots.items())[:421]),
    )
    root = tmp_path / "outputs/new"
    for i, relative in enumerate(checker.NEW_SOURCES):
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# diagnostic source")
        sources[str(source)] = checker.sha256(source)
        snap = root / "preparation/snapshots" / (f"{i:02d}_" + relative.replace("/", "__"))
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_bytes(source.read_bytes())
        snapshots[str(snap)] = checker.sha256(snap)
    prepared = json.loads(
        json.dumps(dict(output_root=str(root), source_sha256=sources, snapshot_sha256=snapshots), sort_keys=True)
    )
    checker.source_closure(prepared, native, checker.Evidence())
    prepared["source_sha256"].pop(str(tmp_path / checker.NEW_SOURCES[-1]))
    with pytest.raises(ValueError, match="registered8"):
        checker.source_closure(prepared, native, checker.Evidence())


def test_cli_saves_false_and_refuses_overwrite_without_real_data(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")

    def rejected(*args):
        raise ValueError("fixture source rejected")

    monkeypatch.setattr(checker, "verify_preparation", rejected)
    output = tmp_path / "outputs/rejected.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--prepared",
            "absent.json",
            "--prepared-sha256",
            "0" * 64,
            "--collection",
            "absent.json",
            "--worker-lifecycle",
            "absent.json",
            "--output",
            str(output),
            "--seed",
            "42",
        ],
    )
    with pytest.raises(SystemExit, match="fixture source rejected"):
        checker.main()
    saved = output.read_bytes()
    actual = json.loads(saved)
    assert (
        actual["passed"]
        is actual["attribution_supported"]
        is actual["original_native_quality_passed"]
        is actual["model_training_authorized"]
        is False
    )
    with pytest.raises(ValueError, match="refuses overwrite"):
        checker.main()
    assert output.read_bytes() == saved


def test_real_main_entry_binds_all_helpers_before_any_io():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_native_terminal_predicate_diagnostic.py"

    class StopBeforeIO(Exception):
        pass

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == str(path) and frame.f_code.co_name == "main":
            assert all(
                callable(frame.f_globals[name])
                for name in (
                    "verify_preparation",
                    "verify_collection",
                    "verify_ledger",
                    "verify_endpoint",
                    "summarize_endpoints",
                    "owned_worker",
                )
            )
            raise StopBeforeIO
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(StopBeforeIO):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.settrace(previous)
