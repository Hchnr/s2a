"""038 metadata preparation with opaque, deliberately non-decodable label files."""

import copy
import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from s2a.artifacts import REPO


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("prepare_wam_clean_training")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def dataset(preparation, tmp_path, monkeypatch):
    module = preparation
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")

    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return module.reference(path)

    def command(name):
        return dict(
            launch=write(f"input/{name}_launch.json", {"child": {"pid": 10, "start_ticks": 20}}),
            exit=write(f"input/{name}_exit.json", dict(actual_wait=True, returncode=0)),
            release=write(
                f"input/{name}_release.json", dict(pid=10, start_ticks=20, reaped=True, returncode=0, signal_sent=False)
            ),
        )

    png = tmp_path / "input/current.png"
    png.parent.mkdir(parents=True)
    # A signature/IHDR witness only. Pixel decoding/re-rendering is out of scope.
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 960))
    image_ref = module.reference(png)
    opaque = tmp_path / "input/trajectory.npz"
    opaque.write_bytes(b"NOT AN NPZ: preparation must hash but never decode future labels")
    trajectory_ref = module.reference(opaque)
    fresh = [
        dict(task_index=t, demo=f"demo_{200 + i}", initial_state_sha256=digest(f"fresh{t}/{i}"))
        for t in range(3)
        for i in range(4)
    ]
    fresh_ref = write("input/fresh_metadata.json", dict(records=fresh))
    tasks = ["put the bowl on the plate", "put the wine bottle on the rack", "put the cream cheese in the bowl"]
    records, case_refs = [], []
    for task in range(3):
        for demo in range(100, 110):
            frames = [0, 16, 33, 50, 130]
            record = dict(
                task_index=task,
                demo=f"demo_{demo}",
                initial_state_sha256=digest(f"train{task}/{demo}"),
                native_steps=90,
                anchor_frames=frames,
                task=tasks[task],
            )
            records.append(record)
            anchors = []
            for frame in frames:
                physics = dict(sim_time=frame * 0.05, sim_state_sha256=digest(f"state{task}/{demo}/{frame}"))
                anchors.append(
                    dict(
                        frame_index=frame,
                        future_rgb_frames=0,
                        canvas_shape=[960, 640, 3],
                        current_endpoint=dict(endpoint_step=frame, **physics),
                        render_before=physics,
                        render_after=dict(physics),
                        canvas_png=image_ref,
                    )
                )
            index = {key: dict(shape=[171, 128], dtype="float64") for key in ("state128", "action128")}
            index.update({key: dict(shape=[171, 128], dtype="bool") for key in ("state_mask", "action_mask")})
            index["action_valid"] = dict(shape=[171], dtype="bool")
            case_refs.append(
                (
                    record,
                    dict(
                        passed=True,
                        complete=True,
                        native_quality_passed=True,
                        record=record,
                        native_steps=90,
                        endpoint_count=171,
                        future_rgb_frames=0,
                        arrays=trajectory_ref,
                        array_index=index,
                        anchors=anchors,
                    ),
                )
            )
    settled = dict(
        records=list(reversed(records)),
        original_prepared=dict(
            fresh_fixtures_ref=fresh_ref,
            selection=dict(tasks=[dict(task_index=t, excluded_demo_ids=["demo_1"]) for t in range(3)]),
            forbidden_initial_state_sha256=[r["initial_state_sha256"] for r in fresh],
            tool_source_from_tcp=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ),
    )
    refs = {"settled_prepared": write("input/settled_prepared.json", settled)}
    collection = []
    for record, case in case_refs:
        case["prepared_ref"] = refs["settled_prepared"]
        collection.append(
            dict(
                name=f"t{record['task_index']}_{record['demo']}",
                case=write(f"input/cases/{record['task_index']}_{record['demo']}.json", case),
            )
        )
    refs["settled_collection"] = write(
        "input/settled_collection.json",
        dict(passed=True, complete=True, cases=collection, prepared_ref=refs["settled_prepared"]),
    )
    refs["settled_audit"] = write(
        "input/settled_audit.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=refs["settled_prepared"],
            eligible_for_model_training_registration=True,
        ),
    )
    refs["settled_owner"] = write(
        "input/settled_owner.json", dict(passed=True, complete=True, verified_ref=refs["settled_audit"])
    )
    failed034 = write("input/failed034.json", dict(passed=False))
    lifecycle = dict(actual_wait=True, returncode=0, reaped=True, signal_sent=False)
    refs["settled_peer"] = write(
        "input/settled_peer.json",
        dict(
            passed=True,
            complete=True,
            prepared_ref=refs["settled_prepared"],
            verified_ref=refs["settled_audit"],
            owner_ref=refs["settled_owner"],
            original_failed_owner_ref=failed034,
            worker_lifecycle=lifecycle,
            audit_lifecycle=lifecycle,
            new_quality=dict(
                total_cases=30,
                stable_native_cases=30,
                stable_native_by_task={str(t): 10 for t in range(3)},
                eligible_for_model_training_registration=True,
            ),
            native_prefix_exact_cases=30,
            sources_and_inputs_before_after_exact=True,
            preserved_original034_owner_passed=False,
        ),
    )
    templates, val, links = [], [], []
    by_episode = [
        (1254, 0, "demo_32"),
        (1126, 0, "demo_16"),
        (868, 1, "demo_27"),
        (1139, 1, "demo_4"),
        (1209, 2, "demo_14"),
        (1176, 2, "demo_24"),
    ]
    for task in range(3):
        request = dict(
            action_chunk_size=40,
            domain_name="franka_panda__parallel_gripper",
            fps=20,
            guidance=1.0,
            image_size=960,
            model_mode="wam",
            name=f"old{task}",
            num_frames=41,
            num_steps=35,
            prompt=tasks[task],
            seed=42,
            shift=5.0,
            vision_path=str(png),
            extra=dict(
                action128=dict(
                    action_dim_mask=[True] * 10 + [False] * 118,
                    action_domain="libero",
                    action_fps=20,
                    camera_layout=[],
                    return_video=False,
                    selected_roles=[],
                    task_title=tasks[task],
                )
            ),
        )
        templates.append(dict(request=request, name=f"old{task}"))
    for episode, task, demo in by_episode:
        links.append(dict(r2_episode=episode, task_index=task, demo=demo, initial_sync_certified=False))
        for index in range(4):
            val.append(
                dict(
                    episode=episode,
                    anchor_index=index,
                    anchor_frame=index * 10,
                    name=f"ep{episode}_a{index}",
                    split="validation",
                    valid_future_steps=40,
                    task_index=task,
                    task=tasks[task],
                    arrays_path=trajectory_ref["path"],
                    arrays_sha256=trajectory_ref["sha256"],
                    image_path=image_ref["path"],
                    image_sha256=image_ref["sha256"],
                    measured_state128=[0.0] * 128,
                    state_mask128=[True] * 10 + [False] * 118,
                    source_episode_id=f"old-source-{episode}",
                )
            )
    historical_ref = write(
        "input/historical.json", dict(anchors=val, conservative_source_exclusion=dict(records=links))
    )
    qualified_source = tmp_path / "input/qualified.py"
    qualified_source.write_text("# previously qualified read-only source\n")
    bridge = dict(
        root=str(tmp_path / "old_model"),
        config=dict(historical_prepared_ref=historical_ref),
        requests3=templates,
        source_sha256={str(qualified_source): module.sha(qualified_source)},
        weight_files={},
    )
    refs["bridge_prepared"] = write("input/bridge_prepared.json", bridge)
    refs["bridge_qualification"] = write(
        "input/bridge_qualification.json",
        dict(
            passed=True,
            complete=True,
            bridge_qualified=True,
            prepared_ref=refs["bridge_prepared"],
            actual_requests=7,
            actual_denoise_calls=245,
        ),
    )
    refs["bridge_gpu_after"] = write(
        "input/gpu.json", dict(released=True, memory_mib=0, compute_apps=[], uuid=module.GPU)
    )
    refs["bridge_failed_owner"] = write(
        "input/bridge_failed_owner.json",
        dict(passed=False, owned_commands=[command("model")], gpu_after_ref=refs["bridge_gpu_after"]),
    )
    refs["native_mask_prepared"] = write(
        "input/native_prepared.json",
        dict(
            original_failed_owner_ref=refs["bridge_failed_owner"],
            original_prepared_ref=refs["bridge_prepared"],
            qualification_ref=refs["bridge_qualification"],
        ),
    )
    refs["native_mask_review"] = write(
        "input/native_review.json",
        dict(
            passed=True,
            complete=True,
            bridge_qualified=True,
            actual_online_points_verified=245,
            nonzero_changed_active_values=102,
        ),
    )
    refs["native_mask_owner"] = write(
        "input/native_owner.json",
        dict(
            passed=True,
            complete=True,
            bridge_qualified=True,
            registration_ref=refs["native_mask_prepared"],
            review_ref=refs["native_mask_review"],
            original_failed_owner_ref=refs["bridge_failed_owner"],
            original_prepared_ref=refs["bridge_prepared"],
            qualification_ref=refs["bridge_qualification"],
            preserved_original_owner_passed=False,
            owned_commands=[command("independent")],
        ),
    )
    refs["native_mask_peer"] = write(
        "input/native_peer.json",
        dict(
            passed=True,
            complete=True,
            consumed_source_and_artifacts_before_after_exact=True,
            owner039_ref=refs["native_mask_owner"],
            formal_ref=refs["native_mask_review"],
            worker_ref=refs["bridge_qualification"],
            prepared_ref=refs["bridge_prepared"],
            consumed_sha256={
                refs[key]["path"]: refs[key]["sha256"] for key in ("native_mask_owner", "native_mask_review")
            },
        ),
    )
    plan = tmp_path / "plans/038.md"
    plan.parent.mkdir()
    plan.write_text("# synthetic metadata-only038 fixture\n")
    source = tmp_path / "unit_source.py"
    source.write_text("# source fingerprint fixture\n")
    config = dict(
        schema_version=module.CONFIG_SCHEMA,
        seed=42,
        bindings_complete=True,
        budget=copy.deepcopy(module.BUDGET),
        resources=copy.deepcopy(module.RESOURCES),
        refs=refs,
        noise_roots=[101, 102],
        evaluation_seeds=[42, 43],
        ridge=0.1,
        feature_terms=[],
        validation_episodes=list(module.VAL_EPISODES),
        gpu_uuid=module.GPU,
        automatic_retry=False,
        preparation_only=True,
        execution_authorized=False,
        output_dir="outputs/wam_stability_20261021/clean_residual_training_20261023",
        plan_path="plans/038.md",
        source_paths=["unit_source.py"],
        execution_entrypoints={"collect": "scripts/missing_collect.py"},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return dict(config=config, path=config_path, write=write, png=png, trajectory=opaque)


def test_complete_metadata_contract_enumerates150_300_and_never_decodes_labels(preparation, dataset, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("future NPZ decoding is not allowed in prepare")

    monkeypatch.setattr(np, "load", forbidden)
    reference = preparation.prepare(dataset["path"])
    result = json.loads(Path(reference["path"]).read_text())
    assert len(result["training_anchors"]) == 150 and len(result["training_requests"]) == 300
    assert len(set(result["noise_roster"]["training_seeds"])) == 300
    assert not {42, 43} & set(result["noise_roster"]["training_seeds"])
    assert len(result["validation_anchors"]) == 24 and len(result["validation_generations"]) == 96
    assert result["training_rounds"][0]["rows"] == result["training_rounds"][1]["rows"] == 420000
    assert result["union_rows"] == 840000 and result["budget"]["total_denoise"] == 55965
    assert result["future_label_archive_decodes"] == result["new_model_calls"] == result["new_fits"] == 0
    assert result["readiness"]["ready_for_execution"] is False and result["execution_authorized"] is False
    assert result["readiness"]["missing_entrypoints"] == ["scripts/missing_collect.py"]
    assert result["exclusion"]["validation_initial_sync_certified"] is False
    assert result["original037_owner_passed"] is False
    first = result["training_requests"][0]
    assert first["task_index"] == 0 and first["demo"] == "demo_100" and first["root_seed"] == 101
    assert all(
        "state" not in item["request"] and "target" not in item["request"] for item in result["training_requests"]
    )
    assert set(result["source_sha256"].values()) == set(result["snapshot_sha256"].values())
    with pytest.raises(ValueError, match="existing"):
        preparation.prepare(dataset["path"])


@pytest.mark.parametrize("fault", ["pending", "budget", "resources", "seed", "output", "authorization"])
def test_preparation_cannot_change_fixed_budget_or_treat_unbound_qualification_as_true(preparation, dataset, fault):
    value = copy.deepcopy(dataset["config"])
    if fault == "pending":
        value["refs"]["native_mask_peer"]["sha256"] = None
    elif fault == "budget":
        value["budget"]["raw_training_requests"] -= 1
    elif fault == "resources":
        value["resources"]["d0_timeout_s"] = 7200
    elif fault == "seed":
        value["seed"] = 43
    elif fault == "output":
        value["output_dir"] = "outputs/fresh/new_write"
    else:
        value["execution_authorized"] = True
    with pytest.raises(ValueError):
        preparation.validate_config(value)


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("native_mask_owner", "passed", False),
        ("native_mask_peer", "complete", False),
        ("bridge_failed_owner", "passed", True),
        ("bridge_gpu_after", "memory_mib", 1),
        ("settled_peer", "preserved_original034_owner_passed", True),
    ],
)
def test_wrong_or_rewritten_joint_evidence_is_rejected(preparation, dataset, key, field, value):
    config = copy.deepcopy(dataset["config"])
    old = json.loads(Path(config["refs"][key]["path"]).read_text())
    old[field] = value
    config["refs"][key] = dataset["write"](f"input/wrong_{key}.json", old)
    with pytest.raises(ValueError):
        preparation.prerequisites(config, preparation.Evidence())


def test_binary_metadata_reader_never_unpickles_or_parses_archive(preparation, dataset):
    evidence = preparation.Evidence()
    ref = preparation.reference(dataset["trajectory"])
    assert evidence.check(ref, kind="opaque_future_label_sha_only") == ref
    with pytest.raises(ValueError, match="refuses binary"):
        evidence.json(ref)


def test_current_png_and_opaque_label_changes_are_rejected_by_sha(preparation, dataset):
    dataset["png"].write_bytes(dataset["png"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA changed"):
        preparation.prepare(dataset["path"])


def test_validation_demo_leak_cannot_enter_training_roster(preparation, dataset):
    lineage = preparation.prerequisites(dataset["config"], preparation.Evidence())
    record = next(r for r in lineage["settled_prepared"]["records"] if r["task_index"] == 0)
    record["demo"] = "demo_32"
    with pytest.raises(ValueError):
        preparation.training_metadata(dataset["config"], preparation.Evidence(), lineage)


def test_template_cannot_receive_future_conditions_and_six_validation_episodes_are_indivisible(preparation, dataset):
    lineage = preparation.prerequisites(dataset["config"], preparation.Evidence())
    template = copy.deepcopy(lineage["bridge_prepared"]["requests3"][0]["request"])
    template["future_state"] = [1]
    with pytest.raises(ValueError, match="future model input"):
        preparation._request(
            template, name="new", image=preparation.reference(dataset["png"]), seed=101, task=template["prompt"]
        )
    historical_ref = lineage["bridge_prepared"]["config"]["historical_prepared_ref"]
    historical = json.loads(Path(historical_ref["path"]).read_text())
    historical["anchors"].pop()
    lineage["bridge_prepared"]["config"]["historical_prepared_ref"] = dataset["write"](
        "input/missing_val.json", historical
    )
    with pytest.raises(ValueError, match="four-anchor"):
        preparation.validation_metadata(lineage, preparation.Evidence())
