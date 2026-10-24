"""Adversarial metadata tests for the independent038 verifier, without producers."""

import ast
import copy
import hashlib
import importlib.util
import json
import struct
from argparse import Namespace
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_wam_clean_training_preparation.py"
SPEC = importlib.util.spec_from_file_location("independent_clean_preparation", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def ref(name):
    return {"path": f"/{name}.json", "sha256": "a" * 64}


class Metadata:
    """Synthetic metadata registry; binary parsing deliberately unavailable."""

    def __init__(self):
        self.documents = {}
        self.binary_checks = []

    def add(self, name, value):
        reference = ref(name)
        self.documents[reference["path"]] = value
        return reference

    def json(self, reference):
        assert reference["path"].endswith(".json")
        return self.documents[reference["path"]]

    def check(self, reference):
        self.binary_checks.append(reference["path"])
        return Path(reference["path"])

    def png(self, reference):
        self.binary_checks.append(reference["path"])
        return reference.copy()


def template(task):
    return dict(
        action_chunk_size=40,
        domain_name="test-domain",
        fps=20,
        guidance=1.0,
        image_size=960,
        model_mode="wam",
        name="qualified",
        num_frames=41,
        num_steps=35,
        prompt=task,
        seed=42,
        shift=5.0,
        vision_path="/qualified.png",
        extra={
            "action128": dict(
                action_dim_mask=[True] * 10 + [False] * 118,
                action_domain="observed-local-delta",
                action_fps=20,
                camera_layout=[["current", "top", 640, 480]],
                return_video=False,
                selected_roles=["current"],
                task_title=task,
            )
        },
    )


def request(task, name, image, seed):
    return dict(template(task), name=name, vision_path=image["path"], seed=seed)


@pytest.fixture
def readiness():
    return dict(
        config=dict(
            schema_version="s2a.clean-training-config.v1",
            seed=42,
            bindings_complete=True,
            budget=copy.deepcopy(audit.BUDGET),
            resources=copy.deepcopy(audit.RESOURCES),
            noise_roots=[101, 102],
            evaluation_seeds=[42, 43],
            validation_episodes=[1254, 1126, 868, 1139, 1209, 1176],
            ridge=0.1,
            feature_terms=[],
            gpu_uuid=audit.GPU,
            automatic_retry=False,
            numerical_contract=copy.deepcopy(audit.NUMERICS),
            execution_entrypoints=audit.ENTRYPOINTS.copy(),
            preparation_only=True,
            execution_authorized=False,
        ),
        budget=copy.deepcopy(audit.BUDGET),
        resources=copy.deepcopy(audit.RESOURCES),
        noise_seed_algorithm=audit.SEED_DESCRIPTION,
        execution_authorized=False,
        fresh_performance_authorized=False,
        readiness=dict(
            metadata_contract_complete=True,
            ready_for_execution=False,
            entrypoint_presence_is_not_qualification=True,
            missing_entrypoints=list(audit.ENTRYPOINTS.values()),
            reasons=["not runtime-qualified"],
        ),
        future_label_archive_decodes=0,
        new_model_initializations=0,
        new_model_calls=0,
        new_fits=0,
        physics_steps=0,
        fresh_performance_reads=0,
        original037_owner_passed=False,
        full_weight_bytes_rehashed=False,
    )


@pytest.fixture
def train():
    ev = Metadata()
    fresh = [
        dict(task_index=t, demo=f"demo_{900 + i}", initial_state_sha256=f"{900 + t * 4 + i:064x}")
        for t in range(3)
        for i in range(4)
    ]
    fresh_ref = ev.add("fresh-fixture-metadata", {"records": fresh})
    original = dict(
        fresh_fixtures_ref=fresh_ref,
        forbidden_initial_state_sha256=["f" * 64],
        selection={"tasks": [dict(task_index=t, excluded_demo_ids=["demo_800"]) for t in range(3)]},
        tool_source_from_tcp=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.0036], [0, 0, 0, 1]],
    )
    records, cases, anchors, requests, seeds = [], [], [], [], []
    settled_ref = ref("settled-prepared")
    for t in range(3):
        for d in range(100, 110):
            source = dict(task_index=t, demo=f"demo_{d}", initial_state_sha256=f"{1 + t * 10 + d - 100:064x}")
            record = dict(source, native_steps=100, anchor_frames=[0, 20, 40, 60, 140], task=f"task{t}")
            records.append(record)
            array = {"path": f"/task{t}_demo{d}.npz", "sha256": "b" * 64}
            currents = []
            for frame in (0, 20, 40, 60, 140):
                image = {"path": f"/t{t}d{d}f{frame}.png", "sha256": "c" * 64}
                current = dict(
                    frame_index=frame,
                    canvas_png=image,
                    canvas_shape=[960, 640, 3],
                    future_rgb_frames=0,
                    render_before={"sim_state_sha256": source["initial_state_sha256"]},
                    render_after={"sim_state_sha256": source["initial_state_sha256"]},
                    current_endpoint=dict(endpoint_step=frame, sim_state_sha256=source["initial_state_sha256"]),
                )
                currents.append(current)
            index = {
                key: dict(shape=[181, 128], dtype=dtype)
                for key, dtype in (
                    ("state128", "float64"),
                    ("action128", "float64"),
                    ("state_mask", "bool"),
                    ("action_mask", "bool"),
                )
            }
            index["action_valid"] = dict(shape=[181], dtype="bool")
            case = dict(
                passed=True,
                complete=True,
                native_quality_passed=True,
                prepared_ref=settled_ref,
                record=record,
                arrays=array,
                native_steps=100,
                endpoint_count=181,
                future_rgb_frames=0,
                anchors=currents,
                array_index=index,
            )
            case_ref = ev.add(f"case{t}_{d}", case)
            cases.append({"case": case_ref})
            for current in currents:
                frame, image = current["frame_index"], current["canvas_png"]
                name = f"task{t}_demo_{d}_a{frame}"
                anchors.append(
                    dict(
                        name=name,
                        source_identity=source.copy(),
                        anchor_frame=frame,
                        task=f"task{t}",
                        native_steps=100,
                        case_ref=case_ref,
                        declared_trajectory_ref=array.copy(),
                        trajectory_ref=array.copy(),
                        image_ref=image.copy(),
                        current_endpoint=current["current_endpoint"].copy(),
                        future_window_steps=40,
                        future_labels_parsed=False,
                        tool_source_from_tcp=original["tool_source_from_tcp"],
                    )
                )
                for root in (101, 102):
                    text = '{"anchor_frame":%d,"demo":"demo_%d","root_seed":%d,"task":%d}' % (frame, d, root, t)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    seed = int(digest[:16], 16) % (2**63)
                    seeds.append(seed)
                    req_name = f"train_{name}_r{root}"
                    requests.append(
                        dict(
                            source,
                            index=len(requests),
                            anchor_frame=frame,
                            root_seed=root,
                            sampling_seed=seed,
                            canonical_json=text,
                            identity_sha256=digest,
                            anchor_name=name,
                            name=req_name,
                            request=request(f"task{t}", req_name, image, seed),
                        )
                    )
    q = dict(
        settled_prepared=dict(records=records, original_prepared=original),
        settled_collection=dict(prepared_ref=settled_ref, cases=cases),
        bridge_prepared={"requests3": [{"request": template(f"task{t}")} for t in range(3)]},
    )
    p = dict(
        prerequisite_refs={"settled_prepared": settled_ref},
        training_anchors=anchors,
        training_requests=requests,
        exclusion=dict(
            fresh_fixture_ref=fresh_ref,
            fresh_sources=fresh,
            validation_demo_keys=[
                [0, "demo_16"],
                [0, "demo_32"],
                [1, "demo_4"],
                [1, "demo_27"],
                [2, "demo_14"],
                [2, "demo_24"],
            ],
            forbidden_demo_keys=[[t, "demo_800"] for t in range(3)],
            forbidden_initial_state_sha256=["f" * 64],
            validation_initial_sync_certified=False,
        ),
        noise_roster=dict(training_seeds=seeds, evaluation_seeds=[42, 43]),
        row_order=audit.ORDER,
        union_rows=840000,
        training_weight_per_row={"W1": "1/420000", "W2": "1/840000"},
        training_rounds=[
            dict(name="D0", policy="raw", request_indices=list(range(300)), rows=420000),
            dict(name="D1", policy="fixed_W1", request_indices=list(range(300)), rows=420000),
        ],
    )
    return p, q, ev


def test_golden_seed_and_native_identity():
    assert audit.seed_identity(0, "demo_8", 0, 101) == (
        '{"anchor_frame":0,"demo":"demo_8","root_seed":101,"task":0}',
        "afa98a9d8f84f6b2a8b298af3436044e146be55e00217f07f3bc58c6a73232fb",
        3434428600169068210,
    )
    assert audit.identity(dict(task_index=2, demo="demo_0", initial_state_sha256="0" * 64)) == (2, "demo_0", "0" * 64)


@pytest.mark.parametrize(
    "task,demo,frame,root",
    [(True, "demo_8", 0, 101), (0, "demo_08", 0, 101), (0, "demo_8", -1, 101), (0, "demo_8", 0, 42)],
)
def test_seed_identity_rejects_alias_or_eval_root(task, demo, frame, root):
    with pytest.raises(ValueError):
        audit.seed_identity(task, demo, frame, root)


def test_metadata_qualification_never_grants_execution(readiness):
    audit.budget_and_readiness(readiness)
    readiness["readiness"]["missing_entrypoints"] = []
    audit.budget_and_readiness(readiness)  # Even eight present entrypoints do not change admission.
    readiness["readiness"]["ready_for_execution"] = True
    with pytest.raises(ValueError, match="execution qualification"):
        audit.budget_and_readiness(readiness)


@pytest.mark.parametrize("kind", ["budget", "noise", "future", "original_false", "numerical", "entrypoints"])
def test_fixed_contract_tampering_rejected(readiness, kind):
    if kind == "budget":
        readiness["config"]["budget"]["raw_training_requests"] = readiness["budget"]["raw_training_requests"] = 299
    elif kind == "noise":
        readiness["config"]["noise_roots"] = [42, 43]
    elif kind == "future":
        readiness["future_label_archive_decodes"] = 1
    elif kind == "original_false":
        readiness["original037_owner_passed"] = True
    elif kind == "numerical":
        readiness["config"]["numerical_contract"]["bias_regularized"] = False
    else:
        readiness["config"]["execution_entrypoints"]["collect"] = "scripts/unreviewed.py"
    with pytest.raises(ValueError):
        audit.budget_and_readiness(readiness)


def test_complete_training_metadata_and_no_archive_parse(train):
    p, q, evidence = train
    assert audit.training(p, q, evidence) == dict(
        training_sources=30, training_anchors=150, training_noise_seeds=300, training_requests=300
    )
    assert len({x for x in evidence.binary_checks if x.endswith(".npz")}) == 30


@pytest.mark.parametrize(
    "kind",
    [
        "drop",
        "duplicate",
        "seed",
        "row_order",
        "future_input",
        "image",
        "future_parsed",
        "zero_labels",
        "state_alias",
        "val_demo",
        "fresh_demo",
        "round",
    ],
)
def test_training_rejects_real_contract_violations(train, kind):
    p, q, ev = train
    if kind == "drop":
        p["training_requests"].pop()
    elif kind == "duplicate":
        q["settled_prepared"]["records"][1]["initial_state_sha256"] = q["settled_prepared"]["records"][0][
            "initial_state_sha256"
        ]
    elif kind == "seed":
        p["training_requests"][0]["sampling_seed"] ^= 1
    elif kind == "row_order":
        p["training_anchors"][0], p["training_anchors"][1] = p["training_anchors"][1], p["training_anchors"][0]
    elif kind == "future_input":
        p["training_requests"][0]["request"]["extra"]["future_state"] = [1]
    elif kind == "image":
        p["training_anchors"][0]["image_ref"]["sha256"] = "d" * 64
    elif kind == "future_parsed":
        p["training_anchors"][0]["future_labels_parsed"] = True
    elif kind == "zero_labels":
        ev.documents["/case0_100.json"]["array_index"]["action128"]["shape"] = [0, 128]
    elif kind == "state_alias":
        q["settled_prepared"]["original_prepared"]["forbidden_initial_state_sha256"].append(
            q["settled_prepared"]["records"][0]["initial_state_sha256"]
        )
    elif kind == "val_demo":
        q["settled_prepared"]["records"][0]["demo"] = "demo_16"
    elif kind == "fresh_demo":
        q["settled_prepared"]["records"][0]["demo"] = "demo_900"
    else:
        p["training_rounds"][1]["policy"] = "best_of_W1_W2"
    with pytest.raises(ValueError):
        audit.training(p, q, ev)


def test_current_png_and_opaque_byte_identity(tmp_path):
    png = tmp_path / "current.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 640, 960))
    reference = dict(path=str(png), sha256=hashlib.sha256(png.read_bytes()).hexdigest())
    ev = audit.Evidence()
    assert ev.png(reference) == reference
    opaque = tmp_path / "future.npz"
    opaque.write_bytes(b"This is deliberately NOT a zip or valid NumPy archive")
    binary_ref = dict(path=str(opaque), sha256=hashlib.sha256(opaque.read_bytes()).hexdigest())
    assert ev.check(binary_ref) == opaque
    with pytest.raises(ValueError, match="opaque"):
        ev.json(binary_ref)
    opaque.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during"):
        ev.unchanged()
    with pytest.raises(ValueError, match="SHA differs"):
        ev.check(binary_ref)


def test_rejects_png_with_wrong_layout_even_with_correct_sha(tmp_path):
    path = tmp_path / "wrong.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0\0\0\rIHDR" + struct.pack(">II", 960, 640))
    with pytest.raises(ValueError, match="IHDR"):
        audit.Evidence().png(dict(path=str(path), sha256=audit.file_sha(path)))


def test_current_only_request_cannot_include_privileged_condition():
    image = {"path": "/current.png", "sha256": "a" * 64}
    actual = request("task0", "n", image, 101)
    audit.request_binding(actual, template("task0"), name="n", image=image, seed=101, task="task0")
    actual["extra"]["action128"]["action_dim_mask"][10] = True
    with pytest.raises(ValueError):
        audit.request_binding(actual, template("task0"), name="n", image=image, seed=101, task="task0")


def test_independent_imports_are_stdlib_only():
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports <= {"argparse", "hashlib", "json", "os", "re", "struct", "pathlib"}
    assert not {"s2a", "numpy", "prepare_wam_clean_training"} & imports


def test_refuses_output_escape_existing_and_non_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    path = tmp_path / "outputs/existing.json"
    path.parent.mkdir()
    path.write_text("original")
    args = Namespace(seed=42, prepared="missing.json", prepared_sha256="a" * 64, output=str(path))
    with pytest.raises(ValueError, match="existing"):
        audit.verify(args)
    assert path.read_text() == "original"
    args.output = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escaped"):
        audit.verify(args)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="empty CUDA"):
        audit.verify(args)


def test_failed_audit_persists_false_receipt_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    prepared = tmp_path / "broken.json"
    prepared.write_text('{"schema_version":"wrong"}')
    output = tmp_path / "outputs/new/verified.json"
    args = Namespace(seed=42, prepared=str(prepared), prepared_sha256=audit.file_sha(prepared), output=str(output))
    with pytest.raises(ValueError, match="schema"):
        audit.verify(args)
    result = json.loads(output.read_text())
    assert result["passed"] is result["complete"] is result["execution_ready"] is False
    assert result["error"]["type"] == "ValueError"
    with pytest.raises(ValueError, match="existing"):
        audit.verify(args)
