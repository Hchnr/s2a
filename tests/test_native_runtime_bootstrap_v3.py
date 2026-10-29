"""Pure file and synthetic runtime tests for the four historical texture paths."""

import ast
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_runtime_bootstrap import case as shared_case

from s2a import native_runtime_bootstrap_v2 as prior
from s2a import native_runtime_bootstrap_v3 as repaired

REPO = prior.frozen.REPO
MANIFEST_REF = dict(
    path=REPO / "outputs/native_runtime_path_snapshot_20261029/runtime_manifest.json",
    sha256="ec8c377eb9fe505d69745f55e33b4ccb6f352fdf6d2e49555d6146c39cb111f5",
)
SNAPSHOT_REF = dict(
    path=REPO / "outputs/native_runtime_path_snapshot_20261029/manifest.json",
    sha256="a9ffc0fc6b883f00676d8211958cf96b752b0f977d72a737fcca17fde6b742cd",
)
HISTORICAL_PREFIX = "/Users/yifengz/workspace/libero-dev/chiliocosm/assets/scenes/../textures/"
TEXTURES = (
    "light-gray-floor-tile.png",
    "martin_novak_wood_table.png",
    "martin_novak_wood_table.png",
    "light-gray-plaster.png",
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    return shared_case.__wrapped__(tmp_path, monkeypatch)


def _read_bound(ref):
    data = ref["path"].read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["sha256"]
    return json.loads(data)


def test_actual_four_record_paths_match_bound_snapshot_files_without_runtime():
    """Read frozen XML/inventory/files only; no simulation imports or calls."""
    manifest = _read_bound(MANIFEST_REF)
    snapshot = _read_bound(SNAPSHOT_REF)
    root = ET.fromstring(manifest["record"]["model_xml"])
    declared = [node.get("file") for node in root.iter() if node.get("file") and ".." in Path(node.get("file")).parts]
    assert declared == [HISTORICAL_PREFIX + name for name in TEXTURES]
    asset_root = Path(manifest["runtime_config"]["libero_root"]) / "libero/libero/assets"
    inventory = {row["snapshot_ref"]["path"]: row["snapshot_ref"]["sha256"] for row in snapshot["file_inventory"]}
    mapped = ET.Element("mujoco")
    assets = ET.SubElement(mapped, "asset")
    for path in declared:
        ET.SubElement(assets, "texture", file=str(asset_root / path.split("/assets/", 1)[1]))
    records = prior._private_asset_records(ET.tostring(mapped, encoding="unicode"), (asset_root,), inventory)
    assert len(records) == 4 and len({row["canonical_path"] for row in records}) == 3
    assert all(row["normalization_changed"] for row in records)
    assert all(row["reference"]["sha256"] == inventory[row["canonical_path"]] for row in records)


def test_all_99_actual_file_mappings_preserve_non_file_xml_without_runtime(monkeypatch):
    """Execute only the bound XML functions and v3 record-remapping AST block."""
    manifest = _read_bound(MANIFEST_REF)
    snapshot = _read_bound(SNAPSHOT_REF)
    libero_root = Path(manifest["runtime_config"]["libero_root"])
    robosuite_root = REPO / "outputs/libero_env311/lib/python3.11/site-packages/robosuite"
    asset_roots = (libero_root / "libero/libero/assets", robosuite_root / "models/assets")
    namespace = dict(
        ET=ET, os=os, deepcopy=deepcopy, robosuite=SimpleNamespace(__file__=str(robosuite_root / "__init__.py"))
    )
    for path, sha, name in (
        (
            robosuite_root / "utils/mjcf_utils.py",
            "85edc2ffc5ba3e79f8372468259aa8d16ba886ede9ab556593f88817c0dfaddd",
            "find_elements",
        ),
        (
            libero_root / "libero/libero/utils/utils.py",
            "30e78e4f05a9ef25d6ccf55e3d7ab5739b39f58c098a9ab91b78fb57919ea6ba",
            "postprocess_model_xml",
        ),
    ):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        function = next(
            node for node in ast.parse(data).body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    # Read no historical /Users or retired /share/project files, even as probes.
    for method in ("exists", "is_symlink", "read_bytes"):
        original = getattr(Path, method)

        def guarded(path, *args, _original=original, **kwargs):
            assert not str(path).startswith(("/Users/", "/share/project/"))
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, guarded)

    run = repaired._repaired_function()
    block = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "original_xml" for target in child.targets)
            for child in node.body
        )
    ).body
    begin = next(
        index
        for index, node in enumerate(block)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "original_xml" for target in node.targets)
    )
    end = next(
        index
        for index, node in enumerate(block)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "reset_from_xml_string"
    )
    namespace.update(
        Path=Path,
        runtime=SimpleNamespace(
            postprocess_xml=namespace["postprocess_model_xml"], get_path=lambda _: str(asset_roots[0])
        ),
        record=manifest["record"],
        roots=asset_roots,
        receipt={},
        _require=prior.frozen._require,
        _xml_structure=prior.frozen._xml_structure,
        _hash=prior.frozen._hash,
    )
    exec(compile(ast.Module(body=block[begin:end], type_ignores=[]), str(repaired.__file__), "exec"), namespace)
    xml = namespace["xml"]
    original_xml = ET.fromstring(manifest["record"]["model_xml"])
    assert prior.frozen._xml_structure(xml) == prior.frozen._xml_structure(original_xml)
    records = prior._private_asset_records(ET.tostring(xml, encoding="unicode"), asset_roots, None)
    assert len(records) == 99 and len({row["canonical_path"] for row in records}) == 96
    assert sum(row["normalization_changed"] for row in records) == 4
    assert [
        row["declared"] for row in namespace["receipt"]["record_xml_redirects"] if ".." in Path(row["declared"]).parts
    ] == [HISTORICAL_PREFIX + name for name in TEXTURES]
    inventory = {row["snapshot_ref"]["path"]: row["snapshot_ref"]["sha256"] for row in snapshot["file_inventory"]}
    for row in records:
        if Path(row["canonical_path"]).is_relative_to(asset_roots[0]):
            assert row["reference"]["sha256"] == inventory[row["canonical_path"]]
        assert row["reference"]["sha256"] == hashlib.sha256(Path(row["canonical_path"]).read_bytes()).hexdigest()


def test_inverse_proof_only_changes_record_predicate():
    proof = repaired.compile_repair_proof()
    assert proof["run_inverse_to_v2_ast_equal"] and proof["record_guard_replacement_count"] == 1
    assert proof["record_xml_precondition_unchanged"] is False
    assert proof["record_xml_absolute_check_preserved"] and proof["asset_validator_and_cleanup_unchanged"]
    changed = repaired._repaired_function()
    next(node for node in ast.walk(changed) if isinstance(node, ast.Constant) and node.value == 256).value = 255
    with pytest.raises(ValueError, match="non-record-guard"):
        repaired.compile_repair_proof(changed)


def test_four_historical_parent_paths_reach_only_mapped_synthetic_assets(case, monkeypatch):
    (case.assets / "scenes").mkdir()
    (case.assets / "textures").mkdir()
    for name in set(TEXTURES):
        (case.assets / "textures" / name).write_bytes(name.encode())
    xml = ET.Element("mujoco")
    ET.SubElement(xml, "option", timestep="0.002", gravity="0 0 -9.81")
    assets = ET.SubElement(xml, "asset")
    for name in TEXTURES:
        ET.SubElement(assets, "texture", file=HISTORICAL_PREFIX + name)
    original_xml = ET.tostring(xml, encoding="unicode")
    case.kwargs["record"]["model_xml"] = original_xml
    original_exists = Path.exists
    original_symlink = Path.is_symlink

    def guarded_exists(path):
        assert not str(path).startswith("/Users/"), "historical external path must not be probed"
        return original_exists(path)

    def guarded_symlink(path):
        assert not str(path).startswith("/Users/"), "historical external path must not be probed"
        return original_symlink(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "is_symlink", guarded_symlink)
    before = dict(vars(prior.frozen))
    result = repaired.run_native_bootstrap(**case.kwargs)
    assert result["passed"] and result["cleanup_complete"]
    assert result["counters"]["model_construct"] == dict(attempted=3, completed=3)
    assert result["counters"]["forward"]["completed"] == 16
    final_xml = Path(result["model_records"][-1]["xml_ref"]["path"]).read_text()
    assert prior.frozen._xml_structure(ET.fromstring(original_xml)) == prior.frozen._xml_structure(
        ET.fromstring(final_xml)
    )
    assert result["original_record_xml_sha256"] == hashlib.sha256(original_xml.encode()).hexdigest()
    assert len(result["record_xml_redirects"]) == 4
    assert [row["declared"] for row in result["record_xml_redirects"]] == [
        HISTORICAL_PREFIX + name for name in TEXTURES
    ]
    records = result["model_records"][-1]["assets"]
    assert len(records) == 4 and all(row["normalization_changed"] for row in records)
    assert [row["canonical_path"] for row in records] == [str(case.assets / "textures" / name) for name in TEXTURES]
    assert vars(prior.frozen) == before


@pytest.mark.parametrize(
    "declared, message",
    [
        ("relative/assets/scenes/../mesh.stl", "invalid declared XML path"),
        ("/Users/old/unknown/mesh.stl", "unrecognized external XML path"),
        ("/Users/old/assets/../outside.stl", "leaves approved root"),
        ("/Users/old/assets/scenes/../../assets/mesh.stl", "leaves approved root"),
    ],
)
def test_bad_record_mapping_fails_before_third_compile(case, declared, message):
    (case.assets / "scenes").mkdir()
    case.kwargs["record"]["model_xml"] = f'<mujoco><asset><mesh file="{declared}"/></asset></mujoco>'
    with pytest.raises(repaired.BootstrapFailure, match=message) as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    receipt = caught.value.receipt
    assert receipt["counters"]["model_construct"]["completed"] == 2
    assert receipt["failure_stage"] == "record_xml"
    assert receipt["cleanup_outcome"] == "closed" and receipt["cleanup_complete"]


def test_mapped_symlink_before_parent_component_is_rejected(case):
    (case.assets / "scenes").symlink_to(case.assets, target_is_directory=True)
    case.kwargs["record"]["model_xml"] = (
        '<mujoco><asset><mesh file="/Users/old/assets/scenes/../mesh.stl"/></asset></mujoco>'
    )
    with pytest.raises(repaired.BootstrapFailure, match="symlink") as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["model_construct"] == dict(attempted=3, completed=2)


def test_output_guard_stays_strict(case):
    case.kwargs["output"] = case.kwargs["output"] / ".." / "unsafe_output"
    with pytest.raises(repaired.BootstrapFailure, match="traversing"):
        repaired.run_native_bootstrap(**case.kwargs)
    assert case.actions.reset_calls == 0


def test_forbidden_physics_is_still_counted_without_dispatch(case):
    case.actions.failure = "mj_step"
    with pytest.raises(repaired.BootstrapFailure) as caught:
        repaired.run_native_bootstrap(**case.kwargs)
    assert caught.value.receipt["counters"]["mj_step"] == dict(attempted=1, completed=0)
    assert case.actions.native_steps == 0
