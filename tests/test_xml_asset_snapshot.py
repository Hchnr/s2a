"""Synthetic asset bytes and XML only; no real asset snapshot or simulator."""

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from s2a import xml_asset_snapshot as snapshot


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))
    return reference(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    repo, safe = tmp_path / "repo", tmp_path / "safe"
    repo.mkdir()
    safe.mkdir()
    monkeypatch.setattr(snapshot, "REPO", repo)
    monkeypatch.setattr(snapshot, "SAFE_ROOT", safe)
    monkeypatch.setattr(snapshot, "OLD_PREFIX", str(tmp_path / "deleted") + "/")
    monkeypatch.setattr(snapshot, "SAFE_PREFIX", str(safe) + "/")
    originals = [safe / branch / "same & ü.msh" for branch in ("a", "b")]
    for index, path in enumerate(originals):
        path.parent.mkdir()
        path.write_bytes(b"synthetic mesh\x00" + bytes([index]))
    declared = [snapshot.OLD_PREFIX + str(path.relative_to(safe)) for path in originals]
    escaped = [escape(path, {'"': "&quot;", "'": "&apos;"}) for path in declared]
    # Comments, mixed quotes, CRLF, numeric character references and an apparent
    # file attribute inside another attribute must survive byte for byte.
    xml = repo / "source.xml"
    xml.write_bytes(
        (
            '<?xml version="1.0" encoding="utf-8"?>\r\n'
            '<mujoco><option density="1.2" viscosity="2e-05"/>\r\n'
            f"<!-- file='{escaped[0]}' is a comment -->\r\n"
            f"<asset><mesh name=\"a\" file='{escaped[0].replace('&amp;', '&#x26;')}'/>\r\n"
            f'<mesh name="b" file="{escaped[1]}"/>\r\n'
            f'<mesh name="repeated" file="{escaped[0]}"/></asset>\r\n'
            '<custom><numeric name="file=\'literal.msh\'" data="1"/></custom></mujoco>\r\n'
        ).encode("utf-8")
    )
    history = repo / "history.json"
    save_json(
        history,
        {"assets": {old: {"sha256": reference(path)["sha256"]} for old, path in zip(declared, originals, strict=True)}},
    )
    return dict(
        xml=xml, history=history, destination=repo / "new_snapshot", originals=originals, declared=declared, repo=repo
    )


def prepare(inputs):
    return snapshot.prepare_asset_snapshot(
        reference(inputs["xml"]), reference(inputs["history"]), inputs["destination"]
    )


def test_complete_closure_is_copied_without_basename_collision_or_other_xml_edits(inputs):
    original = inputs["xml"].read_bytes()
    source_bytes = [path.read_bytes() for path in inputs["originals"]]
    ref = prepare(inputs)
    result = snapshot.verify_asset_snapshot(ref)
    assert result["verified"] is result["byte_exact_inverse"] is True
    assert result["file_reference_count"] == 3 and result["unique_asset_count"] == 2
    assets = result["assets"]
    assert [item["reference_count"] for item in assets] == [2, 1]
    copies = [Path(item["snapshot_ref"]["path"]) for item in assets]
    assert copies[0].name != copies[1].name
    assert [path.read_bytes() for path in copies] == source_bytes
    assert inputs["xml"].read_bytes() == Path(result["original_xml_snapshot_ref"]["path"]).read_bytes() == original
    assert [path.read_bytes() for path in inputs["originals"]] == source_bytes
    migrated = Path(result["migrated_xml_ref"]["path"]).read_bytes()
    assert b'<option density="1.2" viscosity="2e-05"/>' in migrated
    assert b'<numeric name="file=\'literal.msh\'" data="1"/>' in migrated
    assert original.split(b"<!--", 1)[1].split(b"-->", 1)[0] in migrated
    assert b"\r\n" in migrated
    with pytest.raises(ValueError, match="already exists"):
        prepare(inputs)


@pytest.mark.parametrize(
    "target", ["asset", "manifest", "xml", "original_copy", "source_xml", "source_asset", "history"]
)
def test_hash_tampering_is_rejected(inputs, target):
    ref = prepare(inputs)
    manifest = snapshot.verify_asset_snapshot(ref)
    path = {
        "asset": Path(manifest["assets"][0]["snapshot_ref"]["path"]),
        "manifest": Path(ref["path"]),
        "xml": Path(manifest["migrated_xml_ref"]["path"]),
        "original_copy": Path(manifest["original_xml_snapshot_ref"]["path"]),
        "source_xml": inputs["xml"],
        "source_asset": inputs["originals"][0],
        "history": inputs["history"],
    }[target]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        snapshot.verify_asset_snapshot(ref)


def test_rehashed_nonpath_xml_edit_is_rejected_by_exact_byte_inverse(inputs):
    ref = prepare(inputs)
    manifest_path = Path(ref["path"])
    manifest = json.loads(manifest_path.read_text())
    model = Path(manifest["migrated_xml_ref"]["path"])
    model.write_bytes(model.read_bytes().replace(b'density="1.2"', b'density="1.3"'))
    manifest["migrated_xml_ref"] = reference(model)
    with pytest.raises(ValueError, match="beyond registered file"):
        snapshot.verify_asset_snapshot(save_json(manifest_path, manifest))


def test_rehashed_asset_cannot_override_historical_hash(inputs):
    ref = prepare(inputs)
    manifest_path = Path(ref["path"])
    manifest = json.loads(manifest_path.read_text())
    asset = Path(manifest["assets"][0]["snapshot_ref"]["path"])
    asset.write_bytes(b"different geometry")
    manifest["assets"][0]["snapshot_ref"] = reference(asset)
    with pytest.raises(ValueError, match="historical bytes"):
        snapshot.verify_asset_snapshot(save_json(manifest_path, manifest))


@pytest.mark.parametrize(
    "change", ["missing_asset", "wrong_source", "wrong_count", "boolean_count", "xml_outside", "asset_outside"]
)
def test_rehashed_manifest_cannot_change_closure_or_containment(inputs, change):
    ref = prepare(inputs)
    manifest_path = Path(ref["path"])
    manifest = json.loads(manifest_path.read_text())
    if change == "missing_asset":
        manifest["assets"].pop()
    elif change == "wrong_source":
        manifest["assets"][0]["safe_source_ref"] = reference(inputs["originals"][1])
    elif change == "wrong_count":
        manifest["file_reference_count"] = 4
    elif change == "boolean_count":
        manifest["assets"][1]["reference_count"] = True
    elif change == "xml_outside":
        manifest["migrated_xml_ref"] = reference(inputs["xml"])
    else:
        manifest["assets"][0]["snapshot_ref"] = reference(inputs["originals"][0])
    with pytest.raises(ValueError):
        snapshot.verify_asset_snapshot(save_json(manifest_path, manifest))


@pytest.mark.parametrize("change", ["include", "doctype", "missing_hash", "missing_asset", "changed_asset"])
def test_invalid_sources_are_rejected_before_creating_destination(inputs, change):
    if change == "include":
        inputs["xml"].write_bytes(inputs["xml"].read_bytes().replace(b"<asset>", b'<include file="other.xml"/><asset>'))
    elif change == "doctype":
        inputs["xml"].write_bytes(b'<!DOCTYPE mujoco [<!ENTITY leak "hidden">]><mujoco/>')
    elif change == "missing_hash":
        history = json.loads(inputs["history"].read_text())
        history["assets"].pop(inputs["declared"][0])
        save_json(inputs["history"], history)
    elif change == "missing_asset":
        inputs["originals"][0].unlink()
    else:
        inputs["originals"][0].write_bytes(b"not the recorded mesh")
    with pytest.raises(ValueError):
        prepare(inputs)
    assert not inputs["destination"].exists()


@pytest.mark.parametrize("change", ["outside_repo", "parent_symlink", "existing_directory"])
def test_destination_cannot_escape_or_overwrite(inputs, tmp_path, change):
    if change == "outside_repo":
        inputs["destination"] = tmp_path / "outside"
    elif change == "parent_symlink":
        (inputs["repo"] / "linked").symlink_to(inputs["repo"], target_is_directory=True)
        inputs["destination"] = inputs["repo"] / "linked" / "new"
    else:
        inputs["destination"].mkdir()
        (inputs["destination"] / "keep").write_bytes(b"previous evidence")
    with pytest.raises(ValueError):
        prepare(inputs)
    if change == "existing_directory":
        assert (inputs["destination"] / "keep").read_bytes() == b"previous evidence"


@pytest.mark.parametrize("after_snapshot", [False, True])
def test_safe_source_symlink_escape_is_checked_on_prepare_and_verify(inputs, tmp_path, after_snapshot):
    ref = prepare(inputs) if after_snapshot else None
    outside = tmp_path / "outside.msh"
    source = inputs["originals"][0]
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        snapshot.verify_asset_snapshot(ref) if after_snapshot else prepare(inputs)
