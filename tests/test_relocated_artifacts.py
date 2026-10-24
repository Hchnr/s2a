"""Real fixture I/O verifies relocation without modifying historical evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from s2a.relocated_artifacts import RelocatedArtifacts


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_historical_sources_and_other_outputs_map_without_access_or_writes(tmp_path, monkeypatch):
    old, root, recovery = (tmp_path / name for name in ("deleted", "mixed", "recovered"))
    paths = [Path("src/s2a/worker.py"), Path("outputs/qualification/result.json")]
    for relative in paths:
        write_json(old / relative, {"old": "must not be consumed"})
        write_json(root / relative, {"relocated": str(relative)})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert not path.is_relative_to(old), "historical source must never be read"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    locator = RelocatedArtifacts(root, recovery, historical_root=old)
    for relative in paths:
        declared = old / relative
        assert locator.resolve_path(declared) == root / relative
        ref = locator.reference(declared)
        assert ref == dict(path=str(declared), sha256=hashlib.sha256(before[root / relative]).hexdigest())
        assert locator.read(ref) == {"relocated": str(relative)}
    # Neither old files nor new files are rewritten by resolving or reading.
    assert {p: read_bytes(p) for p in tmp_path.rglob("*") if p.is_file()} == before


def test_wam_backup_has_priority_and_missing_backup_falls_back_to_new_root(tmp_path):
    old, root, recovery = (tmp_path / name for name in ("deleted", "mixed", "recovered"))
    suffix = Path("free_trajectory_diagnostic_source_v2/review/verified.json")
    declared = old / "outputs/wam_stability_20261021" / suffix
    ordinary = write_json(root / "outputs/wam_stability_20261021" / suffix, {"source": "mixed"})
    backup = write_json(recovery / suffix, {"source": "safe backup"})
    locator = RelocatedArtifacts(root, recovery, historical_root=old)
    assert locator.resolve_path(declared) == backup
    assert locator.read(locator.reference(declared)) == {"source": "safe backup"}
    backup.unlink()
    assert locator.resolve_path(declared) == ordinary
    assert locator.read(locator.reference(declared)) == {"source": "mixed"}


def test_current_relative_and_symlink_paths_preserve_declared_reference(tmp_path):
    root = tmp_path / "mixed"
    actual = write_json(root / "outputs/data.json", {"ok": True})
    link = root / "named.json"
    link.symlink_to(actual)
    locator = RelocatedArtifacts(root, "outputs/recovery", historical_root=tmp_path / "deleted")
    assert locator.resolve_path("outputs/data.json") == actual
    assert locator.resolve_path(link) == actual
    ref = locator.reference("named.json")
    assert ref["path"] == str(link)
    assert locator.read(ref) == {"ok": True}
    assert ref["path"] != str(actual)


def test_missing_directory_traversal_and_symlink_back_to_old_are_rejected(tmp_path):
    old, root, recovery = (tmp_path / name for name in ("deleted", "mixed", "recovered"))
    old_file = write_json(old / "outputs/lost.json", {"still exists": "must not fall back"})
    root.mkdir()
    locator = RelocatedArtifacts(root, recovery, historical_root=old)
    with pytest.raises(FileNotFoundError, match="no existing relocated"):
        locator.resolve_path(old_file)
    with pytest.raises(FileNotFoundError, match="regular file"):
        locator.resolve_path(root)
    with pytest.raises(FileNotFoundError):
        locator.reference(root / "missing.json")
    for path in ("../outside.json", old / "outputs/../../outside.json", root / "../outside.json"):
        with pytest.raises(ValueError, match="parent traversal"):
            locator.resolve_path(path)
    link = root / "forbidden.json"
    link.symlink_to(old_file)
    with pytest.raises(FileNotFoundError, match="historical root"):
        locator.resolve_path(link)


def test_json_read_rejects_wrong_digest_extra_fields_and_changed_or_gone_bytes(tmp_path, monkeypatch):
    root = tmp_path / "mixed"
    path = write_json(root / "result.json", {"version": 1})
    locator = RelocatedArtifacts(root, root / "outputs/recovery", historical_root=tmp_path / "deleted")
    ref = locator.reference(path)
    with pytest.raises(ValueError, match="SHA256"):
        locator.read(dict(ref, sha256="0" * 64))
    with pytest.raises(ValueError, match="exactly"):
        locator.read(dict(ref, extra=True))
    original_reference = locator.reference

    def change_after_reference(declared):
        checked = original_reference(declared)
        write_json(path, {"version": 2})
        return checked

    monkeypatch.setattr(locator, "reference", change_after_reference)
    with pytest.raises(ValueError, match="changed before JSON consumption"):
        locator.read(ref)
    monkeypatch.setattr(locator, "reference", original_reference)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        locator.read(ref)
