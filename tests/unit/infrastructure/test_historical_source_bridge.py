"""Historical blobs and current sources must both retain their pinned identities."""

import copy
import hashlib

import pytest

from s2a.infrastructure import historical_source_bridge as bridge


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def sources(tmp_path):
    source = tmp_path / "src" / "solver.py"
    source.parent.mkdir()
    source.write_bytes(b"current source")
    stable = tmp_path / "src" / "stable.py"
    stable.write_bytes(b"unchanged")
    helper = tmp_path / "bridge.py"
    helper.write_bytes(b"bridge implementation")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"pinned manifest")
    historical = b"historical source"
    row = dict(
        path="src/solver.py",
        historical_sha256=digest(historical),
        historical_commit="a" * 40,
        current_sha256=digest(source.read_bytes()),
    )
    manifest = dict(schema=bridge.SCHEMA, bridges=[row])
    recorded = [dict(path=str(source), sha256=digest(historical)), bridge.reference(stable)]
    current = [
        bridge.reference(source),
        bridge.reference(stable),
        bridge.reference(helper),
        bridge.reference(manifest_path),
    ]

    def read_blob(root, revision, path):
        assert root == tmp_path and revision == row["historical_commit"] and path == row["path"]
        return historical

    return dict(
        recorded=recorded,
        current=current,
        manifest=manifest,
        root=tmp_path,
        extensions=(helper, manifest_path),
        read_blob=read_blob,
        require_extensions=True,
    )


def test_pinned_historical_blob_and_current_sources_are_both_checked(sources):
    proof = bridge.compare_sources(**sources)
    assert proof["matched_sources"] == 2
    assert proof["bridges"] == sources["manifest"]["bridges"]
    assert len(proof["extensions"]) == 2
    # Later registrations may already contain the bridge files.
    sources["recorded"] += sources["current"][-2:]
    assert bridge.compare_sources(**sources)["matched_sources"] == 4


def test_nested_source_lists_can_repeat_complete_extension_groups(sources):
    sources["current"] += sources["current"][-2:]
    assert len(bridge.compare_sources(**sources)["extensions"]) == 4
    sources["current"].pop()
    with pytest.raises(ValueError, match="complete matching groups"):
        bridge.compare_sources(**sources)


@pytest.mark.parametrize("failure", ["unregistered", "old_hash", "current_hash", "blob", "order", "missing_extension"])
def test_unknown_or_changed_source_identity_is_rejected(sources, failure):
    data = copy.deepcopy(sources)
    if failure == "unregistered":
        data["manifest"]["bridges"] = []
    elif failure == "old_hash":
        data["recorded"][0]["sha256"] = "1" * 64
    elif failure == "current_hash":
        data["manifest"]["bridges"][0]["current_sha256"] = "2" * 64
    elif failure == "blob":
        data["read_blob"] = lambda *_args: b"different historical content"
    elif failure == "order":
        data["current"][:2] = data["current"][1::-1]
    elif failure == "missing_extension":
        data["current"] = data["current"][:-1]
    with pytest.raises(ValueError):
        bridge.compare_sources(**data)


def test_forged_current_reference_is_not_treated_as_unchanged(sources):
    sources["recorded"][1]["sha256"] = sources["current"][1]["sha256"] = "3" * 64
    with pytest.raises(ValueError, match="current source bytes"):
        bridge.compare_sources(**sources)


def test_missing_or_tampered_manifest_is_rejected(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(bridge, "MANIFEST", manifest)
    with pytest.raises(ValueError, match="unavailable"):
        bridge.load_manifest()
    manifest.write_text('{"schema": "changed", "bridges": []}')
    with pytest.raises(ValueError, match="manifest changed"):
        bridge.load_manifest()
