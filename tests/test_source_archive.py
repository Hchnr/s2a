"""Historical archives never conceal changed live sources or unrelated absence."""

from pathlib import Path

import pytest

from s2a.artifacts import sha256
from s2a.source_archive import verify_archived_sources


@pytest.fixture
def lineage(tmp_path):
    external = tmp_path / "old_cosmos"
    external.mkdir()
    archive = tmp_path / "snapshots" / "source.py"
    archive.parent.mkdir()
    archive.write_text("# historical bytes\n")
    return external, archive, sha256(archive)


def test_exact_archive_fallback_reports_absence_and_hashes_every_snapshot(lineage, tmp_path):
    external, archive, digest = lineage
    live = tmp_path / "native.py"
    live.write_text("# actual live native source\n")
    additional = archive.parent / "unselected.py"
    additional.write_text("# also verified\n")
    missing = external / "package" / "model.py"
    checked = []

    def actual_hash(path):
        checked.append(Path(path))
        return sha256(path)

    result = verify_archived_sources(
        {str(missing): digest, str(live): sha256(live)},
        {str(archive): digest, str(additional): sha256(additional)},
        missing_prefix=external,
        hash_function=actual_hash,
    )
    assert result == dict(
        verified_live_source_count=1,
        verified_snapshot_count=2,
        missing_sources=[
            dict(
                original_path=str(missing),
                expected_sha256=digest,
                archive_ref=dict(path=str(archive), sha256=digest),
                live_source_present=False,
            )
        ],
        historical_bytes_only=True,
    )
    assert checked == [archive, additional, live]
    assert not missing.exists()  # Verification does not recreate the external source.


def test_changed_live_source_is_rejected_even_when_exact_old_archive_exists(lineage):
    external, archive, digest = lineage
    live = external / "model.py"
    live.write_text("# changed live bytes\n")
    with pytest.raises(ValueError, match="live source SHA256 mismatch"):
        verify_archived_sources({str(live): digest}, {str(archive): digest}, missing_prefix=external)
    assert live.read_text() == "# changed live bytes\n"


@pytest.mark.parametrize("escape", ["sibling", "dotdot", "prefix_itself"])
def test_missing_sources_must_be_strictly_inside_allowed_directory(lineage, escape):
    external, archive, digest = lineage
    if escape == "sibling":
        missing = external.with_name(external.name + "_other") / "model.py"
    elif escape == "dotdot":
        missing = external / ".." / "unrelated.py"
    else:
        external.rmdir()
        missing = external
    with pytest.raises(ValueError, match="missing source outside allowed prefix"):
        verify_archived_sources({str(missing): digest}, {str(archive): digest}, missing_prefix=external)


@pytest.mark.parametrize("bad", ["changed", "missing", "unselected"])
def test_every_declared_archive_must_exist_and_hash_correctly(lineage, bad):
    external, archive, digest = lineage
    missing = external / "model.py"
    snapshots = {str(archive): digest}
    if bad == "changed":
        archive.write_text("# archive corruption\n")
    elif bad == "missing":
        archive.unlink()
    else:
        unused = archive.parent / "unused.py"
        unused.write_text("# unrelated corrupt archive\n")
        snapshots[str(unused)] = digest
    with pytest.raises(ValueError, match="snapshot (SHA256 mismatch|file missing)"):
        verify_archived_sources({str(missing): digest}, snapshots, missing_prefix=external)


def test_archive_with_other_bytes_cannot_stand_in_for_missing_source(lineage):
    external, archive, digest = lineage
    with pytest.raises(ValueError, match="no exact verified snapshot"):
        verify_archived_sources({str(external / "model.py"): "0" * 64}, {str(archive): digest}, missing_prefix=external)


def test_empty_or_duplicate_normalized_snapshot_roster_rejected(lineage):
    external, archive, digest = lineage
    mapping = {str(external / "model.py"): digest}
    with pytest.raises(ValueError, match="snapshot must be a nonempty"):
        verify_archived_sources(mapping, {}, missing_prefix=external)
    alias = archive.parent / "subdir" / ".." / archive.name
    with pytest.raises(ValueError, match="duplicate normalized snapshot"):
        verify_archived_sources(mapping, {str(archive): digest, str(alias): digest}, missing_prefix=external)


def test_broken_live_symlink_is_not_treated_as_missing_historical_source(lineage):
    external, archive, digest = lineage
    broken = external / "model.py"
    broken.symlink_to(external / "absent.py")
    with pytest.raises(ValueError, match="live source is not a readable file"):
        verify_archived_sources({str(broken): digest}, {str(archive): digest}, missing_prefix=external)
