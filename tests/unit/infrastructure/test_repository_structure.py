"""Migration acceptance must detect lost files, stale aliases and missing tests."""

import hashlib
import json

import pytest

from scripts.maintenance.check_repository_structure import AREAS, CATALOG, check, compare_nodes


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    for area in (*AREAS, "docs/migrations"):
        (tmp_path / area).mkdir(parents=True, exist_ok=True)
    old = "src/s2a/example.py"
    (tmp_path / old).write_bytes(b"original source\n")
    catalog = dict(
        file_count=1,
        entries=[dict(old_path=old, sha256=sha(b"original source\n"))],
    )
    (tmp_path / CATALOG).write_text(json.dumps(catalog))
    return tmp_path


def move(repository, *, current=b"original source\n", original_remains=False):
    old, new = "src/s2a/example.py", "src/s2a/research/example.py"
    target = repository / new
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(current)
    if not original_remains:
        (repository / old).unlink()
    manifest = dict(
        catalog_ref=dict(path=CATALOG, sha256=sha((repository / CATALOG).read_bytes())),
        moves=[
            dict(old_path=old, original_sha256=sha(b"original source\n"), new_path=new, current_sha256=sha(current))
        ],
    )
    relative = "docs/migrations/example.json"
    (repository / relative).write_text(json.dumps(manifest))
    return relative


def test_unchanged_baseline_is_checked_before_any_move(repository):
    result = check(repository, [])
    assert result["passed"] and result["baseline_file_count"] == result["current_byte_identical_count"] == 1


def test_new_current_bytes_are_recorded_separately_from_original_identity(repository):
    result = check(repository, [move(repository, current=b"new import path\n")])
    assert result["passed"] and result["migrated_file_count"] == 1 and result["current_byte_identical_count"] == 0
    assert result["current_changed_sources"][0]["current_path"] == "src/s2a/research/example.py"


def test_a_required_importer_edit_is_registered_without_inventing_a_physical_move(repository):
    old = "src/s2a/example.py"
    current = b"updated canonical import\n"
    (repository / old).write_bytes(current)
    manifest = dict(
        catalog_ref=dict(path=CATALOG, sha256=sha((repository / CATALOG).read_bytes())),
        moves=[
            dict(old_path=old, original_sha256=sha(b"original source\n"), new_path=old, current_sha256=sha(current))
        ],
    )
    relative = "docs/migrations/importers.json"
    (repository / relative).write_text(json.dumps(manifest))
    result = check(repository, [relative])
    assert result["passed"] and result["migrated_file_count"] == 0 and result["registered_in_place_edit_count"] == 1


@pytest.mark.parametrize("defect", ["lost", "changed", "old_alias"])
def test_lost_bytes_source_drift_and_old_flat_file_are_detected(repository, defect):
    manifest = move(repository, original_remains=defect == "old_alias")
    target = repository / "src/s2a/research/example.py"
    if defect == "lost":
        target.unlink()
    elif defect == "changed":
        target.write_bytes(b"unregistered drift")
    result = check(repository, [manifest])
    assert not result["passed"]
    expected = dict(lost="missing_file", changed="current_sha_mismatch", old_alias="old_flat_file_remains")
    assert expected[defect] in {row["error"] for row in result["errors"]}


def test_path_mapping_preserves_parameter_ids_and_distinguishes_added_tests():
    old = "tests/test_path.py"
    moves = {old: dict(new_path="tests/unit/test_path.py")}
    result = compare_nodes(
        [old + "::test_identity[tests/test_path.py]"],
        ["tests/unit/test_path.py::test_identity[tests/test_path.py]", "tests/new.py::test_new"],
        moves,
    )
    assert result["preserved"] and result["added_nodes"] == ["tests/new.py::test_new"]


def test_equal_test_counts_do_not_hide_a_missing_original_node():
    result = compare_nodes(["tests/a.py::test_a"], ["tests/a.py::test_b"], {})
    assert not result["preserved"] and result["missing_nodes"] == ["tests/a.py::test_a"]


def test_duplicate_discovery_cannot_masquerade_as_more_coverage():
    with pytest.raises(ValueError, match="Duplicate"):
        compare_nodes(["tests/a.py::test_a"], ["tests/a.py::test_a"] * 2, {})


def test_batch_success_cannot_be_reported_as_full_restructure(repository):
    for index in range(13):
        (repository / f"src/s2a/unmigrated{index}.py").write_text("# pending\n")
    assert check(repository, [])["passed"]
    result = check(repository, [], require_complete=True)
    assert not result["passed"] and not result["full_restructure_complete"]


def test_a_migration_from_another_catalog_is_rejected(repository):
    manifest = move(repository)
    payload = json.loads((repository / manifest).read_text())
    payload["catalog_ref"]["sha256"] = "0" * 64
    (repository / manifest).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different baseline"):
        check(repository, [manifest])


def test_root_metadata_edits_have_an_explicit_supplement(repository):
    catalog = json.loads((repository / CATALOG).read_text())
    catalog["baseline_commit"] = "explicit-baseline"
    (repository / CATALOG).write_text(json.dumps(catalog))
    original = (repository / "pyproject.toml").read_bytes()
    current = original + b"# package discovery updated\n"
    (repository / "pyproject.toml").write_bytes(current)
    supplement = dict(
        baseline_commit="explicit-baseline",
        file_count=1,
        entries=[dict(old_path="pyproject.toml", sha256=sha(original))],
    )
    supplementary_path = "docs/migrations/root.json"
    (repository / supplementary_path).write_text(json.dumps(supplement))
    manifest = dict(
        catalog_ref=dict(path=CATALOG, sha256=sha((repository / CATALOG).read_bytes())),
        moves=[
            dict(
                old_path="pyproject.toml",
                original_sha256=sha(original),
                new_path="pyproject.toml",
                current_sha256=sha(current),
            )
        ],
    )
    relative = "docs/migrations/root-edit.json"
    (repository / relative).write_text(json.dumps(manifest))
    result = check(repository, [relative], additional_catalog_paths=[supplementary_path])
    assert result["passed"] and result["baseline_file_count"] == result["additional_baseline_file_count"] == 1


def test_an_inventory_cannot_extend_access_to_an_unregistered_root_file(repository):
    catalog = json.loads((repository / CATALOG).read_text())
    catalog["entries"][0]["old_path"] = "private_local.py"
    (repository / CATALOG).write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="explicitly managed"):
        check(repository, [])
