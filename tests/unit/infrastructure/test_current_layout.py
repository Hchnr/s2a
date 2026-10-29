"""Approved current bytes and import identities, with no fixture source execution."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from s2a.infrastructure import current_layout, legacy_script_imports
from s2a.infrastructure.current_layout import CurrentLayout, current_source_path
from s2a.infrastructure.legacy_script_imports import CurrentScriptImports


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return reference(path)


@pytest.fixture
def layout_fixture(tmp_path):
    root = tmp_path / "repository"
    (root / "src/s2a").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    names = ("s2a_fixture_current_alpha", "s2a_fixture_current_beta")
    commit = "1" * 40
    entries, moves = [], []
    for index, name in enumerate(names):
        old_path = f"scripts/{name}.py"
        path = root / old_path
        path.write_text(f"raise RuntimeError('original fixture {index} must never execute')\n")
        original = reference(path)
        entries.append(
            dict(
                old_path=old_path,
                sha256=original["sha256"],
                size_bytes=path.stat().st_size,
                git_mode="100644",
                git_blob_oid="2" * 40,
                baseline_commit=commit,
            )
        )
        if index:
            path = root / "scripts/tracking/fixtures" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"raise RuntimeError('current fixture {index} must never execute')\n")
        moves.append(
            dict(
                old_path=old_path,
                original_sha256=original["sha256"],
                new_path=path.relative_to(root).as_posix(),
                current_sha256=reference(path)["sha256"],
            )
        )
    catalog = dict(schema_version=1, baseline_commit=commit, entries=entries)
    catalog_ref = write_json(root / "docs/migrations/catalog.json", catalog)
    manifest = dict(
        schema_version=1,
        baseline_commit=commit,
        catalog_ref=catalog_ref,
        additional_catalog_refs=[],
        script_namespace="canonical_scripts",
        moves=moves,
    )
    migration_ref = write_json(root / "docs/migrations/current.json", manifest)
    return dict(
        root=root,
        names=names,
        commit=commit,
        catalog=catalog,
        catalog_ref=catalog_ref,
        manifest=manifest,
        migration_ref=migration_ref,
    )


def layout(fixture):
    return CurrentLayout(
        root=fixture["root"],
        migration_ref=fixture["migration_ref"],
        baseline_commit=fixture["commit"],
        catalog_ref=fixture["catalog_ref"],
    )


def guard(fixture, **kwargs):
    return CurrentScriptImports(
        root=fixture["root"],
        migration_ref=fixture["migration_ref"],
        baseline_commit=fixture["commit"],
        catalog_ref=fixture["catalog_ref"],
        allowed_names=fixture["names"],
        **kwargs,
    )


def current_path(fixture, name):
    move = next(row for row in fixture["manifest"]["moves"] if row["old_path"] == f"scripts/{name}.py")
    return fixture["root"] / move["new_path"]


def canonical_name(fixture, name):
    path = current_path(fixture, name).relative_to(fixture["root"])
    return ".".join(path.with_suffix("").parts)


def put_module(monkeypatch, fixture, name, *, runtime_key=None, origin=None):
    expected = canonical_name(fixture, name)
    key = expected if runtime_key is None else runtime_key
    path = current_path(fixture, name) if origin is None else origin
    module = ModuleType(key)
    module.__file__ = str(path)
    module.__package__ = expected.rpartition(".")[0]
    module.__spec__ = importlib.util.spec_from_file_location(expected, path)
    module.untouched_marker = object()
    monkeypatch.setitem(sys.modules, key, module)
    return module


def test_current_query_preserves_distinct_original_bytes_and_one_actual_target(layout_fixture):
    reader = layout(layout_fixture)
    for name in layout_fixture["names"]:
        old = f"scripts/{name}.py"
        entry = reader.entry(old)
        assert entry["original_ref"]["sha256"] != entry["current_ref"]["sha256"]
        assert reader.source_path(old) == current_path(layout_fixture, name)
    assert reader.source_refs == [reference(current_path(layout_fixture, name)) for name in layout_fixture["names"]]
    reader.finish()
    name = layout_fixture["names"][1]
    assert current_source_path(
        f"scripts/{name}.py",
        root=layout_fixture["root"],
        migration_ref=layout_fixture["migration_ref"],
        baseline_commit=layout_fixture["commit"],
        catalog_ref=layout_fixture["catalog_ref"],
    ) == current_path(layout_fixture, name)


@pytest.mark.parametrize("old_path", ["scripts/unknown.py", "../scripts/escape.py", "/scripts/absolute.py"])
def test_current_query_rejects_unknown_or_nonrelative_identity(layout_fixture, old_path):
    with pytest.raises(ValueError):
        layout(layout_fixture).source_path(old_path)


def test_changed_target_does_not_fall_back_to_original_same_bytes_elsewhere(layout_fixture):
    name = layout_fixture["names"][1]
    current_path(layout_fixture, name).write_text("# drift\n")
    assert (layout_fixture["root"] / f"scripts/{name}.py").exists()
    with pytest.raises(ValueError, match="Current physical source SHA drift"):
        layout(layout_fixture).source_path(f"scripts/{name}.py")


@pytest.mark.parametrize("which", ["catalog", "migration"])
@pytest.mark.parametrize("when", ["construction", "finish"])
def test_approved_metadata_drift_is_rejected(layout_fixture, which, when):
    reader = layout(layout_fixture) if when == "finish" else None
    ref = layout_fixture[which + "_ref"]
    Path(ref["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="metadata|identity bytes"):
        reader.finish() if reader is not None else layout(layout_fixture)


@pytest.mark.parametrize(
    "defect", ["original_sha", "unknown_original", "duplicate_old", "duplicate_target", "escape", "absolute"]
)
def test_even_explicitly_referenced_maps_must_preserve_unambiguous_catalog_identities(layout_fixture, defect):
    manifest = copy.deepcopy(layout_fixture["manifest"])
    moves = manifest["moves"]
    if defect == "original_sha":
        moves[0]["original_sha256"] = "0" * 64
    elif defect == "unknown_original":
        moves[0]["old_path"] = "scripts/unregistered.py"
    elif defect == "duplicate_old":
        moves.append(copy.deepcopy(moves[0]))
    elif defect == "duplicate_target":
        moves[1]["new_path"] = moves[0]["new_path"]
    elif defect == "escape":
        moves[0]["new_path"] = "scripts/../outside.py"
    else:
        moves[0]["new_path"] = str(layout_fixture["root"] / moves[0]["new_path"])
    layout_fixture["migration_ref"] = write_json(Path(layout_fixture["migration_ref"]["path"]), manifest)
    with pytest.raises(ValueError):
        layout(layout_fixture)


def test_current_target_symlink_is_rejected(layout_fixture):
    name = layout_fixture["names"][1]
    path = current_path(layout_fixture, name)
    target = path.with_suffix(".original")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        layout(layout_fixture).source_path(f"scripts/{name}.py")


def test_default_cannot_self_approve_an_existing_manifest(layout_fixture, monkeypatch):
    monkeypatch.setattr(current_layout, "CURRENT_MIGRATION_REF", None)
    with pytest.raises(ValueError, match="explicitly approved and pinned"):
        CurrentLayout(root=layout_fixture["root"])
    monkeypatch.setattr(current_layout, "CURRENT_MIGRATION_REF", layout_fixture["migration_ref"])
    reader = CurrentLayout(
        root=layout_fixture["root"], baseline_commit=layout_fixture["commit"], catalog_ref=layout_fixture["catalog_ref"]
    )
    assert reader.migration_ref == layout_fixture["migration_ref"]


def test_current_root_metadata_uses_an_explicit_supplement(layout_fixture):
    path = layout_fixture["root"] / "pyproject.toml"
    original = reference(path)
    entry = dict(
        old_path="pyproject.toml",
        sha256=original["sha256"],
        size_bytes=path.stat().st_size,
        git_mode="100644",
        git_blob_oid="3" * 40,
        baseline_commit=layout_fixture["commit"],
    )
    supplement = write_json(
        layout_fixture["root"] / "docs/migrations/roots.json",
        dict(schema_version=1, baseline_commit=layout_fixture["commit"], entries=[entry]),
    )
    path.write_text(path.read_text() + "\n# current metadata differs from original\n")
    layout_fixture["manifest"]["additional_catalog_refs"] = [supplement]
    layout_fixture["manifest"]["moves"].append(
        dict(
            old_path="pyproject.toml",
            original_sha256=original["sha256"],
            new_path="pyproject.toml",
            current_sha256=reference(path)["sha256"],
        )
    )
    layout_fixture["migration_ref"] = write_json(
        Path(layout_fixture["migration_ref"]["path"]), layout_fixture["manifest"]
    )
    reader = layout(layout_fixture)
    assert reader.source_path("pyproject.toml") == path
    assert reader.additional_catalog_refs == (supplement,)
    assert reader.entry("pyproject.toml")["original_ref"] == dict(path="pyproject.toml", sha256=original["sha256"])


def test_inplace_current_import_is_canonical_and_does_not_require_original_source_sha(layout_fixture, monkeypatch):
    name = layout_fixture["names"][0]
    module = put_module(monkeypatch, layout_fixture, name)
    marker = module.untouched_marker
    reader = guard(layout_fixture)
    access = reader.verify_loaded(name)
    receipt = access.identity_receipt
    assert access.module is module and module.untouched_marker is marker
    assert receipt["namespace"] == "canonical" and receipt["runtime_module_key"] == f"scripts.{name}"
    assert receipt["original_ref"]["sha256"] != receipt["source_ref"]["sha256"]
    assert receipt["source_ref"] == reference(current_path(layout_fixture, name))
    assert receipt["historical_identity_scope"] == "byte_identity_only"
    assert receipt["transitive_dependencies_qualified"] is receipt["main_invoked_by_boundary"] is False
    assert reader.receipts[name] == receipt
    reader.finish()


def test_inplace_current_import_rejects_bare_name(layout_fixture, monkeypatch):
    name = layout_fixture["names"][0]
    put_module(monkeypatch, layout_fixture, name, runtime_key=name)
    with pytest.raises(ValueError, match="Duplicate/obsolete"):
        guard(layout_fixture).verify_present()


@pytest.mark.parametrize("policy", [None, "bare", "mixed_scripts"])
def test_current_guard_requires_the_explicit_canonical_namespace_policy(layout_fixture, policy):
    layout_fixture["manifest"]["script_namespace"] = policy
    layout_fixture["migration_ref"] = write_json(
        Path(layout_fixture["migration_ref"]["path"]), layout_fixture["manifest"]
    )
    with pytest.raises(ValueError, match="canonical scripts namespace required"):
        guard(layout_fixture)


def test_moved_module_must_match_current_canonical_origin(layout_fixture, monkeypatch):
    name = layout_fixture["names"][1]
    module = put_module(monkeypatch, layout_fixture, name)
    receipt = guard(layout_fixture).verify_present([name], require_loaded=True)["verified_modules"][name]
    assert receipt["namespace"] == "canonical" and receipt["module_name"] == module.__name__
    assert receipt["migration_ref"] == layout_fixture["migration_ref"]


@pytest.mark.parametrize("alias_kind", ["bare_only", "qualified_old", "private_origin", "same_object"])
def test_moved_import_rejects_old_or_duplicate_aliases(layout_fixture, monkeypatch, alias_kind):
    name = layout_fixture["names"][1]
    if alias_kind == "bare_only":
        put_module(monkeypatch, layout_fixture, name, runtime_key=name)
    else:
        module = put_module(monkeypatch, layout_fixture, name)
        alias = f"scripts.{name}" if alias_kind == "qualified_old" else f"private_{name}"
        if alias_kind == "same_object":
            monkeypatch.setitem(sys.modules, alias, module)
        else:
            put_module(monkeypatch, layout_fixture, name, runtime_key=alias)
    with pytest.raises(ValueError, match="Duplicate/obsolete"):
        guard(layout_fixture).verify_present()


def test_absent_optional_modules_are_not_imported_or_read(layout_fixture):
    reader = guard(layout_fixture)
    paths, hooks = list(sys.path), list(sys.meta_path)
    for name in layout_fixture["names"]:
        current_path(layout_fixture, name).unlink()
    receipt = reader.verify_present()
    assert receipt["verified_modules"] == {} and receipt["source_refs"] == []
    assert all(canonical_name(layout_fixture, name) not in sys.modules for name in layout_fixture["names"])
    assert sys.path == paths and sys.meta_path == hooks
    with pytest.raises(ValueError, match="not loaded"):
        reader.verify_present(require_loaded=True)


@pytest.mark.parametrize("defect", ["file", "spec_origin", "spec_name", "package", "name"])
def test_current_module_metadata_drift_is_rejected(layout_fixture, monkeypatch, defect):
    name = layout_fixture["names"][1]
    module = put_module(monkeypatch, layout_fixture, name)
    if defect == "file":
        module.__file__ = str(layout_fixture["root"] / f"scripts/{name}.py")
    elif defect == "spec_origin":
        module.__spec__.origin = str(layout_fixture["root"] / f"scripts/{name}.py")
    elif defect == "spec_name":
        module.__spec__.name = name
    elif defect == "package":
        module.__package__ = ""
    else:
        module.__name__ = name
    with pytest.raises(ValueError, match="origin/namespace drift"):
        guard(layout_fixture).verify_loaded(name)


def test_explicit_canonical_main_identity_is_verified_without_running_it(layout_fixture, monkeypatch):
    name = layout_fixture["names"][1]
    expected = canonical_name(layout_fixture, name)
    module = put_module(monkeypatch, layout_fixture, name, runtime_key="__main__")
    access = guard(layout_fixture, approved_main_modules=(expected,)).verify_loaded(name)
    assert access.module is module and access.identity_receipt["runtime_module_key"] == "__main__"
    assert access.identity_receipt["module_name"] == expected
    assert access.identity_receipt["main_invoked_by_boundary"] is False


@pytest.mark.parametrize("defect", ["unapproved", "wrong_spec", "duplicate_same_object", "duplicate_second_object"])
def test_main_cannot_bypass_approval_or_duplicate_identity(layout_fixture, monkeypatch, defect):
    name = layout_fixture["names"][1]
    expected = canonical_name(layout_fixture, name)
    module = put_module(monkeypatch, layout_fixture, name, runtime_key="__main__")
    approved = () if defect == "unapproved" else (expected,)
    if defect == "wrong_spec":
        module.__spec__.name = name
    elif defect == "duplicate_same_object":
        monkeypatch.setitem(sys.modules, expected, module)
    elif defect == "duplicate_second_object":
        put_module(monkeypatch, layout_fixture, name)
    with pytest.raises(ValueError, match="Duplicate"):
        guard(layout_fixture, approved_main_modules=approved).verify_loaded(name)


@pytest.mark.parametrize("defect", ["source", "manifest", "alias", "replacement"])
def test_current_guard_finish_rejects_late_drift(layout_fixture, monkeypatch, defect):
    name = layout_fixture["names"][1]
    module = put_module(monkeypatch, layout_fixture, name)
    reader = guard(layout_fixture)
    reader.verify_loaded(name)
    if defect == "source":
        current_path(layout_fixture, name).write_text("# late drift\n")
    elif defect == "manifest":
        Path(layout_fixture["migration_ref"]["path"]).write_text("{}\n")
    elif defect == "alias":
        monkeypatch.setitem(sys.modules, f"private_{name}", module)
    else:
        put_module(monkeypatch, layout_fixture, name)
    with pytest.raises(ValueError):
        reader.finish()


def test_current_guard_unknown_and_duplicate_requests_are_rejected(layout_fixture):
    reader = guard(layout_fixture)
    with pytest.raises(ValueError, match="Unknown current"):
        reader.verify_present(["unregistered"])
    name = layout_fixture["names"][0]
    with pytest.raises(ValueError, match="Duplicate requested"):
        reader.verify_present([name, name])


def test_legacy_wrapper_delegates_only_to_current_verification(monkeypatch):
    calls = []

    def verify(names=None, **kwargs):
        calls.append((names, kwargs))
        return {"approved_current_receipt": True}

    monkeypatch.setattr(legacy_script_imports, "verify_current_imports", verify)
    result = legacy_script_imports.verify_legacy_modules(["named"], require_loaded=True, root="explicit-root")
    assert result == {"approved_current_receipt": True}
    assert calls == [(["named"], dict(require_loaded=True, root="explicit-root"))]
