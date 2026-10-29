"""Only temporary ordinary-Python helpers execute in legacy boundary tests."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from s2a.infrastructure.legacy_script_imports import LegacyScriptImports


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return reference(path)


@pytest.fixture
def helpers(tmp_path):
    root = tmp_path / "repository"
    (root / "src/s2a").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    names = ("s2a_fixture_legacy_alpha", "s2a_fixture_legacy_beta")
    commit = "1" * 40
    entries = []
    for index, name in enumerate(names):
        path = root / f"scripts/{name}.py"
        path.write_text(
            f"VALUE = {index + 10}\n"
            "def main():\n    raise RuntimeError('main must not execute')\n"
            "if __name__ == '__main__':\n    main()\n"
        )
        entries.append(
            dict(
                old_path=f"scripts/{name}.py",
                sha256=reference(path)["sha256"],
                size_bytes=path.stat().st_size,
                git_mode="100644",
                git_blob_oid="2" * 40,
                baseline_commit=commit,
            )
        )
    catalog = dict(schema_version=1, baseline_commit=commit, entries=entries)
    catalog_path = root / "docs/migrations/catalog.json"
    catalog_ref = write_json(catalog_path, catalog)
    fixture = dict(root=root, names=names, commit=commit, catalog=catalog, catalog_ref=catalog_ref)
    before = set(sys.modules)
    yield fixture
    for key in set(sys.modules) - before:
        if any(name in key for name in names):
            del sys.modules[key]


def boundary(helpers):
    return LegacyScriptImports(
        root=helpers["root"],
        catalog_ref=helpers["catalog_ref"],
        baseline_commit=helpers["commit"],
        allowed_names=helpers["names"],
    )


def load_fixture(helpers, name, *, path=None, module_name=None):
    path = helpers["root"] / f"scripts/{name}.py" if path is None else path
    module_name = name if module_name is None else module_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), vars(module))
    sys.modules[module_name] = module
    return module


def test_package_boot_checks_no_absent_modules_and_changes_no_import_hooks(helpers):
    reader = boundary(helpers)
    paths, hooks = list(sys.path), list(sys.meta_path)
    for name in helpers["names"]:
        (helpers["root"] / f"scripts/{name}.py").unlink()
    receipt = reader.verify_present()
    assert receipt["verified_modules"] == {}
    assert receipt["source_refs"] == []
    assert all(name not in sys.modules for name in helpers["names"])
    assert sys.path == paths
    assert sys.meta_path == hooks


def test_explicit_load_retains_bare_identity_real_origin_and_json_receipts(helpers):
    reader = boundary(helpers)
    name = helpers["names"][0]
    paths, hooks = list(sys.path), list(sys.meta_path)
    access = reader.load(name)
    assert access.module is sys.modules[name]
    assert access.module.__name__ == name
    assert access.module.__file__ == str(helpers["root"] / f"scripts/{name}.py")
    assert access.module.VALUE == 10
    assert access.identity_receipt["original_ref"]["path"] == f"scripts/{name}.py"
    assert access.identity_receipt["source_ref"] == reference(Path(access.module.__file__))
    assert access.identity_receipt["load_kind"] == "loaded_exact_source"
    assert access.identity_receipt["transitive_dependencies_qualified"] is False
    assert access.identity_receipt["main_invoked_by_boundary"] is False
    json.dumps(reader.receipts)
    access.module.VALUE = 99
    assert reader.load(name).module.VALUE == 99
    assert sys.path == paths
    assert sys.meta_path == hooks
    reader.finish()


def test_existing_module_is_verified_without_reexecution(helpers):
    name = helpers["names"][0]
    module = load_fixture(helpers, name)
    module.VALUE = 77
    reader = boundary(helpers)
    access = reader.verify_loaded(name)
    assert access.module is module
    assert access.module.VALUE == 77
    assert access.identity_receipt["load_kind"] == "verified_existing"
    assert reader.load(name).module is module


def test_unknown_name_is_rejected_even_when_a_script_with_that_name_exists(helpers):
    name = "s2a_fixture_legacy_unregistered"
    (helpers["root"] / f"scripts/{name}.py").write_text("raise RuntimeError('must not run')\n")
    reader = boundary(helpers)
    with pytest.raises(ValueError, match="Unknown legacy"):
        reader.load(name)
    with pytest.raises(ValueError, match="Unknown legacy"):
        reader.verify_present([name])
    assert name not in sys.modules


def test_require_loaded_is_explicit_and_does_not_import(helpers):
    name = helpers["names"][0]
    reader = boundary(helpers)
    with pytest.raises(ValueError, match="not loaded"):
        reader.verify_present([name], require_loaded=True)
    assert name not in sys.modules
    reader.load(name)
    receipt = reader.verify_present([name], require_loaded=True)
    assert set(receipt["verified_modules"]) == {name}


def test_loaded_same_basename_from_another_origin_is_rejected(helpers):
    name = helpers["names"][0]
    wrong = helpers["root"] / "other" / f"{name}.py"
    wrong.parent.mkdir()
    wrong.write_bytes((helpers["root"] / f"scripts/{name}.py").read_bytes())
    load_fixture(helpers, name, path=wrong)
    with pytest.raises(ValueError, match="origin/namespace drift"):
        boundary(helpers).verify_loaded(name)


@pytest.mark.parametrize("alias_kind", ["scripts", "private", "same_object"])
def test_qualified_or_duplicate_source_identity_is_rejected(helpers, alias_kind):
    name = helpers["names"][0]
    bare = load_fixture(helpers, name)
    alias = f"scripts.{name}" if alias_kind == "scripts" else f"private_{name}"
    if alias_kind == "same_object":
        sys.modules[alias] = bare
    else:
        load_fixture(helpers, name, module_name=alias)
    with pytest.raises(ValueError, match="duplicate/qualified"):
        boundary(helpers).verify_present()


def test_qualified_only_module_is_not_reused_as_a_bare_module(helpers):
    name = helpers["names"][0]
    load_fixture(helpers, name, module_name=f"scripts.{name}")
    with pytest.raises(ValueError, match="duplicate/qualified"):
        boundary(helpers).load(name)
    assert name not in sys.modules


def test_source_drift_is_rejected_before_any_explicit_load(helpers):
    name, other = helpers["names"]
    reader = boundary(helpers)
    (helpers["root"] / f"scripts/{other}.py").write_text("raise RuntimeError('tampered dependency')\n")
    with pytest.raises(ValueError, match="original SHA/size changed"):
        reader.load(name)
    assert name not in sys.modules
    assert other not in sys.modules


def test_finish_detects_source_and_late_alias_drift(helpers):
    name = helpers["names"][0]
    reader = boundary(helpers)
    access = reader.load(name)
    alias = f"scripts.{name}"
    sys.modules[alias] = access.module
    with pytest.raises(ValueError, match="duplicate/qualified"):
        reader.finish()
    del sys.modules[alias]
    Path(access.module.__file__).write_text("# drift after import\n")
    with pytest.raises(ValueError, match="original SHA/size changed"):
        reader.finish()


def test_finish_detects_module_replacement_even_with_the_correct_origin(helpers):
    name = helpers["names"][0]
    reader = boundary(helpers)
    original = reader.load(name).module
    replacement = load_fixture(helpers, name)
    assert replacement is not original
    with pytest.raises(ValueError, match="object was replaced"):
        reader.finish()


def test_module_spec_cannot_disagree_with_the_bare_name(helpers):
    name = helpers["names"][0]
    module = load_fixture(helpers, name)
    module.__spec__.name = f"scripts.{name}"
    with pytest.raises(ValueError, match="origin/namespace drift"):
        boundary(helpers).verify_loaded(name)


def test_catalog_drift_cannot_repin_original_source_during_a_boundary(helpers):
    reader = boundary(helpers)
    catalog = copy.deepcopy(helpers["catalog"])
    catalog["entries"][0]["sha256"] = "0" * 64
    write_json(Path(helpers["catalog_ref"]["path"]), catalog)
    with pytest.raises(ValueError, match="catalog SHA changed"):
        reader.verify_present()


def test_catalog_must_register_every_allowlisted_original_source(helpers):
    helpers["catalog"]["entries"].pop()
    helpers["catalog_ref"] = write_json(Path(helpers["catalog_ref"]["path"]), helpers["catalog"])
    with pytest.raises(ValueError, match="must be registered"):
        boundary(helpers)


def test_failed_import_removes_only_its_owned_bare_module(helpers):
    name = helpers["names"][0]
    path = helpers["root"] / f"scripts/{name}.py"
    path.write_text("raise RuntimeError('synthetic import failure')\n")
    entry = helpers["catalog"]["entries"][0]
    entry.update(sha256=reference(path)["sha256"], size_bytes=path.stat().st_size)
    helpers["catalog_ref"] = write_json(Path(helpers["catalog_ref"]["path"]), helpers["catalog"])
    other = helpers["names"][1]
    before = load_fixture(helpers, other)
    reader = boundary(helpers)
    with pytest.raises(RuntimeError, match="synthetic import failure"):
        reader.load(name)
    assert name not in sys.modules
    assert sys.modules[other] is before
    assert name not in reader.receipts


def test_ordinary_bare_dependency_is_verified_without_namespace_aliases(helpers, monkeypatch):
    name, dependency = helpers["names"]
    path = helpers["root"] / f"scripts/{name}.py"
    path.write_text(f"import {dependency}\nVALUE = {dependency}.VALUE + 1\n")
    helpers["catalog"]["entries"][0].update(sha256=reference(path)["sha256"], size_bytes=path.stat().st_size)
    helpers["catalog_ref"] = write_json(Path(helpers["catalog_ref"]["path"]), helpers["catalog"])
    monkeypatch.syspath_prepend(str(helpers["root"] / "scripts"))
    reader = boundary(helpers)
    access = reader.load(name)
    assert access.module.VALUE == 12
    assert set(reader.receipts) == {name, dependency}
    assert all(f"scripts.{item}" not in sys.modules for item in helpers["names"])
    reader.finish()
