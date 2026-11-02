"""Historical helper bytes never execute; only approved current origins are used."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from s2a.research.trajectory_search.inputs import SearchInputContext
from scripts.verification.trajectory_search.native_replay_inputs import (
    HELPER_IDENTITY,
    bind_closed_native_owner,
    bind_native_ast_helper,
)


def reference(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return reference(path)


def git(root, *args):
    process = subprocess.Popen(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0"),
    )
    output, error = process.communicate(timeout=30)
    assert process.wait() == 0, error.decode()
    return output


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "src/s2a").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    original = root / HELPER_IDENTITY
    historical = '''"""Synthetic helper: importing the whole module is forbidden."""
from s2a.original_math import VALUE
SENTINEL = "unchanged"
raise RuntimeError("whole historical or current fixture must never execute")
def native_functions(source_ref, names, *, include_eps=False):
    return source_ref, names, include_eps
'''
    original.write_text(historical)
    dependency = root / "src/s2a/original_math.py"
    dependency.write_text("VALUE = 42\n")
    git(root, "init", "--quiet")
    git(root, "add", "pyproject.toml", "src", "scripts")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "base")
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    entries = []
    for path in (original, dependency):
        relative = str(path.relative_to(root))
        mode, _, oid = git(root, "ls-tree", "HEAD", "--", relative).decode().split("\t")[0].split()
        entries.append(
            dict(
                old_path=relative,
                sha256=reference(path)["sha256"],
                size_bytes=path.stat().st_size,
                git_blob_oid=oid,
                git_mode=mode,
                baseline_commit=commit,
            )
        )
    declared = reference(original)
    current = root / "scripts/verification/fixtures/check_tracking_v5_state_updated_response.py"
    current.parent.mkdir(parents=True)
    current.write_text(historical.replace("s2a.original_math", "s2a.foundation.original_math"))
    original.unlink()
    moved_dependency = root / "src/s2a/foundation/original_math.py"
    moved_dependency.parent.mkdir()
    dependency.rename(moved_dependency)
    catalog = write_json(
        root / "docs/migrations/catalog.json",
        dict(
            schema_version=1,
            baseline_commit=commit,
            file_count=len(entries),
            total_size_bytes=sum(entry["size_bytes"] for entry in entries),
            entries=entries,
        ),
    )
    manifest = dict(
        schema_version=1,
        baseline_commit=commit,
        catalog_ref=catalog,
        additional_catalog_refs=[],
        script_namespace="canonical_scripts",
        moves=[
            dict(
                old_path=entry["old_path"],
                original_sha256=entry["sha256"],
                new_path=str(path.relative_to(root)),
                current_sha256=reference(path)["sha256"],
            )
            for entry, path in zip(entries, (current, moved_dependency), strict=True)
        ],
    )
    migration_path = root / "docs/migrations/current.json"
    migration = write_json(migration_path, manifest)

    def context():
        return SearchInputContext(
            root=root,
            baseline_commit=commit,
            catalog_ref=catalog,
            migration_refs=(reference(migration_path),),
            cache_directory=root / "outputs/source_cache",
        )

    module_name = "scripts.verification.fixtures.check_tracking_v5_state_updated_response"
    module = ModuleType(module_name)
    module.__file__ = str(current)
    module.__spec__ = importlib.util.spec_from_file_location(module_name, current)
    # Define only the current synthetic function; neither whole module executes.
    function = next(n for n in ast.parse(current.read_text()).body if isinstance(n, ast.FunctionDef))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(current), "exec"), vars(module))
    monkeypatch.setitem(sys.modules, module_name, module)
    return dict(
        root=root,
        declared=declared,
        historical=historical,
        current=current,
        module=module,
        context=context,
        manifest=manifest,
        migration_path=migration_path,
        migration=migration,
    )


def test_only_registered_namespace_change_preserves_original_git_identity(fixture):
    context = fixture["context"]()
    result = bind_native_ast_helper(context, fixture["declared"], fixture["module"].native_functions)
    assert result["declared_ref"] == fixture["declared"]
    assert result["current_ref"] == reference(fixture["current"])
    assert result["current_ref"]["sha256"] != result["declared_ref"]["sha256"]
    assert result["full_module_namespace_ast_equal"]
    assert result["namespace_changes"] == [dict(original="s2a.original_math", current="s2a.foundation.original_math")]
    assert result["historical_source_executed_here"] is result["historical_source_execution_qualified"] is False
    assert context.registration()["historical_source_identities"][0]["identity_scope"] == "byte_identity_only"


@pytest.mark.parametrize("fault", ["assignment", "extra_import", "function_body", "trailing_statement"])
def test_whole_module_changes_remain_rejected_even_when_current_manifest_binds_them(fixture, fault):
    path = fixture["current"]
    text = path.read_text()
    if fault == "assignment":
        text = text.replace('SENTINEL = "unchanged"', 'SENTINEL = "changed"')
    elif fault == "extra_import":
        text = text.replace(
            "from s2a.foundation.original_math import VALUE",
            "import socket\nfrom s2a.foundation.original_math import VALUE",
        )
    elif fault == "function_body":
        text = text.replace("return source_ref, names, include_eps", "return None")
    else:
        text += '\nSENTINEL = "overridden"\n'
    path.write_text(text)
    fixture["manifest"]["moves"][0]["current_sha256"] = reference(path)["sha256"]
    write_json(fixture["migration_path"], fixture["manifest"])
    with pytest.raises(ValueError, match="whole module changed"):
        bind_native_ast_helper(fixture["context"](), fixture["declared"], fixture["module"].native_functions)


def test_wrong_original_sha_cannot_be_replaced_by_current_identity(fixture):
    wrong = dict(fixture["declared"], sha256=reference(fixture["current"])["sha256"])
    with pytest.raises(ValueError, match="Original AST helper declared path/SHA"):
        bind_native_ast_helper(fixture["context"](), wrong, fixture["module"].native_functions)


@pytest.mark.parametrize("fault", ["module_origin", "function_origin", "alias"])
def test_loaded_function_and_module_require_one_actual_current_origin(fixture, monkeypatch, fault):
    module = fixture["module"]
    if fault == "module_origin":
        module.__file__ = str(fixture["root"] / "scripts/wrong.py")
    elif fault == "function_origin":
        module.native_functions.__module__ = "scripts.wrong"
    else:
        monkeypatch.setitem(sys.modules, "check_tracking_v5_state_updated_response", module)
    with pytest.raises(ValueError, match="origin differs|Duplicate.*AST helper"):
        bind_native_ast_helper(fixture["context"](), fixture["declared"], module.native_functions)


def closed_owner_records(fixture, fault=None):
    """Reseal mutated JSON receipts so rejection exercises each original guard."""
    root = fixture["root"] / "outputs/prerequisite"
    asset_ref = write_json(root / "asset.json", dict(unchanged=True))
    records = dict(
        launch=dict(
            automatic_retry=False,
            selected_environment=dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=""),
            child=dict(pid=12345, start_ticks=67890),
        ),
        exit=dict(automatic_retry=False, actual_wait=True, returncode=0),
        release=dict(reaped=True, returncode=0, signal_sent=False, original_error=None, pid=12345, start_ticks=67890),
        result=dict(complete=True, passed=True),
        registration=dict(
            source_sha256={
                fixture["declared"]["path"]: fixture["declared"]["sha256"],
                asset_ref["path"]: asset_ref["sha256"],
            }
        ),
        owner=dict(complete=True, passed=True),
    )
    if fault is not None:
        document, key, value = fault
        if document == "source":
            records["registration"]["source_sha256"][fixture["declared"]["path"]] = value
        else:
            records[document][key] = value
    references = {
        name: write_json(root / (name + ".json"), value) for name, value in records.items() if name != "owner"
    }
    owner = records["owner"]
    owner.update(
        registration_ref=references["registration"],
        result_ref=references["result"],
        owned_command={name: references[name] for name in ("launch", "exit", "release")},
    )
    return write_json(root / "owner.json", owner), owner, records["result"]


def test_closed_owner_keeps_all_artifact_guards_and_separates_historical_sources(fixture):
    ref, expected_owner, expected_result = closed_owner_records(fixture)
    context = fixture["context"]()
    owner, result = bind_closed_native_owner(context, ref)
    assert owner == expected_owner and result == expected_result
    assert fixture["declared"]["path"] not in context.runtime_refs
    assert not (fixture["root"] / HELPER_IDENTITY).exists()
    identities = context.registration()["historical_source_identities"]
    assert len(identities) == 1 and identities[0]["declared_ref"] == fixture["declared"]
    assert identities[0]["identity_scope"] == "byte_identity_only"
    assert any(path.endswith("outputs/prerequisite/asset.json") for path in context.runtime_refs)


@pytest.mark.parametrize(
    "fault",
    [
        ("owner", "complete", False),
        ("owner", "passed", False),
        ("result", "complete", False),
        ("result", "passed", False),
        ("launch", "automatic_retry", True),
        ("exit", "automatic_retry", True),
        ("launch", "selected_environment", dict(CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH="")),
        ("exit", "actual_wait", False),
        ("exit", "returncode", 1),
        ("release", "reaped", False),
        ("release", "returncode", 1),
        ("release", "signal_sent", True),
        ("release", "original_error", "failure"),
        ("release", "pid", 999),
        ("release", "start_ticks", 999),
        ("source", "sha256", "0" * 64),
    ],
    ids=[
        "owner_complete",
        "owner_passed",
        "result_complete",
        "result_passed",
        "launch_retry",
        "exit_retry",
        "cpu_environment",
        "actual_wait",
        "exit_status",
        "reaped",
        "release_status",
        "signal",
        "error",
        "pid",
        "start_ticks",
        "source_sha",
    ],
)
def test_closed_owner_rejects_resealed_lifecycle_and_source_drift(fixture, fault):
    ref, _, _ = closed_owner_records(fixture, fault)
    with pytest.raises(ValueError):
        bind_closed_native_owner(fixture["context"](), ref)
