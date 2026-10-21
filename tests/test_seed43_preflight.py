"""Prerequisite hashes must survive later actual-source collection unchanged."""

import importlib
import sys
from pathlib import Path

import pytest


def test_qualification_hash_cannot_be_replaced_by_current_execution_source():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        merge = importlib.import_module("prepare_wam_raw8_seed43").merge
        inputs = {"qualified_runtime.py": "prior"}
        with pytest.raises(ValueError, match="immutable input conflict"):
            merge(inputs, {"new.py": "new", "qualified_runtime.py": "changed"})
        assert inputs == {"qualified_runtime.py": "prior"}
        merge(inputs, {"qualified_runtime.py": "prior", "new.py": "new"})
        assert inputs == {"qualified_runtime.py": "prior", "new.py": "new"}
    finally:
        sys.path.remove(scripts)


def test_dry_run_empty_root_is_distinct_from_any_existing_experiment(tmp_path):
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, scripts)
    try:
        unused_root = importlib.import_module("prepare_wam_raw8_seed43").unused_root
        root = tmp_path / "experiment"
        assert unused_root(root)
        root.mkdir()
        assert unused_root(root)
        (root / "partial_case").mkdir()
        assert not unused_root(root)
        file = tmp_path / "file"
        file.write_text("old evidence")
        assert not unused_root(file)
    finally:
        sys.path.remove(scripts)
