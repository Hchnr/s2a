from pathlib import Path

import pytest
from full_pose_execution_sources_resolved import merge

from s2a.artifacts import REPO


def test_historical_relative_and_absolute_sources_merge_with_declarations_retained():
    relative = "src/s2a/velocity_model_planner.py"
    absolute = str((REPO / relative).resolve())
    sources, aliases = {}, {}
    merge(sources, relative, "a" * 64, aliases)
    merge(sources, absolute, "a" * 64, aliases)
    assert sources == {absolute: "a" * 64}
    assert aliases[relative] == aliases[absolute] == dict(path=absolute, sha256="a" * 64)


def test_relative_alias_cannot_hide_conflicting_historical_sha():
    sources = {}
    relative = "src/s2a/velocity_model_planner.py"
    merge(sources, relative, "a" * 64)
    with pytest.raises(ValueError, match="Conflicting source SHA"):
        merge(sources, REPO / relative, "b" * 64)


def test_relative_provenance_is_repository_anchored_independently_of_cwd(monkeypatch):
    monkeypatch.chdir(REPO / "scripts")
    sources = {}
    merge(sources, "src/s2a/velocity_model_planner.py", "a" * 64)
    assert list(sources) == [str((REPO / "src/s2a/velocity_model_planner.py").resolve())]
    with pytest.raises(ValueError, match="escapes repository"):
        merge(sources, "../outside.py", "a" * 64)
    assert all(Path(name).is_absolute() for name in sources)
