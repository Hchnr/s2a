"""Repository discovery remains correct after moving modules into subpackages."""

from pathlib import Path

import pytest

from s2a.infrastructure.repository_paths import (
    FORBIDDEN_HISTORICAL_ROOT,
    repo_path,
    repository_root,
)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "src/s2a/infrastructure/nested").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "s2a"\n')
    source = root / "src/s2a/infrastructure/nested/module.py"
    source.write_text("# existing deeply nested source\n")
    return root, source


def test_discovery_uses_anchor_depth_and_nearest_matching_project(project):
    root, source = project
    assert repository_root(anchor=source) == root
    assert repository_root(anchor=source.parent) == root
    assert repository_root(anchor=root) == root


def test_default_anchor_is_independent_of_current_directory(project, monkeypatch):
    expected = repository_root()
    monkeypatch.chdir(project[0])
    assert repository_root() == expected
    assert repository_root(anchor=Path.cwd()) == project[0]


def test_explicit_root_must_itself_be_a_project(project):
    root, source = project
    with pytest.raises(ValueError, match="Explicit root"):
        repo_path("file.py", root=source.parent)
    assert repo_path("docs/new/topic.md", root=root) == root / "docs/new/topic.md"


@pytest.mark.parametrize("relative", ["", ".", "../escape", "docs/../escape", "/absolute"])
def test_repository_paths_reject_empty_absolute_and_traversal(project, relative):
    with pytest.raises(ValueError, match="repository-relative"):
        repo_path(relative, root=project[0])


def test_missing_anchor_is_not_replaced_with_current_directory(project):
    with pytest.raises(ValueError, match="anchor does not exist"):
        repository_root(anchor=project[0] / "missing")


def test_symlink_components_are_rejected_before_following_their_target(project):
    root, source = project
    link = root / "alias"
    link.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        repo_path("alias/module.py", root=root)
    with pytest.raises(ValueError, match="Symlink"):
        repository_root(anchor=link / "module.py")
    link.unlink()
    link.symlink_to(FORBIDDEN_HISTORICAL_ROOT / "forbidden", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        repo_path("alias/file.py", root=root)


def test_forbidden_old_workspace_is_rejected_lexically():
    with pytest.raises(ValueError, match="Historical workspace"):
        repository_root(anchor=FORBIDDEN_HISTORICAL_ROOT / "not-opened.py")
