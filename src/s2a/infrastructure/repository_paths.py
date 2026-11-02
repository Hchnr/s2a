"""Locate the S2A worktree without depending on a module's directory depth."""

import os
import stat
import tomllib
from pathlib import Path

FORBIDDEN_HISTORICAL_ROOT = Path("/share/project/eai_pwm/home/hcr/repos/s2a")


def _absolute_without_links(path):
    """Inspect each component before following it; never inspect the old root."""
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden")
    path = Path(os.path.abspath(path))
    if path.is_relative_to(FORBIDDEN_HISTORICAL_ROOT):
        raise ValueError("Historical workspace access is forbidden")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError("Symlink path components are forbidden")
    return path


def _is_repository(path):
    marker = _absolute_without_links(path / "pyproject.toml")
    package = _absolute_without_links(path / "src/s2a")
    if not marker.is_file() or not package.is_dir():
        return False
    with marker.open("rb") as stream:
        project = tomllib.load(stream)
    return project.get("project", {}).get("name") == "s2a"


def repository_root(*, anchor=None):
    """Find the nearest S2A project containing an existing file or directory.

    The default anchor is this module, never the caller's current directory.
    A source installation without a project marker fails explicitly.
    """
    path = _absolute_without_links(Path(__file__) if anchor is None else anchor)
    if not path.exists():
        raise ValueError("Repository anchor does not exist")
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, *directory.parents):
        if _is_repository(candidate):
            return candidate
    raise ValueError("No S2A repository marker contains the anchor")


def repo_path(relative, *, root):
    """Return a safe path under an explicitly validated repository root."""
    root = _absolute_without_links(root)
    if not root.is_dir() or not _is_repository(root):
        raise ValueError("Explicit root is not an S2A repository")
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("A nonempty repository-relative path without traversal is required")
    return _absolute_without_links(root / relative)
