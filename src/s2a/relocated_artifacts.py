"""Read relocated artifact bytes while preserving historical reference paths.

This is a file locator and byte-identity check. It does not restore files, change
historical JSON, certify that a historical source still exists, or locate an
external Cosmos checkout. Callers must handle separately registered source
snapshots through their source archive.
"""

import hashlib
import json
from pathlib import Path


def _path_without_parent_steps(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed in artifact paths")
    return path


class RelocatedArtifacts:
    """Locate existing files without ever falling back to a historical root.

    Relative references are rooted at ``root``. Declarations retain their
    absolute lexical path, including symlink names; only I/O follows symlinks.
    The class performs no writes. Both relocation roots are explicit and no
    external dependency path is inferred.
    """

    def __init__(
        self,
        root,
        recovery_root,
        *,
        historical_root="/share/project/eai_pwm/home/hcr/repos/s2a",
    ):
        self.root = _path_without_parent_steps(root).absolute()
        recovery = _path_without_parent_steps(recovery_root)
        self.recovery_root = recovery if recovery.is_absolute() else self.root / recovery
        self.historical_root = _path_without_parent_steps(historical_root)
        if not self.historical_root.is_absolute():
            raise ValueError("historical_root must be absolute")
        if self.root == self.historical_root or self.recovery_root.is_relative_to(self.historical_root):
            raise ValueError("relocation destinations must not use the historical root")

    def _declared_path(self, path):
        path = _path_without_parent_steps(path)
        return path if path.is_absolute() else self.root / path

    def resolve_path(self, path):
        """Return an existing regular file at its new location or raise.

        Historical ``outputs/wam_stability_20261021/<dir>/...`` first searches
        ``recovery_root/<dir>/...``, then the corresponding path under ``root``.
        Other historical files search only their corresponding new-root path.
        A missing file never triggers a read of its historical location.
        """
        declared = self._declared_path(path)
        if declared.is_relative_to(self.historical_root):
            relative = declared.relative_to(self.historical_root)
            candidates = []
            if relative.parts[:2] == ("outputs", "wam_stability_20261021") and len(relative.parts) > 2:
                candidates.append(self.recovery_root.joinpath(*relative.parts[2:]))
            candidates.append(self.root / relative)
        else:
            candidates = [declared]
        for candidate in candidates:
            actual = candidate.resolve(strict=False)
            if actual.is_relative_to(self.historical_root):
                raise FileNotFoundError(f"artifact destination resolves into the historical root: {declared}")
            if actual.is_file():
                return actual
        raise FileNotFoundError(f"artifact has no existing relocated regular file: {declared}")

    def reference(self, path):
        """Bind the unchanged declaration to the actual relocated file's SHA256."""
        declared = self._declared_path(path)
        actual = self.resolve_path(declared)
        return dict(path=str(declared), sha256=hashlib.sha256(actual.read_bytes()).hexdigest())

    def read(self, ref):
        """Read JSON only when the declaration and actual SHA match exactly.

        The consumed bytes are hashed again after reference validation, so a
        concurrent replacement/disappearance does not silently use new bytes.
        """
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("artifact reference must contain exactly path and sha256")
        if self.reference(ref["path"]) != ref:
            raise ValueError("artifact reference declaration or SHA256 does not match")
        data = self.resolve_path(ref["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("artifact SHA256 changed before JSON consumption")
        return json.loads(data)
