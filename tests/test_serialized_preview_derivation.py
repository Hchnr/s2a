"""Recovery changes publication and provenance, never controller or model math."""

import ast
import json

from s2a.artifacts import REPO, sha256


def canonical_ast(source):
    tree = ast.parse(source)
    # Ruff may sort contiguous import groups after module renaming. Do not
    # move imports across any executable statement, including guard preload.
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if not isinstance(value, list):
                continue
            start = 0
            while start < len(value):
                if not isinstance(value[start], (ast.Import, ast.ImportFrom)):
                    start += 1
                    continue
                end = start + 1
                while end < len(value) and isinstance(value[end], (ast.Import, ast.ImportFrom)):
                    end += 1
                value[start:end] = sorted(value[start:end], key=ast.dump)
                start = end
    return ast.dump(tree)


def test_registered_derivation_has_only_declared_changes():
    manifest = json.loads((REPO / "tests/fixtures/serialized_preview_derivation.json").read_text())
    for row in manifest["files"]:
        source = REPO / row["original"]
        assert sha256(source) == row["original_sha256"]
        expected = source.read_text()
        for old, new in sorted(manifest["changes"].items(), key=lambda item: -len(item[0])):
            expected = expected.replace(old, new)
        if source.name == "collect_coupled_preview_wam_development.py":
            expected = expected.replace(
                "from s2a.native_environment import restore_cv2_environment, strict_native_environment",
                "from s2a.native_environment import restore_cv2_environment, strict_native_environment\n"
                "from s2a.preview_result_publication import publish_result",
            ).replace(
                'publish(root / "worker_result.json", result)', 'publish_result(root / "worker_result.json", result)'
            )
        if source.name in (
            "prepare_coupled_preview_wam_development.py",
            "verify_coupled_preview_wam_metadata.py",
        ):
            expected = expected.replace(
                '"tests/test_coupled_preview_control_gate.py"',
                '"tests/test_coupled_preview_control_gate.py", "tests/test_preview_result_publication.py", '
                '"tests/test_serialized_preview_derivation.py"',
            ).replace(
                '"test_coupled_preview_wam_*.py"',
                '"test_serialized_preview_wam_*.py", "test_coupled_preview_wam_protocol.py"',
            )
        for old, new in manifest.get("recovery_binding_changes", {}).get(row["derived"], []):
            assert expected.count(old) == 1
            expected = expected.replace(old, new)
        actual = (REPO / row["derived"]).read_text()
        assert canonical_ast(actual) == canonical_ast(expected), row["derived"]
