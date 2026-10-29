"""Recovery changes publication and provenance, never controller or model math."""

import ast
import hashlib
import json

from s2a.infrastructure.artifacts import REPO
from tests.support.source_layout import current_module_name, current_source_path, historical_source_bytes


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
    # Fixture identities retain the registered baseline byte proofs. Compare
    # current source separately, using only the declared module relocation.
    import_relocations = {
        "s2a." + name: current_module_name("src/s2a/" + name + ".py")
        for name in (
            "native_environment",
            "preview_result_publication",
            "terminal_development_policy",
            "serialized_preview_recovery",
            "clean_training_runtime",
        )
    }

    def current_imports(fragment):
        for old, new in import_relocations.items():
            fragment = fragment.replace("from " + old + " import", "from " + new + " import")
        return fragment

    for row in manifest["files"]:
        historical = historical_source_bytes(dict(path=row["original"], sha256=row["original_sha256"]))
        assert hashlib.sha256(historical).hexdigest() == row["original_sha256"]
        source = current_source_path(row["original"])
        expected = source.read_text()
        for old, new in sorted(manifest["changes"].items(), key=lambda item: -len(item[0])):
            expected = expected.replace(old, new)
        if source.name == "collect_coupled_preview_wam_development.py":
            expected = expected.replace(
                current_imports(
                    "from s2a.native_environment import restore_cv2_environment, strict_native_environment"
                ),
                current_imports(
                    "from s2a.native_environment import restore_cv2_environment, strict_native_environment\n"
                    "from s2a.preview_result_publication import publish_result"
                ),
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
            old, new = current_imports(old), current_imports(new)
            assert expected.count(old) == 1
            expected = expected.replace(old, new)
        actual = current_source_path(row["derived"]).read_text()
        assert canonical_ast(actual) == canonical_ast(expected), row["derived"]
