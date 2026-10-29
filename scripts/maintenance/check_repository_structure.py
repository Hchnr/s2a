"""Verify file migration identities and mapped test discovery without executing code."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from s2a.infrastructure.repository_paths import repo_path, repository_root

AREAS = ("src/s2a", "scripts", "tests", "configs", "plans/v0.0.1", "reports")
FINAL_FLAT_LIMITS = dict(zip(AREAS, (12, 2, 2, 1, 1, 1), strict=True))
CATALOG = "docs/migrations/baseline_20261029.json"
ROOT_METADATA = {".gitignore", "AGENTS.md", "README.md", "pyproject.toml"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    with path.open() as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _moves(root, catalog, manifest_paths):
    reference = dict(path=CATALOG, sha256=digest(repo_path(CATALOG, root=root)))
    entries = {item["old_path"]: item for item in catalog["entries"]}
    moves, destinations, references = {}, set(), []
    for relative in manifest_paths:
        path = repo_path(relative, root=root)
        manifest = read(path)
        require(manifest["catalog_ref"] == reference, "Migration binds a different baseline catalog")
        references.append(dict(path=relative, sha256=digest(path)))
        complete_registration = manifest.get("complete_current_layout") is True
        if complete_registration:
            require(
                {item["old_path"] for item in manifest["moves"]} == set(entries),
                "Complete current layout must register every baseline identity",
            )
        for item in manifest["moves"]:
            require(
                set(item) == {"old_path", "original_sha256", "new_path", "current_sha256"},
                "Exact four-field move schema required",
            )
            old, new = item["old_path"], item["new_path"]
            repo_path(old, root=root)
            repo_path(new, root=root)
            require(old in entries and item["original_sha256"] == entries[old]["sha256"], "Unknown original identity")
            require(old not in moves and new not in destinations, "Ambiguous migration entry")
            require(
                complete_registration or old != new or item["current_sha256"] != item["original_sha256"],
                "Unchanged entries require a complete current layout registration",
            )
            require(re.fullmatch("[0-9a-f]{64}", item["current_sha256"]) is not None, "Current SHA256 required")
            moves[old] = item
            destinations.add(new)
    return moves, references


def compare_nodes(baseline, current, moves):
    """Map file portions only; test names and parameter IDs retain their identity."""
    require(
        isinstance(baseline, list)
        and isinstance(current, list)
        and all(isinstance(node, str) and "::" in node for node in (*baseline, *current)),
        "Explicit pytest node lists required",
    )
    require(len(set(baseline)) == len(baseline) and len(set(current)) == len(current), "Duplicate test node identity")
    expected = set()
    for node in baseline:
        old, suffix = node.split("::", 1)
        expected.add(moves.get(old, {}).get("new_path", old) + "::" + suffix)
    missing = sorted(expected - set(current))
    return dict(
        baseline_count=len(baseline),
        current_count=len(current),
        mapped_baseline_count=len(expected),
        missing_nodes=missing,
        added_nodes=sorted(set(current) - expected),
        preserved=not missing,
    )


def check(
    root,
    manifest_paths,
    *,
    additional_catalog_paths=(),
    baseline_nodes=None,
    current_nodes=None,
    require_complete=False,
):
    root = repository_root(anchor=root)
    catalog_path = repo_path(CATALOG, root=root)
    catalog = read(catalog_path)
    require(catalog["file_count"] == len(catalog["entries"]), "Incomplete baseline inventory")
    require(len({row["old_path"] for row in catalog["entries"]}) == catalog["file_count"], "Duplicate original path")
    primary_count = catalog["file_count"]
    additional_refs = []
    entries = list(catalog["entries"])
    for relative in additional_catalog_paths:
        path = repo_path(relative, root=root)
        supplement = read(path)
        require(
            supplement["baseline_commit"] == catalog["baseline_commit"]
            and supplement["file_count"] == len(supplement["entries"]),
            "Supplement must register the same baseline commit",
        )
        additional_refs.append(dict(path=relative, sha256=digest(path)))
        entries.extend(supplement["entries"])
    require(len({row["old_path"] for row in entries}) == len(entries), "Ambiguous baseline catalogs")
    for row in entries:
        relative = row["old_path"]
        require(
            relative in ROOT_METADATA or any(relative.startswith(area + "/") for area in AREAS),
            "Baseline entry is outside explicitly managed files",
        )
    catalog = dict(catalog, entries=entries)
    moves, manifest_refs = _moves(root, catalog, manifest_paths)
    errors, identity_unchanged, changed_sources = [], 0, []
    for row in catalog["entries"]:
        move = moves.get(row["old_path"])
        relative = row["old_path"] if move is None else move["new_path"]
        expected = row["sha256"] if move is None else move["current_sha256"]
        path = repo_path(relative, root=root)
        if not path.is_file():
            errors.append(dict(old_path=row["old_path"], current_path=relative, error="missing_file"))
            continue
        actual = digest(path)
        if actual != expected:
            errors.append(dict(old_path=row["old_path"], current_path=relative, error="current_sha_mismatch"))
        elif actual == row["sha256"]:
            identity_unchanged += 1
        else:
            changed_sources.append(dict(old_path=row["old_path"], current_path=relative, current_sha256=actual))
        if move is not None and move["old_path"] != move["new_path"] and repo_path(row["old_path"], root=root).exists():
            errors.append(dict(old_path=row["old_path"], error="old_flat_file_remains"))
    flat = {relative: sum(path.is_file() for path in repo_path(relative, root=root).iterdir()) for relative in AREAS}
    expected_depth = all(flat[relative] <= limit for relative, limit in FINAL_FLAT_LIMITS.items())
    nodes = None
    require((baseline_nodes is None) == (current_nodes is None), "Both test discovery lists must be supplied")
    if baseline_nodes is not None:
        nodes = compare_nodes(read(baseline_nodes), read(current_nodes), moves)
        if not nodes["preserved"]:
            errors.append(dict(error="test_nodes_missing", count=len(nodes["missing_nodes"])))
    complete = expected_depth and not errors
    if require_complete and not complete:
        errors.append(dict(error="repository_still_has_unmigrated_flat_areas"))
    return dict(
        schema_version=1,
        passed=not errors,
        scope="Structural storage identity and mapped discovery; no runtime or control qualification",
        catalog_ref=dict(path=CATALOG, sha256=digest(catalog_path)),
        additional_catalog_refs=additional_refs,
        migration_refs=manifest_refs,
        baseline_file_count=primary_count,
        additional_baseline_file_count=len(entries) - primary_count,
        migrated_file_count=sum(move["old_path"] != move["new_path"] for move in moves.values()),
        registered_in_place_edit_count=sum(
            move["old_path"] == move["new_path"] and move["current_sha256"] != move["original_sha256"]
            for move in moves.values()
        ),
        registered_unchanged_file_count=sum(
            move["old_path"] == move["new_path"] and move["current_sha256"] == move["original_sha256"]
            for move in moves.values()
        ),
        current_byte_identical_count=identity_unchanged,
        current_changed_sources=changed_sources,
        flat_file_counts=flat,
        final_flat_limits=FINAL_FLAT_LIMITS,
        full_restructure_complete=complete,
        test_discovery=nodes,
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--additional-catalog", action="append", default=[])
    parser.add_argument("--baseline-nodeids", type=Path)
    parser.add_argument("--current-nodeids", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", help="New repository-relative result path; existing evidence is never overwritten")
    args = parser.parse_args()
    root = repository_root()
    result = check(
        root,
        args.manifest,
        additional_catalog_paths=args.additional_catalog,
        baseline_nodes=args.baseline_nodeids,
        current_nodes=args.current_nodeids,
        require_complete=args.require_complete,
    )
    if args.output is not None:
        output = repo_path(args.output, root=root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
