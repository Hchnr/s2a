"""Make isolated041 entrypoints by enumerated text edits to frozen040 sources.

This module uses only the standard library. It reads Python/JSON metadata and
hashes files, never imports an entrypoint, decodes training labels, or executes
model/control code. Publishing is exclusive; a failed partial bundle is retained.
"""

import ast
import hashlib
import io
import json
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT_REL = "outputs/wam_stability_20261021/clean_residual_training_20261023"
NEW_ROOT_REL = ORIGINAL_ROOT_REL + "/recovery_20261023"
BUNDLE_REL = NEW_ROOT_REL + "/source/entrypoints"
SCHEMA = "s2a.clean-training-continuation-bundle.v1"
SCRIPT_NAMES = (
    "run_wam_clean_training.py",
    "collect_wam_clean_training.py",
    "verify_wam_clean_training_capture.py",
    "run_wam_clean_training_state.py",
    "sample_wam_clean_training_state.py",
    "verify_wam_clean_training_state.py",
    "run_wam_clean_training_development.py",
    "serve_wam_clean_training.py",
    "verify_wam_clean_training_development.py",
    "execute_wam_clean_training_case.py",
)
RUNTIME_REL = "src/s2a/clean_training_runtime.py"
IMPORT_OLD = "from s2a.clean_training_runtime import"
IMPORT_NEW = "from continuation_runtime import"
REPO_GUARD = "REPO = Path(__file__).resolve().parents[1]"

# Counts are measured against the frozen040 source contract, not inferred from
# whatever a later source happens to contain. All replacements are exact tokens.
_COUNTS = (
    {"scripts/collect_wam_clean_training.py": 1, "scripts/verify_wam_clean_training_capture.py": 1, "import": 1},
    {RUNTIME_REL: 1, "import": 1},
    {"scripts/collect_wam_clean_training.py": 2, RUNTIME_REL: 1, ORIGINAL_ROOT_REL: 1, "repo": 1},
    {"scripts/sample_wam_clean_training_state.py": 2, "scripts/verify_wam_clean_training_state.py": 1, "import": 1},
    {RUNTIME_REL: 1, "scripts/collect_wam_clean_training.py": 1, "scripts/run_wam_clean_training.py": 1, "import": 1},
    {"scripts/sample_wam_clean_training_state.py": 2},
    {
        "scripts/execute_wam_clean_training_case.py": 2,
        "scripts/verify_wam_clean_training_development.py": 2,
        "scripts/serve_wam_clean_training.py": 2,
        "import": 1,
    },
    {RUNTIME_REL: 1, "scripts/collect_wam_clean_training.py": 1, "import": 1},
    {"repo": 1},
    {"scripts/run_wam_clean_training_development.py": 1, "import": 1},
    {ORIGINAL_ROOT_REL: 1},
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _repo(repo):
    repo = Path(repo)
    _require(repo.is_absolute() and repo.is_dir() and repo.resolve() == repo, "repository must be canonical absolute")
    return repo


def _safe(path, repo, *, exists):
    path = Path(path)
    _require(path.is_absolute() and path.is_relative_to(repo), "bundle path escapes the repository")
    _require(path.resolve() == path, "bundle/source path contains a symlink or traversal")
    if exists:
        _require(path.is_file() and not path.is_symlink(), "frozen source is missing or symlinked")
    return path


def _transform(source, *, relative, target, repo, expected_counts):
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=relative)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(position):
        return offsets[position[0] - 1] + position[1]

    edits, counts = [], {}

    def add(kind, key, start, end, old, new):
        begin, finish = offset(start), offset(end)
        _require(text[begin:finish] == old, "replacement span does not match exact original text")
        edits.append(dict(kind=kind, old=old, new=new, count=1, start=list(start), end=list(end), offset=begin))
        counts[key] = counts.get(key, 0) + 1

    replacements = {"scripts/" + name: BUNDLE_REL + "/" + name for name in SCRIPT_NAMES}
    replacements[RUNTIME_REL] = BUNDLE_REL + "/continuation_runtime.py"
    replacements[ORIGINAL_ROOT_REL] = NEW_ROOT_REL
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, str) or value not in replacements:
            continue
        _require(
            token.string in ('"' + value + '"', "'" + value + "'"),
            "registered path literal must use exact ordinary quotes without escapes",
        )
        new = token.string[0] + replacements[value] + token.string[-1]
        add("quoted_path", value, token.start, token.end, token.string, new)
    significant = [t for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL)]
    for index in range(len(significant) - 4):
        group = significant[index : index + 5]
        if [t.string for t in group] == ["from", "s2a", ".", "clean_training_runtime", "import"]:
            add("runtime_import", "import", group[0].start, group[-1].end, IMPORT_OLD, IMPORT_NEW)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or ast.get_source_segment(text, node) != REPO_GUARD:
            continue
        add(
            "repository_guard",
            "repo",
            (node.lineno, node.col_offset),
            (node.end_lineno, node.end_col_offset),
            REPO_GUARD,
            "REPO = Path(" + json.dumps(str(repo)) + ")",
        )
    _require(counts == expected_counts, f"finite replacement counts differ for {relative}: {counts}")
    edits.sort(key=lambda e: e["offset"])
    end = 0
    for edit in edits:
        _require(edit["offset"] >= end, "registered edits overlap")
        end = edit["offset"] + len(edit["old"])
    transformed = text
    for edit in reversed(edits):
        begin = edit["offset"]
        transformed = transformed[:begin] + edit["new"] + transformed[begin + len(edit["old"]) :]
    # Compiling both the AST and the full source checks syntax without importing
    # dependencies, writing pyc files, or executing a single statement.
    compiled_ast = compile(transformed, str(target), "exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True)
    compile(compiled_ast, str(target), "exec", dont_inherit=True)
    compile(transformed.encode("utf-8"), str(target), "exec", dont_inherit=True)
    return transformed, edits


def expected_bundle_sources(original_registration, *, repo=REPO):
    """Return the eleven exact byte plans, without creating any output path."""
    repo = _repo(repo)
    _require(isinstance(original_registration, dict), "original registration must be JSON metadata")
    sources = original_registration.get("source_sha256")
    _require(isinstance(sources, dict) and bool(sources), "original registration source closure is absent")
    originals = ["scripts/" + name for name in SCRIPT_NAMES] + [RUNTIME_REL]
    result = {}
    for relative, counts in zip(originals, _COUNTS, strict=True):
        original = _safe(repo / relative, repo, exists=True)
        expected = sources.get(str(original))
        _require(_hex(expected), f"required original source is not registration-bound: {relative}")
        before = original.read_bytes()
        _require(_digest(before) == expected, f"original source SHA drift: {relative}")
        name = original.name if relative != RUNTIME_REL else "continuation_runtime.py"
        target = _safe(repo / BUNDLE_REL / name, repo, exists=False)
        transformed, edits = _transform(before, relative=relative, target=target, repo=repo, expected_counts=counts)
        _require(_sha(original) == expected, f"original source changed during planning: {relative}")
        result[str(original)] = dict(
            target_abs=str(target), source_sha256=expected, transformed_text=transformed, edits=edits
        )
    _require(len({entry["target_abs"] for entry in result.values()}) == 11, "bundle target collision")
    return result


def publish_bundle(original_registration, output_root=REPO / NEW_ROOT_REL, *, repo=REPO, old_admitted_ref=None):
    """Write eleven exclusive files and return metadata for the owning producer.

    The output root may already hold the producer's preparation receipt. Only
    the entrypoint directory is exclusive. The caller publishes the manifest.
    """
    repo = _repo(repo)
    output_root = _safe(output_root, repo, exists=False)
    _require(output_root == repo / NEW_ROOT_REL, "continuation output root differs from the fixed041 location")
    if old_admitted_ref is not None:
        _require(
            isinstance(old_admitted_ref, dict)
            and set(old_admitted_ref) == {"path", "sha256"}
            and _hex(old_admitted_ref["sha256"]),
            "old admitted reference must have exact path/SHA fields",
        )
        admitted = _safe(old_admitted_ref["path"], repo, exists=True)
        _require(admitted == repo / ORIGINAL_ROOT_REL / "execution/admitted.json", "old admitted path differs")
        _require(_sha(admitted) == old_admitted_ref["sha256"], "old admitted SHA differs")
        _require(json.loads(admitted.read_bytes()) == original_registration, "old admitted payload differs")
    planned = expected_bundle_sources(original_registration, repo=repo)
    directory = _safe(output_root / "source/entrypoints", repo, exists=False)
    _require(not directory.exists(), "bundle already exists; refuses overwrite or retry")
    # Check all inputs again before the first mutation. A failed publication is
    # not rolled back or retried: its original partial evidence stays available.
    for original, entry in planned.items():
        _require(_sha(original) == entry["source_sha256"], "source changed before bundle publication")
    directory.mkdir(parents=True, exist_ok=False)
    files, actual_sources = [], {}
    for original, entry in planned.items():
        target = _safe(entry["target_abs"], repo, exists=False)
        data = entry["transformed_text"].encode("utf-8")
        with target.open("xb") as stream:
            stream.write(data)
        _require(target.read_bytes() == data, "bundle written bytes differ from compiled source")
        actual = dict(path=str(target), sha256=_digest(data))
        actual_sources[str(target)] = actual["sha256"]
        files.append(
            dict(
                original_ref=dict(path=original, sha256=entry["source_sha256"]),
                actual_ref=actual,
                edits=entry["edits"],
            )
        )
    for original, entry in planned.items():
        _require(_sha(original) == entry["source_sha256"], "original source changed during publication")
    manifest = dict(
        schema_version=SCHEMA,
        complete=True,
        passed=True,
        original_root=str(repo / ORIGINAL_ROOT_REL),
        output_root=str(output_root),
        bundle_root=str(directory),
        old_admitted_ref=old_admitted_ref,
        files=files,
        source_sha256=actual_sources,
        original_sources_before_after_exact=True,
        ast_and_source_compiled=True,
        entrypoints_executed=0,
        future_archive_decodes=0,
    )
    return manifest
