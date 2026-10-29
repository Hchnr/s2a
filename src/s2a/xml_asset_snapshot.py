"""Copy a hash-bound XML asset closure without changing any model content.

Only ``file`` attribute values are replaced. Their original byte spans are
retained by the source XML, and verification reverses every replacement to
recover those exact bytes. No simulator is imported or queried. Historical
asset hashes describe their recorded snapshot, not an unrecorded later load.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from xml.parsers import expat
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
SAFE_ROOT = Path("/share-mixed")
OLD_PREFIX = "/share/project/"
SAFE_PREFIX = "/share-mixed/project/"
SCHEMA = "s2a.xml-asset-snapshot.v1"
_ATTRIBUTE = re.compile(rb"([^\s=<>/\"']+)\s*=\s*([\"'])(.*?)\2", re.DOTALL)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _contained(path, root):
    path = Path(path)
    _require(path.is_absolute(), "Absolute evidence and asset paths required")
    resolved = path.resolve()
    _require(resolved.is_relative_to(root.resolve()), "Path escapes its permitted root")
    return resolved


def _read_ref(ref, root):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Exact path/SHA reference required")
    _require(isinstance(ref["path"], str) and isinstance(ref["sha256"], str), "String path and SHA required")
    path = _contained(ref["path"], root)
    _require(path.is_file(), "Referenced file is missing")
    data = path.read_bytes()
    _require(_digest(data) == ref["sha256"], "Referenced file SHA differs")
    return path, data


def _reference(path, data):
    return {"path": str(path), "sha256": _digest(data)}


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def _file_attributes(raw):
    """Use parser offsets, never a whole-document regex, to locate attributes."""
    raw.decode("utf-8")
    _require(b"\x00" not in raw, "UTF-8 XML required")
    parser = expat.ParserCreate("utf-8")
    attributes = []

    def reject(*_args):
        raise ValueError("DTD, entities and external XML dependencies are unsupported")

    def declaration(_version, encoding, _standalone):
        _require(encoding is None or encoding.lower() in ("utf-8", "utf8", "ascii"), "UTF-8 XML required")

    def start(name, values):
        _require(name.split(":")[-1] != "include", "XML include is not a closed asset snapshot")
        if "file" not in values:
            return
        begin = parser.CurrentByteIndex
        tag = re.match(rb"<[^\s/>]+", raw[begin:])
        _require(tag is not None, "Element byte offset must identify a start tag")
        cursor, quote = begin + tag.end(), None
        while cursor < len(raw):
            char = raw[cursor]
            if quote is None:
                if char in (34, 39):
                    quote = char
                elif char == 62:
                    break
            elif char == quote:
                quote = None
            cursor += 1
        matches = [m for m in _ATTRIBUTE.finditer(raw, begin + tag.end(), cursor) if m[1] == b"file"]
        _require(len(matches) == 1, "Exactly one parsed file attribute span required")
        match = matches[0]
        attributes.append((match.start(3), match.end(3), values["file"]))

    parser.StartElementHandler = start
    parser.XmlDeclHandler = declaration
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    try:
        parser.Parse(raw, True)
    except expat.ExpatError as error:
        raise ValueError("Malformed XML") from error
    _require(attributes, "At least one external asset required")
    return attributes


def _safe_source(declared):
    _require(isinstance(declared, str), "String asset path required")
    mapped = SAFE_PREFIX + declared[len(OLD_PREFIX) :] if declared.startswith(OLD_PREFIX) else declared
    return _contained(mapped, SAFE_ROOT)


def _source_plan(raw, history):
    _require(isinstance(history, dict) and isinstance(history.get("assets"), dict), "Historical asset map required")
    historical = {}
    for declared, record in history["assets"].items():
        path = _safe_source(declared)
        _require(path not in historical, "Ambiguous historical asset aliases")
        _require(
            isinstance(record, dict)
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
            "Recorded historical SHA required for every asset",
        )
        historical[path] = record["sha256"]
    attributes = _file_attributes(raw)
    counts = Counter(declared for _, _, declared in attributes)
    plan = []
    for declared, count in counts.items():
        path = _safe_source(declared)
        _require(path in historical, "XML asset lacks a historical SHA")
        _require(path.is_file(), "Safe source asset is missing")
        data = path.read_bytes()
        _require(_digest(data) == historical[path], "Safe source differs from historical asset SHA")
        plan.append(dict(declared_path=declared, safe_source_ref=_reference(path, data), reference_count=count))
    return attributes, plan


def _replace(raw, attributes, replacements):
    _require(len(attributes) == len(replacements), "One replacement per file attribute required")
    for (start, end, _), value in reversed(list(zip(attributes, replacements, strict=True))):
        raw = raw[:start] + value + raw[end:]
    return raw


def _encoded_path(path):
    return escape(str(path), {'"': "&quot;", "'": "&apos;"}).encode("utf-8")


def _snapshot_path(destination, index, declared):
    return destination / "assets" / f"{index:04d}_{Path(declared).name}"


def _destination(path, *, new):
    path = Path(path)
    path = path if path.is_absolute() else REPO / path
    destination = _contained(path, REPO)
    _require(destination != REPO.resolve(), "Snapshot must have its own directory")
    # Reject symlink indirection even when it currently resolves inside the repo.
    absolute = path.absolute()
    _require(absolute.is_relative_to(REPO.resolve()), "Destination must be lexically inside repository")
    current = REPO.resolve()
    for part in absolute.relative_to(REPO.resolve()).parts:
        _require(part != "..", "Destination traversal is forbidden")
        current = current / part
        _require(not current.is_symlink(), "Snapshot destination cannot traverse symlinks")
    if new:
        _require(not destination.exists(), "Snapshot destination already exists; no overwrite or retry")
    return destination


def prepare_asset_snapshot(xml_ref, historical_manifest_ref, destination):
    """Create one new repository snapshot; partial failures are never overwritten."""
    destination = _destination(destination, new=True)
    xml_path, original = _read_ref(xml_ref, REPO)
    history_path, history_bytes = _read_ref(historical_manifest_ref, REPO)
    attributes, plan = _source_plan(original, _json(history_bytes))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "assets").mkdir()
    assets = []
    for index, item in enumerate(plan):
        _, data = _read_ref(item["safe_source_ref"], SAFE_ROOT)
        target = _snapshot_path(destination, index, item["declared_path"])
        with target.open("xb") as stream:
            stream.write(data)
        assets.append(dict(item, snapshot_ref=_reference(target, data)))
    paths = {item["declared_path"]: item["snapshot_ref"]["path"] for item in assets}
    migrated = _replace(original, attributes, [_encoded_path(paths[declared]) for _, _, declared in attributes])
    original_copy, model = destination / "original.xml", destination / "model.xml"
    for path, data in ((original_copy, original), (model, migrated)):
        with path.open("xb") as stream:
            stream.write(data)
    manifest = dict(
        schema_version=SCHEMA,
        original_xml_ref=_reference(xml_path, original),
        original_xml_snapshot_ref=_reference(original_copy, original),
        migrated_xml_ref=_reference(model, migrated),
        historical_manifest_ref=_reference(history_path, history_bytes),
        assets=assets,
        file_reference_count=len(attributes),
        unique_asset_count=len(assets),
        byte_exact_inverse=True,
    )
    path = destination / "manifest.json"
    data = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    ref = _reference(path, data)
    verify_asset_snapshot(ref)
    return ref


def verify_asset_snapshot(manifest_ref):
    """Check source hashes, all copied assets and the exact reversible XML edit."""
    path, data = _read_ref(manifest_ref, REPO)
    destination = _destination(path.parent, new=False)
    _require(path == destination / "manifest.json", "Canonical snapshot manifest required")
    manifest = _json(data)
    _require(
        isinstance(manifest, dict)
        and set(manifest)
        == {
            "schema_version",
            "original_xml_ref",
            "original_xml_snapshot_ref",
            "migrated_xml_ref",
            "historical_manifest_ref",
            "assets",
            "file_reference_count",
            "unique_asset_count",
            "byte_exact_inverse",
        }
        and manifest["schema_version"] == SCHEMA
        and manifest["byte_exact_inverse"] is True,
        "Exact snapshot schema required",
    )
    _, original = _read_ref(manifest["original_xml_ref"], REPO)
    _, history_bytes = _read_ref(manifest["historical_manifest_ref"], REPO)
    attributes, plan = _source_plan(original, _json(history_bytes))
    copy_path, copied = _read_ref(manifest["original_xml_snapshot_ref"], destination)
    model_path, migrated = _read_ref(manifest["migrated_xml_ref"], destination)
    _require(copy_path == destination / "original.xml" and copied == original, "Original XML copy differs")
    _require(model_path == destination / "model.xml", "Canonical migrated XML required")
    assets = manifest["assets"]
    _require(isinstance(assets, list) and len(assets) == len(plan), "Complete unique asset closure required")
    for index, (item, expected) in enumerate(zip(assets, plan, strict=True)):
        _require(isinstance(item, dict) and set(item) == {*expected, "snapshot_ref"}, "Exact asset entry schema")
        _require(type(item["reference_count"]) is int, "Integer asset occurrence count required")
        _require(all(item[key] == value for key, value in expected.items()), "Asset source or occurrence count differs")
        target, asset = _read_ref(item["snapshot_ref"], destination / "assets")
        _require(target == _snapshot_path(destination, index, expected["declared_path"]), "Asset destination differs")
        _require(_digest(asset) == expected["safe_source_ref"]["sha256"], "Copied asset differs from historical bytes")
    for key, count in (("file_reference_count", len(attributes)), ("unique_asset_count", len(plan))):
        _require(type(manifest[key]) is int and manifest[key] == count, "Asset closure count differs")
    paths = {item["declared_path"]: item["snapshot_ref"]["path"] for item in assets}
    expected_xml = _replace(original, attributes, [_encoded_path(paths[declared]) for _, _, declared in attributes])
    _require(migrated == expected_xml, "XML changes extend beyond registered file attribute replacements")
    migrated_attributes = _file_attributes(migrated)
    original_values = [original[start:end] for start, end, _ in attributes]
    restored = _replace(migrated, migrated_attributes, original_values)
    _require(restored == original, "Inverse migration does not recover original XML bytes")
    _require(
        {declared for _, _, declared in migrated_attributes} == set(paths.values()),
        "Migrated XML asset closure differs",
    )
    return dict(manifest, verified=True)
