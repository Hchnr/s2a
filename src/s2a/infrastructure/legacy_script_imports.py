"""Explicit access to eleven retained scripts under their original bare names.

This boundary checks origin and baseline source bytes at each access. It does
not modify import search paths, install import hooks, call a script's main, or
qualify its transitive dependencies or any control/runtime behavior. Imports
outside this boundary remain ordinary Python imports; finish detects alias or
source drift introduced after an access.
"""

import copy
import hashlib
import importlib.util
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from s2a.infrastructure.repository_paths import repo_path, repository_root
from s2a.infrastructure.source_identity import _invalid_constant, _json_object, _reference, _stream_digest

BASELINE_COMMIT = "7b2712850dea54090d61e564380f88825ad8b8f6"
CATALOG_REF = dict(
    path="docs/migrations/baseline_20261029.json",
    sha256="1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897",
)
LEGACY_MODULE_NAMES = (
    "accepted_point_sqp_math_audit",
    "check_tracking_v5_accepted_point_sqp",
    "check_tracking_v5_prefix_trust_control",
    "check_tracking_v5_saved_direction_backtracking",
    "check_tracking_v5_tail_translation_reserve_v2",
    "check_tracking_v5_terminal_secant_probe",
    "directional_derivative_audit",
    "probe_tracking_v5_terminal_only_tail_repair",
    "restored_anchor_sqp_ledger_audit",
    "tail_translation_reserve_audit",
    "terminal_secant_audit",
)
_LOCK = threading.RLock()
_SOURCE_BYTE_LIMIT = 4 * 1024 * 1024
_CATALOG_BYTE_LIMIT = 32 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _origins(module):
    """Read module metadata without invoking module-level __getattr__."""
    namespace = vars(module)
    spec = namespace.get("__spec__")
    return namespace.get("__file__"), None if spec is None else spec.origin


def _lexical_origin(value):
    return os.path.abspath(value) if isinstance(value, str) and value and ".." not in Path(value).parts else None


@dataclass(frozen=True)
class LegacyScriptAccess:
    """A module plus a JSON-only receipt, separate from execution admission."""

    module: ModuleType
    identity_receipt: dict


class LegacyScriptImports:
    """Allowlisted bare-module access; explicit alternate catalogues aid tests."""

    def __init__(
        self, *, root=None, catalog_ref=None, baseline_commit=BASELINE_COMMIT, allowed_names=LEGACY_MODULE_NAMES
    ):
        self.root = repository_root() if root is None else repository_root(anchor=root)
        if root is not None:
            _require(self.root == Path(root).absolute(), "Explicit repository root required")
        self.catalog_ref = _reference(CATALOG_REF if catalog_ref is None else catalog_ref)
        self.baseline_commit = baseline_commit
        self.allowed_names = tuple(allowed_names)
        _require(
            bool(self.allowed_names)
            and len(set(self.allowed_names)) == len(self.allowed_names)
            and all(isinstance(name, str) and name.isidentifier() for name in self.allowed_names),
            "Unique bare module names required in explicit allowlist",
        )
        path = Path(self.catalog_ref["path"])
        if path.is_absolute():
            _require(path.is_relative_to(self.root), "Catalog escaped repository root")
            path = path.relative_to(self.root)
        self._catalog_path = repo_path(path, root=self.root)
        self._verify_catalog()
        _require(self._catalog_path.stat().st_size <= _CATALOG_BYTE_LIMIT, "Catalog exceeds byte budget")
        with self._catalog_path.open("rb") as stream:
            raw = stream.read(_CATALOG_BYTE_LIMIT + 1)
        _require(
            len(raw) <= _CATALOG_BYTE_LIMIT and hashlib.sha256(raw).hexdigest() == self.catalog_ref["sha256"],
            "Catalog changed before consumption",
        )
        catalog = json.loads(raw, object_pairs_hook=_json_object, parse_constant=_invalid_constant)
        _require(
            isinstance(catalog, dict)
            and type(catalog.get("schema_version")) is int
            and catalog["schema_version"] == 1
            and catalog.get("baseline_commit") == self.baseline_commit
            and isinstance(catalog.get("entries"), list),
            "Registered baseline catalog required",
        )
        entries = {}
        for entry in catalog["entries"]:
            _require(isinstance(entry, dict) and isinstance(entry.get("old_path"), str), "Catalog entry required")
            old = entry["old_path"]
            _require(old not in entries, "Duplicate catalog source path")
            entries[old] = entry
        self._sources = {}
        for name in self.allowed_names:
            relative = f"scripts/{name}.py"
            entry = entries.get(relative)
            _require(
                entry is not None
                and entry.get("baseline_commit") == self.baseline_commit
                and entry.get("git_mode") in {"100644", "100755"}
                and type(entry.get("size_bytes")) is int
                and 0 <= entry["size_bytes"] <= _SOURCE_BYTE_LIMIT,
                "Allowlisted script must be registered as an original regular source",
            )
            self._sources[name] = dict(
                original_ref=_reference(dict(path=relative, sha256=entry.get("sha256"))), size=entry["size_bytes"]
            )
        self._accesses = {}
        self._load_kinds = {}

    @property
    def receipts(self):
        """JSON-only identities for modules actually verified by this instance."""
        return {name: copy.deepcopy(access.identity_receipt) for name, access in self._accesses.items()}

    def _verify_catalog(self):
        digest, _ = _stream_digest(self._catalog_path)
        _require(digest == self.catalog_ref["sha256"], "Registered catalog SHA changed")

    def _source(self, name):
        _require(name in self._sources, f"Unknown legacy module name: {name}")
        source = self._sources[name]
        path = repo_path(source["original_ref"]["path"], root=self.root)
        digest, size = _stream_digest(path)
        _require(
            digest == source["original_ref"]["sha256"] and size == source["size"],
            f"Legacy script original SHA/size changed: {name}",
        )
        return path, dict(path=str(path), sha256=digest)

    def _no_aliases(self, name):
        _require(name in self._sources, f"Unknown legacy module name: {name}")
        expected = str(repo_path(self._sources[name]["original_ref"]["path"], root=self.root))
        bare = sys.modules.get(name)
        for alias, module in tuple(sys.modules.items()):
            if alias == name or module is None:
                continue
            named_alias = alias == f"scripts.{name}"
            same_object = bare is not None and module is bare
            same_origin = isinstance(module, ModuleType) and any(
                _lexical_origin(value) == expected for value in _origins(module)
            )
            _require(
                not (named_alias or same_object or same_origin),
                f"Legacy script has a duplicate/qualified module identity: {name} as {alias}",
            )

    def verify_loaded(self, name):
        """Verify a present bare module without loading an absent script."""
        with _LOCK:
            self._verify_catalog()
            self._no_aliases(name)
            module = sys.modules.get(name)
            _require(isinstance(module, ModuleType), f"Legacy module is not loaded: {name}")
            path, source_ref = self._source(name)
            namespace, spec = vars(module), vars(module).get("__spec__")
            _require(
                namespace.get("__name__") == name
                and namespace.get("__package__") in {"", None}
                and spec is not None
                and spec.name == name
                and all(value == str(path) for value in _origins(module)),
                f"Legacy module origin/namespace drift: {name}",
            )
            prior = self._accesses.get(name)
            _require(prior is None or prior.module is module, f"Legacy module object was replaced: {name}")
            access = LegacyScriptAccess(
                module=module,
                identity_receipt=dict(
                    module_name=name,
                    namespace="bare",
                    original_ref=copy.deepcopy(self._sources[name]["original_ref"]),
                    source_ref=source_ref,
                    catalog_ref=copy.deepcopy(self.catalog_ref),
                    baseline_commit=self.baseline_commit,
                    load_kind=self._load_kinds.get(name, "verified_existing"),
                    identity_scope="allowlisted_legacy_source_origin_only",
                    transitive_dependencies_qualified=False,
                    main_invoked_by_boundary=False,
                ),
            )
            self._accesses[name] = access
            return access

    def load(self, name):
        """Load verified source under its bare name; never call its main entry."""
        with _LOCK:
            self._verify_catalog()
            _require(name in self._sources, f"Unknown legacy module name: {name}")
            # Direct dependencies can import one another normally. Reject drift
            # in the complete narrow allowlist before any such import executes.
            for registered in self.allowed_names:
                self._source(registered)
                self._no_aliases(registered)
                if registered in sys.modules:
                    self.verify_loaded(registered)
            if name in sys.modules:
                return self.verify_loaded(name)
            path, source_ref = self._source(name)
            with path.open("rb") as stream:
                raw = stream.read(_SOURCE_BYTE_LIMIT + 1)
            _require(
                len(raw) <= _SOURCE_BYTE_LIMIT and hashlib.sha256(raw).hexdigest() == source_ref["sha256"],
                "Legacy source changed before import",
            )
            spec = importlib.util.spec_from_file_location(name, path)
            _require(spec is not None and spec.loader is not None, "Legacy source import specification missing")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                # Compile the checked original bytes, bypassing unverified pyc
                # caches while retaining the real spec, filename and bare name.
                exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
                self._load_kinds[name] = "loaded_exact_source"
                for registered in self.allowed_names:
                    self._no_aliases(registered)
                    if registered in sys.modules:
                        self.verify_loaded(registered)
                return self.verify_loaded(name)
            except BaseException:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
                self._accesses.pop(name, None)
                self._load_kinds.pop(name, None)
                raise

    def verify_present(self, names=None, *, require_loaded=False):
        """Check selected present modules; importing a package need not load all."""
        selected = self.allowed_names if names is None else tuple(names)
        _require(len(set(selected)) == len(selected), "Duplicate requested legacy module name")
        verified = {}
        with _LOCK:
            self._verify_catalog()
            for name in selected:
                self._no_aliases(name)
                if name in sys.modules:
                    verified[name] = self.verify_loaded(name).identity_receipt
                else:
                    _require(not require_loaded, f"Legacy module is not loaded: {name}")
        return dict(
            schema_version="s2a.allowlisted-legacy-script-imports.v1",
            baseline_commit=self.baseline_commit,
            catalog_ref=copy.deepcopy(self.catalog_ref),
            verified_modules=copy.deepcopy(verified),
            source_refs=[copy.deepcopy(receipt["source_ref"]) for receipt in verified.values()],
            require_loaded=require_loaded,
            identity_scope="allowlisted_legacy_source_origin_only",
            transitive_dependencies_qualified=False,
        )

    def finish(self):
        """Detect source, namespace, alias or object replacement after access."""
        with _LOCK:
            self._verify_catalog()
            for name in tuple(self._accesses):
                self.verify_loaded(name)


def verify_legacy_modules(names=None, *, require_loaded=False, root=None):
    """Default D2 package boot check; it never imports an absent old helper."""
    return LegacyScriptImports(root=root).verify_present(names, require_loaded=require_loaded)
