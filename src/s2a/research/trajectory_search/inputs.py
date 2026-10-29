"""Explicit inputs for the migrated active search, separate from historical code.

Historical source receipts prove bytes only. Runtime inputs and currently loaded
source files use their real paths; archived source is never imported here.
"""

import copy
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from s2a.infrastructure.current_layout import CURRENT_MIGRATION_PATH, approved_migration_ref
from s2a.infrastructure.repository_paths import (
    FORBIDDEN_HISTORICAL_ROOT,
    _absolute_without_links,
    repo_path,
    repository_root,
)
from s2a.infrastructure.source_identity import SourceIdentityLocator, _stream_digest

BASELINE_COMMIT = "7b2712850dea54090d61e564380f88825ad8b8f6"
CATALOG_REF = dict(
    path="docs/migrations/baseline_20261029.json",
    sha256="1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897",
)
MIGRATION_PATH = CURRENT_MIGRATION_PATH
ROOT_FILES_REF = dict(
    path="docs/migrations/root_files_20261029.json",
    sha256="6865e39d7d17206a182a7a3038b2a8c13e2ac6af67283358ae63d3d9f90e1b44",
)
BUNDLE_PATH = "outputs/constraint_residual_repair_inputs_20261029/bundle.json"
BUNDLE_SHA = "eadf18a3ed90c29ee70d773741cc310304f4385189baea360e534ca6ef5bd023"
OLD_CONFIG_PATH = "configs/state_vector_objective_scale_20261029.json"
CONFIG_SHA = "fdb9844dd5efcbcaef456f3961feb22f865884b6083a417251b3cc1d214ea42f"
SNAPSHOT_ROOT = "outputs/prefix_trust_asset_snapshot_20261029"
ORIGINAL_XML_PATH = (
    "outputs/tracking_preview_wam_development_v5_20261028/development/cases/"
    "task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/model.xml"
)
HISTORICAL_ASSET_MANIFEST_PATH = "outputs/terminal_development_20261025/historical_oracle_xml_migration.json"
ORIGINAL_REPAIR_SOURCE_PATHS = (
    "src/s2a/xml_asset_snapshot.py",
    "src/s2a/prefix_trust_asset_repair.py",
    "scripts/prepare_prefix_trust_asset_snapshot.py",
    "scripts/check_tracking_v5_prefix_trust_control_v2.py",
    "scripts/execute_tracking_v5_prefix_trust_check_v2.py",
    "plans/v0.0.1/prefix_trust_asset_check_repair_20261029.md",
    "tests/test_xml_asset_snapshot.py",
    "tests/test_prefix_trust_asset_check_v2.py",
    "tests/test_prefix_trust_asset_repair_execution.py",
)
MODEL_PROOF_CURRENT_SOURCES = (
    "src/s2a/infrastructure/snapshots/xml_asset_snapshot.py",
    "src/s2a/infrastructure/native_environment.py",
    "src/s2a/dynamics/mujoco_robot.py",
)
CPU_NATIVE_ENVIRONMENT = dict(
    CUDA_VISIBLE_DEVICES="",
    LD_LIBRARY_PATH="",
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    LP_NUM_THREADS="1",
)
LIBERO_ASSET_ROOT = Path(
    "/share-mixed/project/eai_pwm/datasets/cosmos3_action_tower_training/libero_sim/LIBERO/libero/libero/assets"
)
_ASSET_SUFFIXES = {".png", ".msh", ".stl"}
_SHA = re.compile(r"[0-9a-f]{64}")
_SOURCE_AREAS = {"src", "scripts", "tests", "configs", "plans", "reports", "docs"}
_ROOT_METADATA = {"pyproject.toml", "README.md", "AGENTS.md", ".gitignore"}
CURRENT_SUPPORT_PATHS = (
    "pyproject.toml",
    "src/s2a/research/trajectory_search/inputs.py",
    "src/s2a/infrastructure/repository_paths.py",
    "src/s2a/infrastructure/source_identity.py",
    "src/s2a/infrastructure/legacy_script_imports.py",
    "src/s2a/infrastructure/current_layout.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value):
    require(
        isinstance(value, dict)
        and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str)
        and bool(value["path"])
        and isinstance(value["sha256"], str)
        and _SHA.fullmatch(value["sha256"]),
        "Exact declared path/SHA reference required",
    )
    return copy.deepcopy(value)


class SearchInputContext:
    """Keep historical identities apart from strict current physical inputs."""

    def __init__(
        self, *, root, baseline_commit, catalog_ref, migration_refs=(), additional_catalog_refs=(), cache_directory
    ):
        self.root = repository_root(anchor=root)
        require(self.root == _absolute_without_links(root), "Explicit repository root required")
        self.catalog_ref = _ref(catalog_ref)
        self.migration_refs = tuple(_ref(ref) for ref in migration_refs)
        self.additional_catalog_refs = tuple(_ref(ref) for ref in additional_catalog_refs)
        self.locator = SourceIdentityLocator(
            root=self.root,
            baseline_commit=baseline_commit,
            catalog_ref=self.catalog_ref,
            migration_refs=self.migration_refs,
            additional_catalog_refs=self.additional_catalog_refs,
            cache_directory=cache_directory,
        )
        self.runtime_refs = {}
        self.current_source_refs = {}
        self._historical_sources = {}
        self._historical_artifacts = {}
        self._external_artifact_allowlist = {}
        self._external_artifacts = {}
        self._model_snapshot_identities = []
        catalog = self.read_current_json(self.catalog_ref)
        self._catalog_paths = {entry["old_path"] for entry in catalog["entries"]}
        self._catalog_sha = {entry["old_path"]: entry["sha256"] for entry in catalog["entries"]}
        self.bind_current_source(self.catalog_ref["path"])
        for ref in self.additional_catalog_refs:
            supplemental = self.read_current_json(ref)
            self._catalog_paths.update(entry["old_path"] for entry in supplemental["entries"])
            self._catalog_sha.update({entry["old_path"]: entry["sha256"] for entry in supplemental["entries"]})
            self.bind_current_source(ref["path"])
        for ref in self.migration_refs:
            migration = self.read_current_json(ref)
            self.bind_current_source(ref["path"])
            for move in migration["moves"]:
                actual = dict(path=str(self.path(move["new_path"])), sha256=move["current_sha256"])
                self.checked_current(actual)
                self.current_source_refs[actual["path"]] = actual

    def historical_reference(self, relative):
        require(relative in self._catalog_sha, "Explicit catalogued historical path required")
        return dict(path=str(self.root / relative), sha256=self._catalog_sha[relative])

    def path(self, value):
        value = Path(value)
        path = _absolute_without_links(value) if value.is_absolute() else repo_path(value, root=self.root)
        require(path.is_relative_to(self.root) and path != self.root, "Current input must remain inside repository")
        require(not path.is_relative_to(self.root / ".git"), "Git metadata is not a runtime input")
        return path

    def reference(self, value):
        path = self.path(value)
        return dict(path=str(path), sha256=_stream_digest(path)[0])

    def checked_current(self, ref):
        ref = _ref(ref)
        path = self.path(ref["path"])
        require(_stream_digest(path)[0] == ref["sha256"], "Current input/source SHA256 mismatch")
        actual = dict(path=str(path), sha256=ref["sha256"])
        require(self.runtime_refs.get(str(path), actual) == actual, "Conflicting current physical source identity")
        self.runtime_refs[str(path)] = actual
        return path

    def bind_current_source(self, value):
        relative = self.path(value).relative_to(self.root)
        require(
            relative.parts[0] in _SOURCE_AREAS or relative.as_posix() in _ROOT_METADATA, "Explicit current source area"
        )
        ref = self.reference(value)
        self.checked_current(ref)
        self.current_source_refs[ref["path"]] = ref
        return copy.deepcopy(ref)

    def _historical_relative(self, declared):
        path = Path(declared)
        require(".." not in path.parts and str(path) == declared, "Canonical historical declaration required")
        if path.is_absolute():
            if path.is_relative_to(self.root):
                path = path.relative_to(self.root)
            elif path.is_relative_to(FORBIDDEN_HISTORICAL_ROOT):
                path = path.relative_to(FORBIDDEN_HISTORICAL_ROOT)
            else:
                return None
        return path.as_posix()

    def register_external_bundle_artifacts(self, bundle_ref):
        """Admit only fixed asset references declared in a SHA-bound bundle.

        This does not admit external source code or a different physical path.
        The asset bytes are checked when consumed and again at finish.
        """
        bundle_ref = self.remember_historical(bundle_ref)
        require(
            (bundle_ref["path"], bundle_ref["sha256"]) in self._historical_artifacts,
            "External artifacts require a repository artifact bundle",
        )
        bundle = self.read_historical_json(bundle_ref)
        for value in bundle["input_refs"]:
            ref = _ref(value)
            if self._historical_relative(ref["path"]) is not None:
                continue
            path = _absolute_without_links(ref["path"])
            require(
                str(path) == ref["path"] and path.is_relative_to(LIBERO_ASSET_ROOT) and path.suffix in _ASSET_SUFFIXES,
                "Only explicitly declared fixed LIBERO mesh/texture artifacts are allowed",
            )
            receipt = dict(declared_ref=ref, bundle_ref=copy.deepcopy(bundle_ref))
            prior = self._external_artifact_allowlist.get(ref["path"], receipt)
            require(prior == receipt, "Conflicting external artifact identity or bundle provenance")
            self._external_artifact_allowlist[ref["path"]] = receipt

    def _checked_external_artifact(self, ref):
        receipt = self._external_artifact_allowlist.get(ref["path"])
        require(
            receipt is not None and receipt["declared_ref"] == ref,
            "External artifact requires an exact SHA-bound bundle declaration",
        )
        path = _absolute_without_links(ref["path"])
        require(
            str(path) == ref["path"] and path.is_relative_to(LIBERO_ASSET_ROOT) and path.suffix in _ASSET_SUFFIXES,
            "Only explicitly declared fixed LIBERO mesh/texture artifacts are allowed",
        )
        require(_stream_digest(path)[0] == ref["sha256"], "External artifact SHA256 mismatch")
        return path

    def remember_historical(self, ref):
        """Catalogued source uses the locator; other artifacts retain strict paths."""
        ref = _ref(ref)
        identity = (ref["path"], ref["sha256"])
        relative = self._historical_relative(ref["path"])
        if relative in self._catalog_paths:
            if identity not in self._historical_sources:
                self._historical_sources[identity] = asdict(self.locator.resolve(ref))
        elif relative is None:
            self._checked_external_artifact(ref)
            self._external_artifacts[identity] = copy.deepcopy(self._external_artifact_allowlist[ref["path"]])
        else:
            require(
                relative is not None and Path(relative).parts[0] == "outputs",
                "Uncatalogued historical source cannot fall back to current bytes",
            )
            self.checked_current(ref)
            self._historical_artifacts[identity] = ref
        return copy.deepcopy(ref)

    def read_historical_json(self, ref):
        return json.loads(self.historical_bytes(ref))

    def historical_bytes(self, ref):
        ref = self.remember_historical(ref)
        identity = (ref["path"], ref["sha256"])
        resolved = self._historical_sources.get(identity)
        storage = ref if resolved is None else resolved["storage_ref"]
        path = (
            self._checked_external_artifact(ref) if identity in self._external_artifacts else self.path(storage["path"])
        )
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Historical bytes changed while reading")
        return payload

    def read_current_json(self, ref):
        path = self.checked_current(ref)
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == ref["sha256"], "Current bytes changed while reading")
        return json.loads(payload)

    def finish(self):
        self.locator.finish()
        # Legacy native helpers may add unchanged native/asset inputs to this
        # physical map. They never receive the historical source identity map.
        for ref in list(self.runtime_refs.values()):
            self.checked_current(ref)
        for receipt in self._external_artifacts.values():
            self._checked_external_artifact(receipt["declared_ref"])

    def registration(self):
        self.finish()
        return dict(
            migration_scope="current loader/source adaptation; historical byte identity only",
            catalog_ref=copy.deepcopy(self.catalog_ref),
            migration_refs=copy.deepcopy(list(self.migration_refs)),
            additional_catalog_refs=copy.deepcopy(list(self.additional_catalog_refs)),
            current_source_refs=copy.deepcopy(list(self.current_source_refs.values())),
            runtime_input_refs=copy.deepcopy(list(self.runtime_refs.values())),
            historical_source_identities=copy.deepcopy(list(self._historical_sources.values())),
            historical_artifact_refs=copy.deepcopy(list(self._historical_artifacts.values())),
            external_artifact_identities=copy.deepcopy(list(self._external_artifacts.values())),
            external_artifact_execution_admitted=False,
            model_snapshot_identities=copy.deepcopy(self._model_snapshot_identities),
            identity_git_operations=self.locator.git_operations,
            historical_source_execution_admitted=False,
            previous_failed_checker_qualification_changed=False,
        )


def make_input_context(*, root=None, catalog_ref=CATALOG_REF, migration_refs=None, additional_catalog_refs=None):
    """Bind the approved cumulative current manifest, separately from history."""
    root = repository_root() if root is None else repository_root(anchor=root)
    if migration_refs is None:
        migration_refs = (approved_migration_ref(),)
    if additional_catalog_refs is None:
        additional_catalog_refs = (ROOT_FILES_REF,)
    return SearchInputContext(
        root=root,
        baseline_commit=BASELINE_COMMIT,
        catalog_ref=catalog_ref,
        migration_refs=migration_refs,
        additional_catalog_refs=additional_catalog_refs,
        cache_directory=root / "outputs/source_identity_cache/trajectory_search_20261029",
    )


def validate_closed_dual_inputs(prior, diagnostic):
    require(
        prior["complete"] is prior["passed"] is prior["sources_unchanged"] is True
        and prior["costs"]["variant"]["response_calls_per_route"] == 5
        and prior["costs"]["variant"]["stage_count"] == 10
        and prior["variant"]["witness_indices"] == []
        and prior["variant"]["selected_response_index"] is None
        and prior["variant"]["termination_reason"] == "radius_below_minimum",
        "Closed ten-stage whole-objective search without an adopted trajectory",
    )
    require(
        diagnostic["complete"] is diagnostic["passed"] is diagnostic["sources_unchanged"] is True
        and all(type(value) is int and value == 0 for value in diagnostic["new_activity"].values())
        and diagnostic["original_complete_witness_count"] == 0
        and diagnostic["original_failures_retained"] is True
        and diagnostic["original_ledger_audit"]["passed"] is True
        and len(diagnostic["records"]) == 10
        and sum(row["original_qualified"] is False for row in diagnostic["records"]) == 7
        and all(
            row["original_solver_status"] == 0 and row["original_solver_success"] is True
            for row in diagnostic["records"]
        ),
        "Closed zero-fit saved-dual arithmetic; original failures remain unqualified",
    )


def load_dual_recovery_inputs(
    context,
    *,
    config_path,
    current_source_paths,
    prior_analysis_ref,
    diagnostic_ref,
    bundle_ref=None,
    historical_config_ref=None,
    support_paths=CURRENT_SUPPORT_PATHS,
):
    """Read one original compact stencil and closed motivation, with no cascade."""
    bundle_ref = dict(path=str(context.root / BUNDLE_PATH), sha256=BUNDLE_SHA) if bundle_ref is None else bundle_ref
    bundle = context.read_historical_json(bundle_ref)
    require(
        bundle["base_response_index"] == 3557
        and bundle["trial_response_index"] == 3639
        and bundle["probe_response_indices"] == list(range(3558, 3630))
        and bundle["phase"]["stage_index"] == 53
        and bundle["phase"]["radius"] == 0.005,
        "One original 3557 stencil and unchanged historical direction",
    )
    if bundle["input_refs"]:
        context.register_external_bundle_artifacts(bundle_ref)
    for ref in bundle["input_refs"] + [bundle[key] for key in ("closure_ref", "producer_ref", "source_bundle_ref")]:
        context.remember_historical(ref)
    prior = context.read_historical_json(prior_analysis_ref)
    diagnostic = context.read_historical_json(diagnostic_ref)
    validate_closed_dual_inputs(prior, diagnostic)
    for ref in prior["source_refs_unchanged"] + diagnostic["source_refs_unchanged"]:
        context.remember_historical(ref)
    historical_config_ref = (
        dict(path=str(context.root / OLD_CONFIG_PATH), sha256=CONFIG_SHA)
        if historical_config_ref is None
        else historical_config_ref
    )
    original_config = context.read_historical_json(historical_config_ref)
    config_ref = context.bind_current_source(config_path)
    config = context.read_current_json(config_ref)
    require(
        type(config["max_stages"]) is int
        and config["max_stages"] == 60
        and config == original_config
        and config["max_responses"] == 122
        and context.path(config_ref["path"]).read_bytes() == context.historical_bytes(historical_config_ref),
        "All numeric gates and both upper budgets exactly match the closed whole-objective baseline",
    )
    for path in (*support_paths, *current_source_paths):
        context.bind_current_source(path)
    context.finish()
    return bundle, config, context


def validate_historical_registration(context, registration, *, config, old_config_ref, old_plan_ref):
    """Preserve original declarations while comparing original configuration bytes."""
    require(
        registration["config_ref"] == old_config_ref and registration["plan_ref"] == old_plan_ref,
        "Original declared config/plan identities",
    )
    original = context.read_historical_json(old_config_ref)
    context.remember_historical(old_plan_ref)
    require(
        original == config and registration["config"] == config, "Historical and current numeric contract unchanged"
    )


def model_registration(original_registration, context, bundle, *, current_source_paths=MODEL_PROOF_CURRENT_SOURCES):
    """Keep the original asset proof exact; record current source identity apart.

    The saved compilation is historical evidence. Verifying its asset bytes and
    current source locations does not compile or qualify a migrated execution.
    """
    from s2a.infrastructure.snapshots.xml_asset_snapshot import verify_asset_snapshot

    def input_ref(relative):
        declared = str(context.root / relative)
        found = [ref for ref in bundle["input_refs"] if ref["path"] == declared]
        require(len(found) == 1, "One exact compact-bundle asset input reference required")
        return context.remember_historical(found[0])

    snapshot_ref = input_ref(SNAPSHOT_ROOT + "/manifest.json")
    compile_ref = input_ref(SNAPSHOT_ROOT + "/compile.json")
    original_xml_ref = input_ref(ORIGINAL_XML_PATH)
    history_ref = input_ref(HISTORICAL_ASSET_MANIFEST_PATH)
    recorded = context.read_historical_json(snapshot_ref)
    for key in ("original_xml_ref", "original_xml_snapshot_ref", "migrated_xml_ref", "historical_manifest_ref"):
        context.remember_historical(recorded[key])
    for asset in recorded["assets"]:
        context.remember_historical(asset["safe_source_ref"])
        context.remember_historical(asset["snapshot_ref"])
    snapshot = verify_asset_snapshot(snapshot_ref)
    require(
        snapshot["original_xml_ref"] == original_xml_ref
        and snapshot["historical_manifest_ref"] == history_ref
        and snapshot["file_reference_count"] == 99
        and snapshot["unique_asset_count"] == len(snapshot["assets"]) == 96
        and snapshot["byte_exact_inverse"] is True,
        "The same full 99-reference model and all 96 historically fingerprinted assets",
    )
    compiled = context.read_historical_json(compile_ref)
    require(
        compiled["complete"] is compiled["passed"] is True
        and compiled["manifest_ref"] == snapshot_ref
        and compiled["runtime"]["mujoco"] == "2.3.7"
        and compiled["runtime"]["numpy"] == "1.26.4"
        and compiled["runtime"]["python"].startswith("3.11.")
        and compiled["environment"] == CPU_NATIVE_ENVIRONMENT
        and compiled["model_dimensions"]["nq"] > 0
        and compiled["model_dimensions"]["nv"] > 0
        and compiled["timestep"] == 0.002
        and all(
            compiled[key] == 0
            for key in (
                "response_evaluations",
                "query_count",
                "numerical_inner_steps",
                "physics_steps",
                "world_model_calls",
                "fresh_cases",
            )
        ),
        "Complete native model compilation without a trajectory, forward query or physical step",
    )
    expected_sources = [context.historical_reference(path) for path in ORIGINAL_REPAIR_SOURCE_PATHS]
    require(
        compiled["source_refs"] == expected_sources, "Compilation retains its exact original repair source identities"
    )
    for ref in compiled["source_refs"]:
        context.remember_historical(ref)
    require(original_registration["xml_ref"] == original_xml_ref, "Same original causal full model")
    proof = dict(
        snapshot_ref=snapshot_ref,
        compile_ref=compile_ref,
        original_xml_ref=original_xml_ref,
        historical_manifest_ref=history_ref,
        migrated_xml_ref=snapshot["migrated_xml_ref"],
    )
    require(
        all(context.runtime_refs.get(ref["path"]) == ref for ref in proof.values()),
        "Full original asset proof admitted",
    )
    proof.update(asset_count=96, complete_model_preserved=True, model_body_or_geom_pruning=False)
    current_refs = [context.bind_current_source(path) for path in current_source_paths]
    receipt = dict(
        snapshot_ref=copy.deepcopy(snapshot_ref),
        compile_ref=copy.deepcopy(compile_ref),
        historical_compile_source_refs=copy.deepcopy(compiled["source_refs"]),
        historical_compile_identity_scope="byte_identity_only",
        current_source_refs=current_refs,
        current_source_identity_scope="current_physical_bytes_only",
        historical_compilation_qualification_migrated=False,
        migrated_execution_qualified=False,
        new_model_compilations=0,
        new_forward_queries=0,
        previous_failed_checker_qualification_changed=False,
    )
    if receipt not in context._model_snapshot_identities:
        context._model_snapshot_identities.append(receipt)
    context.finish()
    return dict(original_registration, xml_ref=copy.deepcopy(snapshot["migrated_xml_ref"])), proof
