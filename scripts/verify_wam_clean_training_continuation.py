"""Independent041 root-only metadata and isolated execution admission.

Only metadata JSON, source bytes, opaque file hashes and OS observations are
read. The immutable040 independent verifier supplies its unchanged numerical,
model, control, resource and QA checks; no producer or bundle builder is imported.
Neither phase launches an entrypoint or decodes a future array.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import tokenize
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERIFIER = Path(__file__).resolve()
OLD_ROOT = Path("/share/project/eai_pwm/home/hcr/repos/s2a")
ORIGINAL_ROOT_REL = "outputs/wam_stability_20261021/clean_residual_training_20261023"
NEW_ROOT_REL = ORIGINAL_ROOT_REL + "/recovery_20261023"
BUNDLE_REL = NEW_ROOT_REL + "/source/entrypoints"
CONFIG_REL = "configs/wam_clean_training_continuation_20261023.json"
PLAN_REL = "plans/v0.0.1/041_clean_training_continuation_20261023.md"
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
GPU_PCI = "00000000:BA:00.0"
GPU_MINOR = 6
THREADS = dict(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
ORIGINAL_PREPARED_SHA = "b55c214ffec328ee94f2074007c15dfe62486512aeb660b251fdb21f38e970f7"
ORIGINAL_METADATA_SHA = "eb51208123f6a3bd3f63fef19d44d37de62c52459431cbfcd7f27dbb630b8570"
ORIGINAL_EXECUTION_SHA = "7f4d099f65e6bd98fa671f1d76e35fe9f68bc8fc411718e67d0590422271033b"
ORIGINAL_VERIFICATION_SHA = "702033dd0271897affc372e5c95b0e75a23a623916195b682f78e0ef77d20267"
SEED_DOMAIN = "s2a.clean-training-request-seed31.v1"
STAGES = ("D0", "W1", "D1", "W2", "STATE", "DEVELOPMENT")
TIMEOUTS = dict(D0=88400, W1=15000, D1=88400, W2=15000, STATE=45500, DEVELOPMENT=175000)
AGGREGATE = dict(
    original_complete_requests=20,
    original_partial_requests=1,
    original_saved_denoise=725,
    original_denoise_lower_bound=725,
    original_denoise_upper_bound=735,
    original_model_initializations=1,
    new_request_attempts=1599,
    new_denoise=55965,
    new_model_initializations=4,
    total_request_attempts_upper_bound=1620,
    total_denoise_upper_bound=56700,
    total_model_initializations_upper_bound=5,
    producer_fits=2,
    independent_qr_rebuilds=2,
    physics_steps=7200,
)
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
EXTRA_SOURCES = (
    "configs/wam_clean_training_20261023.json",
    "plans/v0.0.1/038_clean_residual_training_20261023.md",
    "configs/wam_clean_training_execution_20261023.json",
    "plans/v0.0.1/040_clean_training_execution_20261023.md",
    CONFIG_REL,
    PLAN_REL,
    "pyproject.toml",
)
NEW_TESTS = (
    "tests/test_clean_training_continuation_bundle.py",
    "tests/test_durable_stage_supervisor.py",
    "tests/test_clean_training_continuation_verification.py",
    "tests/test_clean_training_continuation_preparation.py",
)
NO_METADATA_WORK = (
    "new_model_calls",
    "new_fit_calls",
    "new_physics_steps",
    "new_fresh_cases",
    "future_label_archives_decoded",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exact(actual, expected):
    return canonical(actual) == canonical(expected)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    declared = Path(path)
    require(declared.is_absolute() and ".." not in declared.parts, "absolute safe path required")
    if declared.is_relative_to(OLD_ROOT):
        relative = declared.relative_to(OLD_ROOT)
        candidates = [REPO / relative]
        if relative.parts[:2] == ("outputs", "wam_stability_20261021"):
            candidates.insert(0, REPO / "outputs/external_deletion_recovery_20261022" / Path(*relative.parts[2:]))
    else:
        candidates = [declared]
    for path in candidates:
        actual = path.resolve()
        require(not actual.is_relative_to(OLD_ROOT), "deleted historical root cannot be consumed")
        if actual.is_file():
            return actual
    raise FileNotFoundError(f"safe metadata/source artifact is missing: {declared}")


class Evidence:
    def __init__(self):
        self.consumed = {}

    def check(self, ref):
        require(
            isinstance(ref, dict)
            and set(ref) == {"path", "sha256"}
            and isinstance(ref["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "exact path/SHA256 reference required",
        )
        path = resolve(ref["path"])
        require(sha(path) == ref["sha256"], f"artifact SHA differs: {path}")
        previous = self.consumed.setdefault(str(path), ref["sha256"])
        require(previous == ref["sha256"], "one actual file has conflicting identities")
        return path

    def json(self, ref):
        path = self.check(ref)
        require(path.suffix == ".json", "JSON decoder refuses an opaque archive/future file")
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == ref["sha256"], "JSON changed while reading")
        return json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def mapping(self, mapping):
        require(isinstance(mapping, dict) and mapping, "nonempty file closure required")
        for path, digest in mapping.items():
            self.check(dict(path=path, sha256=digest))

    def unchanged(self):
        require(all(sha(p) == digest for p, digest in self.consumed.items()), "consumed files changed during audit")


def local_ref(path):
    path = Path(path).absolute()
    return dict(path=str(path), sha256=sha(path))


def root_only_preparation(original, derived):
    expected = copy.deepcopy(original)
    require(original.get("root") == str(REPO / ORIGINAL_ROOT_REL), "original038 root differs")
    expected["root"] = str(REPO / NEW_ROOT_REL)
    require(exact(derived, expected), "derived preparation must change only root, with all types preserved")
    require(
        derived.get("schema_version") == "s2a.clean-training-preparation.v1"
        and derived.get("execution_authorized") is False
        and derived.get("readiness", {}).get("ready_for_execution") is False,
        "derived metadata cannot grant execution",
    )
    return dict(root_only_derivation_exact=True, changed_prepared_fields=["root"])


def continuation_contract(config):
    return dict(
        protocol="041",
        original_execution_ref=config["original_execution_ref"],
        incident_ref=config["incident_ref"],
        old_data_used_for_fit=False,
        one_new_complete_attempt=True,
        automatic_retry=False,
    )


def supervisor_contract():
    return dict(
        python=str(REPO / ".venv/bin/python"),
        stage_order=list(STAGES),
        stage_commands={
            stage: dict(
                entrypoint=BUNDLE_REL
                + "/"
                + (
                    "run_wam_clean_training.py"
                    if stage in ("D0", "W1", "D1", "W2")
                    else "run_wam_clean_training_state.py"
                    if stage == "STATE"
                    else "run_wam_clean_training_development.py"
                ),
                arguments=["--stage", stage] if stage in ("D0", "W1", "D1", "W2") else [],
            )
            for stage in STAGES
        },
        stage_timeouts_s=dict(TIMEOUTS),
        result_paths={stage: f"orchestration/{stage}/result.json" for stage in STAGES},
    )


def seed_plan(new_plan, old_plan, prepared, new_prepared_ref, original_prepared_ref):
    expected = copy.deepcopy(old_plan)
    require(old_plan.get("prepared_ref") == original_prepared_ref, "original seed plan preparation differs")
    expected["prepared_ref"] = new_prepared_ref
    require(exact(new_plan, expected), "new seed plan may only rebind prepared_ref; all300 original seeds stay exact")
    require(
        len(new_plan["entries"]) == len(prepared["training_requests"]) == 300
        and new_plan.get("evaluation_seeds") == [42, 43]
        and new_plan.get("actual_seed_bits") == 31
        and new_plan.get("original_seed_bits") == 63
        and all(type(new_plan.get(k)) is int and new_plan[k] == 0 for k in ("model_calls", "fits", "physics_steps")),
        "full300 seed plan identity/budget differs",
    )
    seen = set()
    for index, (request, entry) in enumerate(zip(prepared["training_requests"], new_plan["entries"], strict=True)):
        original = request["sampling_seed"]
        require(type(original) is int and 0 <= original < 2**63, "original63 seed required")
        payload = json.dumps(
            [SEED_DOMAIN, request["anchor_name"], request["root_seed"], original],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(payload).digest()
        actual = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
        expected_entry = dict(
            index=index,
            name=request["name"],
            anchor_name=request["anchor_name"],
            root_seed=request["root_seed"],
            original_sampling_seed=original,
            sampling_seed=actual,
            payload_sha256=digest.hex(),
        )
        require(exact(entry, expected_entry), "seed31 independent derivation or identity differs")
        require(actual not in seen and actual not in (42, 43), "seed collision/evaluation overlap; no retry")
        seen.add(actual)
    return dict(original300_actual_seeds_exact=True, seed31_independently_rebuilt=True)


def finite_bundle_source(original_text, relative):
    """Independently rebuild the token-limited patch, preserving every other byte."""
    strings = {"scripts/" + name: BUNDLE_REL + "/" + name for name in SCRIPT_NAMES}
    strings[RUNTIME_REL] = BUNDLE_REL + "/continuation_runtime.py"
    strings[ORIGINAL_ROOT_REL] = NEW_ROOT_REL
    lines = original_text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(pos):
        return starts[pos[0] - 1] + pos[1]

    edits = []

    def add(kind, start, end, old, new):
        begin = offset(start)
        require(original_text[begin : offset(end)] == old, "independent edit span differs")
        edits.append(dict(kind=kind, old=old, new=new, count=1, start=list(start), end=list(end), offset=begin))

    tokens = list(tokenize.generate_tokens(io.StringIO(original_text).readline))
    for token in tokens:
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str) and value in strings:
                require(token.string in (repr(value), '"' + value + '"'), "ordinary exact path literal required")
                add(
                    "quoted_path",
                    token.start,
                    token.end,
                    token.string,
                    token.string[0] + strings[value] + token.string[-1],
                )
    significant = [t for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL)]
    for index in range(len(significant) - 4):
        group = significant[index : index + 5]
        if [t.string for t in group] == ["from", "s2a", ".", "clean_training_runtime", "import"]:
            add(
                "runtime_import",
                group[0].start,
                group[-1].end,
                "from s2a.clean_training_runtime import",
                "from continuation_runtime import",
            )
    tree = ast.parse(original_text)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and ast.get_source_segment(original_text, node) == "REPO = Path(__file__).resolve().parents[1]"
        ):
            require(
                relative
                in ("scripts/verify_wam_clean_training_capture.py", "scripts/verify_wam_clean_training_development.py"),
                "unregistered local REPO guard",
            )
            add(
                "repository_guard",
                (node.lineno, node.col_offset),
                (node.end_lineno, node.end_col_offset),
                "REPO = Path(__file__).resolve().parents[1]",
                "REPO = Path(" + json.dumps(str(REPO)) + ")",
            )
    transformed = original_text
    edits.sort(key=lambda e: e["offset"])
    for a, b in zip(edits, edits[1:]):
        require(a["offset"] + len(a["old"]) <= b["offset"], "finite edits overlap")
    for edit in reversed(edits):
        begin = edit["offset"]
        transformed = transformed[:begin] + edit["new"] + transformed[begin + len(edit["old"]) :]
    ast.parse(transformed)
    return transformed, edits


def bundle(manifest_ref, original, config, evidence):
    value = evidence.json(manifest_ref)
    require(
        value.get("schema_version") == "s2a.clean-training-continuation-bundle.v1"
        and value.get("complete") is value.get("passed") is True
        and value.get("old_admitted_ref") == config["original_execution_ref"]
        and value.get("original_root") == str(REPO / ORIGINAL_ROOT_REL)
        and value.get("output_root") == str(REPO / NEW_ROOT_REL)
        and value.get("bundle_root") == str(REPO / BUNDLE_REL),
        "isolated bundle identity/status differs",
    )
    originals = ["scripts/" + name for name in SCRIPT_NAMES] + [RUNTIME_REL]
    require(len(value["files"]) == len(originals) == 11, "exact11 bundle files required")
    actual = {}
    for entry, relative in zip(value["files"], originals, strict=True):
        old_path = REPO / relative
        expected_old = dict(path=str(old_path), sha256=original["source_sha256"][str(old_path)])
        require(entry["original_ref"] == expected_old, "bundle original is not040-frozen")
        source = evidence.check(expected_old).read_text()
        target = REPO / BUNDLE_REL / ("continuation_runtime.py" if relative == RUNTIME_REL else old_path.name)
        require(entry["actual_ref"]["path"] == str(target), "bundle target path differs")
        require(target.resolve() == target and not target.is_symlink(), "bundle target must be canonical regular file")
        expected_text, edits = finite_bundle_source(source, relative)
        require(
            evidence.check(entry["actual_ref"]).read_bytes() == expected_text.encode(),
            "bundle changes an unregistered byte or algorithm",
        )
        require(exact(entry["edits"], edits), "bundle edit witness differs from independent token enumeration")
        actual[str(target)] = entry["actual_ref"]["sha256"]
    require(exact(value["source_sha256"], actual), "bundle11 source map differs")
    require(
        value.get("original_sources_before_after_exact") is True
        and value.get("ast_and_source_compiled") is True
        and value.get("entrypoints_executed") == value.get("future_archive_decodes") == 0,
        "bundle mutation/execution boundary differs",
    )
    return value


def configuration(config):
    require(
        config.get("schema_version") == "s2a.clean-training-continuation-config.v1"
        and type(config.get("seed")) is int
        and config["seed"] == 42
        and config.get("output_dir") == NEW_ROOT_REL
        and config.get("plan_path") == PLAN_REL,
        "fixed041 config/root/seed/plan differs",
    )
    for name, suffix, digest in (
        ("original_prepared_ref", "preparation/prepared.json", ORIGINAL_PREPARED_SHA),
        ("original_metadata_verification_ref", "aux/independent_metadata/verified.json", ORIGINAL_METADATA_SHA),
        ("original_execution_ref", "execution/admitted.json", ORIGINAL_EXECUTION_SHA),
        ("original_execution_verification_ref", "execution/verified.json", ORIGINAL_VERIFICATION_SHA),
    ):
        require(
            config.get(name) == dict(path=str(REPO / ORIGINAL_ROOT_REL / suffix), sha256=digest),
            "original qualified reference differs: " + name,
        )
    require(
        exact(config.get("aggregate_budget"), AGGREGATE)
        and exact(config.get("stage_timeouts_s"), TIMEOUTS)
        and config.get("automatic_retry") is False
        and config.get("fresh_performance_authorized") is False
        and config.get("old_data_used_for_fit") is False,
        "fixed cumulative budget or no-retry/no-old-fit/no-fresh contract differs",
    )
    require(
        config.get("disk_required_bytes") == 200 * 1024**3
        and config.get("ram_required_bytes") == 128 * 1024**3
        and config.get("master_port") == 31338
        and config.get("development_port") == 31339
        and config.get("gpu_uuid") == GPU,
        "fixed resource/port/UUID contract differs",
    )


def old_sources(original, evidence):
    require(
        len(original["source_sha256"]) == len(original["snapshot_sha256"]) == len(original["snapshot_refs"]) == 462,
        "original462 source/snapshot roster differs",
    )
    require(set(original["snapshot_refs"]) == set(original["source_sha256"]), "original snapshot keys differ")
    evidence.mapping(original["source_sha256"])
    evidence.mapping(original["snapshot_sha256"])
    for path, digest in original["source_sha256"].items():
        expected = REPO / ORIGINAL_ROOT_REL / "execution/source" / Path(path).relative_to(REPO)
        ref = original["snapshot_refs"][path]
        require(ref == dict(path=str(expected), sha256=digest), "original462 source/snapshot mapping differs")
        evidence.check(ref)
    return dict(original_sources_unchanged=462, original_snapshots_unchanged=462)


def original_proof(config, original, evidence):
    verified = evidence.json(config["original_execution_verification_ref"])
    require(
        original.get("schema_version") == "s2a.clean-training-execution-registration.v1"
        and original.get("complete") is original.get("passed") is original.get("execution_authorized") is True
        and original.get("readiness", {}).get("ready_for_execution") is True
        and original.get("prepared_ref") == config["original_prepared_ref"]
        and original.get("metadata_verification_ref") == config["original_metadata_verification_ref"]
        and original.get("execution_verification_ref") == config["original_execution_verification_ref"],
        "original040 admission chain differs",
    )
    require(
        verified.get("schema_version") == "s2a.clean-training-execution-verification.v1"
        and verified.get("complete") is verified.get("passed") is verified.get("eligible_for_execution") is True
        and verified.get("execution_authorized") is False
        and verified.get("prepared_ref") == config["original_prepared_ref"]
        and verified.get("proposal_ref") == original["proposal_ref"],
        "original independent040 qualification is missing",
    )
    proposal = evidence.json(original["proposal_ref"])
    expected = copy.deepcopy(proposal)
    expected["schema_version"] = "s2a.clean-training-execution-registration.v1"
    expected["readiness"]["ready_for_execution"] = True
    expected["execution_authorized"] = True
    expected["proposal_ref"] = original["proposal_ref"]
    expected["execution_verification_ref"] = config["original_execution_verification_ref"]
    require(exact(original, expected), "original040 admitted envelope differs from its independently approved proposal")
    evidence.check(verified["verifier_ref"])


def legacy_independent(original, evidence):
    """Load only the already-qualified independent040 audit, never a producer."""
    path = REPO / "scripts/verify_wam_clean_training_execution.py"
    ref = dict(path=str(path), sha256=original["source_sha256"][str(path)])
    evidence.check(ref)
    spec = importlib.util.spec_from_file_location("s2a_immutable040_independent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interrupted_inventory(config, incident, evidence):
    inventory = evidence.json(config["interrupted_inventory_ref"])
    require(
        inventory.get("schema_version") == "s2a.clean-training-interrupted-artifact-inventory.v1"
        and inventory.get("old_registration_ref") == config["original_execution_ref"]
        and inventory.get("scope") == ["capture/D0", "orchestration/D0"]
        and inventory.get("source_count") == inventory.get("snapshot_count") == 462
        and inventory.get("source_changes") == inventory.get("snapshot_changes") == []
        and inventory.get("future_arrays_decoded") == 0,
        "interrupted evidence inventory boundary differs",
    )
    expected_files = {}
    for item in inventory["files"]:
        path = Path(item["path"])
        require(
            any(path.is_relative_to(REPO / ORIGINAL_ROOT_REL / folder) for folder in inventory["scope"]),
            "inventory includes an unregistered path",
        )
        require(str(path) not in expected_files, "inventory repeats a path")
        actual = evidence.check(dict(path=str(path), sha256=item["sha256"]))
        stat = actual.stat()
        require(
            stat.st_size == item["bytes"] and stat.st_mtime_ns == item["mtime_ns"],
            "old interrupted artifact stat differs",
        )
        expected_files[str(path)] = item["sha256"]
    observed_files = {
        str(p) for folder in inventory["scope"] for p in (REPO / ORIGINAL_ROOT_REL / folder).rglob("*") if p.is_file()
    }
    require(
        len(expected_files) == inventory["file_count"] == 1661
        and set(expected_files) == observed_files
        and sum(item["bytes"] for item in inventory["files"]) == inventory["total_bytes"],
        "old interrupted files were added, removed, or changed",
    )
    require(
        incident.get("schema_version") == "s2a.clean-training-interruption-audit.v1"
        and incident.get("audit_complete") is incident.get("audit_consistency_passed") is True
        and incident.get("original_stage_complete") is incident.get("original_stage_passed") is False
        and incident.get("source_files_unchanged") == 462,
        "original incomplete incident was substituted or relabeled",
    )
    capture = incident["capture_summary"]
    require(
        capture["complete_ledgers"] == 20
        and capture["complete_saved_points"] == 700
        and capture["partial_saved_points"] == 25
        and capture["total_saved_point_records"] == 725
        and capture["request_directories"] == 21
        and capture["complete_request_indices"] == list(range(20))
        and capture["incomplete_request_indices"] == [20]
        and capture["full_numerical_capture_audit_performed"] is False,
        "original20 complete plus one25-point partial evidence differs",
    )
    accounting = incident["denoise_accounting"]
    require(
        accounting["completed_original_denoise_lower_bound"] == 725
        and accounting["exact_total_boundary_invocations"] is accounting["exact_total_model_calls"] is None,
        "missing exact old boundary counters cannot be assigned a fabricated total",
    )
    conclusions = incident["lifecycle_conclusions"]
    require(
        conclusions["normal_exit_proven"]
        is conclusions["actual_wait_proven"]
        is conclusions["actual_reap_proven"]
        is False
        and conclusions["original_release_receipt_exists"] is False
        and conclusions["exit_code"] is conclusions["termination_signal"] is conclusions["termination_cause"] is None,
        "unknown original lifecycle must remain unknown",
    )
    return dict(
        old_interrupted_files_unchanged=len(expected_files), original_saved_denoise=725, old_data_used_for_fit=False
    )


def process_absence(incident):
    originals = incident["process_observations"][0]["original_processes"]
    expected = {"1292033": 567473266, "1292382": 567478090}
    require(
        {key: item["expected_start_ticks"] for key, item in originals.items()} == expected,
        "original PID/birth identities differ",
    )
    observations = []
    for key, birth in expected.items():
        path = Path("/proc") / key / "stat"
        current = None
        if path.exists():
            raw = path.read_text()
            current = int(raw[raw.rfind(")") + 2 :].split()[19])
        require(current != birth, "original owned PID/birth is still live; do not start a copy")
        observations.append(
            dict(pid=int(key), expected_start_ticks=birth, current_start_ticks=current, original_identity_absent=True)
        )
    return observations


def common(config_ref, prepared_ref, evidence):
    config = evidence.json(config_ref)
    configuration(config)
    original = evidence.json(config["original_execution_ref"])
    checks = old_sources(original, evidence)
    original_proof(config, original, evidence)
    old_prepared = evidence.json(config["original_prepared_ref"])
    prepared = evidence.json(prepared_ref)
    checks.update(root_only_preparation(old_prepared, prepared))
    require(
        prepared_ref["path"] == str(REPO / NEW_ROOT_REL / "preparation/prepared.json"), "new preparation path differs"
    )
    incident = evidence.json(config["incident_ref"])
    checks.update(interrupted_inventory(config, incident, evidence))
    evidence.json(config["recovery_design_ref"])
    require(
        config["development_control_ref"] == original["development_control_ref"], "qualified control reference changed"
    )
    original_metadata = evidence.json(config["original_metadata_verification_ref"])
    require(original_metadata["prepared_ref"] == config["original_prepared_ref"], "original metadata binding differs")
    evidence.mapping(original_metadata["consumed_sha256"])
    checker = legacy_independent(original, evidence)
    require(config["required_test_paths"] == [*checker.REQUIRED_TESTS, *NEW_TESTS], "fixed23 test modules differ")
    checker.metadata_and_model(original, old_prepared, evidence)
    for key in ("source_sha256", "snapshot_sha256", "input_sha256", "qualified_model_source_sha256"):
        evidence.mapping(prepared[key])
    manifest_ref = local_ref(REPO / NEW_ROOT_REL / "source/bundle_manifest.json")
    manifest = bundle(manifest_ref, original, config, evidence)
    observations = process_absence(incident)
    observed_gpu = checker.gpu_observation()
    require(
        observed_gpu
        == dict(uuid=GPU, pci_bus_id=GPU_PCI, device_minor=GPU_MINOR, memory_mib=0, utilization_gpu=0, compute_apps=[]),
        "fixed allowed GPU is occupied or has a different driver identity",
    )
    checks.update(
        bundle_files_independently_verified=11, original_process_observations=observations, actual_gpu=observed_gpu
    )
    return config, original, prepared, manifest_ref, manifest, checker, checks


def metadata_audit(config_ref, prepared_ref):
    evidence = Evidence()
    config, _, prepared, manifest_ref, _, _, checks = common(config_ref, prepared_ref, evidence)
    evidence.check(local_ref(VERIFIER))
    evidence.unchanged()
    return dict(
        prepared_ref=prepared_ref,
        original_prepared_ref=config["original_prepared_ref"],
        original_metadata_verification_ref=config["original_metadata_verification_ref"],
        bundle_manifest_ref=manifest_ref,
        original_execution_ref=config["original_execution_ref"],
        incident_ref=config["incident_ref"],
        derived_metadata_review=True,
        original_metadata_audit_reexecuted=False,
        checks=checks,
        consumed_sha256=evidence.consumed,
        consumed_before_after_exact=True,
        weight_stat_files=len(prepared["weight_files"]),
        full_weight_content_rehash=False,
    )


def execution_sources(proposal, manifest, original, evidence):
    mapping = proposal["source_sha256"]
    main = {str(p.resolve()) for folder in ("src", "scripts", "tests") for p in (REPO / folder).rglob("*.py")}
    main.update(str(REPO / relative) for relative in EXTRA_SOURCES)
    bundled = set(manifest["source_sha256"])
    require(set(mapping) == main | bundled, "complete main Python/extra7/bundle11 source closure differs")
    require(str(VERIFIER) in mapping, "independent041 verifier is not frozen")
    for relative in (
        "src/s2a/clean_training_continuation_bundle.py",
        "src/s2a/durable_stage_supervisor.py",
        "scripts/run_wam_clean_training_continuation.py",
        "scripts/prepare_wam_clean_training_continuation.py",
        *NEW_TESTS,
    ):
        require(str(REPO / relative) in mapping, "required041 source/test is absent: " + relative)
    require(
        all(mapping.get(path) == digest for path, digest in original["source_sha256"].items())
        and all(mapping.get(path) == digest for path, digest in manifest["source_sha256"].items()),
        "old462 or isolated bundle source changed",
    )
    evidence.mapping(mapping)
    snapshots = proposal["snapshot_refs"]
    require(set(snapshots) == set(mapping), "source/snapshot one-to-one roster differs")
    expected_map = {}
    for path, digest in mapping.items():
        source = Path(path)
        require(source.resolve() == source and not source.is_symlink(), "source path is noncanonical or symlinked")
        if path in main:
            require(
                source.is_relative_to(REPO) and not source.is_relative_to(REPO / "outputs"),
                "main executable source escapes repository",
            )
            relative = Path("repo") / source.relative_to(REPO)
        else:
            require(source.parent == REPO / BUNDLE_REL, "unregistered output Python source")
            relative = Path("bundle") / source.name
        target = REPO / NEW_ROOT_REL / "execution/source" / relative
        ref = dict(path=str(target), sha256=digest)
        require(
            snapshots[path] == ref and str(target) not in expected_map, "source snapshot path/SHA is not uniquely bound"
        )
        require(evidence.check(ref) == target, "snapshot resolves to a different physical file")
        expected_map[str(target)] = digest
    require(exact(proposal["snapshot_sha256"], expected_map), "snapshot map omits/adds/changes a source")
    return dict(final_sources=len(mapping), source_snapshots=len(snapshots), isolated_bundle_sources=11)


def execution_contract(proposal, config, prepared, prepared_ref, manifest_ref, original, checker):
    require(
        proposal.get("schema_version") == "s2a.clean-training-execution-proposal.v1"
        and proposal.get("complete") is proposal.get("passed") is True
        and proposal.get("execution_authorized") is False
        and proposal.get("readiness", {}).get("ready_for_execution") is False
        and proposal.get("automatic_retry") is False
        and proposal.get("fresh_performance_authorized") is False,
        "new proposal must remain complete but unadmitted; retry/fresh forbidden",
    )
    require(
        proposal.get("root") == prepared["root"] == str(REPO / NEW_ROOT_REL)
        and proposal.get("prepared_ref") == prepared_ref
        and proposal.get("original_execution_ref") == config["original_execution_ref"]
        and proposal.get("incident_ref") == config["incident_ref"]
        and proposal.get("bundle_manifest_ref") == manifest_ref
        and proposal.get("development_control_ref") == config["development_control_ref"],
        "new execution provenance/root differs",
    )
    require(
        exact(proposal.get("budget"), prepared["budget"])
        and exact(proposal["budget"], checker.BUDGET)
        and exact(proposal.get("resources"), prepared["resources"])
        and exact(proposal["resources"], checker.RESOURCES)
        and exact(proposal.get("aggregate_budget"), config["aggregate_budget"])
        and exact(proposal["aggregate_budget"], AGGREGATE),
        "new full budget/resources or cross-run cumulative accounting differs",
    )
    require(
        exact(proposal.get("continuation_contract"), continuation_contract(config))
        and exact(proposal.get("supervisor"), supervisor_contract()),
        "single-attempt/no-old-fit continuation or exact durable supervisor contract differs",
    )
    entrypoints = {
        stage: (
            BUNDLE_REL + "/collect_wam_clean_training.py"
            if stage in ("D0", "D1")
            else "scripts/fit_wam_clean_training.py"
            if stage in ("W1", "W2")
            else BUNDLE_REL + "/sample_wam_clean_training_state.py"
            if stage == "STATE"
            else BUNDLE_REL + "/run_wam_clean_training_development.py"
        )
        for stage in STAGES
    }
    require(exact(proposal.get("stage_entrypoints"), entrypoints), "fixed stage entrypoint mapping differs")
    require(set(proposal["entrypoint_refs"]) == set(STAGES), "six stage entrypoint refs required")
    for stage, relative in entrypoints.items():
        path = str(REPO / relative)
        require(
            proposal["entrypoint_refs"][stage] == dict(path=path, sha256=proposal["source_sha256"][path]),
            "stage entrypoint is not bound to the actual registered source",
        )
    require(
        proposal.get("master_port") == 31338
        and exact(proposal.get("development"), dict(port=31339, case_order=checker.case_order()))
        and proposal.get("sampler_trace_ref") == original["sampler_trace_ref"],
        "fixed development cases/ports or qualified actual sampler grid differs",
    )
    for key in ("new_model_calls", "new_fits", "physics_steps", "fresh_cases", "future_archive_decodes"):
        require(type(proposal.get(key)) is int and proposal[key] == 0, "unadmitted proposal consumed execution/data")


def supervisor_source_contract(source):
    """Check the registered launch/run interface and the single ordered owner loop."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require(
        {"main", "execute", "build_stage_argv", "check_owner_result"} <= functions.keys(),
        "supervisor entrypoint functions are missing",
    )
    main = functions["main"]
    groups = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_mutually_exclusive_group"
    ]
    require(
        len(groups) == 1
        and any(
            item.arg == "required" and isinstance(item.value, ast.Constant) and item.value.value is True
            for item in groups[0].keywords
        ),
        "supervisor must require exactly one launch/run phase",
    )
    mode_flags = [
        node.args[0].value
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mode"
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    require(mode_flags == ["--launch", "--run"], "supervisor launch/run flags differ")
    execute = functions["execute"]
    loops = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "stage"
    ]
    require(
        len(loops) == 1
        and isinstance(loops[0].iter, ast.Name)
        and loops[0].iter.id == "STAGES"
        and not any(isinstance(node, ast.While) for node in ast.walk(execute)),
        "supervisor must use one six-stage loop without a restart loop",
    )
    owned = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "owned_stage"
    ]
    require(
        len(owned) == 1 and any(item.arg == "check_result" for item in owned[0].keywords),
        "each stage needs one real owner and its result gate",
    )
    return dict(supervisor_launch_run_interface_checked=True, supervisor_single_ordered_owner_loop_checked=True)


def execution_audit(config_ref, prepared_ref, proposal_ref):
    evidence = Evidence()
    config, original, prepared, manifest_ref, manifest, checker, checks = common(config_ref, prepared_ref, evidence)
    proposal = evidence.json(proposal_ref)
    require(proposal_ref["path"] == str(REPO / NEW_ROOT_REL / "execution/proposal.json"), "new proposal path differs")
    require(proposal.get("config_ref") == config_ref, "proposal does not bind exact041 config")
    execution_contract(proposal, config, prepared, prepared_ref, manifest_ref, original, checker)
    checks.update(execution_sources(proposal, manifest, original, evidence))
    supervisor_path = str(REPO / "scripts/run_wam_clean_training_continuation.py")
    supervisor_source = evidence.check(
        dict(path=supervisor_path, sha256=proposal["source_sha256"][supervisor_path])
    ).read_text()
    checks.update(supervisor_source_contract(supervisor_source))
    evidence.mapping(proposal["input_sha256"])
    metadata_ref = proposal["metadata_verification_ref"]
    require(
        metadata_ref["path"] == str(REPO / NEW_ROOT_REL / "metadata_verified.json"),
        "derived metadata receipt path differs",
    )
    metadata = checker.independent_metadata(proposal, evidence)
    require(
        metadata.get("original_prepared_ref") == config["original_prepared_ref"]
        and metadata.get("original_metadata_verification_ref") == config["original_metadata_verification_ref"]
        and metadata.get("bundle_manifest_ref") == manifest_ref
        and metadata.get("derived_metadata_review") is True
        and metadata.get("original_metadata_audit_reexecuted") is False,
        "new root-only metadata qualification was substituted for a historical rerun claim",
    )
    old_seed_plan = evidence.json(original["training_seed_plan_ref"])
    new_seed_plan = evidence.json(proposal["training_seed_plan_ref"])
    require(
        proposal["training_seed_plan_ref"]["path"] == str(REPO / NEW_ROOT_REL / "execution/seed_plan.json"),
        "new bound seed plan path differs",
    )
    checks.update(seed_plan(new_seed_plan, old_seed_plan, prepared, prepared_ref, config["original_prepared_ref"]))
    checks.update(checker.training_seed_plan(proposal, prepared, evidence))
    bridge = checker.metadata_and_model(proposal, prepared, evidence)
    checks.update(checker.normalization(proposal, bridge, evidence))
    checks.update(checker.sampler(proposal, prepared, evidence))
    checks.update(checker.control(proposal, evidence))
    checks.update(checker.resources(proposal, prepared, evidence))
    qa = evidence.json(proposal["qa_ref"])
    require(exact(qa["test_paths"], config["required_test_paths"]), "final actual23-module CPU QA roster differs")
    checks.update(checker.qa(proposal, evidence))
    require(
        proposal["plan_ref"] == dict(path=str(REPO / PLAN_REL), sha256=proposal["source_sha256"][str(REPO / PLAN_REL)])
        and proposal["source_sha256"].get(config_ref["path"]) == config_ref["sha256"],
        "041 config/plan is not exactly source-frozen",
    )
    evidence.check(proposal["plan_ref"])
    closure = {
        **proposal["source_sha256"],
        **proposal["snapshot_sha256"],
        **proposal["input_sha256"],
        proposal_ref["path"]: proposal_ref["sha256"],
    }
    require(
        all(closure.get(path) == digest for path, digest in evidence.consumed.items()),
        "execution closure omits consumed independent evidence/source/input",
    )
    evidence.unchanged()
    checks.update(
        aggregate_budget_exact=True,
        old_partial_capture_excluded_from_training=True,
        one_new_complete_attempt=True,
        stage_command_specs_exact=True,
        real_d0_dry_run_required_after_admission_before_model=True,
        real_d0_dry_run_performed_by_this_audit=False,
    )
    return dict(
        prepared_ref=prepared_ref,
        original_execution_ref=config["original_execution_ref"],
        incident_ref=config["incident_ref"],
        bundle_manifest_ref=manifest_ref,
        training_seed_plan_ref=proposal["training_seed_plan_ref"],
        checks=checks,
        input_sha256=evidence.consumed,
        consumed_before_after_exact=True,
        weight_stat_files=163,
        full_weight_content_rehash=False,
    )


def verify(args):
    require(
        args.seed == 42
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
        and all(os.environ.get(name) == value for name, value in THREADS.items())
        and Path.cwd().resolve() == REPO,
        "041 independent audit requires S2A cwd, seed42, empty CUDA/LD and three thread values8",
    )
    output = Path(args.output).absolute()
    expected = (
        REPO / NEW_ROOT_REL / ("metadata_verified.json" if args.stage == "metadata" else "execution/verified.json")
    )
    require(
        output == expected and output.resolve() == output and not output.exists(),
        "new fixed audit output cannot overwrite evidence or escape041",
    )
    config_ref = dict(path=str(Path(args.config).absolute()), sha256=args.config_sha256)
    require(config_ref["path"] == str(REPO / CONFIG_REL), "fixed041 configuration path required")
    prepared_ref = dict(path=str(Path(args.prepared).absolute()), sha256=args.prepared_sha256)
    metadata = args.stage == "metadata"
    result = dict(
        schema_version=(
            "s2a.clean-training-preparation-independent.v1"
            if metadata
            else "s2a.clean-training-execution-verification.v1"
        ),
        complete=False,
        passed=False,
        prepared_ref=prepared_ref,
        config_ref=config_ref,
        verifier_ref=local_ref(VERIFIER),
        execution_authorized=False,
        fresh_performance_authorized=False,
        new_model_initializations=0,
        **{key: 0 for key in NO_METADATA_WORK},
        new_fits=0,
        derived_metadata_review=metadata,
        audit_observed_utc=datetime.now(timezone.utc).isoformat(),
    )
    if metadata:
        result.update(metadata_qualified=False, execution_ready=False)
    else:
        result.update(eligible_for_execution=False)
    try:
        if metadata:
            require(
                args.proposal is None and args.proposal_sha256 is None,
                "metadata phase must not consume an execution proposal",
            )
            result.update(metadata_audit(config_ref, prepared_ref))
            result.update(complete=True, passed=True, metadata_qualified=True)
        else:
            require(
                args.proposal is not None and args.proposal_sha256 is not None,
                "execution phase requires exact proposal path/SHA",
            )
            proposal_ref = dict(path=str(Path(args.proposal).absolute()), sha256=args.proposal_sha256)
            result["proposal_ref"] = proposal_ref
            result.update(execution_audit(config_ref, prepared_ref, proposal_ref))
            result.update(complete=True, passed=True, eligible_for_execution=True)
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("metadata", "execution"), required=True)
    for name in ("config", "prepared"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--proposal-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    result = verify(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("schema_version", "complete", "passed", "prepared_ref")}))


if __name__ == "__main__":
    main()
