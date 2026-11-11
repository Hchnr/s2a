"""Load reviewed initial states without importing a simulator before configuration."""

import hashlib
import json
from pathlib import Path

import numpy as np

from s2a.research.native_runtime.development_prefix_replay import map_record_xml
from s2a.research.native_runtime.reset_endpoint import saved_reset_endpoint, verify_reset_endpoint
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
PLAN = ROOT / "plans/v0.0.1/tracking/joint_control/reset_native_admission_20261111.md"
SOURCE_PREFLIGHT = {
    "path": str(ROOT / "outputs/reset_start_source_preflight_20261111/result.json"),
    "sha256": "acb379f3bcac659339cecc45eb2045285c686f1079acee0cdac5ea5380b2ddd4",
}


def validate_case_identity(index, case, manifest):
    """Keep the full preselected case roster, task and sampling seed aligned."""
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError("One of twelve registered development cases is required")
    seed = (42, 42, 43, 43)[index % 4]
    variant = ("raw", "terminal058", "raw", "terminal058")[index % 4]
    name, task = f"task{index // 4}_init0_s{seed}_{variant}", (8, 9, 6)[index // 4]
    if (
        case["case_index"] != index
        or case["case_id"] != name
        or case["sampling_seed"] != seed
        or case["variant"] != variant
        or manifest["record"]["task_id"] != task
        or manifest["runtime_config"]["task_id"] != task
        or manifest["runtime_config"]["seed"] != seed
    ):
        raise ValueError("The original case, task, seed and variant must agree")


def reviewed_case(index, prior, review):
    """Read only the original files named by both prior independent reviews."""
    entry, reviewed = prior["cases"][index], review["cases"][index]
    hashes, source_map = entry["original_source_sha256"], review["source_and_artifact_sha256"]
    prefix.require(entry["case"] == reviewed["case"] and len(hashes) == 7, "One exact reviewed source roster required")
    for path, digest in hashes.items():
        prefix.require(source_map.get(path) == digest, "Every source must retain the original reviewed identity")
        prefix.checked({"path": path, "sha256": digest})
    case_dir = (
        ROOT / "outputs/tracking_preview_wam_development_v4_20261027/development/cases" / entry["case"]["case_id"]
    )
    worker_path = case_dir / "worker_result.json"
    prefix.require(str(worker_path) in hashes, "The original worker path must match its case")
    worker = json.loads(worker_path.read_bytes())
    prefix.require(
        worker["complete"] is worker["passed"] is True and worker["physics_steps"] == 600, "Completed worker required"
    )
    folder = Path(worker["physical_case_dir"])
    prefix.require(
        folder.is_relative_to(case_dir) and str(folder) == reviewed["mechanical"]["case"],
        "Reviewed physical folder required",
    )
    files = {
        name: folder / filename
        for name, filename in (
            ("manifest", "manifest.json"),
            ("xml", "model.xml"),
            ("chunks", "chunks.json"),
            ("controller", "controller.json"),
        )
    }
    prefix.require(all(str(path) in hashes for path in files.values()), "All original physical files must be reviewed")
    manifest = json.loads(files["manifest"].read_bytes())
    validate_case_identity(index, entry["case"], manifest)
    tool_path = Path(manifest["runtime_config"]["tool_contract"])
    arrays_ref = worker["arrays_ref"]
    prefix.require(
        str(tool_path) in hashes
        and arrays_ref["path"] in hashes
        and hashes[arrays_ref["path"]] == arrays_ref["sha256"]
        and worker["endpoint_receipts_ref"] == entry["original_receipt_ref"]
        and source_map.get(entry["original_receipt_ref"]["path"]) == entry["original_receipt_ref"]["sha256"],
        "The reviewed tool, saved state array and endpoint receipt must match",
    )
    with prefix.checked(entry["original_receipt_ref"]).open() as stream:
        endpoint_receipt = json.loads(stream.readline())
    with np.load(prefix.checked(arrays_ref), allow_pickle=False) as archive:
        initial = saved_reset_endpoint(archive)
    chunks = json.loads(files["chunks"].read_bytes())
    prefix.require(
        len(chunks) == 75 and [row["step"] for row in chunks] == list(range(0, 600, 8)),
        "All original query times required",
    )
    controller = json.loads(files["controller"].read_bytes())
    tool = np.asarray(json.loads(tool_path.read_bytes())["source_from_tcp"], dtype=float)
    verified = verify_reset_endpoint(initial, endpoint_receipt, initial, endpoint_receipt, chunks[0], controller, tool)
    prefix.require(
        verified["recorded_state_sha256"] == manifest["record"]["initial_state_sha256"],
        "Manifest reset identity required",
    )
    return dict(
        case=entry["case"],
        manifest=manifest,
        expected=initial,
        expected_receipt=endpoint_receipt,
        chunk=chunks[0],
        controller=controller,
        tool=tool,
        original_source_sha256=hashes,
        original_receipt_ref=entry["original_receipt_ref"],
    )


def load_inputs(index=None, *, verify_assets=True):
    if index is not None and (type(index) is not int or not 0 <= index < 12):
        raise ValueError("One registered development case is required")
    prior = prefix.read(SOURCE_PREFLIGHT)
    boot, migrated = (prefix.read(item) for item in (prefix.BOOTSTRAP, prefix.MIGRATED_MANIFEST))
    prefix.require(
        prior["complete"] is prior["passed"] is prior["source_unchanged"] is True
        and prior["control_qualified"] is False
        and len(prior["cases"]) == 12
        and prior["native_models"] == prior["forward_queries"] == prior["physics_steps"] == 0
        and boot["complete"] is boot["passed"] is boot["bootstrap_verified"] is boot["source_unchanged"] is True,
        "The closed reset preflight and native bootstrap are required",
    )
    study_inputs = prior["inputs"]
    prefix.checked(study_inputs["config_ref"])
    for ref in study_inputs["refs"].values():
        prefix.checked(ref)
    prefix.require(len(study_inputs["asset_refs"]) == 96, "The fixed snapshot asset roster is required")
    if verify_assets:
        for ref in study_inputs["asset_refs"]:
            prefix.checked(ref)
    review = prefix.read(study_inputs["refs"]["review"])
    prefix.require(
        review["complete"] is review["passed"] is True and len(review["cases"]) == 12,
        "Complete reviewed cases required",
    )
    model = boot["bootstrap"]["model_records"][2]
    prefix.require(model["stage"] == "record_xml" and model["completed"] is True, "Pinned mapped record XML required")
    mapped_ref = {key: model["xml_ref"][key] for key in ("path", "sha256")}
    mapped_xml = prefix.checked(mapped_ref).read_bytes()
    asset_refs = {}
    for stage in boot["bootstrap"]["model_records"]:
        prefix.require(stage["completed"] is True, "All original model setup stages are required")
        for item in stage["assets"]:
            ref = item["reference"]
            path, digest = ref["path"], ref["sha256"]
            prefix.require(path not in asset_refs or asset_refs[path] == digest, "Conflicting native asset identity")
            asset_refs[path] = digest
    prefix.require(len(asset_refs) == 96, "The native runtime must bind exactly 96 assets")
    if verify_assets:
        for path, digest in asset_refs.items():
            prefix.checked({"path": path, "sha256": digest})
    first = reviewed_case(0, prior, review)
    template_xml = first["manifest"]["record"]["model_xml"].encode()
    prefix.require(
        hashlib.sha256(template_xml).hexdigest() == boot["bootstrap"]["original_record_xml_sha256"],
        "Pinned XML mapping template required",
    )
    cases = []
    for position in range(12) if index is None else (index,):
        item = first if position == 0 else reviewed_case(position, prior, review)
        manifest = item.pop("manifest")
        config = dict(manifest["runtime_config"])
        for key in ("libero_root", "libero_config", "osmesa_library"):
            config[key] = migrated["runtime_config"][key]
            prefix.require(
                Path(config[key]).is_absolute() and Path(config[key]).is_relative_to(ROOT), "Safe runtime path"
            )
        record_xml = map_record_xml(manifest["record"]["model_xml"].encode(), template_xml, mapped_xml).decode()
        item.update(config=config, record=manifest["record"], record_xml=record_xml)
        cases.append(item)
    return dict(
        cases=cases,
        asset_refs=asset_refs,
        cv2_admission=boot["cv2_admission"],
        mapped_ref=mapped_ref,
        source_preflight_ref=SOURCE_PREFLIGHT,
        boot_ref=prefix.BOOTSTRAP,
        migrated_ref=prefix.MIGRATED_MANIFEST,
        study_inputs=study_inputs,
    )


def input_receipt(inputs):
    """Record identities rather than serializing states or future historical rows."""
    return dict(
        cases=[
            dict(
                case=c["case"],
                original_source_sha256=c["original_source_sha256"],
                original_receipt_ref=c["original_receipt_ref"],
            )
            for c in inputs["cases"]
        ],
        asset_refs=inputs["asset_refs"],
        mapped_ref=inputs["mapped_ref"],
        source_preflight_ref=inputs["source_preflight_ref"],
        boot_ref=inputs["boot_ref"],
        migrated_ref=inputs["migrated_ref"],
        study_inputs=inputs["study_inputs"],
    )
