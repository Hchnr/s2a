"""Bind a preserved zero-query failure to a complete byte-verified asset repair."""

from s2a.prefix_trust_control_evidence import (
    CPU_ENVIRONMENT as CPU_ENVIRONMENT,
)
from s2a.prefix_trust_control_evidence import (
    NATIVE_PYTHON as NATIVE_PYTHON,
)
from s2a.prefix_trust_control_evidence import (
    QA_TESTS,
    REPO,
    SOURCES,
    case_root,
    closed_owner,
    finish,
    load_prerequisites,
    read,
    reference,
    remember,
    require,
)

PLAN = REPO / "plans/v0.0.1/prefix_trust_asset_check_repair_20261029.md"
SNAPSHOT_ROOT = REPO / "outputs/prefix_trust_asset_snapshot_20261029"
HISTORICAL_MANIFEST_REF = dict(
    path=str(REPO / "outputs/terminal_development_20261025/historical_oracle_xml_migration.json"),
    sha256="a4a423a077c2a99ec9817159c48d89c6d6e01a7f816cd4cb13bcca84d531d186",
)
ORIGINAL_XML_REF = dict(
    path=str(
        REPO
        / "outputs/tracking_preview_wam_development_v5_20261028/development/cases/task0_init0_s42_terminal058/simulation/8400_masked_position_hold/task0_init0_s42/model.xml"
    ),
    sha256="50d3ece2777882c860513208d8c5ac75a31f49ff9db31d034211b5879c98f0e6",
)
FAILED_RESULT_REF = dict(
    path=str(case_root("candidate", check=True) / "verified.json"),
    sha256="cc4993e6e0e2772ad4e2172910cf7f93b634b1cb94f61753ae18499f588cea47",
)
FAILED_OWNER_REF = dict(
    path=str(case_root("candidate", check=True, owner=True) / "result.json"),
    sha256="335ea5abf56ea58b41d653b8d5ba7953ef22fc20f2e55ec985c433a6ccdf9a84",
)
FAILED_CAPTURE = case_root("candidate", check=True, owner=True) / "capture"
FAILED_LIFECYCLE_REFS = {
    relative: dict(path=str(FAILED_CAPTURE / relative), sha256=digest)
    for relative, digest in {
        "launch.json": "20f0106d34f8edfef775ec62131b03d90a83fbb3ce648e170318bc103867806c",
        "exit.json": "06bc82811a76aeacd289ab37671b1de45bf96c9669da6f43c4d9525bb1997b6b",
        "ownership/release.json": "a8853b583d2476d9f63e4d616cc1256c78eedc1a1fe954a5a623fc9f1b12a510",
    }.items()
}
PRODUCER_REFS = {
    name: dict(path=str(case_root(name) / "verified.json"), sha256=digest)
    for name, digest in {
        "baseline": "c3814b6d0bf3db48d85865ee23c4104d2cc1fb0cde90f06d37b0d4c3c8c5c841",
        "candidate": "7338118bcad2cc80644555fcee42f9257b411cf9fa6234444fe1790601386d34",
    }.items()
}
PRODUCER_OWNER_REFS = {
    name: dict(path=str(case_root(name, owner=True) / "result.json"), sha256=digest)
    for name, digest in {
        "baseline": "11198fad76716be971aa3b4bb48e62a1cc9af94b0efd75f78938bd28b868cb23",
        "candidate": "1c103741c1da5f97450dfb25a9f2851083cef3bc751070b3390bb30d71264f28",
    }.items()
}
NEW_TESTS = (
    "tests/test_xml_asset_snapshot.py",
    "tests/test_prefix_trust_asset_check_v2.py",
    "tests/test_prefix_trust_asset_repair_execution.py",
)
REPAIR_SOURCES = (
    "src/s2a/xml_asset_snapshot.py",
    "src/s2a/prefix_trust_asset_repair.py",
    "scripts/prepare_prefix_trust_asset_snapshot.py",
    "scripts/check_tracking_v5_prefix_trust_control_v2.py",
    "scripts/execute_tracking_v5_prefix_trust_check_v2.py",
    "plans/v0.0.1/prefix_trust_asset_check_repair_20261029.md",
    *NEW_TESTS,
)
QA_SOURCES = (*SOURCES, *REPAIR_SOURCES)
ALL_QA_TESTS = (*QA_TESTS, *NEW_TESTS)


def repair_root(case_name, *, owner=False):
    require(case_name in PRODUCER_REFS, "One original H12 arm")
    suffix = "_execution" if owner else ""
    return REPO / f"outputs/tracking_v5_prefix_trust_{case_name}_check_v2{suffix}_20261029"


def verify_failed_execution(refs):
    failed = read(remember(FAILED_RESULT_REF, refs))
    owner = read(remember(FAILED_OWNER_REF, refs))
    reg = read(remember(owner["registration_ref"], refs))
    check_reg = read(remember(failed["registration_ref"], refs))
    require(
        failed["complete"] is failed["passed"] is owner["complete"] is owner["passed"] is False
        and failed["case_name"] == owner["case_name"] == reg["case_name"] == "candidate"
        and owner["stage"] == reg["stage"] == "check"
        and "file not found" in failed["error"]
        and "wooden_cabinet_top_vis.msh" in failed["error"]
        and "failed with1" in owner["error"]
        and check_reg["producer_ref"] == PRODUCER_REFS["candidate"]
        and all(
            failed[key] == 0
            for key in (
                "response_evaluations",
                "verified_responses",
                "numerical_inner_steps",
                "mj_forward_queries",
                "query_count",
                "physics_steps",
                "world_model_calls",
                "fresh_cases",
            )
        )
        and failed["boundary_classification_disagreements"] == [],
        "Only the preserved pre-query asset-load failure; no completed response is reused",
    )
    life = {name: read(remember(ref, refs)) for name, ref in FAILED_LIFECYCLE_REFS.items()}
    launch, exited, release = (life[key] for key in ("launch.json", "exit.json", "ownership/release.json"))
    require(
        launch["command"]
        == [str(NATIVE_PYTHON), str(REPO / "scripts/check_tracking_v5_prefix_trust_control.py"), "--case", "candidate"]
        and launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is exited["automatic_retry"] is False
        and exited["actual_wait"] is True
        and exited["returncode"] == 1
        and release["reaped"] is True
        and release["returncode"] == 1
        and release["signal_sent"] is False
        and release["pid"] == launch["child"]["pid"]
        and release["start_ticks"] == launch["child"]["start_ticks"]
        and release["original_error"] == owner["error"],
        "The failed original child was actually waited/reaped once without a signal",
    )
    remember(reference(FAILED_CAPTURE / "command.log"), refs)
    for path, digest in reg["source_sha256"].items():
        remember(dict(path=path, sha256=digest), refs)


def load_snapshot():
    from s2a.xml_asset_snapshot import verify_asset_snapshot

    snapshot_ref = reference(SNAPSHOT_ROOT / "manifest.json")
    snapshot = verify_asset_snapshot(snapshot_ref)
    compile_ref = reference(SNAPSHOT_ROOT / "compile.json")
    compiled = read(compile_ref)
    require(
        snapshot["original_xml_ref"] == ORIGINAL_XML_REF
        and snapshot["historical_manifest_ref"] == HISTORICAL_MANIFEST_REF
        and snapshot["file_reference_count"] == 99
        and snapshot["unique_asset_count"] == len(snapshot["assets"]) == 96
        and snapshot["byte_exact_inverse"] is True,
        "The same full 99-reference model and all 96 historically fingerprinted assets",
    )
    require(
        compiled["complete"] is compiled["passed"] is True
        and compiled["manifest_ref"] == snapshot_ref
        and compiled["runtime"]["mujoco"] == "2.3.7"
        and compiled["runtime"]["numpy"] == "1.26.4"
        and compiled["runtime"]["python"].startswith("3.11.")
        and compiled["environment"] == CPU_ENVIRONMENT
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
    require(
        compiled["source_refs"] == [reference(REPO / path) for path in REPAIR_SOURCES],
        "Compilation bound to the final unchanged repair source",
    )
    return dict(snapshot, snapshot_ref=snapshot_ref, compile_ref=compile_ref)


def repair_prerequisites(require_snapshot=True):
    refs = {}
    verify_failed_execution(refs)
    remember(HISTORICAL_MANIFEST_REF, refs)
    remember(ORIGINAL_XML_REF, refs)
    qa_refs = []
    for name, count in (("baseline", 4000), ("candidate", 3148)):
        owner, producer = closed_owner(PRODUCER_OWNER_REFS[name], refs)
        remember(PRODUCER_REFS[name], refs)
        require(
            owner["result_ref"] == PRODUCER_REFS[name]
            and producer["case_name"] == name
            and producer["response_evaluations"] == producer["prediction_calls"] == count
            and producer["numerical_inner_steps"] == 300 * count
            and producer["query_count"] == 1 + 312 * count
            and len(producer["answer"]["evaluated_candidates"]) == count
            and producer["answer"]["witness_found"] is False,
            "Both exact completed original producers; neither is rerun or reclassified",
        )
        qa_ref = read(owner["registration_ref"])["native_component_qa_ref"]
        qa = read(remember(qa_ref, refs))
        qa_refs.append(qa_ref)
        require(
            all(qa["source_sha256"][path] == reference(REPO / path)["sha256"] for path in SOURCES),
            "Original nine executed sources remain frozen",
        )
        for ref in producer["source_refs_unchanged"]:
            remember(ref, refs)
    require(qa_refs[0] == qa_refs[1], "Original matched-warm arms share one source-bound QA")
    for path in REPAIR_SOURCES:
        remember(reference(REPO / path), refs)
    if require_snapshot:
        snapshot = load_snapshot()
        for key in ("snapshot_ref", "compile_ref", "original_xml_ref", "historical_manifest_ref", "migrated_xml_ref"):
            remember(snapshot[key], refs)
        remember(reference(SNAPSHOT_ROOT / "original.xml"), refs)
        for asset in snapshot["assets"]:
            remember(asset["safe_source_ref"], refs)
            remember(asset["snapshot_ref"], refs)
    finish(refs)
    return refs


def load_asset_repair_prerequisites():
    original, registration, backtracking, refs = load_prerequisites()
    for ref in repair_prerequisites().values():
        remember(ref, refs)
    finish(refs)
    return original, registration, backtracking, refs
