"""Bind screened release cases to their original causal replay inputs."""

import hashlib
from pathlib import Path

import numpy as np

from s2a.research.joint_control.preview.opening_event_admission import aligned_opening_references
from s2a.research.native_runtime.development_prefix_replay import map_record_xml, saved_prefix
from scripts.simulation.runtime import compare_multicase_emergency_retreat as chunks
from scripts.simulation.runtime import replay_causal_event_prefixes as original_prefix
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.simulation.runtime import screen_causal_opening_events as screen

REPO = prefix.REPO
SCREEN = {
    "path": str(REPO / "outputs/causal_opening_event_screen_20261111/result.json"),
    "sha256": "d511e62fafa8817797637859c4035a4986926c4ce3be4dd837a6eb8b5fc297be",
}
CHECK = {
    "path": str(REPO / "outputs/causal_opening_event_screen_check_20261111/result.json"),
    "sha256": "5b08a45148572fdd4b0fb573d037e81057bc36cdd42453f9c56ae90c28b3d5af",
}


def source_refs():
    return screen.source_refs() + [
        prefix.reference(path)
        for path in (
            REPO / "scripts/simulation/runtime/prepare_causal_opening_feedback.py",
            REPO / "scripts/verification/reference_ports/check_causal_opening_event_screen.py",
            REPO / "tests/research/native_runtime/test_causal_opening_feedback_inputs.py",
        )
    ]


def merge_refs(target, source):
    for path, digest in source.items():
        prefix.require(path not in target or target[path] == digest, "Conflicting historical source identity")
        target[path] = digest


def selected_case(case, mode):
    if mode not in ("latest", "remembered"):
        raise ValueError("One registered paired-reference branch required")
    ref = case["references"][mode]
    cursor = ref["cursor"]
    prefix.require(cursor + 15 < 40, "A complete selected action window and preview required")
    return {
        "spec": case["spec"],
        "controller": case["controller"],
        "tool": case["tool"].copy(),
        "targets": ref["poses"].copy(),
        "openings": ref["openings"].copy(),
        "anchor": ref["anchor"].copy(),
        "cursor": cursor,
        "fixed_targets": ref["poses"][cursor : cursor + 8].copy(),
        "fixed_openings": ref["openings"][cursor : cursor + 8].copy(),
    }


def load_inputs():
    refs = {}
    screened, checked = (prefix.read(ref, refs) for ref in (SCREEN, CHECK))
    prefix.require(
        screened["complete"]
        is screened["passed"]
        is screened["sources_unchanged"]
        is checked["complete"]
        is checked["passed"]
        is checked["sources_unchanged"]
        is True
        and screened["admitted_cases"] == checked["admitted_cases"]
        and screened["rejected_cases"] == checked["rejected_cases"]
        and len(screened["case_refs"]) == len(checked["case_refs"]) == 4,
        "Original four-case screen and independent check must agree",
    )
    context = screen.load_context()
    baseline = original_prefix.load_inputs()
    merge_refs(refs, context["source_sha256"])
    merge_refs(refs, baseline["source_sha256"])
    review = prefix.read(prefix.REVIEW, refs)
    migrated = prefix.read(prefix.MIGRATED_MANIFEST, refs)
    source_map = review["source_and_artifact_sha256"]
    template = baseline["cases"][0]
    template_xml = template["record"]["model_xml"].encode()
    mapped_xml = template["record_xml"].encode()
    screened_cases = [prefix.read(ref, refs)["data"] for ref in screened["case_refs"]]
    checked_cases = [prefix.read(ref, refs)["data"] for ref in checked["case_refs"]]
    cases = []
    for item, admission, audit in zip(context["cases"], screened_cases, checked_cases, strict=True):
        spec, item_refs = item["spec"], item["refs"]
        index, n = spec["case_index"], spec["prefix_steps"]
        prefix.require(
            admission["case"] == audit["case"] == spec
            and admission["input_refs"] == item_refs
            and admission["admission"]["recorded_endpoint_admitted"] == audit["recorded_endpoint_admitted"],
            "Each fixed case retains its own independent admission and inputs",
        )
        if index not in screened["admitted_cases"]:
            prefix.require(not audit["recorded_endpoint_admitted"], "Rejected case cannot enter physical feedback")
            continue
        prefix.require(
            audit["recorded_endpoint_admitted"] and spec["task_id"] in (8, 9), "Supported recorded physical case"
        )
        manifest = prefix.read(item_refs["manifest"], refs)
        controller = prefix.read(item_refs["controller"], refs)
        worker = prefix.read(item_refs["worker"], refs)
        params_ref = manifest["experiment"]["response_parameters_ref"]
        prefix.checked(params_ref, refs)
        prefix.require(source_map.get(params_ref["path"]) == params_ref["sha256"], "Reviewed response parameters")
        rows = prefix.read_recorded_prefix(prefix.checked(item_refs["steps"], refs), n, 600)
        receipts = prefix.read_recorded_prefix(prefix.checked(item_refs["endpoints"], refs), n + 1, 601)
        commands = np.asarray([row["command"] for row in rows], dtype=float)
        with np.load(prefix.checked(item_refs["arrays"], refs), allow_pickle=False) as archive:
            saved = saved_prefix(archive, n)
        prefix.require(
            commands.shape == (n, 7)
            and np.isfinite(commands).all()
            and hashlib.sha256(np.asarray(saved["sim_states"][0], dtype=np.float64).tobytes()).hexdigest()
            == manifest["record"]["initial_state_sha256"]
            and hashlib.sha256(np.asarray(saved["sim_states"][-1], dtype=np.float64).tobytes()).hexdigest()
            == admission["current_state_sha256"],
            "Original initial state, current state and completed command prefix",
        )
        original_xml = manifest["record"]["model_xml"].encode()
        record_xml = map_record_xml(original_xml, template_xml, mapped_xml).decode()
        config = dict(manifest["runtime_config"])
        for key in ("libero_root", "libero_config", "osmesa_library"):
            config[key] = migrated["runtime_config"][key]
            prefix.require(
                Path(config[key]).is_absolute() and Path(config[key]).is_relative_to(REPO),
                "Migrated repository-only runtime path",
            )
        tool = np.asarray(prefix.read(item_refs["tool"], refs)["source_from_tcp"], dtype=float)
        chunk_path = prefix.checked(item_refs["chunks"], refs)
        old = chunks.read_array_item(chunk_path, spec["source_step"] // 8)
        latest = chunks.read_array_item(chunk_path, n // 8)
        pair = aligned_opening_references(
            spec, old, latest, item["shadow"], item["checked"], saved["source_pose"][-1] @ tool, saved["opening"][-1]
        )
        prefix.require(
            worker["arrays_ref"] == item_refs["arrays"]
            and worker["endpoint_receipts_ref"] == item_refs["endpoints"]
            and manifest["experiment"]["gripper_velocity_envelope"] == controller["gripper_velocity_envelope"]
            and manifest["experiment"]["gripper_position_hold"] == controller["gripper_config"],
            "Original complete controller and endpoint contract",
        )
        cases.append(
            dict(
                spec=spec,
                config=config,
                record=manifest["record"],
                record_xml=record_xml,
                commands=commands,
                saved=saved,
                receipts=receipts,
                controller=controller,
                references=pair,
                tool=tool,
                rows=rows,
                gripper_inputs=dict(chunks_ref=item_refs["chunks"], parameters_ref=params_ref),
            )
        )
    prefix.require(
        [item["spec"]["case_index"] for item in cases] == screened["admitted_cases"], "Keep all admitted cases"
    )
    branches = [(case["spec"]["case_index"], mode) for case in cases for mode in ("latest", "remembered")]
    max_control = sum(2 * (case["spec"]["prefix_steps"] + 8) for case in cases)
    return {
        "config_ref": context["config_ref"],
        "config": context["config"],
        "cases": cases,
        "branches": branches,
        "admitted_cases": screened["admitted_cases"],
        "rejected_cases": screened["rejected_cases"],
        "asset_refs": baseline["asset_refs"],
        "source_sha256": refs,
        "cv2_admission": baseline["cv2_admission"],
        "phase_budget": {
            "control_steps": max_control,
            "native_steps": max_control * 25,
            "model_initializations": len(branches) * 3,
            "contexts": len(branches) * 3,
            "gripper_planner_calls": max_control,
            "gripper_candidate_sequences": max_control * 3**8,
            "refiner_calls": len(branches) * 8,
            "refiner_response_evaluations": len(branches) * 8 * 480,
            "quality_calls": len(branches) * 8,
            "quality_response_evaluations": len(branches) * 8 * 160,
        },
    }
