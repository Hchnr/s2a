"""Bind reset feedback to admitted cases and original gripper parameters."""

import json
from pathlib import Path

from scripts.simulation.runtime import prepare_reset_development as preparation
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix

ROOT = prefix.REPO
ADMISSION = {
    "path": str(ROOT / "outputs/reset_native_admission_v2_20261111/result.json"),
    "sha256": "b6fc7fd1bd62485297e2b33ad9299a9c682cf4df0cd95da3ec58ac264539d411",
}
AUDIT = {
    "path": str(ROOT / "outputs/reset_native_admission_v2_audit_20261111/result.json"),
    "sha256": "6cb975bb51f2641122423e31c8cd7eac7c81c553d77f6525e9abae5e0d2385a2",
}
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def validate_gripper_contract(manifest, controller, source_map):
    """Bind the same no-governor reference and original loaded-gripper model."""
    experiment, variant = manifest["experiment"], manifest["variant"]
    ref = experiment["response_parameters_ref"]
    if not (
        source_map.get(ref["path"]) == ref["sha256"]
        and experiment["control_fps"] == 20
        and variant["governor"] is False
        and experiment["gripper_velocity_envelope"] == controller["gripper_velocity_envelope"]
        and experiment["gripper_position_hold"] == controller["gripper_config"]
        and controller["gripper_algorithm"] == "081 native-close loading and frozen079 tracking"
    ):
        raise ValueError("The original reviewed 20 Hz gripper parameters and controller must match")
    return ref


def load_inputs(index=None, *, verify_assets=True):
    inputs = preparation.load_inputs(index, verify_assets=verify_assets)
    admission, audit = prefix.read(ADMISSION), prefix.read(AUDIT)
    prefix.require(
        admission["complete"] is admission["passed"] is admission["sources_unchanged"] is True
        and admission["control_qualified"] is False
        and len(admission["case_refs"]) == 12
        and admission["totals"] == dict(models=36, contexts=36, control_steps=0, physics_steps=0)
        and admission["image_matches"] == admission["exact_state_matches"] == 12
        and audit["complete"] is audit["passed"] is True
        and audit["control_qualified"] is False
        and audit["result_ref"] == ADMISSION
        and len(audit["cases"]) == 12
        and audit["actual_workers_waited_and_reaped"] == 12
        and audit["native_models"] == audit["forward_queries"] == audit["physics_steps"] == 0,
        "The exact actual reset admission and independent image audit are required",
    )
    original_inputs = admission["inputs"]
    own_inputs = preparation.input_receipt(inputs)
    if index is not None:
        original_inputs = dict(original_inputs, cases=[original_inputs["cases"][index]])
    prefix.require(own_inputs == original_inputs, "The admitted source identity must match")
    review = prefix.read(inputs["study_inputs"]["refs"]["review"])
    for case in inputs["cases"]:
        i = case["case"]["case_index"]
        ref, checked = admission["case_refs"][i], audit["cases"][i]
        worker = prefix.read(ref)
        prefix.require(
            checked["case"] == case["case"] == worker["case"]
            and checked["worker_ref"] == ref
            and checked["state_sha256"] == case["expected_receipt"]["sim_state_sha256"]
            and checked["image_rgb_sha256"] == case["chunk"]["image_sha256"]
            and worker["complete"] is worker["passed"] is True
            and worker["runtime"]["reset_image_match"] is True,
            "Each physical reset and original reference must be independently admitted",
        )
        manifests = [path for path in case["original_source_sha256"] if path.endswith("/manifest.json")]
        prefix.require(len(manifests) == 1, "Exactly one reviewed original manifest required")
        manifest_path = prefix.checked({"path": manifests[0], "sha256": case["original_source_sha256"][manifests[0]]})
        manifest = json.loads(Path(manifest_path).read_bytes())
        params_ref = validate_gripper_contract(manifest, case["controller"], review["source_and_artifact_sha256"])
        case["gripper_parameters"] = prefix.read(params_ref)
        case["gripper_parameters_ref"] = params_ref
        case["target_object"] = TARGETS[i // 4]
        case["reset_admission_ref"] = ref
    inputs.update(reset_admission_ref=ADMISSION, reset_audit_ref=AUDIT)
    return inputs


def input_receipt(inputs):
    return dict(
        base=preparation.input_receipt(inputs),
        reset_admission_ref=inputs["reset_admission_ref"],
        reset_audit_ref=inputs["reset_audit_ref"],
        cases=[
            dict(
                case_index=c["case"]["case_index"],
                gripper_parameters_ref=c["gripper_parameters_ref"],
                target_object=c["target_object"],
                reset_admission_ref=c["reset_admission_ref"],
            )
            for c in inputs["cases"]
        ],
    )
