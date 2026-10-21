"""Prepare029's finite paired diagnostic without reopening observation files."""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

from prepare_wam_flow_readout import Evidence, external_identity, reference, require, weight_stats
from prepare_wam_scaled_readout import forbidden_observation_paths

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import write_once

CONFIG = REPO / "configs/wam_free_trajectory_diagnostic_20261021.json"
CONFIG_SHA = "a271cacaf0e011f0dd497173437f09cf3f256cb384e66bdf9d8a71928dd2cb73"
PLAN = REPO / "plans/v0.0.1/029_free_trajectory_diagnostic_20261021.md"
PLAN_SHA = "3616046f32d6baa6939b02ecab619c31297c9b187cc26b50520f9a38a8b4b731"
ADDENDUM = REPO / "plans/v0.0.1/029_scalar_promotion_addendum_20261021.md"
ADDENDUM_SHA = "a83792831cd9aefd31adfb3b6f92827171411b087cd91d084f8fd98a5142ca6d"
SOURCE_ADDENDUM = REPO / "plans/v0.0.1/029_source_revision_addendum_20261021.md"
SOURCE_ADDENDUM_SHA = "4abe4742eaee961d371f129031d79fa41d3c3a1ea8252defa5b37c4da74b7946"
UNIT = REPO / "outputs/wam_stability_20261021/free_trajectory_arithmetic_unit_observer/model_python_cpu_scalar.json"
UNIT_SHA = "20622da8808d62e8d45db32993dcf138593f34b8d97404bbf3c5d9db758eb018"
BACKEND_CONTRACT = dict(
    cpu_bfloat16="sigma_bf16_then_product_fp32_then_bf16_then_sub_fp32",
    cuda_bfloat16="sigma_fp32_then_product_fp32_then_bf16_then_sub_fp32",
    float32="product_fp32_then_sub_fp32",
)
NEW_SOURCES = (
    "src/s2a/wam_trajectory_capture.py",
    "scripts/collect_wam_free_trajectory.py",
    "tests/test_wam_trajectory_capture.py",
    "scripts/verify_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_verification.py",
    "scripts/prepare_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_preparation.py",
    "scripts/run_wam_free_trajectory.py",
    "tests/test_wam_free_trajectory_runner.py",
    "scripts/probe_wam_free_trajectory_precision.py",
    "tests/test_wam_free_trajectory_precision.py",
)
STATE_REFS = ("prepared", "sampling", "verification", "owner", "final_peer")


def safe_hashes(mapping, forbidden, evidence):
    """Observation SHAs remain declarations; preparation never opens their data."""
    skipped = {}
    for filename, digest in mapping.items():
        path = Path(filename).resolve()
        if str(path) in forbidden or path.suffix.lower() == ".png":
            skipped[str(path)] = digest
        else:
            evidence.check(path, digest)
    return skipped


def state_admission(values, refs):
    source, sampling, audit, owner, peer = (values[k] for k in STATE_REFS)
    require(
        source["schema_version"] == "s2a.scaled-state-preparation.v1"
        and len(source["source_sha256"]) == len(source["snapshot_sha256"]) == 96,
        "029 must inherit original028 source closure",
    )
    require(
        sampling["schema_version"] == "s2a.scaled-state-sampling.v1"
        and sampling["complete"] is sampling["passed"] is True
        and sampling["prepared_ref"] == refs["prepared"]
        and sampling["denoise_calls"] == 3465
        and len(sampling["generations"]) == 96
        and sampling["all_original_parameters_frozen"] is True,
        "029 requires complete original028 generation evidence",
    )
    require(
        audit["schema_version"] == "s2a.scaled-state-independent-verification.v1"
        and audit["passed"] is audit["sampling_verified"] is True
        and audit["prepared"] == refs["prepared"]
        and audit["input_sha256"].get(refs["sampling"]["path"]) == refs["sampling"]["sha256"]
        and audit["sampling_gate"]["passed"] is False
        and audit["eligible_for_development_physics_registration"] is audit["physics_authorized"] is False,
        "029 must retain original028 failed quality gate",
    )
    require(
        owner["schema_version"] == "s2a.scaled-state-owner.v1"
        and owner["complete"] is owner["passed"] is True
        and owner["producer_result"] == refs["sampling"]
        and owner["prepared_ref"] == refs["prepared"]
        and owner["eligible_for_development_physics_registration"] is owner["physics_authorized"] is False
        and owner["automatic_retry"] is False
        and len(owner["owned_commands"]) == 2,
        "029 original028 owner did not complete normally",
    )
    require(
        peer["schema_version"] == "s2a.scaled-state-final-readonly-peer.v1"
        and peer["complete"] is peer["passed"] is True
        and peer["prepared"] == refs["prepared"]
        and peer["sampling_result"] == refs["sampling"]
        and peer["verification"] == refs["verification"]
        and peer["owner_result"] == refs["owner"]
        and peer["frozen_sources_verified"] == peer["source_snapshots_verified"] == 96
        and peer["original8400_weight_stats_verified"] == 163
        and peer["source_before_after_unchanged"] is True,
        "029 requires the completed independent028 final peer",
    )


def rosters(source, sampling, config, evidence):
    anchors = [copy.deepcopy(a) for a in source["anchors"] if a["split"] == "validation"]
    require(
        len(anchors) == 24
        and list(dict.fromkeys(a["episode"] for a in anchors)) == config["validation_episodes"]
        and all(a["valid_future_steps"] == 40 for a in anchors),
        "029 six validation episodes/four complete anchors differ",
    )
    by_name = {a["name"]: a for a in anchors}
    entries = [e for e in sampling["generations"] if e["variant"] == "raw"]
    expected = [(a["name"], seed) for a in anchors for seed in (42, 43)]
    require([(e["anchor_name"], e["sampling_seed"]) for e in entries] == expected, "029 fixed raw48 order differs")
    free, teacher, nodes = [], [], None
    for i, entry in enumerate(entries):
        anchor = by_name[entry["anchor_name"]]
        require(entry["request"]["seed"] == entry["sampling_seed"], "029 actual request seed differs")
        trace = evidence.ref(entry["actual_sigma_trace"])
        require(
            trace["complete"] is True
            and len(trace["steps"]) == 35
            and trace["actual_sampler_arguments"] == dict(num_steps=35, shift=5.0, seed=[entry["sampling_seed"]]),
            "029 original actual solver trace differs",
        )
        current_nodes = [
            dict(
                step_index=j,
                sigma_actual=trace["steps"][j]["solver_sigma"],
                network_timestep=trace["steps"][j]["solver_timestep"],
            )
            for j in config["precision_step_indices"]
        ]
        if nodes is None:
            nodes = current_nodes
        require(current_nodes == nodes, "029 prior raw requests used different actual schedules")
        selected = [n for n in nodes if n["step_index"] in config["selected_step_indices"]]
        require(
            [n["sigma_actual"] for n in selected] == config["selected_actual_sigmas"],
            "029 selected actual sigmas differ",
        )
        member = copy.deepcopy(entry)
        cache = {k: member.pop(k) for k in ("response", "conditioning", "initial", "actual_sigma_trace")}
        member.update(generation_index=i, source_generation_index=entry["generation_index"], cache=cache)
        require(member["request"]["vision_path"] == anchor["image_path"], "029 online image reference differs")
        free.append(member)
        for node in selected:
            teacher.append(
                dict(
                    call_index=len(teacher),
                    name=f"{entry['name']}_teacher_step{node['step_index']}",
                    anchor_name=anchor["name"],
                    sampling_seed=entry["sampling_seed"],
                    free_generation_index=i,
                    **node,
                )
            )
    require(len(free) == 48 and len(teacher) == 144, "029 raw/teacher budget differs")
    return anchors, free, teacher, nodes


def read_prepared(path, digest):
    evidence = Evidence()
    prepared = evidence.json(path, digest)
    require(
        prepared["schema_version"] == "s2a.free-trajectory-preparation.v1"
        and prepared["config"] == evidence.json(CONFIG, CONFIG_SHA)
        and prepared["config_ref"] == reference(CONFIG)
        and prepared["registration"] == dict(path=str(PLAN), sha256=PLAN_SHA)
        and prepared["numerical_addendum_ref"] == dict(path=str(ADDENDUM), sha256=ADDENDUM_SHA)
        and prepared["scalar_promotion_unit_ref"] == dict(path=str(UNIT), sha256=UNIT_SHA)
        and prepared["backend_arithmetic_contract"] == BACKEND_CONTRACT
        and prepared.get("source_revision_ref") == dict(path=str(SOURCE_ADDENDUM), sha256=SOURCE_ADDENDUM_SHA)
        and len(prepared["source_sha256"]) == len(prepared["snapshot_sha256"]) == 111,
        "029 preparation/registration/source closure differs",
    )
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(ADDENDUM, ADDENDUM_SHA)
    evidence.check(SOURCE_ADDENDUM, SOURCE_ADDENDUM_SHA)
    for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
        for filename, expected in prepared[key].items():
            evidence.check(filename, expected)
    weight_stats(prepared["weight_files"])
    require(
        external_identity(prepared["config"]["cosmos_repo"]) == prepared["external_cosmos"],
        "029 external source differs",
    )
    require(
        Path(prepared["output_root"]) == REPO / (prepared["config"]["output_dir"] + "_source_v2"),
        "029 registered source_v2 output root differs",
    )
    return prepared


def prepare():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "029 preparation requires repository cwd and emptyCUDA/LD",
    )
    evidence = Evidence()
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    evidence.check(ADDENDUM, ADDENDUM_SHA)
    evidence.check(SOURCE_ADDENDUM, SOURCE_ADDENDUM_SHA)
    unit = evidence.json(UNIT, UNIT_SHA)
    require(
        unit["torch_version"] == "2.10.0+cu130"
        and unit["all_bfloat16_scalar_first_exact"] is True
        and len(unit["cases"]) == 6
        and unit["environment"] == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="")
        and all(
            unit[k] == 0
            for k in (
                "original_scheduler_calls",
                "model_weights_loaded",
                "wam_denoise_calls",
                "fit_groups",
                "episode_reads",
                "physics_steps",
            )
        ),
        "029 registered CPU arithmetic unit observation differs",
    )
    torch_root = Path(config["cosmos_repo"]) / ".venv/lib/python3.13/site-packages/torch"
    arithmetic_inputs = [
        reference(evidence.check(torch_root / name))
        for name in (
            "version.py",
            "include/ATen/native/cuda/Loops.cuh",
            "include/ATen/TensorIterator.h",
            "include/ATen/OpMathType.h",
        )
    ]
    root = REPO / (config["output_dir"] + "_source_v2")
    require(root.is_relative_to(REPO / "outputs") and not root.exists(), "029 refuses existing/non-output root")
    refs = {k: reference(REPO / config[f"source_state_{k}"]) for k in STATE_REFS}
    values = {k: evidence.json(refs[k]["path"], config[f"source_state_{k}_sha256"]) for k in STATE_REFS}
    state_admission(values, refs)
    source = values["prepared"]
    forbidden = forbidden_observation_paths(source)
    declarations = {}
    for value in values.values():
        for key in ("input_sha256", "source_sha256", "snapshot_sha256"):
            declarations.update(safe_hashes(value.get(key, {}), forbidden, evidence))
    for command in values["owner"]["owned_commands"]:
        launch, end, release = (evidence.ref(command[k]) for k in ("launch", "exit", "release"))
        require(
            end == dict(actual_wait=True, automatic_retry=False, returncode=0)
            and release["reaped"] is True
            and release["returncode"] == 0
            and release["signal_sent"] is False
            and release["original_error"] is None
            and all(launch["child"][k] == release[k] for k in ("pid", "start_ticks")),
            "029 prior owned normal exit missing",
        )
    released = evidence.ref(values["owner"]["gpu_after_ref"])
    require(
        released["memory_mib"] == 0 and released["compute_apps"] == [] and released["released"] is True,
        "029 prior GPU release missing",
    )
    anchors, free, teacher, nodes = rosters(source, values["sampling"], config, evidence)
    sources = dict(source["source_sha256"])
    for name in (*NEW_SOURCES, str(CONFIG), str(PLAN), str(ADDENDUM), str(SOURCE_ADDENDUM)):
        p = REPO / name
        sources[str(p.resolve())] = sha256(evidence.check(p))
    require(len(sources) == 111, "029 requires original96 plus new11, plan/config and two addenda")
    weight_stats(source["weight_files"])
    require(external_identity(config["cosmos_repo"]) == source["external_cosmos"], "029 external source differs")
    directory = root / "preparation"
    directory.mkdir(parents=True)
    snapshots = {}
    for i, (filename, digest) in enumerate(sorted(sources.items())):
        target = directory / "source_snapshot" / f"{i:03d}_{Path(filename).name}"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(filename, target)
        snapshots[str(target)] = sha256(evidence.check(target, digest))
    inherited = (
        "source_prepared_ref",
        "source_features_ref",
        "scaled_prepared_ref",
        "scaled_fit_ref",
        "scaled_fit_verification_ref",
        "adapted_model_ref",
        "domain",
        "checkpoint",
        "weight_files",
        "external_cosmos",
        "conservative_source_exclusion",
    )
    prepared = {k: copy.deepcopy(source[k]) for k in inherited}
    prepared.update(
        schema_version="s2a.free-trajectory-preparation.v1",
        config=config,
        config_ref=reference(CONFIG),
        registration=reference(PLAN),
        numerical_addendum_ref=reference(ADDENDUM),
        source_revision_ref=reference(SOURCE_ADDENDUM),
        scalar_promotion_unit_ref=reference(UNIT),
        torch_arithmetic_inputs=arithmetic_inputs,
        backend_arithmetic_contract=BACKEND_CONTRACT,
        output_root=str(root),
        anchors=anchors,
        free_generations=free,
        teacher_calls=teacher,
        precision_nodes=nodes,
        source_sha256=sources,
        snapshot_sha256=snapshots,
        inherited_observation_sha256=declarations,
        **{f"source_state_{k}_ref": v for k, v in refs.items()},
    )
    evidence.recheck()
    prepared["input_sha256"] = evidence.hashes.copy()
    write_once(directory / "prepared.json", prepared)
    return reference(directory / "prepared.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    parser.parse_args()
    print(json.dumps(prepare()))


if __name__ == "__main__":
    main()
