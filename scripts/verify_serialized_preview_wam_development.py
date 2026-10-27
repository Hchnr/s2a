"""Independent preview WAM development audit: full mechanical replay, all31500 nodes and scoring.

Does not import the development owner, worker, service, readout producer or fit.
The frozen014 seed-aware independent controller checker and the independent
compact numerical checker are reused explicitly; inference inputs are verified
against each physical current-image query, rather than expert future labels.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import verify_wam_clean_training_capture as numerics
from audit_wam_position import Evidence as PhysicsEvidence
from coupled_preview_wam_measured_audit import verify_measured_endpoints
from serialized_preview_wam_acceptance_checker import verify_case
from verify_wam_terminal_development_points import audit_points as audit_terminal_points
from verify_wam_terminal_development_points import head_parameters

REPO = Path(__file__).resolve().parents[1]
GPU = "GPU-ad74a609-3a39-43cb-1392-24cbd279ce7f"
FIXTURE_SHA = "f522ee333827d51ab66b5b8d1c2a7ebd8ceca3832403b3c54dfa56a821fb2a8e"
CONTROL_SHA = "106f45543f8bc0aa0e336c0ad9b5c01e0bce252a5e5dad0ec3ebbba349e5fda0"


def require(value, message):
    if not value:
        raise ValueError(message)


def order():
    return [
        dict(
            case_index=i,
            case_id=f"task{t}_init0_s{s}_{v}",
            task_index=t,
            record_name=f"task{t}_init0",
            sampling_seed=s,
            variant=v,
            expected_requests=75,
        )
        for i, (t, s, v) in enumerate((t, s, v) for t in range(3) for s in (42, 43) for v in ("raw", "terminal058"))
    ]


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def current_query_evidence_scope(*args, **kwargs):
    """The inherited callback records scope; all75 service bindings follow below."""
    return dict(
        historical_003_regression_performed=False,
        current_query_bindings="independently enforced for every new-service request",
    )


def label_current_mechanical_evidence(mechanical):
    """Avoid presenting the inherited callback slot as a historical003 result."""
    labeled = dict(mechanical)
    scope = labeled.pop("first_query003_regression")
    require(scope == current_query_evidence_scope(), "unexpected claim in the inherited first-query evidence slot")
    labeled["current_query_evidence_scope"] = scope
    return labeled


def outcome(success):
    require(
        isinstance(success, list) and len(success) == 600 and all(type(v) is bool for v in success),
        "all600 actual native booleans required; partial/placeholder success rejected",
    )
    trailing = 0
    for value in reversed(success):
        if not value:
            break
        trailing += 1
    return dict(
        ever=any(success),
        terminal=success[-1],
        stable=all(success[-40:]),
        successful_steps=sum(success),
        trailing_success_steps=trailing,
    )


def score(cases):
    require(
        len(cases) == 12 and [row["case"] for row in cases] == order(),
        "fixed complete twelve-case task/seed/variant order required",
    )
    result = {}
    for variant in ("raw", "terminal058"):
        selected = [row for row in cases if row["case"]["variant"] == variant]
        result[variant] = dict(
            trials=6,
            ever=sum(row["outcome"]["ever"] for row in selected),
            terminal=sum(row["outcome"]["terminal"] for row in selected),
            stable=sum(row["outcome"]["stable"] for row in selected),
            stable_by_task={
                str(t): sum(row["outcome"]["stable"] for row in selected if row["case"]["task_index"] == t)
                for t in range(3)
            },
        )
    gate = result["terminal058"]["stable"] >= 5 and all(
        n >= 1 for n in result["terminal058"]["stable_by_task"].values()
    )
    adaptation = result["terminal058"]["stable"] > result["raw"]["stable"]
    return dict(
        **result,
        development_stability_gate_passed=gate,
        strict_adaptation_gain_passed=adaptation,
        eligible_for_later_fresh_candidate_freeze=gate,
        fresh_performance_authorized=False,
        final_fresh_target=dict(trials=24, stable_min=20, stable_each_task_min=4),
    )


def normal_owned(ev, command, cuda):
    launch, exited, release = (ev.json(command[k]) for k in ("launch", "exit", "release"))
    require(
        exited == dict(returncode=0, actual_wait=True, automatic_retry=False), "owned process lacks normal actual wait"
    )
    require(
        release["reaped"] is True
        and release["returncode"] == 0
        and release["signal_sent"] is False
        and release["original_error"] is None
        and release["only_owned_child_signalled"] is True
        and all(release[k] == launch["child"][k] for k in ("pid", "start_ticks")),
        "owned process normal birth/reap differs",
    )
    require(
        launch["selected_environment"] == dict(CUDA_VISIBLE_DEVICES=cuda, LD_LIBRARY_PATH="")
        and launch["automatic_retry"] is False,
        "owned actual CUDA/LD/automatic retry differs",
    )
    return launch


def admission(ev, prepared_ref, registration_ref, run_ref):
    from serialized_preview_wam_audit_admission import admission as independent_admission

    return independent_admission(
        ev, prepared_ref, registration_ref, run_ref, require=require, normal_owned=normal_owned, gpu=GPU, order=order
    )


def verify_service_request(ev, record, case, index, directory, physical_request, physical_response, *, head_ref):
    global_index = case["case_index"] * 75 + index
    require(
        record["complete"] is True
        and record["case_id"] == case["case_id"]
        and record["case_index"] == case["case_index"]
        and record["request_index"] == index
        and record["global_request_index"] == global_index
        and record["variant"] == case["variant"]
        and record["sampling_seed"] == case["sampling_seed"]
        and record["anchor_name"] == case["case_id"]
        and record["readout_ref"] == head_ref,
        "service case/seed/head/row binding differs",
    )
    require(
        ev.json(record["request_ref"]) == physical_request and ev.json(record["response_ref"]) == physical_response,
        "physical HTTP and actual service artifacts differ",
    )
    provenance = {
        key: record[key]
        for key in (
            "case_index",
            "case_id",
            "request_index",
            "global_request_index",
            "variant",
            "sampling_seed",
            "readout_ref",
        )
    }
    require(
        physical_response["clean_training"] == provenance, "HTTP response is not bound to this exact head/case/request"
    )
    image = directory / f"input_{index:04d}.png"
    require(
        ev.path(record["image_ref"]) == image
        and physical_request["vision_path"] == str(image)
        and physical_request["seed"] == case["sampling_seed"],
        "physical current image/seed differs",
    )


def audit(args):
    ev, physics = numerics.Evidence(), PhysicsEvidence()
    refs = {
        key: dict(path=str(Path(getattr(args, key)).resolve()), sha256=getattr(args, key + "_sha256"))
        for key in ("prepared", "execution_registration", "development")
    }
    prepared, reg, run, state, control, sources = admission(
        ev, refs["prepared"], refs["execution_registration"], refs["development"]
    )
    root = Path(prepared["root"]) / "development"
    require(
        ev.path(refs["development"]) == root / "orchestration/result.json", "development producer output route differs"
    )
    service = ev.json(run["service_ref"])
    require(
        service["schema_version"] == "s2a.serialized-preview-wam-development-service.v1"
        and service["complete"] is service["passed"] is True
        and service["actual_requests"] == len(service["requests"]) == 900
        and service["actual_denoise_calls"] == service["actual_head_calls"] == 31500
        and service["model_initializations"] == service["model_initialization_attempts"] == 1
        and service["parameters_before"] == service["parameters_after"]
        and service["source_sha256"] == sources
        and service["gpu_uuid"] == GPU
        and service["automatic_retry"] is False,
        "complete frozen private service budget/source failed",
    )
    for key in (
        "prepared_ref",
        "execution_registration_ref",
        "state_owner_ref",
        "state_verification_ref",
        "candidate_proof_ref",
        "state_score_ref",
        "readout_ref",
    ):
        require(service[key] == run[key], "service model/candidate provenance differs")
    for key in ("future_label_reads", "producer_fits", "physics_steps", "fresh_cases"):
        require(service[key] == 0, "private model service consumed forbidden data/physics budget")
    terminal_head = head_parameters(ev.json(run["readout_ref"]))
    baseline_head = head_parameters(ev.json(prepared["baseline_params_ref"]))
    require(
        numerics.np.array_equal(baseline_head[0][:, 3:9], terminal_head[0][:, 3:9])
        and numerics.np.array_equal(baseline_head[1], terminal_head[1])
        and baseline_head[2] == terminal_head[2],
        "061 rotation and feature transform must remain frozen",
    )
    head = (baseline_head, terminal_head)
    bridge = ev.json(prepared["prerequisite_refs"]["bridge_qualification"])
    grid = [(s["solver_sigma"], s["solver_timestep"]) for s in ev.json(bridge["generations"][0]["trace_ref"])["steps"]]
    for relative in (
        "scripts/serialized_preview_wam_acceptance_checker.py",
        "scripts/coupled_preview_wam_measured_audit.py",
        "scripts/serialized_preview_wam_audit_admission.py",
        "scripts/coupled_preview_wam_physical_admission.py",
    ):
        path = str(REPO / relative)
        require(path in sources, "Independent preview checker not frozen")
        physics.check(path, sources[path])
    qualified_control = ev.json(reg["physical_control_admission"]["control_config_ref"])
    base = physics.json(REPO / "configs/wam_short_replan_8_alpha1_20261021.json")
    require(base["fixture_sha256"] == FIXTURE_SHA, "original fixture declaration differs")
    physics.check(REPO / base["fixture"], FIXTURE_SHA)
    fixture = physics.json(REPO / base["fixture"])
    all_cases, first_pairs, changed, verified_points = [], {}, 0, 0
    for item, case in zip(run["cases"], order(), strict=True):
        launch = normal_owned(ev, item["owned_command"], "")
        require(item["owned_command"] == run["owned_commands"][case["case_index"]], "case owner source differs")
        worker = ev.json(item["worker_ref"])
        config = ev.json(item["config_ref"])
        require(
            worker["schema_version"] == "s2a.serialized-preview-wam-development-case.v1"
            and worker["passed"] is worker["complete"] is True
            and worker["case"] == case
            and worker["config_ref"] == item["config_ref"]
            and worker["physics_steps"] == 600
            and worker["actual_requests"] == 75
            and worker["http_timeout_cap_s"] == 300
            and worker["actual_model_gpu_uuid"] == GPU
            and worker["preview_controller"] is True,
            "isolated CPU wrapper qualification failed",
        )
        require(
            all(
                worker[key] == run[key]
                for key in ("prepared_ref", "execution_registration_ref", "state_owner_ref", "state_verification_ref")
            )
            and worker["served_reference_variant"] == case["variant"]
            and worker["raw_checkpoint_performance_evidence"] is (case["variant"] == "raw"),
            "worker run identity or raw/adapted evidence scope differs",
        )
        omitted = {
            "output_dir",
            "notes",
            "records",
            "sampling_seeds",
            "simulation_python",
            "fixture",
            "checkpoints",
            "policy_template",
            "cosmos_repo",
        }
        preview_fields = (
            "arm_pose_policy",
            "arm_pose_objective",
            "source_provisional_inverse_weight",
            "arm_residual_policy",
            "arm_forward_parameter_ref",
            "arm_forward_qualification_ref",
            "pose_preview_chain_ref",
            "response_parameters_ref",
            "planner_preflight_ref",
            "gripper_velocity_envelope",
        )
        omitted.update(preview_fields)
        omitted.update(("gripper_preview_steps", "preview_control_closure_ref", "wam_development"))
        require(
            all(config[k] == qualified_control[k] for k in preview_fields)
            and config["gripper_preview_steps"] == 0
            and config["preview_control_closure_ref"] == prepared["preview_control_closure_ref"]
            and config["wam_development"]
            == dict(
                case=case,
                prepared_ref=refs["prepared"],
                execution_registration_ref=refs["execution_registration"],
                service_source_ref=reg["service_source_ref"],
                readout_ref=prepared["candidate_params_ref"],
            ),
            "Qualified preview configuration differs",
        )
        require(
            {k: v for k, v in config.items() if k not in omitted} == {k: v for k, v in base.items() if k not in omitted}
            and config["records"] == [case["record_name"]]
            and config["sampling_seeds"] == [case["sampling_seed"]]
            and config["checkpoints"]
            == [dict(prepared["model_config"]["checkpoint"], name="8400_masked", gpu_uuid=GPU, port=31365)]
            and Path(config["fixture"]).resolve() == (REPO / base["fixture"]).resolve(),
            "fixed raw8 full controller contract changed",
        )
        relocation = ev.json(prepared["tool_relocation_ref"])
        require(
            config["policy_template"] == relocation["new_template_ref"]["path"]
            and config["cosmos_repo"] == prepared["model_config"]["cosmos_source_root"],
            "061 active template/Cosmos paths differ",
        )
        directory = (
            root
            / "cases"
            / case["case_id"]
            / "simulation/8400_masked_position_hold"
            / f"{case['record_name']}_s{case['sampling_seed']}"
        )
        require(worker["physical_case_dir"] == str(directory), "physical case output path differs")
        snapshot = ev.json(worker["execution_snapshot_ref"])
        manifest = physics.json(directory / "manifest.json")
        require(
            snapshot["source_sha256"].get(item["config_ref"]["path"]) == item["config_ref"]["sha256"]
            and all(
                manifest["execution_sha256"].get(path) == digest for path, digest in snapshot["source_sha256"].items()
            ),
            "pre-execution snapshot/config is not bound to the physical source closure",
        )
        readiness = ev.json(worker["cv2_readiness_ref"])
        require(
            readiness["passed"] is readiness["complete"] is True
            and readiness["restore_performed"] is True
            and readiness["cv2_probe_ref"] == control["cv2_guard_registration"]["exact_probe_ref"]
            and readiness["cv2_runtime_source_refs"] == control["cv2_guard_registration"]["runtime_source_refs"]
            and readiness["after_restore"]
            == worker["final_environment"]
            == dict(CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH="", cwd=str(REPO)),
            "registered cv2 preload/strict CPU environment differs",
        )
        require(
            readiness["pid"] == launch["child"]["pid"], "preload receipt does not belong to this owned physics worker"
        )
        calls = [json.loads(line) for line in ev.path(worker["http_ref"]).read_text().splitlines()]
        require(
            len(calls) == 76
            and calls[0]["method"] == "GET"
            and calls[0]["url"] == "http://127.0.0.1:31365/info"
            and calls[0]["timeout_s"] == 15
            and all(
                c["method"] == "POST"
                and c["url"] == "http://127.0.0.1:31365/generate"
                and c["timeout_s"] == 300
                and c["completed"] is True
                and c["status_code"] == 200
                for c in calls[1:]
            ),
            "actual transport cap/full75 queries differ",
        )
        record = next(r for r in fixture["records"] if r["name"] == case["record_name"])
        import ctypes

        manifest = physics.json(directory / "manifest.json")
        ctypes.CDLL(manifest["runtime_config"]["osmesa_library"], mode=ctypes.RTLD_GLOBAL)
        mechanical = verify_case(
            config,
            record,
            config["variants"][0],
            directory,
            physics,
            source="wam",
            checkpoint="8400_masked",
            full=True,
            sampling_seed=case["sampling_seed"],
            first_query_validator=current_query_evidence_scope,
            oracle_reference=None,
        )
        mechanical = label_current_mechanical_evidence(mechanical)
        mechanical["measured_endpoint_audit"] = verify_measured_endpoints(physics, config, worker, directory)
        rows = [json.loads(line) for line in physics.read(directory / "steps.jsonl").splitlines()]
        result = outcome([row["success"] for row in rows])
        require(result == mechanical["outcome"], "independent native600/tail40 score differs")
        for index in range(75):
            service_record = service["requests"][case["case_index"] * 75 + index]
            request, response = (
                physics.json(directory / f"request_{index:04d}.json"),
                physics.json(directory / f"response_{index:04d}.json"),
            )
            actual_head = None if case["variant"] == "raw" else run["readout_ref"]
            verify_service_request(ev, service_record, case, index, directory, request, response, head_ref=actual_head)
            require(
                calls[index + 1]["request_name"] == request["name"]
                and calls[index + 1]["request_canonical_sha256"] == canonical_sha(request),
                "actual HTTP body witness differs",
            )
            service_dir = (
                root / "service/requests" / f"{case['case_index'] * 75 + index:03d}_{case['case_id']}_q{index:02d}"
            )
            require(
                Path(service_record["request_ref"]["path"]).resolve().parent == service_dir,
                "service request escaped fixed output root",
            )
            trace = ev.json(service_record["trace_ref"])
            numerics.verify_trace(
                trace,
                dict(name=request["name"], sampling_seed=case["sampling_seed"]),
                grid,
                sources,
                prepared["noise_roster"],
                purpose="evaluation",
            )
            require(
                trace["source_frozen"] is True and trace["mask_policy"] == "native_and_registered_active_slot_wrapper",
                "source/mask trace provenance differs",
            )
            conditioning = ev.json(service_record["conditioning_ref"])
            caption = json.loads(conditioning["caption"])
            require(
                caption["task_title"] == caption["command"] == record["task"]
                and caption["task_plan"] == []
                and conditioning["inference_only_current_image"] is True,
                "actual conditioning uses privileged or wrong task input",
            )
            require(
                conditioning["caption_sha256"] == hashlib.sha256(conditioning["caption"].encode()).hexdigest()
                and conditioning["context"]["normalization_id"] == 31
                and conditioning["context"]["projection_id"] == 0
                and conditioning["context"]["domain"] == request["extra"]["action128"]["action_domain"]
                and conditioning["context"]["action_dim_mask"] == [True] * 10 + [False] * 118,
                "actual inference domain/normalization/projection/mask differs",
            )
            ledger = ev.json(service_record["compact_ledger_ref"])
            header = ev.json(ledger["request_ref"])
            require(
                header["request"]
                == dict(
                    request_name=request["name"],
                    noise_seed=case["sampling_seed"],
                    purpose="evaluation",
                    round_name="development",
                    anchor_id=case["case_id"],
                    readout_sha256=None if actual_head is None else actual_head["sha256"],
                    current_image_sha256=service_record["image_ref"]["sha256"],
                )
                and ledger["complete"] is True
                and ledger["future_labels_read"] == 0
                and ledger["omitted_vision_reconstructible"] is ledger["qualification_claim"] is False
                and header["schema_version"] == "s2a.clean-compact-request.v1"
                and header["expected_nodes"] == 35
                and header["expected_vision_shape"] == [1, 48, 11, 60, 40]
                and header["expected_grid"] == [list(pair) for pair in grid]
                and header["labels_accepted"] is header["omitted_vision_reconstructible"] is False,
                "current-only compact request/head ledger differs",
            )
            first, count, mutation = audit_terminal_points(
                ev,
                ledger,
                header,
                service_dir,
                trace["steps"],
                head=None if actual_head is None else head,
                purpose="evaluation",
            )
            verified_points += count
            changed += mutation
            initial = ev.json(service_record["initial_ref"])
            require(
                initial["variant"] == case["variant"] and initial["pair_id"] == case["case_id"],
                "initial current condition belongs to a different policy/case",
            )
            arrays = ev.arrays(initial["arrays"], service_dir / "initial.npz")
            numerics.verify_initial(
                initial,
                arrays,
                dict(name=request["name"], index=case["case_index"] * 75 + index, sampling_seed=case["sampling_seed"]),
                conditioning["caption"],
                first,
            )
            if index == 0:
                key = (case["task_index"], case["sampling_seed"])
                if case["variant"] == "raw":
                    first_pairs[key] = (initial, arrays, service_record["image_ref"])
                else:
                    old, old_arrays, old_image = first_pairs[key]
                    require(
                        old_image["sha256"] == service_record["image_ref"]["sha256"],
                        "raw/W2 first physical image differs",
                    )
                    numerics.paired_initial(initial, arrays, old, old_arrays)
        all_cases.append(dict(case=case, outcome=result, mechanical=mechanical, actual_requests=75, physics_steps=600))
    require(verified_points == 31500 and len(first_pairs) == 6, "full31500 points/six first-current pairs missing")
    physics.recheck()
    ev.finish()
    return dict(
        cases=all_cases,
        score=score(all_cases),
        actual_requests=900,
        actual_denoise_calls=31500,
        actual_head_calls=31500,
        physics_steps=7200,
        model_initializations=1,
        paired_first_current_conditions=6,
        independently_verified_compact_points=verified_points,
        changed_active_values=changed,
        original_state_gate_passed=state["state_gate_passed"],
        source_and_artifact_sha256=dict(ev.inputs, **physics.hashes),
        consumed_before_after_exact=True,
        future_label_archive_decodes=0,
        fresh_cases=0,
        evidence_level="new full simulator task outcomes plus independent saved-log/numerical replay; latency excluded",
    )


def verify(args):
    require(
        args.seed == 42 and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "independent development audit requires emptyCUDA/LD and seed42",
    )
    output = Path(args.output).absolute()
    require(
        output.resolve() == output and output.is_relative_to(REPO / "outputs") and not output.exists(),
        "audit output outside repo outputs or exists",
    )
    from s2a.native_environment import restore_cv2_environment, strict_native_environment

    strict_native_environment()
    guard_doc = json.loads(
        (REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json").read_text()
    )
    from s2a.artifacts import sha256

    require(
        sha256(REPO / "outputs/clean_training_development_preflight_20261023/dependency_addendum.json") == CONTROL_SHA,
        "Qualified native guard changed",
    )
    guard = guard_doc["cv2_guard_registration"]
    restore_cv2_environment(guard["exact_probe_ref"], guard["runtime_source_refs"])
    result = dict(
        schema_version="s2a.serialized-preview-wam-development-verification.v1",
        complete=False,
        passed=False,
        prepared_ref=dict(path=str(Path(args.prepared).resolve()), sha256=args.prepared_sha256),
        execution_registration_ref=dict(
            path=str(Path(args.execution_registration).resolve()), sha256=args.execution_registration_sha256
        ),
        development_ref=dict(path=str(Path(args.development).resolve()), sha256=args.development_sha256),
        new_model_calls=0,
        new_physics_steps=0,
        new_fits=0,
        fresh_performance_authorized=False,
    )
    try:
        result.update(audit(args))
        result.update(complete=True, passed=True)
    except BaseException as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "execution-registration", "development"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42,), required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
