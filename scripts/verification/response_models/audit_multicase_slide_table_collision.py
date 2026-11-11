"""Recheck every saved target pairing and positive multi-case collision witness."""

from pathlib import Path

import numpy as np

from s2a.infrastructure.publication import publish_result as publish
from s2a.research.wam.training.native.native_training import state_sha256
from scripts.simulation.runtime import replay_multicase_development_prefix as prefix
from scripts.tracking.response_models import multicase_slide_table_collision as study
from scripts.verification.response_models.audit_slide_range_table_collision import independent_witness_slack

ROOT = prefix.REPO
RESULT = ROOT / "outputs/multicase_slide_table_collision_20261111/result.json"
RESULT_SHA = "12295d9c6feaa309ebcb6fa0fa6fb0968d31a9b85d9d7de8f46425de1c41ee13"
OUTPUT = ROOT / "outputs/multicase_slide_table_collision_audit_20261111/result.json"


def audit_queries(analysis, chunks, states, config):
    """Match every target/state to original files and independently bound positive witnesses."""
    queries = analysis["queries"]
    interval = config["finger_joint_range_m"]
    prefix.require(
        len(queries) == len(chunks) == 75 and len(states) == 601 and analysis["finger_joint_range_m"] == interval,
        "The complete saved query, source and joint-range roster is required",
    )
    table, size = np.asarray(analysis["table_pose"]), np.asarray(analysis["table_half_size_m"])
    prefix.require(
        table.shape == (4, 4)
        and size.shape == (3,)
        and np.isfinite(table).all()
        and np.isfinite(size).all()
        and (size > 0).all(),
        "Finite fixed table geometry is required",
    )
    counts = {"fixed_current": 0, "full_joint_range": 0}
    by_cursor, outside, with_full, checks = [0] * 8, 0, 0, []
    for i, (query, chunk) in enumerate(zip(queries, chunks, strict=True)):
        step = 8 * i
        prefix.require(
            query["query"] == i
            and query["step"] == chunk["step"] == step
            and query["state_sha256"] == state_sha256(np.asarray(states[step], dtype=float))
            and len(query["targets"]) == 8,
            "Query order, original state and eight targets must match",
        )
        vertices, axis = np.asarray(query["vertices_tcp"]), np.asarray(query["slide_axis_tcp"])
        q = float(query["measured_joint_m"])
        in_range = interval[0] <= q <= interval[1]
        prefix.require(
            vertices.ndim == 2
            and vertices.shape[1] == 3
            and len(vertices) >= 4
            and np.isfinite(vertices).all()
            and axis.shape == (3,)
            and np.isfinite(axis).all()
            and abs(np.linalg.norm(axis) - 1) < 1e-10
            and np.isfinite(q)
            and query["measured_joint_in_range"] is bool(in_range),
            "Finite query geometry and the measured joint-range flag are required",
        )
        outside += not in_range
        original = np.asarray(chunk["effective_targets_tcp"], dtype=float)
        prefix.require(original.shape == (40, 4, 4), "The original forty generated targets are required")
        found_full = False
        for cursor, target in enumerate(query["targets"]):
            pose = np.asarray(target["target_tcp"])
            prefix.require(
                target["cursor"] == cursor and np.array_equal(pose, original[cursor]) and np.isfinite(pose).all(),
                "Each target must match its original query and cursor exactly",
            )
            decision = {"original_target_tcp": pose, "decision_vertices_tcp": vertices}
            for kind in counts:
                witness = target[kind]
                target_range = [q, q] if kind == "fixed_current" else interval
                prefix.require(
                    isinstance(witness["certified"], bool)
                    and witness["joint_target_interval_m"] == target_range
                    and witness["measured_joint_m"] == q
                    and witness["numerical_guard_m"] == config["numerical_margin_m"],
                    "Each saved witness must carry its original interval and guard",
                )
                if not witness["certified"]:
                    continue
                vertex = witness["vertex_index"]
                prefix.require(
                    isinstance(vertex, int)
                    and 0 <= vertex < len(vertices)
                    and np.array_equal(witness["vertex_tcp"], vertices[vertex]),
                    "Each positive witness must identify the original material vertex",
                )
                slack = independent_witness_slack(decision, axis, q, target_range, table, size, vertex, config)
                prefix.require(slack > witness["numerical_guard_m"], "Independent bound must prove strict interior")
                counts[kind] += 1
                checks.append(dict(query=i, step=step, cursor=cursor, kind=kind, independent_slack_m=slack))
                if kind == "full_joint_range":
                    by_cursor[cursor] += 1
                    found_full = True
                    prefix.require(not in_range or target["fixed_current"]["certified"], "Full range contains current")
        with_full += found_full
    return {
        "case": analysis["case"],
        "targets": 600,
        "queries": 75,
        "fixed_current_certified": counts["fixed_current"],
        "full_joint_range_certified": counts["full_joint_range"],
        "queries_with_full_range_certificate": with_full,
        "queries_current_joint_outside_range": outside,
        "full_range_by_cursor": by_cursor,
    }, checks


def execute():
    inputs = study.load_inputs(verify_assets=True)
    result_ref = {"path": str(RESULT), "sha256": RESULT_SHA}
    result = prefix.read(result_ref)
    prefix.require(
        result["complete"] is result["passed"] is result["sources_unchanged"] is True
        and result["control_qualified"] is False
        and result["inputs"] == study.input_receipt(inputs)
        and len(result["case_refs"]) == len(result["summaries"]) == 12
        and result["totals"]
        == dict(
            models_completed=12,
            forward_queries_completed=900,
            certificate_calculations=14400,
            physics_steps_attempted=0,
        )
        and result["world_model_calls"] == result["fresh_cases"] == 0,
        "A complete pinned twelve-case static study is required",
    )
    for ref in result["source_refs"]:
        prefix.checked(ref)
    lifecycle = prefix.read(result["workers_ref"])
    prefix.require(
        lifecycle["launch_error"] is None
        and [w["key"] for w in lifecycle["workers"]] == [f"case_{i:02d}" for i in range(12)]
        and all(
            w["actual_wait"] is w["reaped"] is True and w["returncode"] == 0 and not w["signal_sent"]
            for w in lifecycle["workers"]
        ),
        "All twelve owned workers must have actually exited and been reaped",
    )
    summaries, checks = [], []
    for index, ref in enumerate(result["case_refs"]):
        case = prefix.read(ref)
        source = study.case_inputs(index, inputs)
        prefix.require(
            case["complete"] is case["passed"] is True
            and case["case_index"] == index
            and case["analysis"]["case"] == source["case"]
            and case["analysis"]["input_sha256"] == source["source_sha256"]
            and case["stats"]
            == dict(
                models_attempted=1,
                models_completed=1,
                forward_queries_attempted=75,
                forward_queries_completed=75,
                certificate_calculations=1200,
                physics_steps_attempted=0,
            ),
            "Each result must bind its complete original source and static budget",
        )
        with np.load(source["files"]["arrays"], allow_pickle=False) as archive:
            states = archive["sim_states"].copy()
            prefix.require(archive["endpoint_step"].tolist() == list(range(601)), "Complete source states required")
        summary, case_checks = audit_queries(case["analysis"], source["chunks"], states, inputs["config"])
        prefix.require(summary == result["summaries"][index], "Independently recounted case summary must match")
        summaries.append(summary)
        checks.extend(dict(case_index=index, **check) for check in case_checks)
    sources = [
        prefix.reference(Path(__file__).resolve()),
        prefix.reference(ROOT / "tests/research/response_models/test_multicase_slide_table_collision_audit.py"),
        prefix.reference(Path(independent_witness_slack.__code__.co_filename).resolve()),
    ]
    print(
        publish(
            OUTPUT,
            dict(
                complete=True,
                passed=True,
                control_qualified=False,
                result_ref=result_ref,
                source_refs=sources,
                cases=12,
                queries_checked=900,
                targets_checked=7200,
                positive_witnesses={
                    kind: sum(s[kind + "_certified"] for s in summaries)
                    for kind in ("fixed_current", "full_joint_range")
                },
                checks=checks,
                summaries=summaries,
                native_models=0,
                forward_queries=0,
                physics_steps=0,
                world_model_calls=0,
                fresh_cases=0,
                scope="Source pairing and positive saved-geometry witnesses only; negative results remain inconclusive",
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
