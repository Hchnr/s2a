"""Audit completed recovery physics without rerunning or changing frozen inputs.

The initial coordinator stopped on bitwise float equality during postprocessing.
Raw physical evidence remains untouched. This auditor explicitly allows1e-12
absolute float roundoff and creates only previously absent report artifacts.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from s2a.artifacts import load_config, sha256

FLOAT_ATOL = 1e-12


def compare_metrics(actual, expected, path=""):
    """Require exact schema/discrete values and bounded numerical roundoff."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        return [
            difference
            for key in expected
            for difference in compare_metrics(actual[key], expected[key], f"{path}/{key}")
        ]
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        return [
            difference
            for index, (a, e) in enumerate(zip(actual, expected, strict=True))
            for difference in compare_metrics(a, e, f"{path}/{index}")
        ]
    if isinstance(expected, float):
        difference = abs(actual - expected)
        assert difference <= FLOAT_ATOL, (path, actual, expected)
        return (
            []
            if actual == expected
            else [dict(path=path, actual=actual, stored=expected, absolute_difference=difference)]
        )
    assert actual == expected, (path, actual, expected)
    return []


def independent_endpoint_metrics(rows, references, config):
    """Direct world translation/SO(3) checks, separate from worker pose_error."""
    down = rows[config["requery_step"] - 1]
    down_endpoint = np.asarray(references[0]["targets_tcp"])[-1]
    up_endpoint = np.asarray(references[1]["targets_tcp"])[-1]
    final = np.asarray(rows[-1]["after_tcp"])
    rotation = Rotation.from_matrix(up_endpoint[:3, :3] @ final[:3, :3].T).magnitude()
    return dict(
        down_endpoint_error_m=float(np.linalg.norm(down_endpoint[:3, 3] - np.asarray(down["after_tcp"])[:3, 3])),
        final_endpoint_error_m=float(np.linalg.norm(up_endpoint[:3, 3] - final[:3, 3])),
        final_endpoint_rotation_error_rad=float(rotation),
        physical_elapsed_s=float(sum(row["actual_dt"] for row in rows)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reference_recovery_qualification_20261021.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("requires explicit CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH=''")
    config, root = load_config(args.config)
    assert args.seed == config["seed"] == 42
    for name in (
        "raw_verification.json",
        "qualification_summary.json",
        "output_manifest.json",
        "audit_manifest.json",
        "figures",
    ):
        if (root / name).exists():
            raise FileExistsError(root / name)
    script = Path("scripts/qualify_reference_recovery.py")
    spec = importlib.util.spec_from_file_location("frozen_recovery_qualification", script)
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    frozen_inputs = json.loads((root / "input_manifest.json").read_text())["files_sha256"]
    qualification.check_inputs(frozen_inputs)
    auditor_digest = sha256(__file__)
    results, canonical_results, audit_cases, raw_sha = [], [], [], {}
    for record in config["records"]:
        for method in config["methods"]:
            dest = root / record / method
            result_path = dest / "result.json"
            result = json.loads(result_path.read_text())
            assert result["error"] is None and result["steps"] == config["steps"] == 200
            assert result["reference_queries"] == [0, 100]
            rows = [json.loads(line) for line in (dest / "steps.jsonl").read_text().splitlines()]
            references = [json.loads((dest / f"reference_step{step}.json").read_text()) for step in (0, 100)]
            recomputed = qualification.summarize_rows(rows, references, config)
            differences = compare_metrics(recomputed, result["metrics"])
            independent = independent_endpoint_metrics(rows, references, config)
            for actual, expected in (
                (independent["down_endpoint_error_m"], result["metrics"]["first_down"]["generated_endpoint_error_m"]),
                (
                    independent["final_endpoint_error_m"],
                    result["metrics"]["after_requery"]["final_generated_endpoint_position_error_m"],
                ),
                (
                    independent["final_endpoint_rotation_error_rad"],
                    result["metrics"]["after_requery"]["final_generated_endpoint_rotation_error_rad"],
                ),
                (independent["physical_elapsed_s"], 10.0),
            ):
                assert abs(actual - expected) <= FLOAT_ATOL, (actual, expected)
            initial = json.loads((dest / "initial.json").read_text())
            with np.load(dest / "initial_physics.npz") as physics:
                import hashlib

                assert (
                    hashlib.sha256(physics["state"].tobytes()).hexdigest()
                    == initial["restored_simulation_state_sha256"]
                )
            results.append(result)
            # The frozen geometric/command/raw checker uses bitwise comparison.
            # Its input gets the locally recomputed metrics only after the
            # stored metrics passed the explicit independent tolerance check.
            canonical_results.append(dict(result, metrics=recomputed))
            audit_cases.append(
                dict(
                    record=record,
                    method=method,
                    raw_steps=len(rows),
                    independent_metrics=independent,
                    numerical_differences=differences,
                    all_metrics_within_float_tolerance=True,
                    fixture_initial_state_sha256=initial["fixture_initial_state_sha256"],
                    restored_initial_state_sha256=initial["restored_simulation_state_sha256"],
                )
            )
            for path in (
                result_path,
                dest / "steps.jsonl",
                dest / "initial.json",
                dest / "initial_physics.npz",
                dest / "reference_step0.json",
                dest / "reference_step100.json",
                dest / "manifest.json",
            ):
                raw_sha[str(path.resolve())] = sha256(path)
    qualification.verify_and_plot(config, root, canonical_results)
    # Preserve original recorded worker metrics in the report, rather than
    # replacing any original evidence with the canonical local roundoff.
    qualification.write_new(
        root / "qualification_summary.json",
        dict(
            seed=args.seed,
            expected=4,
            completed=len(results),
            complete=len(results) == 4,
            physical_steps=sum(r["steps"] for r in results),
            results=results,
            metric_recomputation_absolute_tolerance=FLOAT_ATOL,
            limitations="Synthetic reference recovery only. Neither scene met the registered physical-plateau blockage criterion. No WAM task success or real-robot evidence.",
        ),
    )
    qualification.check_inputs(frozen_inputs)
    qualification.check_inputs(raw_sha)
    assert auditor_digest == sha256(__file__)
    qualification.write_new(
        root / "audit_manifest.json",
        dict(
            independent_inputs_sha256=dict(frozen_inputs, **{str(Path(__file__).resolve()): auditor_digest}),
            physical_raw_outputs_sha256=raw_sha,
            cases=audit_cases,
            float_absolute_tolerance=FLOAT_ATOL,
            original_postprocessing_failure="strict dict equality rejected floating roundoff after all4 physics workers completed; frozen physical inputs and raw output remain unchanged",
            all_checks_passed=True,
            verified_physical_steps=800,
        ),
    )
    output_files = {
        str(path.resolve()): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not any(part in ("numba_cache", "matplotlib") for part in path.parts)
    }
    qualification.write_new(
        root / "output_manifest.json",
        dict(
            files_sha256=output_files,
            inputs_sha256=dict(frozen_inputs, **{str(Path(__file__).resolve()): auditor_digest}),
            output_manifest_self_excluded=True,
        ),
    )
    print(
        json.dumps(
            dict(
                audited_steps=800,
                summary=str(root / "qualification_summary.json"),
                audit_sha256=sha256(root / "audit_manifest.json"),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
