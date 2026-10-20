"""Read-only finite017 prefix observation; never substitute for final1200 audit.

Snapshot only the requested complete JSONL prefix from each live log. Every
completed query file is immutable and SHA checked; the growing original JSONL
is explicitly not claimed immutable. No model, render, HTTP or physics calls.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_long_wam import OLD, OLD_SHA
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    assert 8 <= args.steps <= 600 and args.steps % 8 == 0
    assert args.output.resolve().is_relative_to(REPO / "outputs") and not args.output.exists()
    args.output.mkdir(parents=True)
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(OLD, OLD_SHA)
    previous = evidence.json(OLD)
    old = {c["record"]: Path(c["case"]) for c in previous["cases"] if c["profile"] == "interval8_alpha1"}
    results = []
    for part in range(3):
        name = f"task{part}_init0"
        config_path = REPO / f"configs/wam_long_raw_task{part}_20261021.json"
        config = evidence.json(config_path)
        assert config["records"] == [name] and config["sampling_seeds"] == [42]
        case = REPO / config["output_dir"] / "simulation/8400_masked_position_hold" / f"{name}_s42"
        source = case / "steps.jsonl"
        captured = source.read_bytes()
        complete_lines = captured.splitlines(keepends=True)
        assert len(complete_lines) >= args.steps and all(p.endswith(b"\n") for p in complete_lines[: args.steps])
        prefix_bytes = b"".join(complete_lines[: args.steps])
        archive = args.output / f"{name}_steps_prefix.jsonl"
        with archive.open("xb") as stream:
            stream.write(prefix_bytes)
        rows = [json.loads(line) for line in evidence.read(archive).splitlines()]
        old_rows = [json.loads(line) for line in evidence.read(old[name] / "steps.jsonl").splitlines()]
        assert [r["step"] for r in rows] == list(range(args.steps))
        physical_equal = canonical(rows) == canonical(old_rows[: args.steps])
        differences = [
            dict(step=i, fields=[k for k in set(a) | set(b) if canonical(a.get(k)) != canonical(b.get(k))])
            for i, (a, b) in enumerate(zip(rows, old_rows[: args.steps], strict=True))
            if canonical(a) != canonical(b)
        ]
        queries = []
        for index in range(args.steps // 8):
            png_equal = evidence.read(case / f"input_{index:04d}.png") == evidence.read(
                old[name] / f"input_{index:04d}.png"
            )
            current = evidence.json(case / f"response_{index:04d}.json")["outputs"][0]["content"]
            prior = evidence.json(old[name] / f"response_{index:04d}.json")["outputs"][0]["content"]
            raw_equal = np.array_equal(current["action"], prior["action"])
            normalized_equal = np.array_equal(
                current["action128"]["normalized_action"], prior["action128"]["normalized_action"]
            )
            assert (
                np.asarray(current["action"]).shape
                == np.asarray(current["action128"]["normalized_action"]).shape
                == (40, 128)
            )
            requests = [evidence.json(p / f"request_{index:04d}.json") for p in (case, old[name])]
            condition_keys = (
                "model_mode",
                "seed",
                "fps",
                "num_frames",
                "action_chunk_size",
                "guidance",
                "num_steps",
                "shift",
                "prompt",
                "extra",
            )
            conditions_equal = all(canonical(requests[0][k]) == canonical(requests[1][k]) for k in condition_keys)
            queries.append(
                dict(
                    index=index,
                    png_bytes_exact=png_equal,
                    raw_full40x128_exact=raw_equal,
                    normalized_full40x128_exact=normalized_equal,
                    common_conditions_exact=conditions_equal,
                )
            )
        results.append(
            dict(
                record=name,
                case=str(case),
                old_case=str(old[name]),
                steps=args.steps,
                captured_prefix_path=str(archive.resolve()),
                captured_prefix_sha256=sha256(archive),
                live_original_path=str(source),
                live_original_bytes_at_read_sha256=hashlib.sha256(captured).hexdigest(),
                live_original_full_file_not_claimed_immutable=True,
                all_physical_prefix_canonical_tokens_exact=physical_equal,
                differences=differences,
                completed_queries_compared=len(queries),
                queries=queries,
            )
        )
    evidence.recheck()
    passed = all(
        c["all_physical_prefix_canonical_tokens_exact"]
        and all(
            all(
                q[k]
                for k in (
                    "png_bytes_exact",
                    "raw_full40x128_exact",
                    "normalized_full40x128_exact",
                    "common_conditions_exact",
                )
            )
            for q in c["queries"]
        )
        for c in results
    )
    output = args.output / "prefix_observation.json"
    write_new(
        output,
        dict(
            schema_version="s2a.long-raw-available-prefix-observation.v1",
            created_utc=datetime.now(timezone.utc).isoformat(),
            available_prefix_consistent=passed,
            observed_steps_per_task=args.steps,
            tasks=results,
            full1200_independent_audit=False,
            performance_acceptance_claimed=False,
            no_task_success_rate_computed=True,
            new_model_requests=0,
            new_physics_steps=0,
            input_sha256=evidence.hashes,
        ),
    )
    print(
        json.dumps(
            dict(
                path=str(output),
                sha256=sha256(output),
                available_prefix_consistent=passed,
                compared_physical_endpoints=3 * args.steps,
                compared_queries=3 * args.steps // 8,
            )
        )
    )
    if not passed:
        raise SystemExit("recorded prefix discrepancy; no runtime, retry or budget change made")


if __name__ == "__main__":
    main()
