"""Preserve all native-success intervals and final40 endpoints of a complete batch."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intervals(flags):
    result = []
    start = None
    for step, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = step
        elif not flag and start is not None:
            result.append(dict(start_step=start, end_step_exclusive=step, successful_endpoints=step - start))
            start = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path, help="optional new PNG path; only a complete12-case batch is plotted")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("outcome audit requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    outputs = [args.output] + ([args.plot] if args.plot else [])
    for path in outputs:
        if path.exists() or not path.resolve().is_relative_to(Path("outputs").resolve()):
            raise ValueError("outputs must be new files below repository outputs")
    audit_sha = sha(args.audit)
    audit = json.loads(args.audit.read_text())
    assert audit["complete"] and audit["all12_cases_independently_verified"] and len(audit["cases"]) == 12
    records, hashes = [], {str(args.audit.resolve()): audit_sha, str(Path(__file__).resolve()): sha(__file__)}
    for item in audit["cases"]:
        case = Path(item["case"])
        path = case / "steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        flags = [row["success"] for row in rows]
        assert len(flags) == 600 and all(type(flag) is bool for flag in flags)
        runs = intervals(flags)
        trailing = 600 - runs[-1]["start_step"] if flags[-1] else 0
        expected = dict(
            ever=any(flags),
            terminal=flags[-1],
            stable=all(flags[-40:]),
            successful_steps=sum(flags),
            trailing_success_steps=trailing,
        )
        result_path = case / "result.json"
        result = json.loads(result_path.read_text())
        assert expected == result["task_outcome"] == item["outcome"]
        assert result["error"] is None and result["complete"] and result["steps"] == 600
        base_time = rows[0]["physical_time_before_s"]
        for run in runs:
            run["first_success_endpoint_elapsed_s"] = rows[run["start_step"]]["physical_time_after_s"] - base_time
            run["last_success_endpoint_elapsed_s"] = (
                rows[run["end_step_exclusive"] - 1]["physical_time_after_s"] - base_time
            )
        hashes[str(path.resolve())], hashes[str(result_path.resolve())] = sha(path), sha(result_path)
        records.append(
            dict(
                case=str(case),
                variant=item["variant"],
                record=item["record"],
                raw_checkpoint_evidence=item["raw_checkpoint_evidence"],
                component_ablation=item["component_ablation"],
                outcome=expected,
                success_intervals=runs,
                success_flags=flags,
                final40=[
                    dict(
                        step=row["step"],
                        success=row["success"],
                        physical_endpoint_elapsed_s=row["physical_time_after_s"] - base_time,
                    )
                    for row in rows[-40:]
                ],
            )
        )
    for path, digest in hashes.items():
        assert sha(path) == digest
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,
        source_audit=str(args.audit),
        source_audit_sha256=audit_sha,
        expected_cases=12,
        verified_steps=7200,
        ever_successes=sum(record["outcome"]["ever"] for record in records),
        stable_successes=sum(record["outcome"]["stable"] for record in records),
        records=records,
        input_sha256=hashes,
        model_calls=0,
        physics_steps=0,
        scope="Independently aggregate recorded native predicate endpoints; no new predicate/physics execution. "
        "All event runs and final40 endpoint booleans are preserved. Alpha0 component success is not raw "
        "full-checkpoint success; only old development scenes are included.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for index, record in enumerate(records):
            ranges = [(run["start_step"] / 20, run["successful_endpoints"] / 20) for run in record["success_intervals"]]
            axis.broken_barh(ranges, (index - 0.3, 0.6), facecolors="#26975b")
        axis.axvspan(28, 30, color="#6e9fd1", alpha=0.16)
        axis.set(yticks=range(12), yticklabels=[f"{r['variant']} / {r['record']}" for r in records], xlim=(0, 30))
        axis.invert_yaxis()
        axis.set_xlabel("Physical elapsed time (s); native predicate sampled at 20 Hz")
        axis.set_title("8400 guidance1.0: complete12-case development batch")
        axis.legend(
            handles=[
                Patch(color="#26975b", label="Recorded native-success endpoints"),
                Patch(color="#6e9fd1", alpha=0.16, label="Final40 endpoint scoring window"),
            ],
            loc="upper right",
        )
        axis.grid(axis="x", alpha=0.2)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=180)
        plt.close(figure)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha(args.output),
                stable=value["stable_successes"],
                ever=value["ever_successes"],
            )
        )
    )


if __name__ == "__main__":
    main()
