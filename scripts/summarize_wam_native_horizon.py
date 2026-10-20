"""Summarize the predeclared latest-checkpoint native-window physics diagnosis."""

import argparse
import json
import shutil

from analyze_wam_acceptance import physics_diagnostics

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_native_h40_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--primary-config", default="configs/wam_checkpoint_masked_20261019v2.json")
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    if args.prepare:
        primary, source = load_config(args.primary_config)
        for key in set(primary) | set(c):
            if key not in ("output_dir", "online_chunk_size", "checkpoints", "notes") and primary[key] != c[key]:
                raise ValueError(f"unexpected supplementary protocol change: {key}")
        if c["online_chunk_size"] != 40 or c["checkpoints"] != [
            r for r in primary["checkpoints"] if r["name"] == "8800_masked"
        ]:
            raise ValueError("supplement must evaluate latest 8800 at native horizon40")
        if any(root.iterdir()):
            raise FileExistsError("supplement preparation refuses to overwrite existing evidence")
        (root / "simulation").mkdir()
        for relative in ("simulation/fixtures.json", "checkpoint_fingerprints.json"):
            shutil.copyfile(source / relative, root / relative)
        write_json(
            root / "protocol.json",
            dict(
                code=code_fingerprint(),
                config=c,
                config_sha256=sha256(args.config),
                primary_config_sha256=sha256(args.primary_config),
                fixtures_sha256=sha256(root / "simulation/fixtures.json"),
                declared_differences=[
                    "online_chunk_size:16 -> 40",
                    "only latest 8800; supplementary, not checkpoint ranking",
                ],
                reason="Exclude shortened sampling horizon before attributing closed-loop failure to frozen weights.",
            ),
        )
        return
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["config_sha256"] != sha256(args.config):
        raise ValueError("native-window protocol changed after freezing")
    fixtures = json.loads((root / "simulation/fixtures.json").read_text())["records"]
    if protocol["fixtures_sha256"] != sha256(root / "simulation/fixtures.json"):
        raise ValueError("initial states changed")
    expected = {(r["name"], seed) for r in fixtures for seed in c["sampling_seeds"]}
    paths = sorted((root / "simulation/8800_masked").glob("*/result.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    valid = [r for r in rows if r["error"] is None and r["steps"] == r["expected_steps"] == c["online_max_steps"]]
    keys = [(r["record"], r["sampling_seed"]) for r in valid]
    if len(keys) != len(set(keys)) or not set(keys).issubset(expected):
        raise ValueError("unexpected or duplicate initial-state/seed pair")
    complete = len(valid) == len(rows) == len(expected)
    if not args.partial and not complete:
        raise ValueError("native-window evidence is incomplete")
    report = dict(
        complete=complete,
        code=code_fingerprint(),
        config=c,
        checkpoint_matrices=[],
        physics={
            "8800_masked": dict(
                expected=len(expected),
                completed=len(valid),
                complete=complete,
                success=sum(r["success"] for r in valid),
                terminal_success=sum(r["terminal_success"] for r in valid),
                cases=rows,
                diagnostics=physics_diagnostics(paths, "8800_masked"),
            )
        },
        fingerprints={str(p): sha256(p) for p in paths},
        interpretation=[
            "Supplementary latest-checkpoint assessment under its native 40-step generation window.",
            "The four-checkpoint primary comparison retains the identical 16-step protocol for every checkpoint.",
            "Only the generated horizon changes; execution horizon, task input, seeds, initial states and physics budget are fixed.",
            "Six initial states with two sampling seeds; not certified unseen during pretraining.",
        ],
    )
    out = root / ("summary_partial.json" if args.partial else "summary.json")
    if out.exists() and not args.partial:
        raise FileExistsError("complete native-window summaries must not be silently overwritten")
    write_json(out, report)
    print(
        json.dumps(
            dict(
                complete=complete,
                completed=len(valid),
                expected=len(expected),
                success=sum(r["success"] for r in valid),
            )
        )
    )


if __name__ == "__main__":
    main()
