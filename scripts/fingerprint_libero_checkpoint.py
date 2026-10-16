"""Hash the complete frozen checkpoint and the read-only inference sources."""

import argparse
import subprocess
from pathlib import Path

from s2a.artifacts import load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_simulation.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config)
    if args.seed != config["seed"]:
        raise ValueError("seed differs from frozen config")
    run = Path(config["checkpoint_run"])
    iteration = run / "checkpoints" / config["checkpoint_iteration"]
    paths = [run / "config.yaml", iteration / ".save_complete", *sorted((iteration / "model").iterdir())]
    hashes = {}
    total = 0
    for path in paths:
        if path.is_file():
            hashes[str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            total += path.stat().st_size
    repo = Path(config["cosmos_repo"])
    sources = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "cosmos_framework/inference",
            "cosmos_framework/model",
            "cosmos_framework/checkpoint",
        ],
        text=True,
    ).splitlines()
    write_json(
        output / "checkpoint_fingerprints.json",
        dict(
            checkpoint_files=hashes,
            total_checkpoint_bytes=total,
            cosmos_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            cosmos_sources_sha256={name: sha256(repo / name) for name in sources if (repo / name).is_file()},
        ),
    )
    print(f"Hashed {len(hashes)} checkpoint files, {total} bytes, {len(sources)} inference/model/checkpoint sources")


if __name__ == "__main__":
    main()
