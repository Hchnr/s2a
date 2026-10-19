"""Fetch only missing official LIBERO task demonstrations into S2A outputs."""

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

from s2a.artifacts import code_fingerprint, load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, out = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    dest = out / "assets"
    dest.mkdir(exist_ok=False)
    repo = "yifengzhu-hf/LIBERO-datasets"
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as stream:
        revision = json.load(stream)["sha"]
    with urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/libero_goal?expand=true", timeout=30
    ) as stream:
        listing = json.load(stream)
    wanted = ["libero_goal/" + task.replace(" ", "_") + "_demo.hdf5" for task in c["tasks"][1:]]
    selected = [r for r in listing if r["path"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("missing official task files")
    write_json(
        dest / "source_manifest.json", dict(code=code_fingerprint(), repo=repo, revision=revision, files=selected)
    )

    def fetch(row):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['path']}"
        path = dest / Path(row["path"]).name
        partial = path.with_suffix(".partial")
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=60) as stream, partial.open("xb") as target:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
                size += len(block)
                target.write(block)
        expected = row.get("lfs", {}).get("oid")
        if size != row["size"] or (expected and digest.hexdigest() != expected):
            raise ValueError("download size or SHA mismatch")
        partial.rename(path)
        result = dict(path=str(path), url=url, bytes=size, sha256=digest.hexdigest(), verified_lfs_oid=expected)
        write_json(path.with_suffix(".json"), result)
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, selected))
    write_json(dest / "download_complete.json", dict(files=results))


if __name__ == "__main__":
    main()
