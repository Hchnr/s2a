"""Prepare the explicitly configured serving entry without replacing prior evidence."""

import argparse
import json
from pathlib import Path

from s2a.artifacts import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_followup_20261019.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, output = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs from frozen config")
    policy = json.loads(Path(c["policy_config"]).read_text())
    if not 0 < c["async_max_age_s"] <= policy["execution_horizon"] / policy["control_fps"]:
        raise ValueError("reference validity exceeds the fixed execution horizon")
    for key in ("source_identity", "native_bridge", "published_bridge", "simulation_python"):
        if not Path(c[key]).is_file():
            raise FileNotFoundError(f"required frozen prerequisite {key}: {c[key]}")
    policy["output_dir"] = c["output_dir"]
    policy["server_url"] = f"http://127.0.0.1:{c['server_port']}"
    policy["server_urls"] = [policy["server_url"]] * len(c["trials"])
    target = output / "serving_config.json"
    if target.exists():
        if json.loads(target.read_text()) != policy:
            raise FileExistsError("existing serving config differs; archive evidence or use a new output directory")
    else:
        write_json(target, policy)
    print(target)


if __name__ == "__main__":
    main()
