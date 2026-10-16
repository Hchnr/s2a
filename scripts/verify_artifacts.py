"""Verify persisted fingerprints and all final sent-command bounds."""

import argparse
import json

import numpy as np

from s2a.artifacts import REPO, load_config, sha256
from s2a.data import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/droid_episode0.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config, out = load_config(args.config)
    require(args.seed == config["seed"], "seed differs from saved configuration")
    manifest = json.loads((out / "sample_manifest.json").read_text())
    hashes = json.loads((out / "artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        require(sha256(out / name) == digest, f"artifact changed: {name}")
    for name, digest in manifest["code"]["files_sha256"].items():
        require(sha256(REPO / name) == digest, f"experiment source changed: {name}")
    require(sha256(out / "sample.npz") == manifest["sample_sha256"], "sample fingerprint mismatch")
    params = json.loads((out / "adapter_params.json").read_text())
    closed = json.loads((out / "closed_loop_model_metrics.json").read_text())
    require(closed["executed"], "model rollout was not executed; cannot validate as passed")
    dt = params["dt_command_planned_s"]
    interface = config["reference_interface"]["parameters"]
    limits = config["rollout"]["pid_limits"]
    rates = [
        limits["max_translation_acceleration"] * dt / interface["max_translation_delta"],
        limits["max_rotation_acceleration"] * dt / interface["max_rotation_delta"],
    ]
    count = 0
    with np.load(out / "closed_loop_traces.npz", allow_pickle=False) as traces:
        for key in traces.files:
            require(np.isfinite(traces[key]).all(), f"nonfinite rollout trace: {key}")
        for feedforward, methods in closed["results"].items():
            for method, values in methods.items():
                for scenario, value in values["test"].items():
                    require(value["valid"], f"invalid rollout: {feedforward}/{method}/{scenario}")
                    prefix = f"{feedforward}__{method}__{scenario}__"
                    command, times = traces[prefix + "command"], traces[prefix + "times"]
                    require((np.diff(times) > 0).all(), "invalid feedback clock")
                    change = np.diff(command, axis=0) / np.diff(times)[:, None]
                    for part, rate in [(slice(0, 3), rates[0]), (slice(3, 6), rates[1])]:
                        require(
                            np.max(np.linalg.norm(command[:, part], axis=1)) <= 1 + 1e-10,
                            f"sent-command norm bound failed: {prefix}",
                        )
                        require(
                            np.max(np.linalg.norm(change[:, part], axis=1)) <= rate + 1e-10,
                            f"sent-command rate bound failed: {prefix}",
                        )
                    count += 1
    expected = (
        len(config["rollout"]["methods"])
        * len(config["rollout"]["feedforward_kinds"])
        * len(config["rollout"]["scenarios"])
    )
    require(count == expected, "missing rollout cases")
    print(
        f"Verified {len(hashes)} artifact hashes, {len(manifest['code']['files_sha256'])} source hashes, {count} test rollouts and final command bounds."
    )
    print("This validates artifact integrity and constraints; control-performance acceptance is reported separately.")


if __name__ == "__main__":
    main()
