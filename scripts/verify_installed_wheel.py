"""Verify an isolated installed wheel, its sources and optional-dependency boundary."""

import argparse
import hashlib
import importlib
import json
import pkgutil
import sys
from pathlib import Path

import numpy as np

import s2a
from s2a.geometry import advance
from s2a.gripper import OpeningIntent
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.perturbations import ActuatorDelay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/libero_audit_20261019/wheel_smoke_final.json")
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("seed differs from frozen validation")
    repo = Path(__file__).resolve().parents[1]
    installed = Path(s2a.__file__).parent
    if not installed.is_relative_to(Path(sys.prefix)) or installed.is_relative_to(repo / "src"):
        raise ValueError("run with the isolated wheel interpreter and -I")
    for module in pkgutil.iter_modules(s2a.__path__):
        importlib.import_module("s2a." + module.name)
    if any(k.startswith(("mujoco", "libero.libero", "cosmos_framework")) for k in sys.modules):
        raise RuntimeError("importing the package loaded optional simulator/model code")
    hashes = {}
    for path in (repo / "src/s2a").glob("*.py"):
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = hashlib.sha256((installed / path.name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"wheel source differs: {path.name}")
        hashes[path.name] = actual
    interface = OSCInterface(
        -np.ones(6), np.ones(6), -np.array([0.05] * 3 + [0.5] * 3), np.array([0.05] * 3 + [0.5] * 3)
    )
    converter = OSCStateToAction(interface, np.full(6, 150), np.full(6, 2 * np.sqrt(150)), 0.05, compensate=True)
    motion = np.array([0.002, 0, 0, 0, 0, 0.005])
    result = converter.convert(np.eye(4), advance(np.eye(4), motion, 1), np.zeros(6))
    np.testing.assert_allclose(result["predicted_motion"], motion, atol=1e-12)
    gripper = OpeningIntent(1)
    assert gripper.update(0.7) == 1 and gripper.update(0.7) == 1
    delay = ActuatorDelay(1, 1)
    np.testing.assert_array_equal(delay.update([1]), [0])
    np.testing.assert_array_equal(delay.update([2]), [1])
    output = (repo / args.output).resolve()
    if not output.is_relative_to(repo / "outputs"):
        raise ValueError("wheel validation output must stay in repository outputs")
    output.write_text(
        json.dumps(
            dict(
                passed=True,
                seed=args.seed,
                installed_package=str(installed),
                source_sha256=hashes,
                modules=len(hashes),
                heavy_simulator_or_cosmos_imported=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Installed wheel smoke and {len(hashes)} source SHA checks passed")


if __name__ == "__main__":
    main()
