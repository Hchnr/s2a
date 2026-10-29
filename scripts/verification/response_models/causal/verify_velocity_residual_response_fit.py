"""079 independent raw-state rows, split/weights and augmented least squares audit."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path("outputs/velocity_residual_response_20261025")
COLLECTION = Path("outputs/native_gripper_response_interventions_20261025_v3")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    fit = json.loads((ROOT / "fit_result.json").read_text())
    params = json.loads((ROOT / "parameters.json").read_text())
    hashes = dict(fit["source_sha256"])
    hashes.update(
        {
            str(ROOT / "fit_result.json"): digest(ROOT / "fit_result.json"),
            str(ROOT / "parameters.json"): fit["parameters_sha256"],
            str(ROOT / "fit_cache.npz"): fit["cache_sha256"],
            __file__: digest(__file__),
        }
    )
    assert all(digest(p) == s for p, s in hashes.items())
    roster = json.loads(Path("outputs/gripper_forward_identification_20261025/fit_result.json").read_text())[
        "trajectories"
    ]
    velocity_audit = json.loads(Path("outputs/gripper_velocity_state_audit_20261025/audit.json").read_text())
    cases = {d["ref"]["path"]: d["case_ref"] for d in velocity_audit["cases"]}
    specs = []
    for entry in roster:
        if entry["split"] == "fit":
            ref = cases[entry["ref"]["path"]]
            assert digest(ref["path"]) == ref["sha256"]
            hashes[ref["path"]] = ref["sha256"]
            record = json.loads(Path(ref["path"]).read_text())["record"]
            specs.append(
                dict(
                    kind="original_native_fit",
                    ref=entry["ref"],
                    task=entry["task"],
                    xml=record["model_xml"],
                    start=0,
                    total_weight=0.5 / 24,
                )
            )
    assert len(specs) == 24
    registration_path = Path("configs/native_gripper_response_interventions_20261025_v3.json")
    hashes[str(registration_path)] = digest(registration_path)
    registration = json.loads(registration_path.read_text())
    for case in (1, 3, 5, 7, 9, 11):
        profile = registration["profiles"][case]
        assert profile["branch"] in ("open", "close")
        path = COLLECTION / f"case{case:02d}" / "collection_result.json"
        result = json.loads(path.read_text())
        assert profile == result["profile"]
        xml_path = path.parent / "model.xml"
        hashes[str(path)] = digest(path)
        hashes[str(xml_path)] = digest(xml_path)
        specs.append(
            dict(
                kind="new_intervention_fit",
                ref=result["arrays_ref"],
                task=profile["task"],
                case=case,
                xml=xml_path.read_text(),
                start=profile["anchor_cycle"],
                total_weight=0.5 / 6,
            )
        )
    raw = []
    target = []
    weights = []
    groups = []
    for group, (spec, declared) in enumerate(zip(specs, fit["datasets"], strict=True)):
        assert spec["kind"] == declared["kind"] and spec["ref"] == declared["ref"] and spec["task"] == declared["task"]
        path = spec["ref"]["path"]
        assert digest(path) == spec["ref"]["sha256"]
        hashes[path] = spec["ref"]["sha256"]
        nq = nv = 0
        indices = {}
        for node in ET.fromstring(spec["xml"]).find("worldbody").iter():
            if node.tag not in ("joint", "freejoint"):
                continue
            kind = "free" if node.tag == "freejoint" else node.get("type", "hinge")
            if node.get("name") in ("gripper0_finger_joint1", "gripper0_finger_joint2"):
                indices[node.get("name")] = (nq, nv)
            a, b = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}[kind]
            nq += a
            nv += b
        with np.load(path, allow_pickle=False) as data:
            q = data["opening"]
            states = data["sim_states"]
            u = data["sent_commands"][:, -1]
            off = data["gripper_current_action_offsets"]
            flat = data["gripper_current_action_flat"]
            assert states.shape == (len(q), 1 + nq + nv)
            pids = [1 + indices[f"gripper0_finger_joint{i}"][0] for i in (1, 2)]
            vids = [1 + nq + indices[f"gripper0_finger_joint{i}"][1] for i in (1, 2)]
            np.testing.assert_array_equal(states[:, pids], data["gripper_qpos"])
            v = 0.05 * ((states[:, vids[0]] - states[:, vids[1]]) / 0.079)
            start = spec["start"]
            stop = len(u) if spec["kind"] == "original_native_fit" else start + 24
            assert (
                declared["source_start"] == start
                and declared["source_stop"] == stop
                and declared["row_start"] == len(raw)
            )
            for k in range(start, stop):
                m = float(flat[off[k]])
                nextm = m
                increment = -0.01 if u[k] > 0 else (0.01 if u[k] < 0 else 0.0)
                for _ in range(25):
                    nextm = max(-1.0, min(1.0, nextm + increment))
                assert abs(nextm - float(flat[off[k + 1]])) < 1e-12
                raw.append([q[k], m, nextm, v[k]])
                target.append([q[k + 1] - q[k] - v[k], v[k + 1] - v[k]])
                weights.append(spec["total_weight"] / (stop - start))
                groups.append(group)
            assert declared["row_stop"] == len(raw)
    raw, target, weights, groups = np.array(raw), np.array(target), np.array(weights), np.array(groups)
    with np.load(ROOT / "fit_cache.npz", allow_pickle=False) as cached:
        for values, name in [(raw, "raw"), (target, "target"), (weights, "weights")]:
            np.testing.assert_allclose(values, cached[name], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(groups, cached["groups"])
    assert len(raw) == 5009 and params["ridge"] == 0.001
    np.testing.assert_allclose(
        [weights[groups < 24].sum(), weights[groups >= 24].sum()], [0.5, 0.5], rtol=0, atol=1e-12
    )
    mean = (weights[:, None] * raw).sum(axis=0)
    scale = np.maximum(np.sqrt((weights[:, None] * (raw - mean) ** 2).sum(axis=0)), 1e-8)
    np.testing.assert_allclose(mean, params["mean"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(scale, params["scale"], rtol=0, atol=1e-12)
    z = (raw - mean) / scale
    x = np.column_stack([z] + [z[:, i] * z[:, j] for i in range(4) for j in range(i, 4)] + [np.ones(len(raw))])
    penalty = np.eye(15) * np.sqrt(0.001)
    penalty[-1, -1] = 0
    a = np.vstack([np.sqrt(weights)[:, None] * x, penalty])
    b = np.vstack([np.sqrt(weights)[:, None] * target, np.zeros((15, 2))])
    independent = np.linalg.lstsq(a, b, rcond=None)[0]
    coef = np.array(params["coefficients"])
    difference = float(np.max(np.abs(independent - coef)))
    gradient = float(np.max(np.abs(a.T @ (a @ coef - b))))
    assert difference < 1e-9 and gradient < 1e-10
    assert all(digest(p) == s for p, s in hashes.items())
    output = dict(
        numerically_verified=True,
        verified_rows=len(raw),
        coefficient_max_abs_difference=difference,
        gradient_max_abs=gradient,
        source_sha256=hashes,
        validation_executed=False,
        physical_qualification=False,
        new_candidate_fits=0,
        model_calls=0,
        physics_steps=0,
        fresh_cases=0,
    )
    with (ROOT / "independent_fit_verified.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print("verified rows", len(raw), "coefficient difference", difference, "gradient", gradient)


if __name__ == "__main__":
    main()
