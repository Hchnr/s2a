"""Independent CPU post-checks of the fixed nine-case position/hold protocol."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from audit_wam_recovery import Evidence, RecoveryReplay, decode, error, mapping

from s2a.artifacts import REPO, sha256
from s2a.geometry import inverse
from s2a.osc import OSCInterface, OSCStateToAction
from s2a.task_acceptance import task_outcome

RECORDS = ("task0_init0", "task1_init0", "task2_init0")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_baseline(config, root):
    checks = []
    signatures = {}
    for name in RECORDS:
        old = REPO / config["qualification"]["historical_baseline_root"] / f"{name}_s42"
        new = root / "simulation/oracle_baseline_v2" / f"{name}_s42"
        previous, current = rows(old / "steps.jsonl"), rows(new / "steps.jsonl")
        assert len(previous) == len(current) == 600
        fields = {}
        keys = set(previous[0]) | set(current[0])
        for key in sorted(keys):
            changed = [
                i
                for i, (a, b) in enumerate(zip(previous, current, strict=True))
                if key not in a or key not in b or a[key] != b[key]
            ]
            fields[key] = dict(exact=not changed, first_difference_step=None if not changed else changed[0])
        old_chunks, new_chunks = read(old / "chunks.json"), read(new / "chunks.json")
        chunk_checks = []
        assert len(old_chunks) == len(new_chunks) == 15
        for i, (a, b) in enumerate(zip(old_chunks, new_chunks, strict=True)):
            differences = [key for key in sorted(set(a) | set(b)) if key != "generation_s" and a.get(key) != b.get(key)]
            chunk_checks.append(
                dict(chunk=i, exact_except_generation_seconds=not differences, differing_fields=differences)
            )
        controller_exact = read(old / "controller.json") == read(new / "controller.json")
        for folder in (old, new):
            for file in ("steps.jsonl", "chunks.json", "controller.json", "result.json", "model.xml"):
                signatures[str(folder / file)] = sha256(folder / file)
        checks.append(
            dict(
                record=name,
                fields=fields,
                chunks=chunk_checks,
                controller_exact=controller_exact,
                model_xml_exact=(old / "model.xml").read_bytes() == (new / "model.xml").read_bytes(),
                exact=all(v["exact"] for v in fields.values())
                and all(v["exact_except_generation_seconds"] for v in chunk_checks)
                and controller_exact,
            )
        )
    return dict(
        all_physical_fields_exact=all(c["exact"] and c["model_xml_exact"] for c in checks),
        cases=checks,
        input_sha256=signatures,
        comparison_excludes_only_wall_generation_time=True,
    )


class GripperReplay:
    """Scalar protocol replay, independent of both adapter implementations."""

    def __init__(self, initial, config, position):
        self.c, self.position = config, position
        self.extreme, self.direction = initial, -1.0
        self.latched, self.minimum, self.width = False, initial, None
        self.confirm, self.last_evidence = 0.0, False

    def step(self, reference, measured, motor, previous, dt):
        c = self.c
        if self.direction < 0:
            self.extreme = max(self.extreme, reference)
            if reference < self.extreme - c["reference_hysteresis"]:
                self.direction, self.extreme = 1.0, reference
        else:
            self.extreme = min(self.extreme, reference)
            if reference > self.extreme + c["reference_hysteresis"]:
                self.direction, self.extreme = -1.0, reference
        valid = dt <= c["max_evidence_interval_s"] + 1e-12
        if not valid or previous < 0:
            self.confirm, self.last_evidence, self.width = 0.0, False, None
        if not self.latched and self.direction > 0 and (not self.position or reference <= c["close_threshold"]):
            self.latched, self.minimum = True, reference
        if self.latched:
            self.minimum = min(self.minimum, reference)
            if motor <= c["motor_limit_tolerance"]:
                self.width = measured if self.width is None else min(self.width, measured)
            limit = min(max(measured, self.width or 0.0) + c["release_margin"], 1 - c["reference_hysteresis"] / 2)
            release = (
                valid
                and self.direction < 0
                and reference >= limit
                and reference - self.minimum >= min(c["release_excursion"], (1 - self.minimum) / 2)
            )
            self.confirm = self.confirm + dt if release and self.last_evidence else 0.0
            self.last_evidence = release
            if self.confirm + 1e-12 >= c["release_confirmation_s"]:
                self.latched, self.width = False, None
                return self.position_command(reference, motor) if self.position else -1.0
            return 1.0 if motor > c["motor_limit_tolerance"] else 0.0
        self.confirm, self.last_evidence = 0.0, False
        if self.position:
            return self.position_command(reference, motor)
        return -1.0 if motor < 1 - c["motor_limit_tolerance"] else 0.0

    def position_command(self, reference, motor):
        if reference > motor + self.c["motor_deadband"]:
            return -1.0
        if reference < motor - self.c["motor_deadband"]:
            return 1.0
        return 0.0


def verify_case(c, record, variant, folder, evidence):
    result = evidence.json(folder / "result.json")
    manifest = evidence.json(folder / "manifest.json")
    controller = evidence.json(folder / "controller.json")
    chunks = evidence.json(folder / "chunks.json")
    saved = [json.loads(line) for line in evidence.read(folder / "steps.jsonl").splitlines()]
    assert result["steps"] == result["expected_steps"] == len(saved) == 600 and result["error"] is None
    assert manifest["experiment"] == c and manifest["source"] == result["source"] == "oracle"
    assert manifest["variant"] == variant and manifest["record"] == record
    assert (
        result["model_rotation_alpha"] == 1
        and not result["raw_checkpoint_reference"]
        and not result["oracle_is_wam_success_evidence"]
    )
    assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == manifest["environment"]["LD_LIBRARY_PATH"] == ""
    for path, digest in manifest["execution_sha256"].items():
        evidence.check(path, digest)
    assert sha256(folder / "model.xml") == manifest["initial_model_xml_sha256"]
    evidence.read(folder / "model.xml")
    assert controller["actual_initial_physics_sha256"] == record["initial_state_sha256"]
    h = np.asarray(evidence.json(manifest["runtime_config"]["tool_contract"])["source_from_tcp"])
    native = REPO / c["oracle_reference_root"] / f"{record['name']}_s42"
    evidence.read(native / "reference.npz")
    with np.load(native / "reference.npz") as cache:
        poses, openings = cache["poses"], cache["openings"]
    contract = controller["contract"]
    assert contract["control_freq"] == 20 and contract["model_timestep"] == 0.002
    interface = OSCInterface(*(contract[k] for k in ("input_min", "input_max", "output_min", "output_max")))
    arm = OSCStateToAction(
        interface, contract["kp"], contract["kd"], 0.05, compensate=True, inverse_weight=0.5, max_command_rate=5.0
    )
    position = variant["gripper"] == "position_hold"
    grip = GripperReplay(
        saved[0]["opening_before"], c["gripper_position_hold"] if position else c["gripper_intent_hold"], position
    )
    governor = RecoveryReplay(c["reference_governor"])
    active = -1
    target = None
    progress = None
    offset = 0
    zero_commands = 0
    for i, row in enumerate(saved):
        assert row["step"] == i
        np.testing.assert_allclose(row["actual_dt"], 0.05, atol=1e-9, rtol=0)
        measured = np.asarray(row["measured_source"])
        if i:
            np.testing.assert_array_equal(measured, saved[i - 1]["after_source"])
        need = (
            active < 0
            or (progress["complete"] if variant["governor"] else offset >= len(target))
            or i - chunks[active]["step"] >= 80
        )
        if need:
            active += 1
            chunk = chunks[active]
            assert (
                chunk["step"] == i
                and chunk["physics_unchanged"]
                and chunk["physics_before_sha256"] == chunk["physics_after_sha256"]
            )
            assert chunk["physics_before_time_s"] == chunk["physics_after_time_s"]
            assert chunk["response_identity"] is None and chunk["image_sha256"] is None
            actions = np.asarray(chunk["raw128"])
            np.testing.assert_array_equal(actions, chunk["effective128"])
            target = decode(measured @ h, actions)
            count = len(target)
            np.testing.assert_allclose(target, poses[i : i + count] @ h, atol=1e-10, rtol=0)
            np.testing.assert_allclose(target, chunk["raw_targets_tcp"], atol=1e-10, rtol=0)
            np.testing.assert_array_equal(chunk["openings"], np.clip(openings[i : i + count], 0, 1))
            if variant["governor"]:
                mapping(
                    chunk["retiming"],
                    governor.chunk(measured @ h, target, np.asarray(chunk["openings"]), row["opening_before"], 0.05),
                )
            offset = 0
        assert row["chunk_index"] == active
        if variant["governor"]:
            progress, _ = governor.step(measured @ h, 0.05)
            mapping(row["reference_progress"], progress)
            desired, opening = progress["target_tcp"], progress["opening"]
        else:
            assert row["reference_progress"] is None
            desired, opening = target[offset], chunks[active]["openings"][offset]
            offset += 1
        np.testing.assert_allclose(row["desired_tcp"], desired, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["desired_source"], desired @ inverse(h), atol=1e-10, rtol=0)
        np.testing.assert_allclose(row["opening_target"], opening, atol=1e-10, rtol=0)
        converted = arm.convert(measured, np.asarray(row["desired_source"]), row["measured_twist"])
        np.testing.assert_allclose(row["command"][:6], converted["command"], atol=1e-12, rtol=0)
        assert row["saturated"] == converted["saturated"] and row["rate_limited"] == converted["rate_limited"]
        command = grip.step(
            opening,
            row["opening_before"],
            row["motor_opening_before"],
            0 if i == 0 else saved[i - 1]["command"][-1],
            0.05,
        )
        assert command == row["command"][-1] and grip.latched == row["gripper"]["closing_episode"]
        before, after = np.asarray(row["motor_target_before"]), np.asarray(row["motor_target_after"])
        if i:
            np.testing.assert_array_equal(before, saved[i - 1]["motor_target_after"])
        else:
            np.testing.assert_array_equal(before, controller["aligned_initial_motor_target"])
        motor_before = float(np.clip((0.04 + 0.02 * (before[0] - before[1]) - 0.001) / 0.079, 0, 1))
        np.testing.assert_allclose(row["motor_opening_before"], motor_before, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            after, np.clip(before + 0.25 * np.array([-1.0, 1.0]) * np.sign(command), -1, 1), atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            row["actual_gripper_actuator_control"], np.array([0.02, -0.02]) + 0.02 * after, atol=1e-12, rtol=0
        )
        if command == 0:
            zero_commands += 1
            np.testing.assert_array_equal(after, before)
        np.testing.assert_allclose(
            row["tcp_error"], error(desired, np.asarray(row["after_source"]) @ h), atol=1e-10, rtol=0
        )
    assert active + 1 == len(chunks)
    outcome = task_outcome([r["success"] for r in saved], expected_steps=600, stable_steps=40)
    assert outcome == result["task_outcome"]
    errors = np.asarray([r["tcp_error"] for r in saved])
    position_rmse = float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1))))
    rotation_rmse = float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1))))
    np.testing.assert_allclose(result["tcp_position_rmse_m"], position_rmse, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["rotation_rmse_rad"], rotation_rmse, atol=1e-12, rtol=0)
    qualified = outcome["stable"] and position_rmse <= 0.01 and rotation_rmse <= 0.05
    return dict(
        record=record["name"],
        variant=variant["name"],
        steps=600,
        outcome=outcome,
        tcp_rmse_m=position_rmse,
        so3_rmse_rad=rotation_rmse,
        native_zero_motor_hold_intervals=zero_commands,
        oracle_reference_causality_verified=True,
        gripper_replayed_without_privileged_inputs=True,
        qualified=qualified,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU verifier requires empty CUDA/LD paths")
    c = read(args.config)
    root = REPO / c["output_dir"]
    evidence = Evidence()
    launch = evidence.json(root / "qualification_launch_manifest.json")
    for path, digest in launch["input_sha256"].items():
        evidence.check(path, digest)
    for path, digest in launch["review_source_sha256"].items():
        evidence.check(path, digest)
    records = {r["name"]: r for r in evidence.json(c["fixture"])["records"]}
    baseline = verify_baseline(c, root)
    assert baseline["all_physical_fields_exact"]
    for path, digest in baseline["input_sha256"].items():
        evidence.check(path, digest)
    cases = []
    for variant in c["variants"]:
        for name in RECORDS:
            cases.append(
                verify_case(
                    c,
                    records[name],
                    variant,
                    root / "simulation" / f"oracle_{variant['name']}" / f"{name}_s42",
                    evidence,
                )
            )
    evidence.recheck()
    output = root / "qualification_independent_verification.json"
    write_new(
        output,
        dict(
            seed=42,
            complete=True,
            verified_cases=9,
            verified_physics_steps=5400,
            qualified=all(case["qualified"] for case in cases),
            baseline_exact=baseline,
            cases=cases,
            input_sha256=evidence.hashes,
            model_or_gpu_started=False,
        ),
    )
    print(
        json.dumps(dict(output=str(output), sha256=sha256(output), qualified=all(case["qualified"] for case in cases))),
        flush=True,
    )


if __name__ == "__main__":
    main()
