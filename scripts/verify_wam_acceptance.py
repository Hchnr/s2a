"""Fingerprint frozen weights and verify saved state/physics evidence end to end."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from s2a.action_reference import decode_action_targets
from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.checkpoint_audit import state_metrics
from s2a.geometry import integrate_local, inverse, pose9_to_transform
from s2a.libero_simulation import sha256_bytes


def fingerprint(c, root):
    destination = root / "checkpoint_fingerprints.json"
    if destination.exists():
        raise FileExistsError(destination)
    weights = []
    for ck in c["checkpoints"]:
        iteration = Path(ck["run"]) / "checkpoints" / ck["iteration"]
        paths = [Path(ck["run"]) / "config.yaml", iteration / ".save_complete"]
        paths += sorted(p for p in (iteration / "model").rglob("*") if p.is_file())
        files = {}
        for p in paths:
            before = p.stat()
            digest = sha256(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ino,
            ):
                raise ValueError(f"checkpoint changed while hashing: {p}")
            files[str(p)] = dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest)
        weights.append(dict(name=ck["name"], checkpoint=str(iteration / "model"), files=files))
        print(
            json.dumps(dict(checkpoint=ck["name"], files=len(files), bytes=sum(x["bytes"] for x in files.values()))),
            flush=True,
        )
    write_json(destination, dict(code=code_fingerprint(), checkpoints=weights))


def verify_object_progress(result, steps):
    """Bind each recorded object's progress to its endpoint log, not geometry."""
    summary_fields = {"grasp_contact_steps", "max_height_gain_m", "max_displacement_m"}
    step_fields = {"position", "grasp_contact", "height_gain_m", "displacement_m"}
    summaries = result.get("object_progress")
    if not steps or not isinstance(summaries, dict) or not summaries:
        raise ValueError("object progress requires nonempty summaries and endpoint logs")

    def finite_number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    observed = {
        name: dict(grasp_contact_steps=0, max_height_gain_m=-math.inf, max_displacement_m=-math.inf)
        for name in summaries
    }
    for row in steps:
        objects = row.get("objects")
        if not isinstance(objects, dict) or set(objects) != set(summaries):
            raise ValueError("object progress object set differs between summaries and endpoint logs")
        for name, obj in objects.items():
            if not isinstance(obj, dict) or set(obj) != step_fields:
                raise ValueError(f"object progress endpoint schema differs: {name}")
            position = obj["position"]
            if (
                not isinstance(position, list)
                or len(position) != 3
                or not all(finite_number(value) for value in position)
                or type(obj["grasp_contact"]) is not bool
                or not finite_number(obj["height_gain_m"])
                or not finite_number(obj["displacement_m"])
                or obj["displacement_m"] < 0
            ):
                raise ValueError(f"object progress endpoint has invalid boolean/finite physical fields: {name}")
            accumulated = observed[name]
            accumulated["grasp_contact_steps"] += obj["grasp_contact"]
            accumulated["max_height_gain_m"] = max(accumulated["max_height_gain_m"], obj["height_gain_m"])
            accumulated["max_displacement_m"] = max(accumulated["max_displacement_m"], obj["displacement_m"])
    for name, summary in summaries.items():
        if (
            not isinstance(summary, dict)
            or set(summary) != summary_fields
            or type(summary["grasp_contact_steps"]) is not int
            or not 0 <= summary["grasp_contact_steps"] <= len(steps)
            or not finite_number(summary["max_height_gain_m"])
            or not finite_number(summary["max_displacement_m"])
            or summary["max_displacement_m"] < 0
        ):
            raise ValueError(f"object progress summary schema or boolean/finite fields differ: {name}")
        expected = observed[name]
        if summary["grasp_contact_steps"] != expected["grasp_contact_steps"] or any(
            not math.isclose(summary[key], expected[key], rel_tol=0, abs_tol=1e-12)
            for key in ("max_height_gain_m", "max_displacement_m")
        ):
            raise ValueError(f"object progress summary differs from endpoint logs: {name}")


def verify_case(path, c, fixtures, variants):
    """Reject inconsistent protocol, response identity, target decoding or budget."""
    result = json.loads(path.read_text())
    manifest = json.loads(path.with_name("manifest.json").read_text())
    config = manifest["config"]
    record = fixtures[result["record"]]
    source = result["source"]
    if source not in ("native", "oracle", "wam", "probe"):
        raise ValueError("unknown physics source")
    if source == "wam":
        label = result["checkpoint"]
        if label not in variants:
            raise ValueError("unknown physics checkpoint")
    else:
        if result["checkpoint"] is not None:
            raise ValueError("non-WAM physics case has a checkpoint")
        label = source
        if source == "probe":
            label += "_" + manifest["probe"]["name"]
    if path.parent.parent.name != label or path.parent.name != f"{record['name']}_s{result['sampling_seed']}":
        raise ValueError("physics directory source/checkpoint or case identity differs")
    if manifest["record"] != record or manifest["source"] != source:
        raise ValueError("physics fixture/source identity changed")
    if result["error"] is not None or result["steps"] != result["expected_steps"]:
        raise ValueError("physics result has an error or incomplete budget")
    if source != "probe" and result["steps"] != c["online_max_steps"]:
        raise ValueError("physics budget differs")
    for key in (
        "sampling_steps",
        "guidance",
        "shift",
        "control_fps",
        "inverse_weight",
        "max_command_rate",
        "opening_hysteresis",
    ):
        # Early manifests inherited these fixed controller defaults from the template.
        if config[key] != c[key]:
            raise ValueError(f"physics protocol differs: {key}")
    if config["task_text"] != record["task"] or config["seed"] != result["sampling_seed"]:
        raise ValueError("physics task or seed differs")
    steps = [json.loads(line) for line in path.with_name("steps.jsonl").read_text().splitlines()]
    if len(steps) != result["steps"] or [r["step"] for r in steps] != list(range(len(steps))):
        raise ValueError("physics step log is missing or duplicated")
    np.testing.assert_allclose([r["actual_dt"] for r in steps], 1 / c["control_fps"], atol=1e-10, rtol=0)
    if result["success"] != any(r["success"] for r in steps) or result["terminal_success"] != steps[-1]["success"]:
        raise ValueError("success summary differs from native step predicates")
    verify_object_progress(result, steps)
    files = {
        str(p): sha256(p)
        for p in (
            path,
            path.with_name("manifest.json"),
            path.with_name("steps.jsonl"),
            path.with_name("controller.json"),
        )
    }
    if source in ("wam", "oracle"):
        chunks = json.loads(path.with_name("chunks.json").read_text())
        if [r["step"] for r in chunks] != list(range(0, len(steps), c["online_execution_horizon"])):
            raise ValueError("reference chunk boundaries differ")
        if result["chunks"] != len(chunks) or not all(r["physics_unchanged"] for r in chunks):
            raise ValueError("physics advanced during inference or chunk count differs")
        tool_path = Path(config["tool_contract"])
        if sha256(tool_path) != manifest["tool_sha256"]:
            raise ValueError("TCP contract changed")
        h = np.asarray(json.loads(tool_path.read_text())["source_from_tcp"])
        info = manifest.get("server_info")
        if source == "wam":
            ck = variants[result["checkpoint"]]
            expected = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
            if Path(info["checkpoint_path"]) != expected or info["ema_weights"]:
                raise ValueError("physics model identity differs")
            intervention = info["acceptance_intervention"]
            if not intervention["active_slot_mask"]:
                raise ValueError("masked protocol used an unmasked service")
            if result["checkpoint"] == "warm6200_masked" and not intervention["parameters_unchanged"]:
                raise ValueError("warm-start migration changed learned parameters")
        for i, chunk in enumerate(chunks):
            start = chunk["step"]
            np.testing.assert_allclose(
                chunk["anchor_tcp"], np.asarray(steps[start]["measured_source"]) @ h, atol=1e-12, rtol=0
            )
            targets, opening = decode_action_targets(
                np.asarray(chunk["anchor_tcp"]), np.asarray(chunk["actions"]), horizon=c["online_execution_horizon"]
            )
            np.testing.assert_allclose(targets, chunk["targets_tcp"], atol=1e-12, rtol=0)
            for j, step in enumerate(steps[start : start + len(targets)]):
                np.testing.assert_allclose(targets[j] @ inverse(h), step["desired_source"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(opening[j], step["opening_target"], atol=1e-12, rtol=0)
            if source == "wam":
                request_path = path.with_name(f"request_{i:04d}.json")
                response_path = path.with_name(f"response_{i:04d}.json")
                request = json.loads(request_path.read_text())
                response = json.loads(response_path.read_text())
                expected_request = dict(
                    name=f"s2a_{path.parent.name}_{i}",
                    model_mode="wam",
                    prompt=record["task"],
                    domain_name="franka_panda__parallel_gripper",
                    image_size=960,
                    action_chunk_size=c["online_chunk_size"],
                    fps=c["control_fps"],
                    num_frames=c["online_chunk_size"] + 1,
                    seed=result["sampling_seed"],
                    num_steps=c["sampling_steps"],
                    guidance=c["guidance"],
                    shift=c["shift"],
                    extra={
                        "action128": dict(
                            action_domain=config["action_domain"],
                            action_dim_mask=[True] * 10 + [False] * 118,
                            action_fps=c["control_fps"],
                            task_title=record["task"],
                            selected_roles=["auxiliary_0", "left_wrist"],
                            camera_layout=[["auxiliary_0", "top", 640, 480], ["left_wrist", "bottom", 640, 480]],
                            return_video=config.get("return_video", True),
                        )
                    },
                )
                if set(request) != set(expected_request) | {"vision_path"}:
                    raise ValueError("online conditioning fields differ")
                for key, value in expected_request.items():
                    if request.get(key) != value:
                        raise ValueError(f"online conditioning differs: {key}")
                image_path = path.with_name(f"input_{i:04d}.png")
                if Path(request["vision_path"]).resolve() != image_path.resolve():
                    raise ValueError("online input does not reference the current chunk image")
                if response["status"] != "success":
                    raise ValueError("online response failed")
                content = response["outputs"][0]["content"]
                if (
                    content["action128"]["checkpoint_path"] != str(expected)
                    or content["action128"]["seed"] != result["sampling_seed"]
                ):
                    raise ValueError("online response checkpoint/seed differs")
                np.testing.assert_array_equal(content["action"], chunk["actions"])
                pixels = np.asarray(Image.open(image_path))
                if pixels.shape != (960, 640, 3) or pixels.dtype != np.uint8:
                    raise ValueError("online input is not a 960x640 RGB uint8 canvas")
                if sha256_bytes(pixels.tobytes()) != chunk["image_sha256"]:
                    raise ValueError("online input pixels changed")
                for p in (request_path, response_path, image_path):
                    files[str(p)] = sha256(p)
        files[str(path.with_name("chunks.json"))] = sha256(path.with_name("chunks.json"))
    return dict(
        record=result["record"],
        source=source,
        checkpoint=result["checkpoint"],
        sampling_seed=result["sampling_seed"],
        steps=len(steps),
        files=files,
    )


def verify_state(c, root, summary):
    if not summary["checkpoint_matrices"]:
        return []
    samples = {r["name"]: r for r in json.loads((root / "samples/manifest.json").read_text())["cases"]}
    states = {}
    for sample in samples.values():
        if sample["episode"] not in states:
            if sha256(sample["arrays_path"]) != sample["arrays_sha256"]:
                raise ValueError("published sample arrays changed")
            with np.load(sample["arrays_path"]) as arrays:
                states[sample["episode"]] = arrays["states"]
        if sha256(sample["image_path"]) != sample["image_sha256"]:
            raise ValueError("published sample pixels changed")
    rows = []
    for matrix in summary["checkpoint_matrices"]:
        directory = root / "state_queries" / matrix["name"]
        manifest = json.loads((directory / "manifest.json").read_text())
        info = manifest["server_info"]
        if matrix["name"].endswith("_masked") and not info["acceptance_intervention"]["active_slot_mask"]:
            raise ValueError("masked state protocol used an unmasked service")
        files = {}
        for path in sorted(directory.glob("*/result.json")):
            result = json.loads(path.read_text())
            sample = samples[result["sample"]]
            data = states[sample["episode"]]
            request = json.loads(path.with_name("request.json").read_text())
            response_path = path.with_name("response.json")
            if sha256(response_path) != result["response_sha256"]:
                raise ValueError("state response changed")
            response = json.loads(response_path.read_text())["outputs"][0]["content"]
            if response["action128"]["checkpoint_path"] != info["checkpoint_path"] or info["ema_weights"]:
                raise ValueError("state response identity differs")
            if request["seed"] != result["sampling_seed"] or response["action128"]["seed"] != request["seed"]:
                raise ValueError("state response seed differs")
            prompt = (
                sample["task"]
                if result["prompt_mode"] == "task_only"
                else sample["published_prompts"][str(result["horizon"])]
            )
            if (
                request["prompt"] != prompt
                or Path(request["vision_path"]).resolve() != Path(sample["image_path"]).resolve()
            ):
                raise ValueError("state input sample or task differs")
            if (
                result["split"] != sample["split"]
                or result["episode"] != sample["episode"]
                or result["anchor_frame"] != sample["anchor_frame"]
            ):
                raise ValueError("state result sample alignment differs")
            frame = sample["anchor_frame"]
            count = min(result["horizon"], len(data) - frame - 1)
            poses = pose9_to_transform(data[:, :9])
            action = np.asarray(response["action"])
            actual = state_metrics(
                poses[frame + 1 : frame + 1 + count],
                integrate_local(poses[frame], action[:count, :9])[1:],
                data[frame + 1 : frame + 1 + count, 9],
                np.clip(action[:count, 9], 0, 1),
            )
            if result["valid_future_steps"] != count:
                raise ValueError("state score includes padded future labels")
            for key, value in actual.items():
                np.testing.assert_allclose(value, result["metrics"][key], atol=1e-12, rtol=0)
            for p in (path, response_path, path.with_name("request.json")):
                files[str(p)] = sha256(p)
        rows.append(dict(checkpoint=matrix["name"], cases=len(files) // 3, files=files))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("fingerprint", "verify"), required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    if args.stage == "fingerprint":
        fingerprint(c, root)
        return
    masked, _ = load_config("configs/wam_checkpoint_masked_20261019v2.json")
    variants = {r["name"]: r for r in masked["checkpoints"]}
    fixtures = {r["name"]: r for r in json.loads((root / "simulation/fixtures.json").read_text())["records"]}
    summary = json.loads((root / ("summary_partial.json" if args.partial else "summary.json")).read_text())
    if not args.partial and not summary["complete"]:
        raise ValueError("full verification requires the complete summary")
    records = []
    for directory in ["native", "oracle", *variants]:
        for path in sorted((root / "simulation" / directory).glob("*/result.json")):
            records.append(verify_case(path, c, fixtures, variants))
    for path in sorted((root / "simulation").glob("probe_*/*/result.json")):
        records.append(verify_case(path, c, fixtures, variants))
    state = verify_state(c, root, summary)
    weights_path = root / "checkpoint_fingerprints.json"
    weights_sha = None
    if weights_path.exists():
        for checkpoint in json.loads(weights_path.read_text())["checkpoints"]:
            for name, signature in checkpoint["files"].items():
                stat = Path(name).stat()
                if stat.st_size != signature["bytes"] or stat.st_mtime_ns != signature["mtime_ns"]:
                    raise ValueError("checkpoint changed since its full-file SHA256 fingerprint")
        weights_sha = sha256(weights_path)
    elif not args.partial:
        raise ValueError("full verification requires complete checkpoint fingerprints")
    out = root / ("integrity_partial.json" if args.partial else "integrity.json")
    if out.exists() and not args.partial:
        raise FileExistsError(out)
    write_json(
        out,
        dict(
            complete=summary["complete"],
            code=code_fingerprint(),
            checkpoint_fingerprints_sha256=weights_sha,
            state=state,
            physics=records,
        ),
    )
    print(
        json.dumps(
            dict(complete=summary["complete"], physics_cases=len(records), state_cases=sum(r["cases"] for r in state))
        )
    )


if __name__ == "__main__":
    main()
