"""Prepare026 from025's verified caches, without reopening source episodes.

Only metadata, frozen source files and the144 existing teacher feature archives
are read. Original episode arrays and PNGs remain inherited references.
"""

import argparse
import copy
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from s2a.artifacts import REPO, sha256
from s2a.worker_handshake import birth, write_once

CONFIG = REPO / "configs/wam_flow_consistent_readout_20261021.json"
CONFIG_SHA = "2d0e45b3ae02ecec4ee45bcaa1928596e347417f945f8903416f0df314a68534"
PLAN = REPO / "plans/v0.0.1/026_flow_consistent_readout_20261021.md"
PLAN_SHA = "42c0435773306852d4aa79be4d319f9de9cc71ff7823b7ad830286c2c41fc2fa"
EXTERNAL_HEAD = "8ed68a9722a14e8f6c21fa0c6c700e1617b7b73e"
GPU_UUID = "GPU-7f176867-170c-b508-5c89-1cd819ad3b8e"
DOMAIN = "franka_panda__parallel_gripper/eef20/observed_local_delta/libero_closed_pad_observed_endpoint_eef10_v1"
ACTIVE = np.asarray([True] * 10 + [False] * 118)
RELEASE = REPO / "outputs/wam_stability_20261021/readout_release_review/current_release_verified.json"
RELEASE_SHA = "c9b076afb246f614d5f4d1a5019964d9972badc5a9c33090b41764485bdff7f8"
OWNER_EXIT = REPO / "outputs/wam_stability_20261021/readout_runtime_observer/features_owner_tool_exit.json"
OWNER_EXIT_SHA = "16393f5e5cb6842eb3bdf86abc933366aac50876029b1633cfa3db8ded211056"
LAUNCH_PROOF = REPO / "outputs/wam_stability_20261021/readout_runtime_observer/features_launch_verified.json"
LAUNCH_PROOF_SHA = "0dbad932ba4ebb649aae966b0c158f3d744e03718aad76c0472ff8c08b1a1818"
LOCAL_SOURCES = (
    "scripts/prepare_wam_flow_readout.py",
    "tests/test_wam_flow_readout_preparation.py",
    "src/s2a/wam_flow_readout.py",
    "tests/test_wam_flow_readout.py",
    "scripts/fit_wam_flow_readout.py",
    "tests/test_wam_flow_readout_fit.py",
    "scripts/verify_wam_flow_readout.py",
    "tests/test_wam_flow_readout_verification.py",
    "scripts/run_wam_flow_readout.py",
    "tests/test_wam_flow_readout_runner.py",
    "scripts/run_wam_flow_readout_sampling.py",
    "tests/test_wam_flow_readout_sampling.py",
    "scripts/probe_wam_flow_readout_quantization.py",
    "tests/test_wam_flow_readout_quantization.py",
    "src/s2a/sampler_trace.py",
    "src/s2a/worker_handshake.py",
    "src/s2a/artifacts.py",
    "scripts/verify_wam_readout.py",
    "scripts/audit_wam_sampler_steps.py",
    "scripts/audit_wam_earlier_screen.py",
)
EXTERNAL_SOURCES = (
    "model/generator/diffusion/samplers/unipc.py",
    "model/generator/diffusion/samplers/fm_solvers_unipc.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f"frozen source SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, f"source changed: {path}")
        self.hashes[str(path)] = digest
        return path

    def json(self, path, expected=None):
        return json.loads(self.check(path, expected).read_text())

    def ref(self, ref):
        require(set(ref) == {"path", "sha256"}, "invalid evidence reference")
        return self.json(ref["path"], ref["sha256"])

    def recheck(self):
        for path, digest in self.hashes.copy().items():
            self.check(path, digest)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def weight_stats(files):
    require(len(files) == 163, "026 requires the original163-file8400 stat roster")
    for path, expected in files.items():
        stat = Path(path).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]),
            f"8400 weight stat changed: {path}",
        )


def external_identity(repo):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    require(head == EXTERNAL_HEAD and not dirty.strip(), "external tracked source must remain clean8ed")
    return dict(head=head, tracked_source_clean=True)


def validate_request(request, anchor):
    metadata = request["extra"]["action128"]
    require(
        request["model_mode"] == "wam"
        and request["vision_path"] == anchor["image_path"]
        and request["prompt"] == metadata["task_title"] == anchor["task"]
        and request["action_chunk_size"] == 40
        and request["num_frames"] == 41
        and request["num_steps"] == 35
        and request["fps"] == 20
        and request["guidance"] == 1.0
        and request["shift"] == 5.0
        and request["seed"] in (42, 43)
        and metadata["action_domain"] == DOMAIN
        and metadata["action_dim_mask"] == ACTIVE.tolist(),
        "inherited online request/domain/clock differs",
    )


def validate_rosters(config, old, admitted_inputs):
    """Validate inherited whole-EP/mask/clock metadata without opening EP files."""
    expected_domain = dict(name=DOMAIN, normalization_id=31, projection_id=0, active_slots=10)
    require(config["domain"] == expected_domain, "026 normalization31/projection0/full domain differs")
    require(all(old["domain"][k] == v for k, v in expected_domain.items()), "025 domain admission differs")
    require(old["domain"]["active_mask"] == ACTIVE.tolist(), "025 active10 domain mask differs")
    episodes, anchors = old["episodes"], old["anchors"]
    require(len(episodes) == 12 and len(anchors) == 48, "expected12 whole EPs/48 fullH40 anchors")
    require(len({a["name"] for a in anchors}) == 48, "duplicate inherited anchor")
    require(len({e["source_episode_id"] for e in episodes}) == 12, "whole-source EP leaks across split entries")
    expected_ids = config["train_episodes"] + config["validation_episodes"]
    require([e["episode"] for e in episodes] == expected_ids, "registered whole-EP order/split differs")
    for index, episode in enumerate(episodes):
        split = "train" if index < 6 else "validation"
        members = [a for a in anchors if a["episode"] == episode["episode"]]
        require(episode["split"] == split and len(members) == 4, "whole-EP split/anchor count differs")
        require(
            [a["anchor_index"] for a in members] == list(range(4))
            and [a["name"] for a in members] == episode["anchors"]
            and [a["anchor_frame"] for a in members] == episode["frames"]
            and episode["frames"] == [j * episode["frames"][-1] // 3 for j in range(4)]
            and len(set(episode["frames"])) == 4,
            "inherited uniform anchor clock differs",
        )
        require(
            len({(a["arrays_path"], a["arrays_sha256"]) for a in members}) == 1,
            "whole-source array aliases disagree",
        )
        for a in members:
            require(
                a["source_episode_id"] == episode["source_episode_id"]
                and a["task_index"] == episode["task_index"]
                and a["split"] == split
                and a["original_split"] == ("diagnostic" if split == "train" else "study_heldout")
                and a["valid_future_steps"] == 40
                and a["state_mask128"] == ACTIVE.tolist()
                and np.asarray(a["measured_state128"]).shape == (128,)
                and np.isfinite(a["measured_state128"]).all(),
                "inherited EP identity/valid mask/measured state differs",
            )
            require(old["input_sha256"].get(a["arrays_path"]) == a["arrays_sha256"], "EP ref lost original admission")
            require(admitted_inputs.get(a["image_path"]) == a["image_sha256"], "PNG ref lost independent admission")
            validate_request(a["request"], a)
    expected_teacher = [
        dict(anchor_name=a["name"], call_index=i * 3 + j, noise_seed=42, sigma=s, split=a["split"])
        for i, a in enumerate(anchors)
        for j, s in enumerate((0.2, 0.5, 0.8))
    ]
    require(old["teacher_calls"] == expected_teacher, "inherited144 teacher roster differs")
    expected_sampling = []
    for a in anchors:
        if a["split"] != "validation":
            continue
        for seed in (42, 43):
            for variant in ("raw", "adapt"):
                request = copy.deepcopy(a["request"])
                request["seed"] = seed
                expected_sampling.append(
                    dict(
                        anchor_name=a["name"],
                        generation_index=len(expected_sampling),
                        name=f"{a['name']}_{variant}_s{seed}",
                        pair_id=f"{a['name']}_s{seed}",
                        request=request,
                        sampling_seed=seed,
                        variant=variant,
                    )
                )
    require(old["validation_generations"] == expected_sampling, "inherited96 paired generation roster differs")
    zero = old["zero_identity_cases"]
    require(len(zero) == 3 and [z["task_index"] for z in zero] == [0, 1, 2], "inherited3 zero cases differ")
    return dict(whole_source_episodes=12, anchors=48, teacher_calls=144, conditional_generations=96, zero_cases=3)


def verify_cached_arrays(metadata, arrays, call):
    expected = dict(
        hidden=((40, 1280), "float32"),
        head_base=((40, 128), "float32"),
        projection_ids=((40,), "int64"),
        base=((40, 128), "float32"),
        target=((40, 128), "float32"),
        xt=((40, 128), "float32"),
        valid=((40, 128), "bool"),
        epsilon=((40, 128), "float32"),
        x0_normalized=((40, 128), "float32"),
        sigma_action=((40, 1), "float32"),
        condition_action=((40, 1), "float32"),
    )
    require(set(arrays) == set(expected) == set(metadata["array_index"]), "cached teacher array roster differs")
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        require(
            value.shape == shape
            and str(value.dtype) == dtype
            and np.isfinite(value).all()
            and metadata["array_index"][name] == dict(shape=list(shape), dtype=dtype),
            f"cached teacher shape/dtype/finite contract differs: {name}",
        )
    require(np.array_equal(arrays["valid"], np.broadcast_to(ACTIVE, (40, 128))), "cached real active10 mask differs")
    require(np.all(arrays["projection_ids"] == 0), "cached projection is not shared0")
    require(
        np.array_equal(arrays["sigma_action"], np.full((40, 1), call["sigma"], dtype=np.float32))
        and np.all(arrays["condition_action"] == 0),
        "cached actual FP32 sigma/action condition differs",
    )


def feature_admission(old, features, verification, failed_fit, refs, evidence):
    require(
        features["complete"] is True
        and features["passed"] is True
        and features["all_original_parameters_frozen"] is True
        and features["denoise_calls"] == 249
        and features["prepared_ref"] == refs["source_prepared_ref"],
        "original025 features are not complete frozen249",
    )
    for phase, checked in (("features", verification), ("fit", failed_fit)):
        require(
            checked["passed"] is True
            and checked["phase"] == phase
            and checked["prepared"] == refs["source_prepared_ref"]
            and checked["teacher_calls_verified"] == 144
            and checked["zero_identity_generations_verified"] == 3
            and checked["actual_denoise_calls_verified"] == 249
            and checked["features_verified"] is True,
            "025 independent complete-feature/fit provenance differs",
        )
        for key in ("source_prepared_ref", "source_features_ref"):
            ref = refs[key]
            require(checked["input_sha256"].get(ref["path"]) == ref["sha256"], "independent audit not bound to source")
    require(
        failed_fit["fit_verified"] is True
        and failed_fit["independent_model_reconstructions"] == 2
        and failed_fit["eligible_for_sampling"] is False
        and failed_fit["teacher_gate"]["passed"] is False
        and failed_fit["teacher_gate"]["checks"]["all_sigmas_nonregressing"] is False,
        "025 failed gate must remain False",
    )
    calls = features["teacher_calls"]
    require(len(calls) == 144, "source teacher cache count differs")
    for planned, actual in zip(old["teacher_calls"], calls, strict=True):
        require({k: v for k, v in actual.items() if k != "bundle"} == planned, "teacher cache order differs")
        metadata = evidence.ref(actual["bundle"])
        require(
            all(metadata[k] == v for k, v in planned.items()), "teacher metadata belongs to another anchor/split/clock"
        )
        path = evidence.check(metadata["arrays"]["path"], metadata["arrays"]["sha256"])
        require(
            verification["input_sha256"].get(str(path)) == metadata["arrays"]["sha256"],
            "teacher archive not independently verified",
        )
        with np.load(path, allow_pickle=False) as source:
            verify_cached_arrays(metadata, {name: source[name] for name in source.files}, planned)
        for key in ("conditioning_ref", "rng_before", "rng_after"):
            evidence.ref(metadata[key])
    return dict(verified_teacher_cache_count=144, old_fit_eligible_for_sampling=False, new_teacher_calls=0)


def terminal(item):
    try:
        actual = birth(item["pid"], allow_exited=True)
    except (FileNotFoundError, ProcessLookupError):
        return dict(expected=item, original_birth_terminal=True, current_pid_absent=True)
    require(
        actual["start_ticks"] != item["start_ticks"] or actual["proc_state"] in ("Z", "X"),
        "original025 owned birth is still live",
    )
    return dict(expected=item, original_birth_terminal=True, observed=actual)


def current_resource():
    raw = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    index, uuid, memory = [p.strip() for p in raw.split(",")]
    apps_raw = subprocess.check_output(
        ["nvidia-smi", "--id=6", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    apps = [line for line in apps_raw.splitlines() if len(line.split(",")) >= 2 and line.split(",")[1].strip() == uuid]
    require(index == "6" and uuid == GPU_UUID and memory == "0" and not apps, "original GPU6 release not current")
    return dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        physical_gpu=6,
        uuid=uuid,
        memory_mib=0,
        compute_apps=apps,
        actual_nvidia_smi=raw,
        raw_compute_query=apps_raw,
    )


def resource_admission(old, evidence):
    root = Path(old["output_root"]) / "orchestration/features"
    original = evidence.json(root / "result.json")
    require(original["complete"] is False and original["passed"] is False, "original025 owner False was changed")
    launch = evidence.json(root / "model/launch.json")
    exit_record = evidence.json(root / "model/exit.json")
    release = evidence.json(root / "model/ownership/release.json")
    require(exit_record == dict(actual_wait=True, automatic_retry=False, returncode=0), "025 model actual wait differs")
    require(
        all(release[k] == launch["child"][k] for k in ("pid", "start_ticks"))
        and release["returncode"] == 0
        and release["reaped"] is True
        and release["signal_sent"] is False
        and release["original_error"] is None,
        "025 model owned normal exit/reap differs",
    )
    tool = evidence.json(OWNER_EXIT, OWNER_EXIT_SHA)
    require(
        tool["tool_handle_terminal"] is True
        and tool["tool_reported_exit_code"] == 1
        and tool["original_owner_result"] == reference(root / "result.json"),
        "025 owner actual tool exit1 differs",
    )
    peer = evidence.json(RELEASE, RELEASE_SHA)
    require(
        peer["passed"] is True
        and peer["current_resource_release_verified"] is True
        and peer["original_owner_passed"] is False
        and peer["prepared"] == reference(REPO / old["config"]["output_dir"] / "preparation/prepared.json"),
        "025 independent terminal resource proof differs",
    )
    observed = evidence.json(LAUNCH_PROOF, LAUNCH_PROOF_SHA)
    model, owner = observed["actual_model"], observed["actual_owner"]
    require(
        observed["passed"] is True
        and observed["launch"] == reference(root / "model/launch.json")
        and all(model[k] == launch["child"][k] for k in ("pid", "start_ticks", "ppid"))
        and model["ppid"] == owner["pid"]
        and (model["pid"], model["start_ticks"], owner["pid"], owner["start_ticks"])
        == (862631, 170222073, 862480, 170221627),
        "025 original actual owner/model birth proof differs",
    )
    births = [terminal(model), terminal(owner)]
    return dict(
        original_owner_passed=False,
        original_owner_tool_exit=1,
        original_model_actual_wait_exit=0,
        original_model_reaped=True,
        original_model_signal_sent=False,
        original_births=births,
        independent_release_ref=reference(RELEASE),
        actual_launch_proof_ref=reference(LAUNCH_PROOF),
        tool_exit_ref=reference(OWNER_EXIT),
        current_resource=current_resource(),
        historical_poll_detail_persisted=False,
        historical_release_timeout_root_cause="unknown",
    )


def source_closure(old, config, evidence):
    paths = [REPO / p for p in LOCAL_SOURCES]
    paths += [Path(config["cosmos_repo"]) / "cosmos_framework" / p for p in EXTERNAL_SOURCES]
    paths += [CONFIG, PLAN]
    sources = dict(old["source_sha256"])
    for path in paths:
        sources[str(path.resolve())] = sha256(evidence.check(path))
    return sources


def prepare():
    require(
        Path.cwd().resolve() == REPO
        and os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == "",
        "026 preparation requires S2A cwd and explicitly empty CUDA/LD",
    )
    evidence = Evidence()
    config = evidence.json(CONFIG, CONFIG_SHA)
    evidence.check(PLAN, PLAN_SHA)
    root = REPO / config["output_dir"]
    require(not root.exists(), "026 refuses an existing output root")
    refs, ancestors = {}, {}
    for key in ("source_prepared", "source_features", "source_features_verification", "source_failed_fit_verification"):
        path = REPO / config[key]
        ancestors[key] = evidence.json(path, config[f"{key}_sha256"])
        refs[f"{key}_ref"] = reference(path)
    old = ancestors["source_prepared"]
    require(old["schema_version"] == "s2a.frozen-readout-preparation.v1", "unexpected025 preparation schema")
    require(len(old["source_sha256"]) == 53, "original025 source roster is not53")
    for mapping in (old["source_sha256"], old["snapshot_sha256"]):
        for path, digest in mapping.items():
            evidence.check(path, digest)
    weight_stats(old["weight_files"])
    identity = external_identity(config["cosmos_repo"])
    require(identity == old["external_cosmos"], "original025 external source identity differs")
    expected_checkpoint = str(
        Path(config["checkpoint"]["run"]) / "checkpoints" / config["checkpoint"]["iteration"] / "model"
    )
    require(old["checkpoint"] == expected_checkpoint, "registered8400 checkpoint differs")
    roster = validate_rosters(config, old, ancestors["source_features_verification"]["input_sha256"])
    cached = feature_admission(
        old,
        ancestors["source_features"],
        ancestors["source_features_verification"],
        ancestors["source_failed_fit_verification"],
        refs,
        evidence,
    )
    resource = resource_admission(old, evidence)
    fit_path = Path(old["output_root"]) / "fit/fit_result.json"
    fit_sha = ancestors["source_failed_fit_verification"]["input_sha256"][str(fit_path)]
    old_fit = evidence.json(fit_path, fit_sha)
    require(
        old_fit["complete"] is True
        and old_fit["passed"] is True
        and old_fit["eligible_for_sampling"] is False
        and old_fit["teacher_gate"]["passed"] is False,
        "025 actual producer fit failure was changed",
    )
    sources = source_closure(old, config, evidence)
    # No directory or output is created until every required source exists.
    root.mkdir()
    directory = root / "preparation"
    directory.mkdir()
    try:
        snapshots = directory / "source_snapshot"
        snapshots.mkdir()
        copies = {}
        for i, (path, digest) in enumerate(sources.items()):
            target = snapshots / f"{i:03d}_{Path(path).name}"
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "026 frozen source snapshot differs")
            copies[str(target)] = digest
        evidence.recheck()
        weight_stats(old["weight_files"])
        require(external_identity(config["cosmos_repo"]) == identity, "external source changed during preparation")
        result = dict(
            schema_version="s2a.flow-readout-preparation.v1",
            config=config,
            config_ref=reference(CONFIG),
            registration=reference(PLAN),
            output_root=str(root),
            checkpoint=old["checkpoint"],
            weight_files=old["weight_files"],
            external_cosmos=identity,
            domain=copy.deepcopy(old["domain"]),
            **refs,
            **{
                key: copy.deepcopy(old[key])
                for key in (
                    "anchors",
                    "episodes",
                    "teacher_calls",
                    "validation_generations",
                    "zero_identity_cases",
                    "conservative_source_exclusion",
                )
            },
            source_admission=dict(
                **roster,
                **cached,
                resource=resource,
                original_source_count=53,
                original_weight_stat_count=163,
                inherited_episode_refs_not_reopened=True,
                inherited_png_refs_not_reopened=True,
                unopened_refs_scope="026 preparation only; later independently authorized state scoring is separate",
                ancestor_input_maps_rehashed_in_full=False,
                historical_initial_physical_sync_certified=False,
                pretraining_visibility="unknown",
            ),
            quantization_admission_required_after_preparation=True,
            budget=config["budget"],
            source_sha256=sources,
            snapshot_sha256=copies,
            input_sha256=evidence.hashes,
            new_model_calls=0,
            new_teacher_calls=0,
            new_episode_reads=0,
            fit_count=0,
            physics_steps=0,
            fresh_performance_cases=0,
            full_weight_bytes_rehashed=False,
        )
        path = directory / "prepared.json"
        write_once(path, result)
        return path
    except BaseException as error:
        write_once(directory / "error.json", dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    require(args.seed == 42, "026 requires registered seed42")
    path = prepare()
    print(json.dumps(dict(prepared=reference(path), model_calls=0, fit_count=0, physics_steps=0)))


if __name__ == "__main__":
    main()
