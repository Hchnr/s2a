"""CPU-only preparation/review for the single frozen012 guidance3 state screen."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from s2a.artifacts import REPO, sha256

CONFIG = "configs/wam_stability_screen_8400_g30_20261021.json"
PLAN = "plans/v0.0.1/012_guidance_state_20261021.md"
COMMIT = "17844f2"
BASELINE = "outputs/wam_stability_20261021/earlier_screening/8400_g10/samples/manifest.json"
FINGERPRINTS = "outputs/wam_independent_20261020/checkpoint_fingerprints.json"
SOURCES = (
    "scripts/run_wam_stability_screen.py",
    "scripts/run_wam_state_matrix.py",
    "scripts/serve_wam_acceptance.py",
    "src/s2a/wam_runtime.py",
    "src/s2a/artifacts.py",
    "src/s2a/checkpoint_audit.py",
    "src/s2a/geometry.py",
    "src/s2a/prediction_audit.py",
    "configs/libero_wam_effect_20261019.json",
)


def write_new(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(REPO / "outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42,))
    parser.parse_args()
    assert Path.cwd().resolve() == REPO
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == os.environ.get("LD_LIBRARY_PATH") == ""
    signatures = {}
    for relative in (CONFIG, PLAN, *SOURCES):
        frozen = subprocess.check_output(["git", "show", f"{COMMIT}:{relative}"])
        expected = hashlib.sha256(frozen).hexdigest()
        assert sha256(relative) == expected, relative
        signatures[str(REPO / relative)] = expected
    signatures[str(Path(__file__).resolve())] = sha256(__file__)
    c = json.loads(Path(CONFIG).read_text())
    original_path = Path(c["source_samples_manifest"])
    assert sha256(original_path) == "0ffc7115c6af4f1b7c944b2572805956f6db229d2e63d602442e20a8e1cc1049"
    original = json.loads(original_path.read_text())
    baseline = json.loads(Path(BASELINE).read_text())
    for path in (original_path, Path(BASELINE), Path(FINGERPRINTS)):
        signatures[str(path.resolve())] = sha256(path)
    cases = [s for s in original["cases"] if s["split"] == "diagnostic"]
    assert cases == baseline["cases"] and len(cases) == 24
    assert len({s["episode"] for s in cases}) == 6
    assert len({(s["episode"], s["anchor_frame"]) for s in cases}) == 22
    omit = {"output_dir", "notes", "checkpoints", "guidance"}
    assert {k: v for k, v in c.items() if k not in omit} == {
        k: v for k, v in baseline["config"].items() if k not in omit
    }
    ck = c["checkpoints"][0]
    old_ck = baseline["config"]["checkpoints"][0]
    assert {k: v for k, v in ck.items() if k not in {"gpu", "port"}} == {
        k: v for k, v in old_ck.items() if k not in {"gpu", "port"}
    }
    assert (ck["name"], ck["gpu"], ck["port"]) == ("8400_masked", 7, 9577)
    assert c["guidance"] == 3 and c["sampling_seeds"] == [42, 43]
    assert c["offline_horizons"] == [40] and c["offline_prompt_modes"] == ["task_only"]
    files = {}
    for sample in cases:
        for field in ("arrays", "image"):
            path = Path(sample[f"{field}_path"])
            expected = sample[f"{field}_sha256"]
            assert path.is_absolute() and sha256(path) == expected
            signatures[str(path)] = expected
            files[str(path)] = dict(kind=field, sha256=expected)
        with np.load(sample["arrays_path"]) as data:
            states = data["states"]
        assert states.ndim == 2 and states.shape[1] >= 10 and np.isfinite(states).all()
        assert 0 <= sample["anchor_frame"] < len(states) - 1
        assert len(states) - 1 - sample["anchor_frame"] == sample["valid_future_steps"]
        pixels = np.asarray(Image.open(sample["image_path"]))
        assert pixels.shape == (960, 640, 3) and pixels.dtype == np.uint8
    assert len(files) == 30
    fingerprints = json.loads(Path(FINGERPRINTS).read_text())
    weights = next(x for x in fingerprints["checkpoints"] if x["name"] == "8400_masked")
    expected_model = Path(ck["run"]) / "checkpoints" / ck["iteration"] / "model"
    assert Path(weights["checkpoint"]).resolve() == expected_model.resolve()
    assert len(weights["files"]) == 163
    for filename, frozen in weights["files"].items():
        stat = Path(filename).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (frozen["bytes"], frozen["mtime_ns"])
        assert len(frozen["sha256"]) == 64
    output = REPO / c["output_dir"]
    assert output.is_relative_to(REPO / "outputs")
    assert not (output / "orchestration").exists() and not (output / "state_queries").exists()
    sample_path = output / "samples/manifest.json"
    samples = copy.deepcopy(baseline)
    samples["config"] = c
    write_new(sample_path, samples)
    signatures[str(sample_path)] = sha256(sample_path)
    command = [
        str(REPO / ".venv/bin/python"),
        "scripts/run_wam_stability_screen.py",
        "--config",
        CONFIG,
        "--seed",
        "42",
        "--fingerprints",
        FINGERPRINTS,
    ]
    dry = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
    result = json.loads(dry.stdout)
    assert (result["gpu"], result["port"], result["queries"]) == (7, 9577, 48)
    for filename, expected in signatures.items():
        assert sha256(filename) == expected, filename
    report = dict(
        registration_commit=COMMIT,
        plan=PLAN,
        config=CONFIG,
        config_sha256=sha256(CONFIG),
        cpu_only=True,
        new_model_processes_started=0,
        new_physics_processes_started=0,
        new_model_requests=0,
        sample_manifest=str(sample_path),
        sample_manifest_sha256=sha256(sample_path),
        exact_original_diagnostic_cases=True,
        cases=24,
        episodes=6,
        distinct_episode_anchors=22,
        unique_observation_files=files,
        unique_observation_files_count=30,
        registered_queries=48,
        registered_physics_steps=0,
        complete_weight_fingerprint=dict(
            path=FINGERPRINTS,
            sha256=sha256(FINGERPRINTS),
            checkpoint_path=str(expected_model),
            files_stat_rechecked=163,
            full_weight_bytes_rehashed_now=False,
        ),
        comparison_to_cached_guidance1=dict(
            only_inference_parameter_changed="guidance1.0_to3.0",
            other_changes=["output_dir", "notes", "physical_gpu_and_port"],
            actual_noise_witness_available=False,
            actual_noise_bitexact_pairing_claimed=False,
        ),
        frozen_admission=dict(
            horizons=[8, 16],
            all_six_position_so3_opening_nonregression=True,
            absolute_comparison_tolerance=1e-10,
            at_least_one_position_rmse_fractional_improvement=0.1,
            permits_automatic_physics=False,
        ),
        dry_run=dict(command=command + ["--dry-run"], returncode=dry.returncode, result=result, stderr=dry.stderr),
        launch_command=command,
        source_before_after_sha256=signatures,
        limitations=[
            "Only48 state-development requests are registered; no task success, fresh holdout or pretraining-unseen claim.",
            "Identical seed is a request condition, not an actual-noise witness;011 noise-paired study is separate.",
            "Root must recheck allowed GPU7/port9577 and source SHA before launch; coordinator locks/checks GPU/port again.",
        ],
    )
    preflight = REPO / "outputs/wam_stability_20261021/guidance3_preflight/prepared_cpu_review.json"
    write_new(preflight, report)
    print(
        json.dumps(
            dict(
                preflight=str(preflight),
                sha256=sha256(preflight),
                sample_manifest_sha256=sha256(sample_path),
                registered_queries=48,
                model_started=False,
                physics_started=False,
            )
        )
    )


if __name__ == "__main__":
    main()
