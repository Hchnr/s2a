"""Prepare a bounded guidance1.0 state screen for old22-source8400/8600 weights."""

import argparse
import collections
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

from s2a.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(42,))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU preparation requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    source = Path("outputs/wam_checkpoint_acceptance_20261019v2/samples/manifest.json")
    expected_source_sha = "0ffc7115c6af4f1b7c944b2572805956f6db229d2e63d602442e20a8e1cc1049"
    if sha256(source) != expected_source_sha:
        raise ValueError("original frozen source manifest changed")
    original = json.loads(source.read_text())
    initial = Path("outputs/wam_stability_20261021/screening/8800_g10/samples/manifest.json")
    selected = json.loads(initial.read_text())
    cases = [s for s in original["cases"] if s["split"] == "diagnostic"]
    if len(cases) != 24 or len({s["episode"] for s in cases}) != 6 or cases != selected["cases"]:
        raise ValueError("must retain the exact original24 diagnostic phase anchors")
    for case in cases:
        for field in ("arrays", "image"):
            path = Path(case[f"{field}_path"])
            if not path.is_absolute() or sha256(path) != case[f"{field}_sha256"]:
                raise ValueError("source observation path or SHA differs")
    fingerprints = Path("outputs/wam_independent_20261020/checkpoint_fingerprints.json")
    frozen_weights = json.loads(fingerprints.read_text())
    # Enumerate existing request records before allocating any additional inference.
    files = subprocess.check_output(["rg", "--files", "outputs"], text=True).splitlines()
    records = []
    for path in files:
        p = Path(path)
        if p.suffix == ".json" and p.name.startswith("request"):
            for stage in (8400, 8600):
                if any(part == str(stage) or part.startswith(f"{stage}_") for part in p.parts):
                    records.append((stage, p))
                    break
    searches = {}
    for stage in (8400, 8600):
        paths = sorted(p for s, p in records if s == stage)
        distribution = collections.Counter()
        g10 = []
        for path in paths:
            request = json.loads(path.read_text())
            distribution[(request.get("guidance"), request.get("action_chunk_size"), request.get("seed"))] += 1
            if request.get("guidance") == 1.0:
                g10.append(str(path))
        searches[str(stage)] = dict(
            stage_associated_request_records=len(paths),
            examined_path_list_sha256=hashlib.sha256("\n".join(map(str, paths)).encode()).hexdigest(),
            guidance_horizon_seed_counts=[
                dict(guidance=k[0], horizon=k[1], seed=k[2], records=v) for k, v in sorted(distribution.items())
            ],
            guidance1_request_paths=g10,
            exact_reusable_predictions=0 if not g10 else None,
            search_scope="all repository outputs request*.json with an explicit8400/8600 path component/prefix",
        )
        if g10:
            raise ValueError(
                "guidance1 records exist; inspect their full model/input contract before allocating queries"
            )
    root = Path("outputs/wam_stability_20261021/earlier_screening")
    root.mkdir(exist_ok=False)
    prepared = []
    for stage, gpu, port in ((8400, 4, 9144), (8600, 6, 9146)):
        name = f"{stage}_masked"
        weight = next(x for x in frozen_weights["checkpoints"] if x["name"] == name)
        model_path = Path(weight["checkpoint"])
        if model_path.parent.name != f"iter_{stage:09d}" or model_path.name != "model":
            raise ValueError("frozen checkpoint identity differs")
        stat_records = {}
        for filename, frozen in weight["files"].items():
            stat = Path(filename).stat()
            if stat.st_size != frozen["bytes"] or stat.st_mtime_ns != frozen["mtime_ns"]:
                raise ValueError(f"checkpoint file stat changed since its full byte SHA: {filename}")
            stat_records[filename] = dict(frozen, stat_size_mtime_match=True)
        output = root / f"{stage}_g10"
        config_path = Path(f"configs/wam_stability_screen_{stage}_g10_20261021.json")
        if config_path.exists():
            raise FileExistsError(config_path)
        config = dict(
            seed=args.seed,
            output_dir=str(output),
            policy_template="configs/libero_wam_effect_20261019.json",
            cosmos_repo="/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm",
            checkpoints=[
                dict(
                    name=name,
                    run=str(model_path.parents[2]),
                    iteration=model_path.parent.name,
                    gpu=gpu,
                    port=port,
                )
            ],
            offline_horizons=[40],
            offline_prompt_modes=["task_only"],
            sampling_seeds=[42, 43],
            sampling_steps=35,
            guidance=1.0,
            shift=5.0,
            control_fps=20,
            source_samples_manifest=str(source.resolve()),
            notes=[
                "Additional2026-10-21 development-only state screen: old22-source stage at guidance1.0; "
                "not a23-source continuation or a latest/best-weight designation.",
                "Exactly the same24 diagnostic phase anchors/6 original episodes as the first8800/15400 "
                "screen. H40, task-only, seeds42/43,35denoising steps,shift5;48queries per checkpoint.",
                "Use the existing complete checkpoint byte fingerprints after path/size/mtime checks. "
                "No identical guidance1 request was found in explicitly stage-associated repository outputs.",
                "No task-success conclusion follows from this state-only comparison. Historical "
                "guidance1.5 physical scores are separate. No fresh holdout is opened.",
                f"Only root-owned orchestration may start this service on physicalGPU{gpu}/port{port}; "
                "preparation and dry-run use empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH.",
            ],
        )
        write_json(config_path, config)
        samples = copy.deepcopy(selected)
        samples["config"] = config
        sample_path = output / "samples/manifest.json"
        sample_path.parent.mkdir(parents=True)
        write_json(sample_path, samples)
        command = [
            str(Path(".venv/bin/python").absolute()),
            "scripts/run_wam_stability_screen.py",
            "--config",
            str(config_path),
            "--seed",
            str(args.seed),
            "--fingerprints",
            str(fingerprints),
            "--dry-run",
        ]
        dry = subprocess.run(command, check=True, capture_output=True, text=True)
        dry_result = json.loads(dry.stdout)
        if dry_result["queries"] != 48 or dry_result["gpu"] != gpu or dry_result["port"] != port:
            raise ValueError("CPU dry-run differs from the bounded budget")
        dry_path = output / "dry_run.json"
        write_json(dry_path, dict(command=command, result=dry_result, stderr=dry.stderr, returncode=dry.returncode))
        prepared.append(
            dict(
                checkpoint=name,
                checkpoint_path=str(model_path),
                config_path=str(config_path),
                config_sha256=sha256(config_path),
                samples_path=str(sample_path),
                samples_sha256=sha256(sample_path),
                dry_run_path=str(dry_path),
                dry_run_sha256=sha256(dry_path),
                queries=48,
                physical_gpu=gpu,
                port=port,
                existing_byte_fingerprints=stat_records,
                full_weight_bytes_rehashed_now=False,
            )
        )
    preflight = dict(
        purpose="old22-source8400/8600 guidance1.0 diagnostic-only state screen",
        seed=args.seed,
        cases_per_checkpoint=24,
        query_budget=96,
        new_model_processes_started=0,
        new_physics_processes_started=0,
        inference_performed=False,
        fresh_holdout_opened=False,
        source_manifest=dict(path=str(source), sha256=sha256(source)),
        initial_g10_samples=dict(path=str(initial), sha256=sha256(initial)),
        existing_complete_byte_fingerprints=dict(path=str(fingerprints), sha256=sha256(fingerprints)),
        selected_episode_ids=sorted({s["episode"] for s in cases}),
        source_case_sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        reuse_search=searches,
        prepared=prepared,
        preparation_code_sha256={
            p: sha256(p)
            for p in (
                "scripts/prepare_wam_earlier_screen.py",
                "scripts/run_wam_stability_screen.py",
                "scripts/run_wam_state_matrix.py",
                "scripts/serve_wam_acceptance.py",
                "src/s2a/wam_runtime.py",
                "src/s2a/checkpoint_audit.py",
                "src/s2a/geometry.py",
            )
        },
        interpretation="Only complete paired48-query matrices support a state comparison; "
        "state errors do not establish closed-loop task success.",
    )
    path = root / "preflight.json"
    write_json(path, preflight)
    print(json.dumps(dict(preflight=str(path), sha256=sha256(path), queries=96, gpu_started=False)))


if __name__ == "__main__":
    main()
