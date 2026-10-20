"""Run isolated CPU counterfactuals on saved WAM references, gated by exact replay."""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from verify_wam_acceptance import verify_case

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json
from s2a.task_acceptance import task_outcome
from s2a.wam_reference_replay import run_replay


def report_destination(c):
    repo = Path(__file__).resolve().parents[1]
    path = (repo / c.get("report_path", "reports/wam_gripper_replay_20261020.md")).resolve()
    if not path.is_relative_to(repo / "reports"):
        raise ValueError("replay report_path must resolve inside repository reports/")
    return path


def prepare(c, root):
    destination = root / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    source_c, source_root = load_config(c["source_config"])
    fixtures_path = source_root / "simulation/fixtures.json"
    fixture_data = json.loads(fixtures_path.read_text())
    if fixture_data["config"] != source_c:
        raise ValueError("source fixture protocol changed")
    fixtures = {x["name"]: x for x in fixture_data["records"]}
    variants = {x["name"]: x for x in source_c["checkpoints"]}
    cases = []
    for spec in c["cases"]:
        path = source_root / f"simulation/{spec['checkpoint']}/{spec['record']}_s{spec['sampling_seed']}/result.json"
        integrity = verify_case(path, source_c, fixtures, variants)
        source_manifest = json.loads(path.with_name("manifest.json").read_text())
        config = source_manifest["config"]
        if integrity["steps"] != c["steps"] or config["execution_horizon"] != 8:
            raise ValueError("source reference budget or execution protocol differs")
        files = dict(integrity["files"])
        for item in (Path(config["tool_contract"]), fixtures_path, Path(c["source_config"])):
            files[str(item)] = sha256(item)
        cases.append(
            dict(
                spec,
                source_result=str(path),
                record=source_manifest["record"],
                simulation_config=config,
                inputs_sha256=files,
            )
        )
    write_json(destination, dict(code=code_fingerprint(), config=c, cases=cases))


def execute(c, root, config_path, modes):
    prepared = json.loads((root / "manifest.json").read_text())
    if prepared["config"] != c:
        raise ValueError("replay protocol changed after preparation")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        LD_LIBRARY_PATH="",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        LP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )

    def worker(job):
        case, mode = job
        logfile = root / f"{case['name']}_{mode}.log"
        if logfile.exists():
            raise FileExistsError(logfile)
        cmd = [
            str(Path(c["simulation_python"]).absolute()),
            __file__,
            "--config",
            config_path,
            "--seed",
            str(c["seed"]),
            "--worker",
            case["name"],
            "--mode",
            mode,
        ]
        with logfile.open("x") as stream:
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"replay worker failed: {logfile}")
        print(json.dumps(dict(case=case["name"], mode=mode, complete=True)), flush=True)

    if any(mode != "intent" for mode in modes):
        for case in c["cases"]:
            result = json.loads((root / f"physics/{case['name']}/intent/result.json").read_text())
            if result["error"] is not None or not result["original_trajectory_reproduced"]:
                raise ValueError("counterfactual interpretation requires source trajectory reproduction")
    with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
        list(pool.map(worker, [(case, mode) for case in c["cases"] for mode in modes]))


def report(c, root):
    dest = root / "summary.json"
    report_path = report_destination(c)
    if dest.exists() or report_path.exists():
        raise FileExistsError("replay report already exists")
    prepared = json.loads((root / "manifest.json").read_text())
    if prepared["config"] != c:
        raise ValueError("replay report protocol changed")
    for case in prepared["cases"]:
        for path, digest in case["inputs_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("replay source evidence changed")
    rows, files = [], {}
    for case in c["cases"]:
        source = next(x for x in prepared["cases"] if x["name"] == case["name"])
        original = [
            json.loads(line) for line in Path(source["source_result"]).with_name("steps.jsonl").read_text().splitlines()
        ]
        for mode in c["gripper_modes"]:
            directory = root / f"physics/{case['name']}/{mode}"
            result = json.loads((directory / "result.json").read_text())
            if result["error"] is not None or result["steps"] != result["expected_steps"]:
                raise ValueError("incomplete replay evidence")
            if result["steps"] != c["steps"]:
                raise ValueError("replay budget differs")
            if mode == "intent" and not result["original_trajectory_reproduced"]:
                raise ValueError("source trajectory not reproduced")
            steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            if len(steps) != c["steps"] or [s["step"] for s in steps] != list(range(c["steps"])):
                raise ValueError("replay step coverage differs")
            for key in ("desired_source", "opening_target"):
                np.testing.assert_array_equal([s[key] for s in steps], [s[key] for s in original])
            np.testing.assert_allclose(
                [s["actual_dt"] for s in steps], 1 / source["simulation_config"]["control_fps"], atol=1e-10, rtol=0
            )
            if (
                task_outcome([s["success"] for s in steps], expected_steps=c["steps"], stable_steps=c["stable_steps"])
                != result["outcome"]
            ):
                raise ValueError("replay summary differs from native task predicates")
            for path in directory.glob("*"):
                if path.is_file():
                    files[str(path)] = sha256(path)
            rows.append(result)
    files[str(root / "manifest.json")] = sha256(root / "manifest.json")
    lines = [
        "# 保存WAM参考的夹爪执行诊断 — 2026-10-20",
        "",
        f"事后选择{len(c['cases'])}条已完成WAM固定参考，比较{len(c['gripper_modes'])}种夹爪方式，共{len(rows)}条CPU独立物理重放，每条{c['steps']}步，不查询模型。",
        "",
        "固定原world参考、arm算法/参数/限速；各反事实使用自身当前反馈，因此arm命令数值可以不同。原intent必须复现原始轨迹和判定才执行其余夹爪。",
        "",
        "| source | gripper | 曾成功 | 终态 | 稳定 | TCP RMSE mm | SO(3) RMSE rad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        o = row["outcome"]
        lines.append(
            f"| {row['name']} | {row['mode']} | {int(o['ever'])} | {int(o['terminal'])} | {int(o['stable'])} | "
            f"{row['position_rmse_tcp_m'] * 1000:.3f} | {row['rotation_rmse_rad']:.5f} |"
        )
    lines += [
        "",
        "这些是保存参考的执行反事实，后续参考来自原轨迹观测，不是对反事实当前图像重新推理；结果不能计入WAM闭环成功率或独立验收。",
        "Latch复用历史0.8/0.9阈值，同时改变闭合和释放条件；position改变夹爪接口，均仅回答策略总影响，不能将成败差异唯一归于释放机制。原始完整轨迹、当前反馈及图像保存在outputs。",
        "",
    ]
    with report_path.open("x") as stream:
        stream.write("\n".join(lines))
    write_json(
        dest, dict(code=code_fingerprint(), config=c, cases=rows, files_sha256=files, report_sha256=sha256(report_path))
    )
    print(report_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_gripper_replay_20261020.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", choices=("prepare", "baseline", "counterfactual", "report", "all"), default="all")
    parser.add_argument("--worker")
    parser.add_argument("--mode", choices=("intent", "position", "latch"))
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("study seed differs")
    if args.worker:
        print(json.dumps(run_replay(root, args.worker, args.mode)), flush=True)
        return
    if args.stage in ("prepare", "all"):
        prepare(c, root)
    if args.stage in ("baseline", "all"):
        execute(c, root, args.config, ["intent"])
    if args.stage in ("counterfactual", "all"):
        execute(c, root, args.config, [x for x in c["gripper_modes"] if x != "intent"])
    if args.stage in ("report", "all"):
        report(c, root)


if __name__ == "__main__":
    main()
