"""Publish complete effect and horizon reports without promoting partial trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.geometry import pose_error
from s2a.task_acceptance import task_outcome

ROOT = Path("outputs/wam_independent_20261020")
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


def failure_stages(rows):
    """Partition unsuccessful trials by target contact first, then task progress."""
    counts = dict(no_target_grasp=0, grasp_never_success=0, grasp_ever_not_stable=0)
    failed = [row for row in rows if not row["outcome"]["stable"]]
    for row in failed:
        target = TARGETS[int(row["record"].split("_")[0][4:])]
        if row["object_progress"][target]["grasp_contact_steps"] == 0:
            counts["no_target_grasp"] += 1
        elif not row["outcome"]["ever"]:
            counts["grasp_never_success"] += 1
        else:
            counts["grasp_ever_not_stable"] += 1
    if sum(counts.values()) != len(failed):
        raise ValueError("failure stages do not partition all unsuccessful trials")
    return dict(unstable=len(failed), **counts)


def verify_gripper_replay():
    """Bind the completed 3x3 CPU diagnosis to its source and raw evidence."""
    config_path = Path("configs/wam_gripper_replay_20261020.json")
    config = json.loads(config_path.read_text())
    root = Path(config["output_dir"])
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    report_path = Path("reports/wam_gripper_replay_20261020.md")
    summary = json.loads(summary_path.read_text())
    prepared = json.loads(manifest_path.read_text())
    if summary["config"] != config or prepared["config"] != config:
        raise ValueError("CPU replay configuration changed")
    names = [case["name"] for case in config["cases"]]
    modes = config["gripper_modes"]
    if (
        len(names) != 3
        or len(set(names)) != 3
        or modes != ["intent", "position", "latch"]
        or config["steps"] != 600
        or config["stable_steps"] != 40
        or config["seed"] != 42
    ):
        raise ValueError("CPU replay requires the registered three cases and three modes")
    expected = {(name, mode) for name in names for mode in modes}
    rows = {(row["name"], row["mode"]): row for row in summary["cases"]}
    sources = {case["name"]: case for case in prepared["cases"]}
    if set(rows) != expected or len(summary["cases"]) != 9 or set(sources) != set(names) or len(prepared["cases"]) != 3:
        raise ValueError("CPU replay matrix is missing or duplicates a case")
    raw_names = (
        "manifest.json",
        "controller.json",
        "steps.jsonl",
        "result.json",
        "render_hashes.json",
        "initial_input.png",
        "final_input.png",
    )
    expected_files = {manifest_path.resolve()} | {
        (root / "physics" / name / mode / filename).resolve() for name, mode in expected for filename in raw_names
    }
    signatures = summary["files_sha256"]
    if {Path(path).resolve() for path in signatures} != expected_files or len(signatures) != len(expected_files):
        raise ValueError("CPU replay raw evidence fingerprint coverage differs")
    files = {}
    for path, digest in signatures.items():
        if sha256(path) != digest:
            raise ValueError("CPU replay raw evidence changed")
        files[path] = digest
    if sha256(report_path) != summary["report_sha256"]:
        raise ValueError("CPU replay report changed")
    for path in (summary_path, report_path, config_path):
        files[str(path)] = sha256(path)
    for spec in config["cases"]:
        source = sources[spec["name"]]
        if (
            source["checkpoint"] != spec["checkpoint"]
            or source["sampling_seed"] != spec["sampling_seed"]
            or source["record"]["name"] != spec["record"]
        ):
            raise ValueError("CPU replay source identity differs")
        for path, digest in source["inputs_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("CPU replay original WAM evidence changed")
            files[path] = digest
        original_path = Path(source["source_result"])
        source_manifest = json.loads(original_path.with_name("manifest.json").read_text())
        if source_manifest["config"] != source["simulation_config"] or source_manifest["record"] != source["record"]:
            raise ValueError("CPU replay prepared source differs from the original manifest")
        original = [json.loads(line) for line in original_path.with_name("steps.jsonl").read_text().splitlines()]
        if len(original) != 600 or [row["step"] for row in original] != list(range(600)):
            raise ValueError("CPU replay original reference budget differs")
        h = np.asarray(json.loads(Path(source["simulation_config"]["tool_contract"]).read_text())["source_from_tcp"])
        for mode in modes:
            directory = root / "physics" / spec["name"] / mode
            result = json.loads((directory / "result.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            if (
                result != rows[spec["name"], mode]
                or manifest["source"] != source
                or manifest["config"] != config
                or manifest["mode"] != mode
            ):
                raise ValueError("CPU replay summary or worker identity differs")
            if result["error"] is not None or result["steps"] != 600 or result["expected_steps"] != 600:
                raise ValueError("CPU replay has an error or incomplete budget")
            if result["original_trajectory_reproduced"] is not (mode == "intent"):
                raise ValueError("CPU replay original intent reproduction gate differs")
            steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
            if len(steps) != 600 or [row["step"] for row in steps] != list(range(600)):
                raise ValueError("CPU replay actual step coverage differs")
            np.testing.assert_allclose(
                [row["actual_dt"] for row in steps],
                1 / source["simulation_config"]["control_fps"],
                atol=1e-10,
                rtol=0,
            )
            outcome = task_outcome([row["success"] for row in steps], expected_steps=600, stable_steps=40)
            if outcome != result["outcome"]:
                raise ValueError("CPU replay outcome differs from its native task predicates")
            for key in ("desired_source", "opening_target"):
                np.testing.assert_array_equal([row[key] for row in steps], [row[key] for row in original])
            if mode == "intent":
                for key in ("measured_source", "after_source", "command", "opening_after"):
                    np.testing.assert_allclose(
                        [row[key] for row in steps], [row[key] for row in original], atol=config["replay_atol"], rtol=0
                    )
                if [row["success"] for row in steps] != [row["success"] for row in original]:
                    raise ValueError("CPU replay baseline task predicates differ")
                for target in original[0]["objects"]:
                    np.testing.assert_allclose(
                        [row["objects"][target]["position"] for row in steps],
                        [row["objects"][target]["position"] for row in original],
                        atol=config["replay_atol"],
                        rtol=0,
                    )
                    if [row["objects"][target]["grasp_contact"] for row in steps] != [
                        row["objects"][target]["grasp_contact"] for row in original
                    ]:
                        raise ValueError("CPU replay baseline grasp contacts differ")
            errors = pose_error(
                np.asarray([row["desired_source"] for row in steps]) @ h,
                np.asarray([row["after_source"] for row in steps]) @ h,
            )
            for key, columns in (("position_rmse_tcp_m", slice(0, 3)), ("rotation_rmse_rad", slice(3, 6))):
                np.testing.assert_allclose(
                    result[key], float(np.sqrt(np.mean(np.sum(errors[:, columns] ** 2, axis=1)))), atol=1e-12, rtol=0
                )
    return summary, files


def gripper_replay_section(summary):
    lines = [
        "",
        "## CPU固定参考夹爪执行诊断（单列）",
        "",
        "[完整重放报告](wam_gripper_replay_20261020.md)及[原始汇总](../outputs/wam_gripper_replay_20261020/summary.json)。事后选择三条已完成WAM轨迹，比较intent、position和latch，共9条/5400步；三条intent均已复现原始轨迹及任务/抓取判定。",
        "",
        "| 原轨迹 | 夹爪方式 | 曾成功 | 终态 | 稳定 | TCP RMSE mm | SO(3) RMSE rad |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["cases"]:
        outcome = row["outcome"]
        lines.append(
            f"| {row['name']} | {row['mode']} | {int(outcome['ever'])} | {int(outcome['terminal'])} | {int(outcome['stable'])} | "
            f"{row['position_rmse_tcp_m'] * 1000:.3f} | {row['rotation_rmse_rad']:.5f} |"
        )
    lines += [
        "",
        "原init0的稳定成功在position下保留，latch使该成功丢失；其余两条原失败轨迹在position/latch下均未救回，且均未曾成功。上述差异只适用于这三条事后选择的固定参考。",
        "",
        "重放固定原world参考，arm使用各自当前反馈，不查询模型；后续参考来自原轨迹观测，未对反事实当前图像重新推理，因此不能计入WAM闭环成功率或独立验收。Latch同时改变闭合与释放条件，position改变夹爪接口及与arm跟踪的交互，不能将差异唯一归于释放机制。",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("fixed reporting seed is 42")
    report_path = Path("reports/wam_independent_20261020.md")
    manifest_path = ROOT / "report_manifest.json"
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError("formal report already exists")
    roots = [ROOT] + [Path(f"outputs/wam_horizon_diagnosis_20261020/execute{x}") for x in (16, 40)]
    summaries = []
    files = {}
    matrix_steps = 0
    for root in roots:
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        integrity_path = root / "integrity.json"
        integrity = json.loads(integrity_path.read_text())
        config = summary["config"]
        expected = (
            len(config["tasks"])
            * config["online_initial_states_per_task"]
            * len(config["sampling_seeds"])
            * (len(config["checkpoints"]) + 2)
        )
        if len(summary["cases"]) != expected or len(integrity["cases"]) != expected:
            raise ValueError("missing complete protocol matrix")
        for case in integrity["cases"]:
            if case["steps"] != config["online_max_steps"] or case["steps"] != 600:
                raise ValueError("verified physics budget differs")
            matrix_steps += case["steps"]
            for source, digest in case["files"].items():
                if sha256(source) != digest:
                    raise ValueError("verified raw physics evidence changed")
        for source in (path, integrity_path):
            files[str(source)] = sha256(source)
        summaries.append(summary)
    if [len(summary["cases"]) for summary in summaries] != [144, 36, 36] or matrix_steps != 129600:
        raise ValueError("complete main/window experiment count differs")
    replay, replay_files = verify_gripper_replay()
    files.update(replay_files)
    current = summaries[0]
    historical = Path("outputs/wam_checkpoint_acceptance_20261019v2/native_horizon40")
    old_integrity_path = historical / "integrity.json"
    old_integrity = json.loads(old_integrity_path.read_text())
    if not old_integrity["complete"] or len(old_integrity["physics"]) != 12:
        raise ValueError("historical execute8 evidence incomplete")
    for case in old_integrity["physics"]:
        for source, digest in case["files"].items():
            if sha256(source) != digest:
                raise ValueError("historical execute8 evidence changed")
    old_fixtures = json.loads((historical / "simulation/fixtures.json").read_text())["records"]
    if old_fixtures != json.loads((roots[1] / "simulation/fixtures.json").read_text())["records"]:
        raise ValueError("diagnosis fixtures differ from historical execute8")
    files[str(old_integrity_path)] = sha256(old_integrity_path)
    old_outcomes = []
    for path in sorted((historical / "simulation/8800_masked").glob("*/steps.jsonl")):
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        old_outcomes.append(task_outcome([s["success"] for s in steps], expected_steps=600, stable_steps=40))
    if len(old_outcomes) != 12:
        raise ValueError("historical execute8 step logs incomplete")
    lines = [
        "# WAM 独立物理评测与失败诊断 — 2026-10-20",
        "",
        "WAM图像闭环评测暂停物理时钟等待推理，CPU固定参考重放不查询模型。权重、mask修正版服务与解析S2A固定；未执行真机，外部Cosmos生产代码未修改。",
        "",
        "## 独立物理评测",
        "",
        "3任务×4初态×2采样seed。每次600控制步，生成40步、执行8步；稳定成功要求最后40步均满足原生任务谓词。",
        "",
        "| 来源 | 曾成功 | 终态成功 | 稳定成功 | 工作性门槛 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, group in current["groups"].items():
        n = group["trial_count"]
        lines.append(
            f"| {name} | {group['ever']}/{n} | {group['terminal']}/{n} | {group['stable']}/{n} | {'达到' if group['meets_operational_threshold'] else '未达到'} |"
        )
    lines += [
        "",
        "门槛在结果前固定：总体稳定成功率≥80%、各任务≥50%。这是本实验的工作性标准，不能直接认证总体可靠性。12个初态才是场景单位，24次运行不是24个独立场景。",
        "",
        "### 逐任务与不确定性",
        "",
        "| 来源 | bowl | wine | cream | 初态分组bootstrap 95%区间 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, group in current["groups"].items():
        values = " | ".join(f"{v:.1%}" for v in group["per_task_stable_fraction"])
        lo, hi = group["interval95"]
        lines.append(f"| {name} | {values} | [{lo:.1%}, {hi:.1%}] |")
    lines += [
        "",
        "区间按任务分层、以初态为单位重采样，两个seed一起保留。样本量小，全零/全一时百分位区间可能退化，不能解读为没有总体不确定性。",
        "",
        "## 失败现象与执行误差",
        "",
        "下表只统计未达到稳定成功的WAM trial。三类互斥且合计等于未稳定条数：无目标grasp优先归类（包括曾成功但无grasp）；其余按从未成功或曾成功但未稳定划分。grasp仅指模拟器原生双侧指垫接触，不保证稳定抓持。",
        "",
        "| 来源 | 未稳定成功 | 无目标grasp | 有目标grasp但从未成功 | 有目标grasp且曾成功但未稳定 | 有远期闭合留在未执行后缀 | TCP跟踪RMSE中位数 mm | SO(3) RMSE中位数 rad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in current["config"]["checkpoints"]:
        rows = [x for x in current["cases"] if x["label"] == checkpoint["name"] and not x["outcome"]["stable"]]
        stages = failure_stages(rows)
        deferred = sum(r["opening_diagnostics"]["deferred_closure_chunks"] > 0 for r in rows)
        position = float(np.median([r["tracking"]["position_rmse_m"] for r in rows])) * 1000 if rows else 0
        rotation = float(np.median([r["tracking"]["rotation_rmse_rad"] for r in rows])) if rows else 0
        lines.append(
            f"| {checkpoint['name']} | {stages['unstable']} | {stages['no_target_grasp']} | {stages['grasp_never_success']} | "
            f"{stages['grasp_ever_not_stable']} | {deferred} | {position:.3f} | {rotation:.4f} |"
        )
    lines += [
        "",
        "这些分类描述观测阶段，不能唯一归因于权重、参考、夹爪转换或S2A；无grasp也可能来自位置/朝向参考不合适。远期闭合计数采用预先固定的opening<0.35参考描述阈值，不等于未发送实际闭合命令；应结合执行窗口、逐case轨迹与实际命令判断。",
        "",
        "## 旧集合执行窗口对照",
        "",
        "使用上一轮6个旧初态×2seed，只作开发诊断，不并入上面的独立物理评测，也不根据结果改动其冻结协议。",
        "",
        "| 执行步数 | 来源 | 曾成功 | 终态成功 | 稳定成功 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    lines.append(
        f"| 8（历史完整结果） | 8800_masked | {sum(r['ever'] for r in old_outcomes)}/12 | "
        f"{sum(r['terminal'] for r in old_outcomes)}/12 | {sum(r['stable'] for r in old_outcomes)}/12 |"
    )
    for summary in summaries[1:]:
        for name, group in summary["groups"].items():
            lines.append(
                f"| {summary['config']['online_execution_horizon']} | {name} | {group['ever']}/12 | {group['terminal']}/12 | {group['stable']}/12 |"
            )
    lines += [
        "",
        "窗口改变同时影响重规划频率与执行参考长度，因此若控制跟踪恶化，不能把任务变化全部归因于模型。完整case指标在各summary.json中。",
    ]
    lines += gripper_replay_section(replay)
    lines += [
        "",
        "## 数据独立性",
        "",
        "新初态与已记录的物理评测/控制调参集合分离。完整command匹配找到8条训练catalog来源，其中1条与旧离线state样本重叠；4条来源尚未匹配。未匹配不是训练未见的证明。历史训练消费账本未恢复。",
        "",
        "| 来源 | 旧离线重叠稳定/次数 | 匹配且旧离线不重叠稳定/次数 | 来源未解决稳定/次数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, group in current["groups"].items():
        strata = group["source_strata"]
        values = " | ".join(
            f"{strata[k]['stable']}/{strata[k]['trials']}"
            for k in ("prior_offline_overlap", "matched_without_prior_offline_overlap", "unresolved_source")
        )
        lines.append(f"| {name} | {values} |")
    lines += [
        "",
        "## 完整性",
        "",
        "主独立矩阵及旧集合执行16/40诊断合计216条完整物理实验、129600控制步；CPU固定参考重放另计9条、5400步，不并入216条或WAM闭环成功率。历史执行8的12条仅引用既有证据，不重复计入本轮实验。",
        "",
        "主/窗口WAM试验保留实际请求、响应、当前锚点、参考解码、输入像素和物理时间证据；CPU重放独立核对9条完整预算、3条intent复现、全部重放/原始来源文件SHA及重放报告SHA。正式汇总与报告拒绝缺项或覆盖。每个服务先逐项复现已有审计样本，四份权重全量SHA与上一轮相同。",
        "",
        "该报告由完整证据生成。主要失败原因的最终判断还须结合逐case复核记录，不把统计共现直接写成唯一因果。",
        "",
    ]
    report_path.write_text("\n".join(lines))
    write_json(manifest_path, dict(code=code_fingerprint(), inputs_sha256=files, report_sha256=sha256(report_path)))
    print(report_path)


if __name__ == "__main__":
    main()
