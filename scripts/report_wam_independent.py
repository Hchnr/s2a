"""Publish complete effect and horizon reports without promoting partial trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from s2a.artifacts import code_fingerprint, sha256, write_json
from s2a.task_acceptance import task_outcome

ROOT = Path("outputs/wam_independent_20261020")
TARGETS = ("akita_black_bowl_1", "wine_bottle_1", "cream_cheese_1")


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
            for source, digest in case["files"].items():
                if sha256(source) != digest:
                    raise ValueError("verified raw physics evidence changed")
        for source in (path, integrity_path):
            files[str(source)] = sha256(source)
        summaries.append(summary)
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
        "所有协议暂停物理时钟等待推理。固定权重、mask修正版服务与解析S2A；未执行真机，外部Cosmos生产代码未修改。",
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
        "下表只统计未达到稳定成功的WAM trial。接触采用模拟器原生抓取接触检查；没有接触不自动等于夹爪模型错误，也可能是位置/朝向参考不合适。",
        "",
        "| 来源 | 未稳定成功 | 目标从未有抓取接触 | 有远期闭合留在未执行后缀 | TCP跟踪RMSE中位数 mm | SO(3) RMSE中位数 rad |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for checkpoint in current["config"]["checkpoints"]:
        rows = [x for x in current["cases"] if x["label"] == checkpoint["name"] and not x["outcome"]["stable"]]
        no_contact = sum(
            r["object_progress"][TARGETS[int(r["record"].split("_")[0][4:])]]["grasp_contact_steps"] == 0 for r in rows
        )
        deferred = sum(r["opening_diagnostics"]["deferred_closure_chunks"] > 0 for r in rows)
        position = float(np.median([r["tracking"]["position_rmse_m"] for r in rows])) * 1000 if rows else 0
        rotation = float(np.median([r["tracking"]["rotation_rmse_rad"] for r in rows])) if rows else 0
        lines.append(
            f"| {checkpoint['name']} | {len(rows)} | {no_contact} | {deferred} | {position:.3f} | {rotation:.4f} |"
        )
    lines += [
        "",
        "远期闭合计数使用预先固定的opening<0.35描述阈值：执行前缀未低于阈值而未执行后缀低于阈值。它是现象证据，不单独证明失败因果；应结合下方执行窗口对照、逐case轨迹及实际抓取检查判断。",
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
        "本轮216条完整物理实验，共129600控制步。全部协议保留实际请求、响应、当前锚点、参考解码、输入像素和物理时间证据；正式汇总与报告拒绝缺项或覆盖。每个服务先逐项复现已有审计样本，四份权重全量SHA与上一轮相同。",
        "",
        "该报告由完整证据生成。主要失败原因的最终判断还须结合逐case复核记录，不把统计共现直接写成唯一因果。",
        "",
    ]
    report_path.write_text("\n".join(lines))
    write_json(manifest_path, dict(code=code_fingerprint(), inputs_sha256=files, report_sha256=sha256(report_path)))
    print(report_path)


if __name__ == "__main__":
    main()
