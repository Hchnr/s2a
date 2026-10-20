"""Render019 only from a complete signed audit, preserving separate seed trials."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_wam_position import Evidence
from run_wam_short_replan import cpu_environment, write_new

from s2a.artifacts import REPO, sha256

OLD_SHA = "a4560bb70cc21e204dc7bd0e35fc3aa2a7cab7365b394a6a4c1727473b60ad12"


def success_runs(rows, outcome):
    """Recompute native-success run lengths; no object/contact-based surrogate."""
    assert len(rows) == 600 and [r["step"] for r in rows] == list(range(600))
    flags = [r["success"] for r in rows]
    assert all(type(x) is bool for x in flags)
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    assert outcome == dict(
        ever=any(flags),
        terminal=flags[-1],
        stable=all(flags[-40:]),
        successful_steps=sum(flags),
        trailing_success_steps=current,
    )
    errors = np.asarray([r["opening_target"] - r["opening_after"] for r in rows])
    assert np.isfinite(errors).all()
    return dict(
        first_success_endpoint=next((i + 1 for i, x in enumerate(flags) if x), None),
        longest_success_s=longest / 20,
        trailing_success_s=current / 20,
        total_success_s=sum(flags) / 20,
        opening_mae=float(np.mean(np.abs(errors))),
        opening_rmse=float(np.sqrt(np.mean(errors**2))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/wam_raw8_seed43_20261021.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cpu_environment()
    report, output = args.report.resolve(), args.output_dir.resolve()
    assert report.is_relative_to(REPO / "reports") and not report.exists()
    assert output.is_relative_to(REPO / "outputs") and not output.exists()
    evidence = Evidence()
    evidence.read(__file__)
    evidence.check(args.audit, args.audit_sha256)
    data = evidence.json(args.audit)
    assert data["schema_version"] == "s2a.raw8-seed43-independent-audit.v1"
    assert data["complete"] and data["all_cases_independently_verified"] and data["normal_process_release"]
    assert not data["errors"] and (
        data["verified_cases"],
        data["verified_physics_steps"],
        data["verified_queries"],
        data["verified_http_queries"],
    ) == (3, 1800, 225, 225)
    assert data["sampling_seed"] == 43 and data["experiment_seed"] == 42
    assert data["registered_budget"] == dict(
        wam_trials=3, physics_steps=1800, h40_queries=225, new_oracle_trials=0, fresh_initial_states=0
    )
    assert len(data["resources"]) == 3 and all(r["owned_service_released_normally"] for r in data["resources"])
    assert sum(len(r["processes"]) for r in data["resources"]) == 12
    assert not any(p["present"] for r in data["resources"] for p in r["processes"].values())
    old_path = Path(data["seed42_comparison"]["source_path"])
    assert data["seed42_comparison"]["source_sha256"] == OLD_SHA
    evidence.check(old_path, OLD_SHA)
    old = evidence.json(old_path)
    paired = {
        42: [c for c in old["cases"] if c["profile"] == "interval8_alpha1"],
        43: [c["result"] for c in data["cases"] if c["status"] == "verified"],
    }
    assert all(len(values) == 3 for values in paired.values())
    details, summary = [], {}
    for seed, cases in paired.items():
        assert {c["record"] for c in cases} == {f"task{i}_init0" for i in range(3)}
        summary[seed] = {k: sum(c["outcome"][k] for c in cases) for k in ("ever", "terminal", "stable")}
        for case in sorted(cases, key=lambda x: x["record"]):
            folder = (REPO / case["case"]).resolve()
            path = folder / "steps.jsonl"
            signatures = data["input_sha256"] if seed == 43 else old["source_before_after_hashes"]
            evidence.check(path, signatures[str(path)])
            rows = [json.loads(line) for line in evidence.read(path).splitlines()]
            runs = success_runs(rows, case["outcome"])
            if seed == 43:
                assert runs["opening_mae"] == case["opening_tracking"]["mae"]
                assert runs["opening_rmse"] == case["opening_tracking"]["rmse"]
                assert case["first_query019_provenance"]["actual_sampling_seed"] == 43
                assert not case["first_query019_provenance"]["seed43_raw_equals_seed42_required"]
            details.append(
                dict(
                    seed=seed,
                    record=case["record"],
                    case=str(folder),
                    outcome=case["outcome"],
                    tracking=case["tracking"]["admitted"],
                    target_grasp_contact_steps=case["target_grasp_contact_steps"],
                    **runs,
                )
            )
    assert data["group_outcome"] == {
        k: dict(successes=summary[43][k], trials=3, rate=summary[43][k] / 3) for k in ("ever", "terminal", "stable")
    }
    labels = {"task0_init0": "bowl", "task1_init0": "wine", "task2_init0": "cream"}
    lines = [
        "# 019：raw8 第二采样种子重复性",
        "",
        f"固定第二采样 seed43 已完成：ever {summary[43]['ever']}/3、末端 {summary[43]['terminal']}/3、稳定 {summary[43]['stable']}/3。原 seed42 对应30秒对照分别为 {summary[42]['ever']}/3、{summary[42]['terminal']}/3、{summary[42]['stable']}/3。两个种子各自保持3条分母，不挑成功种子拼接策略；三旧初态不能证明多初态或未见数据稳定性。",
        "",
        "| sampling seed | task/init0 | ever / terminal / stable | 首次成功端点 | 最长 / 尾部连续成功 s | TCP RMSE mm | SO(3) RMSE rad | opening MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in details:
        outcome, tracking = item["outcome"], item["tracking"]
        flags = " / ".join("1" if outcome[k] else "0" for k in ("ever", "terminal", "stable"))
        first = "—" if item["first_success_endpoint"] is None else str(item["first_success_endpoint"])
        lines.append(
            f"| {item['seed']} | {labels[item['record']]} | {flags} | {first} | {item['longest_success_s']:.2f} / {item['trailing_success_s']:.2f} | {tracking['position_rmse_m'] * 1000:.3f} | {tracking['so3_rmse_rad']:.5f} | {item['opening_mae']:.5f} |"
        )
    lines.extend(
        [
            "",
            "稳定谓词沿用最后40个原生task success端点全部为真（2秒）；首次端点从1开始计。最长连续与累计成功不同，不把短暂触发算稳定。",
            "",
            "固定协议为8400非EMA、有效10槽mask、SDPA、35采样步、guidance1、shift5、20Hz/H40，每8真实步重新观测，完整模型旋转/opening、v3位置/保持夹爪、response inverse .5/rate5、无governor。不计推理延迟：逐query前后完整物理SHA/时钟不变。三条均600端点，75个完整H40请求；没有按成功提前停止或自动重试。",
            "",
            "首query认证实际seed43的request/response、checkpoint、mask、图像、锚点与归一化；同fixture初态、PNG、controller/motor初始化及除seed外条件与42一致。43生成值无需等于42，未借用42输出。后续闭环输入随轨迹分叉，不能称全部生成参考配对。夹爪和控制算法未因本项变化。",
            "",
            "原015同协议seed43正确参考资格3/3只作为执行准入，不计本项WAM分母；017的1200步持续效果亦单独保留，本项沿用原600步/30秒验收。原42/43各固定三旧任务；未读取fresh性能、未增加oracle或真机预算。",
            "",
            "014字面检查器独立重建全部1800端点的原始/有效参考、局部增量积分、仿真暂停、OSC/限幅/电机、qpos/opening连续性、三值夹爪及task谓词；源代码和输入SHA前后重核。225个实际生成目录与saved args/outputs逐一对应，225个HTTP POST /generate均200且位于登记服务生命周期内。",
            "",
            "模型实际只使用物理GPU4/5/6；CPU/OSMesa为空CUDA/LD。另行保存的live-ready证据补齐12个coordinator/model/matrix/physics PID、startticks、PPid、CVD及实际GET /info，最终登记身份全部非live，服务正常shutdown。泛用release仅保存model returncode，matrix结果保存worker returncode；本报告不编造额外Popen wait状态，也不声明完整prepared-caption/退出内存noise ledger。",
            "",
            "低TCP误差不能单独证明任务能力，也不能把失败唯一归于权重或S2A。opening是开度误差，不能替代闭合力/抓放判断；目标双指contact仅作指标，不能以contact=0否定原生task成功。该第二种子重复性对照不自动推广新控制方法、增加网格或开启fresh。",
            "",
            f"完整审计：`{args.audit.resolve().relative_to(REPO)}`，SHA `{args.audit_sha256}`。",
            "",
            f"原42对照：`{old_path.relative_to(REPO)}`，SHA `{OLD_SHA}`。",
            "",
            "源码/固定协议见 `plans/v0.0.1/019_raw8_seed43_replication_20261021.md`、`scripts/audit_wam_raw8_seed43.py`；数值审计与报告重算产物中的SHA ledger保留完整来源。",
            "",
        ]
    )
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as stream:
        stream.write("\n".join(lines))
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        report_path=str(report),
        report_sha256=sha256(report),
        complete_audit_path=str(args.audit.resolve()),
        complete_audit_sha256=args.audit_sha256,
        details=details,
        summary_by_seed=summary,
        input_sha256=evidence.hashes,
        new_model_requests=0,
        new_physics_steps=0,
        fresh_performance_read=False,
        previously_verified_full_audit_input_closure_rehashed_by_report=False,
        complete_audit_signature_and_six_reported_step_logs_rechecked=True,
    )
    write_new(output / "report_manifest.json", manifest)
    print(
        json.dumps(
            dict(
                report=str(report),
                report_sha256=sha256(report),
                manifest_sha256=sha256(output / "report_manifest.json"),
            )
        )
    )


if __name__ == "__main__":
    main()
