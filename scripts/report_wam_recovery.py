"""Render the completed registered 24-case recovery pilot; no model or physics."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "intent_hold",
    "recovering_intent_hold",
    "recovering_intent_hold_alpha05",
    "recovering_intent_hold_alpha0",
)
LABELS = ("no governor / alpha1", "governor / alpha1", "governor / alpha0.5", "governor / alpha0")
COLORS = ("#777777", "#0072B2", "#D55E00", "#009E73")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_inputs(mapping):
    for path, expected in mapping.items():
        if digest(path) != expected:
            raise ValueError(f"changed audit input: {path}")


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(value)


def load_rows(case):
    rows = [json.loads(line) for line in (Path(case["case"]) / "steps.jsonl").read_text().splitlines()]
    assert len(rows) == 600 and [row["step"] for row in rows] == list(range(600))
    flags = np.asarray([row["success"] for row in rows], dtype=bool)
    trailing = next((i for i, flag in enumerate(flags[::-1]) if not flag), len(flags))
    outcome = dict(
        ever=bool(flags.any()),
        terminal=bool(flags[-1]),
        stable=bool(trailing >= 40),
        successful_steps=int(flags.sum()),
        trailing_success_steps=trailing,
    )
    assert case["outcome"] == outcome
    assert np.max(np.abs(np.asarray([r["actual_dt"] for r in rows]) - 0.05)) < 1e-9
    return rows


def cumulative(rows, field):
    total, prior, previous_chunk = 0.0, 0.0, None
    values = []
    for row in rows:
        if row["chunk_index"] != previous_chunk:
            total += prior
            previous_chunk = row["chunk_index"]
        prior = row[field]
        values.append(total + prior)
    return values


def draw_pair(checkpoint, record, cases, data, output, pdf):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    time = (np.arange(600) + 1) * 0.05
    for variant, label, color in zip(VARIANTS, LABELS, COLORS, strict=True):
        case = next(
            c for c in cases if c["checkpoint"] == checkpoint and c["record"] == record and c["variant"] == variant
        )
        rows = data[case["case"]]
        admitted = np.asarray([r["desired_tcp"] for r in rows])
        # Logged TCP position error is desired minus actual in the same world frame.
        actual = admitted[:, :3, 3] - np.asarray([r["tcp_error"][:3] for r in rows])
        axes[0, 0].plot(actual[:, 0], actual[:, 1], color=color, lw=1.25, label=label)
        axes[0, 0].scatter(*actual[0, :2], color=color, marker="o", s=12)
        axes[0, 0].scatter(*actual[-1, :2], color=color, marker="x", s=20)
        for axis, section in ((axes[0, 1], slice(0, 3)), (axes[2, 1], slice(3, 6))):
            scale = 1000 if section.start == 0 else 1
            axis.plot(
                time, np.linalg.norm([r["tcp_error"][section] for r in rows], axis=1) * scale, color=color, lw=0.9
            )
            if variant != "intent_hold":
                axis.plot(
                    time,
                    np.linalg.norm([r["raw_model_tcp_error"][section] for r in rows], axis=1) * scale,
                    color=color,
                    ls="--",
                    lw=0.7,
                    alpha=0.65,
                )
        if variant != "intent_hold":
            axes[1, 0].plot(time, [r["opening_after"] for r in rows], color=color, lw=1.0, label=label)
            axes[1, 0].plot(time, [r["opening_target"] for r in rows], color=color, ls="--", lw=0.7)
            axes[1, 0].plot(time, [r["motor_opening_after"] for r in rows], color=color, ls=":", lw=0.7, alpha=0.75)
            axes[1, 1].plot(time, cumulative(rows, "model_time_s"), color=color, lw=1.1)
            axes[1, 1].plot(time, cumulative(rows, "retimed_clock_s"), color=color, ls=":", lw=0.8)
    flags = np.asarray(
        [
            [
                r["success"]
                for r in data[
                    next(
                        c["case"]
                        for c in cases
                        if c["checkpoint"] == checkpoint and c["record"] == record and c["variant"] == variant
                    )
                ]
            ]
            for variant in VARIANTS
        ]
    )
    axes[2, 0].imshow(
        flags, aspect="auto", extent=(0, 30, 3.5, -0.5), vmin=0, vmax=1, cmap="Greens", interpolation="nearest"
    )
    axes[2, 0].set_yticks(range(4), ["no gov a1", "gov a1", "gov a0.5", "gov a0"])
    axes[0, 0].set(title="Actual TCP XY; circle=start, cross=last", xlabel="world x (m)", ylabel="world y (m)")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(
        title="Position error: admitted solid; raw model-clock dashed", ylabel="mm", xlabel="physical time (s)"
    )
    axes[1, 0].set(
        title="Opening: measured solid; reference dashed; motor target dotted",
        ylabel="normalized opening",
        xlabel="physical time (s)",
        ylim=(-0.05, 1.05),
    )
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(time, time, color="black", ls="--", lw=0.7, label="physical clock")
    axes[1, 1].set(
        title="Cumulative model clock solid; retimed clock dotted",
        ylabel="consumed seconds",
        xlabel="physical time (s)",
    )
    axes[1, 1].legend(fontsize=7)
    axes[2, 0].set(title="Native task predicate (metrics only); green=true", xlabel="physical time (s)")
    axes[2, 1].set(
        title="SO(3) error: admitted solid; raw model-clock dashed", ylabel="rad", xlabel="physical time (s)"
    )
    for axis in axes.flat:
        if axis != axes[2, 0]:
            axis.grid(alpha=0.18)
    fig.suptitle(f"{checkpoint} / {record} / seed42: paired causal recovery interventions", fontsize=13)
    path = output / f"{checkpoint}_{record}.png"
    fig.savefig(path, dpi=140)
    pdf.savefig(fig)
    plt.close(fig)
    return path


def number(value, scale=1):
    return f"{value * scale:.3f}"


def make_report(audit_path, audit, cases, groups, data, plots, output):
    stable_count = sum(case["outcome"]["stable"] for case in cases)
    ever_count = sum(case["outcome"]["ever"] for case in cases)
    stable_statement = (
        "本批四个方案均未取得末尾连续40端点的稳定成功，不能提升为已完成稳定任务能力。"
        if stable_count == 0
        else f"本批末尾稳定成功{stable_count}/24，仅限这组既有开发场景，不能视为泛化验收。"
    )
    lines = [
        "# WAM recovery 开发对照：2026-10-21",
        "",
        "完整执行与独立审计了 **24 条 WAM 闭环开发评测（14,400 个 20 Hz 端点）**，以及独立的 6 条 oracle 资格正控（3,600 端点）。"
        + stable_statement,
        "",
        "本报告只统计两个 checkpoint、共同 guidance=1.0、3 个旧开发场景 init0/seed42、4 个方案、每条600步/30秒。无 governor 的方案执行40步前缀；governor 使用恢复桥及模型时间重定时，最迟80物理步重新查询。推理等待不推进物理时间。oracle按当前物理步选取特权 fixed-world chunks，仅用于执行资格，不能计入 WAM 分母。8400/8600 的附加评测、后继 v3 oracle 资格以及 fresh reservation 均不属于本报告。",
        "",
        "## 完整任务结果",
        "",
        "`stable` = 最后40个端点全部native success；`terminal` = 最后一个端点success；`ever` = 任一端点success。计分来自模拟器原生谓词，目标双指contact只作观测指标，不能替代任务谓词。",
        "",
        "| checkpoint | 方案 | complete | ever | terminal | stable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in groups:
        selected = [c for c in cases if c["checkpoint"] == group["checkpoint"] and c["variant"] == group["variant"]]
        lines.append(
            f"| {group['checkpoint']} | {group['variant']} | {len(selected)}/3 | {sum(c['outcome']['ever'] for c in selected)}/3 | {sum(c['outcome']['terminal'] for c in selected)}/3 | {sum(c['outcome']['stable'] for c in selected)}/3 |"
        )
    lines += ["", f"完整观察到的 {ever_count} 个 ever 案例（仍以末尾native谓词单独验收）：", ""]
    for case in cases:
        if case["outcome"]["ever"]:
            flags = [r["step"] for r in data[case["case"]] if r["success"]]
            description = (
                f"{flags[0]}–{flags[-1]}，共{len(flags)}端点"
                if flags == list(range(flags[0], flags[-1] + 1))
                else "、".join(map(str, flags)) + f"（非连续，共{len(flags)}端点）"
            )
            lines.append(
                f"- {case['checkpoint']} / {case['variant']} / {case['record']}：native成功step {description}，末尾连续成功0；目标双指contact {case['target_grasp_contact_steps']}端点。"
            )
    lines += [
        "",
        "15400 alpha0 的连续成功199–426共228端点，即11.4秒；随后427–599全部失败，不能把中途维持成功视为末尾稳定完成。8800 alpha0.5仅step154和157非连续通过，不能写成连续0.1秒成功。成功后仍继续查询/执行是冻结协议的一部分；本批没有用特权成功谓词触发停机。",
        "",
        "## 参考、控制及误差的不同含义",
        "",
        "四方案均使用已资格验证的 task-agnostic GripperIntentHold 和同一 response arm inverse/OSC 接口。adapter只使用当前/历史开度、真实motor目标和已生成参考，不输入对象、接触、任务标签或未来真值。motor达到闭合限位后command=0保持原闭合motor目标；观测开度在物体/机械阻塞下可以明显大于motor目标，不能将0命令解释为零握力，也不能把闭合目标当作实际夹爪开度/力。",
        "",
        "alpha1保留每个raw WAM旋转增量；alpha0.5把每个增量的SO(3)对数乘0.5后重新积分，改变了模型参考。alpha0将每个旋转增量置为单位旋转：**只在该次query的实测TCP anchor朝向保持**，下一次query重新取当前实测anchor，绝非全episode固定朝向；平移和开度仍来自WAM。alpha0仅为部分输出分量消融，不能用于认证原checkpoint完整输出。raw/effective128及各query anchor均已保留并独立核验。",
        "",
        "`admitted`：实际状态相对governor本步允许执行的目标；`raw/effective model-clock`：相对当前已消费模型时间的原始/修改参考；`raw/effective nominal-clock`：相对该query后按物理20Hz推进、耗尽后夹到40端点末端的原始/修改参考。重定时会暂停/桥接/丢弃剩余suffix，因此admitted误差下降不能证明同一条名义WAM轨迹获得更好执行，也不能单靠低RMSE证明任务成功。不同variant闭环观测变化后会生成不同后续参考，并非全episode固定同一轨迹。",
        "",
        "下表对每个3-case组等长端点合并计算RMSE，位置单位mm，朝向rad；R=raw、E=effective。",
        "",
        "| checkpoint / 方案 | admitted TCP / SO3 | R模型时钟 TCP / SO3 | E模型时钟 TCP / SO3 | R名义时钟 TCP / SO3 | E名义时钟 TCP / SO3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    checkpoint_lines = ["", "本批服务实际加载的权重路径（均non-EMA，guidance1.0，seed42）：", ""]
    for checkpoint in ("8800_masked", "15400_masked"):
        item = next(c for c in cases if c["checkpoint"] == checkpoint)
        info = read(Path(item["case"]) / "manifest.json")["server_info"]
        checkpoint_lines.append(f"- {checkpoint}：`{info['checkpoint_path']}`；saved config：`{info['config_file']}`。")
    checkpoint_lines += [""]
    lines[6:6] = checkpoint_lines
    for group in groups:
        metrics = group["tracking_pooled_equal_length_cases"]
        values = [
            f"{number(metrics[k]['position_rmse_m'], 1000)} / {number(metrics[k]['so3_rmse_rad'])}"
            for k in (
                "admitted",
                "raw_at_model_clock",
                "effective_at_model_clock",
                "raw_at_nominal_clock",
                "effective_at_nominal_clock",
            )
        ]
        lines.append(f"| {group['checkpoint']} / {group['variant']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 模型时间消费与恢复桥",
        "",
        "每个case以30物理秒为分母；discard仅统计重query时尚未消费的40端点执行前缀尾部，剩余预算末尾未消费量另列。桥恢复是独立复算的两类误差不退化gate，不是对象接触或任务成功gate。",
        "",
        "| checkpoint / 方案 / 场景 | 消费模型秒 / 30s | generated fraction | discard秒 | 预算末剩余秒 | bridge / advance / pause端点 | 目标双指contact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        diag = case["governor_diagnostics"]
        if not diag:
            continue
        lines.append(
            f"| {case['checkpoint']} / {case['variant']} / {case['record']} | {number(diag['consumed_model_s'])} | {number(diag['generated_reference_fraction'])} | {number(diag['discarded_on_requery_s'])} | {number(diag['remaining_at_budget_s'])} | {diag['bridging_endpoint_count']} / {diag['bridge_recovery_advanced_count']} / {diag['paused_clock_endpoints']} | {case['target_grasp_contact_steps']} |"
        )
    lines += [
        "",
        "## 正控、复核与边界",
        "",
        "- chunked oracle资格为intent_hold 3/3、recovering+intent_hold 3/3稳定成功；原native成功参考也经审计。正控选择是按当前物理step索引的特权真值，不代表WAM在线能力，也不证明所有大角度/阻塞响应范围已合格。",
        "- 全30个case均600步、20Hz，WAM24与oracle6分开计数。冻结auditor重算因果响应、夹爪命令、motor仿射目标、gripper substep、任务谓词、raw/effective积分、10-slot mask、query身份/seed/固定anchor、查询前后physics SHA，以及独立v2 governor桥gate/时钟/速度界。",
        "- 仅controller goal旋转重建使用统一1e-6数值容差，原因是vendor quat2mat的float32路径及NumPy标量转换；body quaternion重排本身未作量化修正。其余状态/误差复核保持原1e-10、实际dt1e-9，任务stable阈值没有放宽。",
        "- 报告生成前及生成后重新核验完整audit snapshot列出的所有输入SHA；图只读取这些既有日志，0新推理、0新物理。物理仿真为CPU/OSMesa，模型仅登记物理GPU5/7，属于用户允许的4–7；无真机。",
        "- 3个旧开发初态和一个simulation seed不足以证明新初态、多任务或训练未见泛化。未把Cosmos只读来源合同与实际历史训练消费/图像-pose物理同步混为一谈；本批也不认证checkpoint权重损坏或唯一归因WAM/S2A。",
        "",
        "## 图与可恢复证据",
        "",
        "每个paired图包含三种governor alpha方案与无governor基线的实际TCP XY、admitted/raw-model-clock误差、实测/参考/motor开度、累计模型/重定时时钟与native成功谓词。不同线型表示不同物理量；motor目标并非测得开度或力。",
        "",
    ]
    for plot in plots:
        relative = os.path.relpath(plot, ROOT / "reports")
        lines.append(f"![{plot.stem}]({relative})")
    lines += [
        "",
        f"- 完整独立snapshot：`{audit_path.relative_to(ROOT)}`，SHA256 `{digest(audit_path)}`。",
        f"- 六页PDF：`{(output / 'paired_recovery.pdf').relative_to(ROOT)}`。",
        f"- 逐case报告指标、输出指纹与全部输入复核：`{(output / 'report_manifest.json').relative_to(ROOT)}`。",
        "",
        "下一阶段先解决中途完成后再失效的稳定性，并以不使用对象标签/接触/成功谓词的在线规则单独资格验收；保留统一预算和原分母。源数据排除链尚未闭合的fresh reservation不能提前宣称已独立验收。本批结果支持继续诊断参考旋转、开度与执行时间的组合，尚不能宣称已达到用户的稳定成功目标。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="reports/wam_recovery_20261021.md")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("CPU-only report requires empty CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH")
    assert Path.cwd().resolve() == ROOT
    output, report, audit_path = (Path(p).absolute() for p in (args.output, args.report, args.audit))
    assert output.is_relative_to(ROOT / "outputs") and report.is_relative_to(ROOT / "reports")
    assert not output.exists() and not report.exists(), "refuse to overwrite artifacts or report"
    audit = read(audit_path)
    assert audit["whole_registered_protocol_complete"] and audit["oracle_full_independent_gate_passed"]
    assert audit["independent_governor_replay_complete"]
    check_inputs(audit["source_before_after_hashes"])
    cases = [case for case in audit["cases"] if case["source"] == "wam"]
    oracles = [case for case in audit["cases"] if case["source"] == "oracle"]
    assert len(cases) == 24 and len(oracles) == 6
    assert all(c["status"] == "verified" and c["steps"] == 600 for c in cases + oracles)
    assert all(c["outcome"]["stable"] for c in oracles)
    assert {c["checkpoint"] for c in cases} == {"8800_masked", "15400_masked"}
    assert {c["variant"] for c in cases} == set(VARIANTS)
    assert {c["record"] for c in cases} == {"task0_init0", "task1_init0", "task2_init0"}
    groups = [g for g in audit["groups"] if g["source"] == "wam"]
    groups.sort(key=lambda g: (g["checkpoint"], VARIANTS.index(g["variant"])))
    assert len(groups) == 8 and all(g["complete"] and g["verified_cases"] == 3 for g in groups)
    data = {case["case"]: load_rows(case) for case in cases}
    output.mkdir(parents=True)
    plots = []
    with PdfPages(output / "paired_recovery.pdf") as pdf:
        for checkpoint in ("8800_masked", "15400_masked"):
            for record in ("task0_init0", "task1_init0", "task2_init0"):
                plots.append(draw_pair(checkpoint, record, cases, data, output, pdf))
    text = make_report(audit_path, audit, cases, groups, data, plots, output)
    check_inputs(audit["source_before_after_hashes"])
    write_new(report, text)
    produced = [report, output / "paired_recovery.pdf", *plots]
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        audit=str(audit_path),
        audit_sha256=digest(audit_path),
        report_entry_sha256=digest(__file__),
        verified_input_sha256=audit["source_before_after_hashes"],
        complete_wam_cases=24,
        oracle_qualification_cases=6,
        endpoints_wam=14400,
        endpoints_oracle=3600,
        outcome_from_raw_steps={c["case"]: c["outcome"] for c in cases},
        groups=groups,
        before_after_all_inputs_unchanged=True,
        output_sha256={str(p): digest(p) for p in produced},
        new_model_calls=0,
        new_physics_steps=0,
    )
    write_new(output / "report_manifest.json", json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(report=str(report), output=str(output), manifest_sha256=digest(output / "report_manifest.json"))
        )
    )


if __name__ == "__main__":
    main()
