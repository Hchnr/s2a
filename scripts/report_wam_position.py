"""Render the completed005 position-hold pilot, forked from the recovery report."""

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
    "position_hold",
    "recovering_position_hold",
    "recovering_position_hold_alpha05",
    "recovering_position_hold_alpha0",
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
            if variant != "position_hold":
                axis.plot(
                    time,
                    np.linalg.norm([r["raw_model_tcp_error"][section] for r in rows], axis=1) * scale,
                    color=color,
                    ls="--",
                    lw=0.7,
                    alpha=0.65,
                )
        if variant != "position_hold":
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
    fig.suptitle(f"{checkpoint} / {record} / seed42: position-hold development comparison", fontsize=13)
    path = output / f"{checkpoint}_{record}.png"
    fig.savefig(path, dpi=140)
    pdf.savefig(fig)
    plt.close(fig)
    return path


def number(value, scale=1):
    return f"{value * scale:.3f}"


def runs(indices):
    result = []
    for value in indices:
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in result)


def draw_first_query(diagnostic, output, pdf):
    """Use the no-governor pairs, whose entire first40 reference clock is equal."""
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for column, checkpoint in enumerate(("8800_masked", "15400_masked")):
        for row, record in enumerate(("task0_init0", "task1_init0", "task2_init0")):
            case = next(
                c
                for c in diagnostic["cases"]
                if c["checkpoint"] == checkpoint and c["record"] == record and c["variant"] == "position_hold"
            )
            old = [json.loads(line) for line in (Path(case["old_case"]) / "steps.jsonl").read_text().splitlines()]
            new = [json.loads(line) for line in (Path(case["new_case"]) / "steps.jsonl").read_text().splitlines()]
            count = case["paired_first_query_physical_steps"]
            assert count == 40
            np.testing.assert_array_equal(
                [r["opening_target"] for r in old[:count]], [r["opening_target"] for r in new[:count]]
            )
            time = (np.arange(count) + 1) * 0.05
            axis = axes[row, column]
            axis.plot(
                time,
                [r["opening_target"] for r in new[:count]],
                color="black",
                ls="--",
                label="same generated reference",
            )
            for data, color, label in ((old, "#777777", "003 v2"), (new, "#0072B2", "005 v3")):
                axis.plot(time, [r["opening_after"] for r in data[:count]], color=color, label=label + " observed")
                axis.plot(
                    time,
                    [r["motor_opening_after"] for r in data[:count]],
                    color=color,
                    ls=":",
                    label=label + " motor target",
                )
            paired = case["paired_first_query"]
            counts = [paired[version]["high_reference_low_observed_after"] for version in ("old_v2", "new_v3")]
            changes = paired["first_exact_numeric_differences"]
            first = None if changes["gripper_command"] is None else changes["gripper_command"]["step"]
            axis.set(
                title=f"{checkpoint} / {record}: high-ref low-open {counts[0]}→{counts[1]}; first cmd Δ step {first}",
                xlabel="physical time (s)",
                ylabel="normalized aperture",
                ylim=(-0.05, 1.05),
            )
            axis.grid(alpha=0.2)
            if row == column == 0:
                axis.legend(fontsize=7)
    figure.suptitle(
        "First query only, no governor: same pixels and full raw H40; fixed physical/reference clock", fontsize=13
    )
    path = output / "first_query_gripper_pairs.png"
    figure.savefig(path, dpi=140)
    pdf.savefig(figure)
    plt.close(figure)
    return path


def make_report(
    audit_path, audit, old_audit, diagnostic_path, diagnostic, cases, groups, data, plots, opening_plot, output
):
    totals = {key: sum(c["outcome"][key] for c in cases) for key in ("ever", "terminal", "stable")}
    lines = [
        "# 连续开度映射下的 WAM 开发评测 — 2026-10-21",
        "",
        f"005固定24条WAM闭环均完成600步/20Hz并独立核验，原生任务 ever **{totals['ever']}/24**、terminal **{totals['terminal']}/24**、末尾连续40端点 stable **{totals['stable']}/24**。004的9条oracle资格全部通过，单独计数，不能计入WAM分母。",
        "",
        "本报告固定两个checkpoint（8800/15400）、共同guidance1.0、旧task0/1/2 init0、sampling seed42、四方案，共14,400个WAM物理端点；oracle为9条/5,400端点。H40、执行前缀40、最长80物理步重观测、35采样步/shift5、response inverse .5和arm command rate5/s与003相同，推理等待不推进物理时间。006/其他checkpoint、新初态验收及真机均不属于本批分母。",
        "",
        "权重完整路径与saved config沿用[003报告](wam_recovery_20261021.md)中列出的8800/15400 non-EMA模型；本批每query另核checkpoint身份、seed、10-slot mask及物理单位。",
        "",
        "## 完整矩阵与003对照",
        "",
        "`ever`为任一端点原生成功，`terminal`为最后端点成功，`stable`为最后40端点全部成功。各组3条全数计分，未按成功病例拼接。",
        "",
        "| checkpoint / 方案 | 005 ever / terminal / stable | 003对应方案 ever / terminal / stable |",
        "|---|---:|---:|",
    ]
    for group in groups:
        old_name = group["variant"].replace("position_hold", "intent_hold")
        current = [c for c in cases if c["checkpoint"] == group["checkpoint"] and c["variant"] == group["variant"]]
        previous = [
            c
            for c in old_audit["cases"]
            if c["source"] == "wam" and c["checkpoint"] == group["checkpoint"] and c["variant"] == old_name
        ]
        assert len(current) == len(previous) == 3

        def summary(values):
            return " / ".join(f"{sum(c['outcome'][key] for c in values)}/3" for key in ("ever", "terminal", "stable"))

        lines.append(f"| {group['checkpoint']} / {group['variant']} | {summary(current)} | {summary(previous)} |")
    lines += ["", "原生成功端点的完整范围（零基step，仍继续执行预定600步）：", ""]
    successes = [case for case in cases if case["outcome"]["ever"]]
    if not successes:
        lines.append("- 本批24条没有原生成功端点。")
    for case in successes:
        flags = [row["step"] for row in data[case["case"]] if row["success"]]
        lines.append(
            f"- {case['checkpoint']} / {case['variant']} / {case['record']}：{runs(flags)}，共{len(flags)}端点；末尾连续{case['outcome']['trailing_success_steps']}端点。"
        )
    lines += [
        "",
        "## 同一首query内，执行映射改了什么",
        "",
        "24条的第一张输入PNG、完整40×128 raw/normalized预测、模型条件及初态physics/anchor与003对应方案逐值相同。首query后的夹爪命令会改变机器人/对象状态，后续观测和WAM预测随之分叉，因此全600步对照不代表始终执行同一参考。",
        "",
        "旧v2把超过.02的下降趋势直接锁存为close-to-limit。新v3在未锁存时按absolute reference与已知motor目标之差作三值位置跟踪（deadband .07）；仅reference≤.60且关闭方向成立时进入close-to-limit。释放仍需当前/历史开度余量、参考回升及完整.05s证据，之后恢复部分开度位置跟踪。参数是当前Panda/20Hz接口的固定开发候选，不是握力估计或通用抓取保证。",
        "",
        "下表每行限制在两次执行都尚未发出第二次query的共同物理区间；阈值为描述性reference>.8、observed_after<.2，不用于策略或验收。`高ref锁存`为在该区间新发生的关闭锁存；高reference下的单次位置减量命令不等于full-close锁存。",
        "",
        "| checkpoint / 方案 / 场景 | 共同步数 | 高ref低actual 003→005 | 高ref新锁存 003→005 | 首gripper / arm命令分叉step |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in diagnostic["cases"]:
        pair = item["paired_first_query"]
        a, b = pair["old_v2"], pair["new_v3"]
        changes = pair["first_exact_numeric_differences"]

        def first(key):
            return "—" if changes[key] is None else str(changes[key]["step"])

        lines.append(
            f"| {item['checkpoint']} / {item['variant']} / {item['record']} | {item['paired_first_query_physical_steps']} | {a['high_reference_low_observed_after']}→{b['high_reference_low_observed_after']} | {a['high_reference_close_latch_onsets']}→{b['high_reference_close_latch_onsets']} | {first('gripper_command')} / {first('arm_command')} |"
        )
    lines += [
        "",
        "分叉按精确数值非相等记录，不设置拟合后的显著性阈值；极小数值残差本身不代表实际运动差异。该极端错配计数为零也不代表所有开度都正确；高ref/低actual可能发生在先前低ref合法锁存后的释放等待。motor target、实际开度和握力是不同量。更长600步的同类计数、首次reason/command/motor明细保存在配对诊断JSON，只作后续输入不同条件下的描述，不能唯一归因模型或控制。",
        "",
        f"![首query开度对照]({os.path.relpath(opening_plot, ROOT / 'reports')})",
        "",
        "## 生成参考与实际允许目标的跟踪",
        "",
        "alpha1保留raw WAM增量；alpha.5逐增量在SO(3)上缩放旋转，alpha0令每个旋转增量为单位，因此只保持本次query实测anchor朝向，后续query重新取anchor。local平移slot数值不变也会因修改后的朝向积分改变world路径；alpha.5/0都不是纯执行补偿，alpha0不能认证原checkpoint完整输出。",
        "",
        "admitted表示实际状态相对本步允许目标；R/E模型时钟表示raw/effective参考按已消费模型时间采样；R/E名义时钟按每query后的物理20Hz推进，耗尽后夹到执行前缀末端。低admitted误差可能来自暂停/桥接/丢弃后缀，不能单独证明同一名义轨迹跟踪改善或任务成功。位置单位mm，SO3单位rad。",
        "",
        "| checkpoint / 方案 | admitted TCP / SO3 | R模型时钟 TCP / SO3 | E模型时钟 TCP / SO3 | R名义时钟 TCP / SO3 | E名义时钟 TCP / SO3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        metrics = group["tracking_pooled_equal_length_cases"]
        cells = [
            f"{number(metrics[key]['position_rmse_m'], 1000)} / {number(metrics[key]['so3_rmse_rad'])}"
            for key in (
                "admitted",
                "raw_at_model_clock",
                "effective_at_model_clock",
                "raw_at_nominal_clock",
                "effective_at_nominal_clock",
            )
        ]
        lines.append(f"| {group['checkpoint']} / {group['variant']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "每组3条合计90物理秒。下表模型时间与discard/bridge/pause均按实际日志计，不把暂停减少误差当能力；逐case消费fraction、未执行后缀及原始/有效目标保留在完整审计JSON。advance标志可包括完成桥的端点，故不必小于post-step bridging计数。",
        "",
        "| checkpoint / 方案 | 消费模型秒 / 90s | consumed / generated | discard模型秒 | bridge / advance / pause端点 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in groups:
        values = [
            c["governor_diagnostics"]
            for c in cases
            if c["checkpoint"] == group["checkpoint"] and c["variant"] == group["variant"]
        ]

        def total(key):
            return sum(v[key] for v in values)

        lines.append(
            f"| {group['checkpoint']} / {group['variant']} | {number(total('consumed_model_s'))} | {number(total('consumed_model_s') / total('generated_model_s'))} | {number(total('discarded_on_requery_s'))} | {total('bridging_endpoint_count')} / {total('bridge_recovery_advanced_count')} / {total('paused_clock_endpoints')} |"
        )
    lines += [
        "",
        "## 核验与范围",
        "",
        "- 004旧v2无gov、新v3无gov、新v3加recovering各3/3 oracle稳定，TCP≤10mm/SO3≤.05rad；旧v2物理序列逐值复现。reference按当前物理step索引成功真值，仅是特权执行资格，9条从24条WAM分母中排除。独立机械阻塞的observed_blockage/recovery_from_blockage门禁仍未通过，不提升为广义恢复资格。",
        "- 新auditor显式fork旧checker，复算响应逆、controller goal、三值电机积分/零保持、qpos→开度/跨步连续性，以及独立GripperReplay关闭/释放锁存。RecoveryReplay独立复算桥gate、时钟和速度界；各query核checkpoint/seed/mask/域/反归一化、raw/effective128和physics前后SHA。",
        "- controller goal旋转重建仍为预先声明的1e-6 vendor数值容差；其余状态/误差1e-10、dt1e-9保持，未因结果放宽task阈值。查询前后full-state哈希由冻结logger记录，未保存每次full-state数组以重新独立哈希；源码/包版本也不等于完整OS/共享库二进制闭包。",
        "- 物理仿真CPU/OSMesa，模型登记仅物理GPU5/7；外部Cosmos与R1/R2只读，无真机。3旧开发初态、1 sampling seed与当前Panda范围不能证明训练未见或新初态泛化，不能唯一判定权重损坏或把余失败全部归于WAM/仅S2A。",
        "- 首query回归与开度诊断回答局部接口行为；原生稳定任务谓词回答执行结果。contact/目标高度仅作观测指标，未作为adapter/governor输入；contact0也不等于任务必失败。",
        "",
        "## 图与证据",
        "",
        "每个case图分别记录三种governor alpha与无gov对照的实际路径、开度/motor、成功谓词和消费时钟，不将后续不同观测的轨迹画成同一参考回放。",
        "",
    ]
    for plot in plots:
        lines.append(f"![{plot.stem}]({os.path.relpath(plot, ROOT / 'reports')})")
    lines += [
        "",
        f"- 完整33审计：`{audit_path.relative_to(ROOT)}`，SHA `{digest(audit_path)}`。",
        f"- 24对局部开度诊断：`{diagnostic_path.relative_to(ROOT)}`，SHA `{digest(diagnostic_path)}`。",
        f"- 图、七页PDF与输出/输入指纹：`{(output / 'report_manifest.json').relative_to(ROOT)}`。",
        "",
        "稳定成功只按完整冻结矩阵报告；若有成功，仍限本开发协议，后续须固定候选另做新初态验收。若仍失败，保留已识别的映射改善与参考/消费/任务指标，继续分开验证模型参考和执行范围。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--gripper-diagnostic", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="reports/wam_position_hold_20261021.md")
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != ROOT
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("report requires repository cwd and empty CPU CUDA/LD")
    output, report, audit_path, diagnostic_path = (
        Path(p).absolute() for p in (args.output, args.report, args.audit, args.gripper_diagnostic)
    )
    assert output.is_relative_to(ROOT / "outputs") and report.is_relative_to(ROOT / "reports")
    assert not output.exists() and not report.exists(), "refuse overwrite"
    source_hash = digest(__file__)
    audit, diagnostic = read(audit_path), read(diagnostic_path)
    assert audit["schema_version"] == "s2a.position-readonly-audit.v1"
    assert (
        audit["whole_registered_protocol_complete"]
        and audit["oracle_full_independent_gate_passed"]
        and audit["independent_governor_replay_complete"]
    )
    assert diagnostic["complete_registered_wam_matrix"] and diagnostic["verified_wam_pairs"] == 24
    check_inputs(audit["source_before_after_hashes"])
    check_inputs(diagnostic["input_sha256"])
    assert diagnostic["input_sha256"][str(audit_path)] == digest(audit_path)
    old_path = ROOT / "outputs/wam_stability_20261021/recovery_full_complete_20261021.json"
    assert digest(old_path) == "b5ad882e7e696ec7d57a05987585c7ca5eea2ffc4c957c48ac00459c99a30eef"
    old_audit = read(old_path)
    cases = [c for c in audit["cases"] if c["source"] == "wam"]
    oracles = [c for c in audit["cases"] if c["source"] == "oracle"]
    assert len(cases) == 24 and len(oracles) == 9
    assert all(c["status"] == "verified" and c["steps"] == 600 for c in cases + oracles)
    assert all(c["outcome"]["stable"] for c in oracles)
    assert {c["case"] for c in cases} == {c["new_case"] for c in diagnostic["cases"]}
    groups = [g for g in audit["groups"] if g["source"] == "wam"]
    groups.sort(key=lambda g: (g["checkpoint"], VARIANTS.index(g["variant"])))
    assert len(groups) == 8 and all(g["complete"] and g["verified_cases"] == 3 for g in groups)
    data = {c["case"]: load_rows(c) for c in cases}
    output.mkdir(parents=True)
    plots = []
    with PdfPages(output / "paired_position.pdf") as pdf:
        for checkpoint in ("8800_masked", "15400_masked"):
            for record in ("task0_init0", "task1_init0", "task2_init0"):
                plots.append(draw_pair(checkpoint, record, cases, data, output, pdf))
        opening_plot = draw_first_query(diagnostic, output, pdf)
    text = make_report(
        audit_path, audit, old_audit, diagnostic_path, diagnostic, cases, groups, data, plots, opening_plot, output
    )
    check_inputs(audit["source_before_after_hashes"])
    check_inputs(diagnostic["input_sha256"])
    assert source_hash == digest(__file__)
    write_new(report, text)
    produced = [report, output / "paired_position.pdf", *plots, opening_plot]
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        audit=str(audit_path),
        audit_sha256=digest(audit_path),
        gripper_diagnostic=str(diagnostic_path),
        gripper_diagnostic_sha256=digest(diagnostic_path),
        entry_sha256=source_hash,
        complete_wam_cases=24,
        oracle_qualification_cases=9,
        endpoints_wam=14400,
        endpoints_oracle=5400,
        original003_snapshot_sha256=digest(old_path),
        verified_input_sha256={**audit["source_before_after_hashes"], **diagnostic["input_sha256"]},
        all_inputs_sha_unchanged_before_after=True,
        output_sha256={str(p): digest(p) for p in produced},
        groups=groups,
        new_model_calls=0,
        new_physics_steps=0,
    )
    write_new(output / "report_manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(report=str(report), manifest_sha256=digest(output / "report_manifest.json"))))


if __name__ == "__main__":
    main()
