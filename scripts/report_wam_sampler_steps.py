"""Export011 figures and a final report only from a complete independent audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_wam_sampler_steps import METRIC_KEYS, PREFIXES, ROOT, STEPS, STUDY, Evidence, process_snapshot, promotion
from matplotlib.backends.backend_pdf import PdfPages


def triple(values):
    return f"{values['position_rmse_mm']:.3f} / {values['rotation_rmse_rad']:.6f} / {values['opening_mae']:.6f}"


def figures(audit, output):
    labels = ("Source EEF position RMSE (mm)", "SO(3) RMSE (rad)", "Opening MAE (normalized)")
    metrics = {n: audit["groups"][str(n)]["metrics"] for n in STEPS}
    saved = []
    pdf_path = output / "sampler_steps.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for row, prefix in enumerate(PREFIXES):
            for col, key in enumerate(METRIC_KEYS):
                ax = axes[row, col]
                ax.plot(STEPS, [metrics[n][str(prefix)]["wam"][key] for n in STEPS], "o-", label="WAM")
                for method, style in (("hold", "--"), ("constant_velocity", ":")):
                    ax.axhline(metrics[35][str(prefix)][method][key], linestyle=style, color="0.45", label=method)
                ax.set(title=f"H{prefix}: {labels[col]}", xlabel="UniPC velocity evaluations", xticks=STEPS)
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("011 | Same8400 / input / actual initial state;48 requests per step setting")
        path = output / "prefix_state_metrics.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        saved.append(path)

        episodes = sorted(audit["groups"]["35"]["by_episode"])
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
        x = np.arange(len(episodes))
        for index, (ax, key, label) in enumerate(zip(axes, METRIC_KEYS, labels, strict=True)):
            for offset, steps in enumerate(STEPS):
                values = [audit["groups"][str(steps)]["by_episode"][e]["16"]["wam"][key] for e in episodes]
                ax.bar(x + (offset - 1) * 0.25, values, width=0.24, label=str(steps))
            hold = [audit["groups"]["35"]["by_episode"][e]["16"]["hold"][key] for e in episodes]
            ax.plot(x, hold, "kx", label="hold")
            ax.set(ylabel=label, xticks=x, xticklabels=episodes, title=f"H16 | {label}")
            ax.tick_params(axis="x", rotation=15)
            ax.grid(axis="y", alpha=0.2)
            if index == 0:
                ax.legend(ncol=4)
        fig.suptitle("Six historical development episodes; phases/seeds pooled within episode")
        path = output / "per_episode_h16.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        saved.append(path)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, key, label in zip(axes, METRIC_KEYS, labels, strict=True):
            for pair in ("35_vs70", "70_vs140"):
                ax.plot(
                    PREFIXES, [audit["generated_state_distance"][pair][str(h)][key] for h in PREFIXES], "o-", label=pair
                )
            ax.set(title=label, xlabel="Prediction prefix", xticks=PREFIXES)
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle("Generated-state distances between step settings; smaller distance does not establish correctness")
        path = output / "generated_state_distance.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        saved.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for steps in STEPS:
            first_name = next(iter(audit["cases"][str(steps)]))
            trace_path = STUDY / f"service/traces{steps}/{first_name}.json"
            trace = json.loads(trace_path.read_text())
            axes[0].plot(np.arange(steps + 1) / steps, trace["sigma_values"], ".-", label=str(steps))
            axes[1].plot(np.arange(steps) / steps, trace["timesteps"], ".-", label=str(steps))
        axes[0].set(title="Observed actual FP32 scheduler sigmas", xlabel="Evaluation index / N", ylabel="sigma")
        axes[1].set(title="Observed actual int64 network timesteps", xlabel="Evaluation index / N", ylabel="timestep")
        for ax in axes:
            ax.legend(title="NFE")
            ax.grid(alpha=0.2)
        path = output / "actual_schedules.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        saved.append(path)
    return saved + [pdf_path]


def report(audit, audit_path, output):
    gate = audit["promotion"]
    selected = gate["selected_steps"]
    checkpoint = json.loads((STUDY / "orchestration/prepared.json").read_text())["checkpoint"]
    lead = (
        "70/140均未达到预注册的更多步数state候选门槛，本轮不增加对应物理预算。"
        if selected is None
        else f"{selected}步达到预注册的state候选门槛；它只可作为另行注册的物理开发候选，本轮没有新增物理评测。"
    )
    lines = [
        "# 8400采样步数与实际噪声配对诊断 — 2026-10-21",
        "",
        f"**{lead}** 本次完成35/70/140各48个H40请求，共144个state请求、0物理步。先前缓存与新traced35的全部48个full40×128 raw/normalized逐值exact；70/140分别与新35的实际sampler初态和conditioning配对通过。结果描述单个checkpoint在历史开发样本上的state质量，不是闭环任务成功率。",
        "",
        "固定8400非EMA、guidance1、shift5、十槽active mask、20Hz、task-only、seed42/43。24个phase样本来自6条历史episode，包含22个独特anchor、44个独特anchor-seed组合；48请求包含重复phase，不能当48个独立episode或训练未见数据。",
        f"Checkpoint：`{checkpoint}`。",
        "",
        "API字段action按冻结发布合同解释为local Δstate与absolute opening，本轮直接比较source EEF state，未经过S2A/TCP重映射或控制器。opening按既有协议clip到[0,1]后计分。位置展示单位mm，朝向为SO(3)弧度，夹爪为归一化opening MAE。每个设置H8/H16/H40有效未padding端点数分别384/768/1778。表中每格为位置RMSE / SO(3) RMSE / opening MAE。",
        "",
        "| 前缀 | 35步 | 70步 | 140步 | hold | constant velocity |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for h in PREFIXES:
        groups = audit["groups"]
        entries = [triple(groups[str(n)]["metrics"][str(h)]["wam"]) for n in STEPS]
        entries += [triple(groups["35"]["metrics"][str(h)][m]) for m in ("hold", "constant_velocity")]
        lines.append(f"| H{h} | " + " | ".join(entries) + " |")
    lines += [
        "",
        "## 预注册候选门槛",
        "",
        "比较H8/H16位置、SO(3)、opening六项，全部不退化（位置先从展示mm转回生产主要指标m，再用1e-10 m；角度1e-10 rad、opening 1e-10），且至少一个位置RMSE改善10%。若两者都合格，默认70；140只有六项均不差70且至少一个位置再改善10%时优先。数值门槛在任何011请求前以1ad56d4冻结；位置容差采用SI的单位口径在任何70/140完整结果和晋级决策前明确，与012约定一致，不修改冻结plan/运行输入或根据结果调门槛。",
        "",
        "| 候选对比 | 六项不退化 | 至少一个位置改善10% | 合格 |",
        "| --- | --- | --- | --- |",
    ]
    for n in (70, 140):
        c = gate["candidates"][str(n)]
        lines.append(
            f"| {n} vs35 | {c['all_six_nonregression']} | {c['at_least_one_position_improves10pct']} | {c['eligible']} |"
        )
    c = gate["extra140_vs70"]
    lines.append(
        f"| 140 vs70的追加优先条件 | {c['all_six_nonregression']} | {c['at_least_one_position_improves10pct']} | {c['eligible']} |"
    )
    lines += [
        "",
        "## 逐episode与phase",
        "",
        "每格仍为位置mm / SO(3)rad / opening MAE；同一episode内按可用端点池化phase与seed，完整audit保留逐请求误差和积分轨迹。",
        "",
        "| Episode | 前缀 | 35 | 70 | 140 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for e in sorted(audit["groups"]["35"]["by_episode"]):
        for h in PREFIXES:
            lines.append(
                f"| {e} | H{h} | "
                + " | ".join(triple(audit["groups"][str(n)]["by_episode"][e][str(h)]["wam"]) for n in STEPS)
                + " |"
            )
    lines += ["", "| Phase | 前缀 | 35 | 70 | 140 |", "| --- | --- | --- | --- | --- |"]
    for phase in sorted(audit["groups"]["35"]["by_phase"]):
        for h in PREFIXES:
            lines.append(
                f"| {phase} | H{h} | "
                + " | ".join(triple(audit["groups"][str(n)]["by_phase"][phase][str(h)]["wam"]) for n in STEPS)
                + " |"
            )
    lines += [
        "",
        "## 更多步数之间的生成state距离",
        "",
        "这里比较两份生成轨迹，采用与GT误差相同的可用future端点mask；距离本身不以GT位置作为目标。较小距离只能描述输出更接近；即使数值收敛，也可能共同偏离真实future，不能证明checkpoint正确或任务成功。",
        "",
        "| 对比 | 前缀 | 生成state位置 / SO(3) / opening距离 |",
        "| --- | --- | --- |",
    ]
    for pair, values in audit["generated_state_distance"].items():
        for h in PREFIXES:
            lines.append(f"| {pair} | H{h} | {triple(values[str(h)])} |")
    comparisons = []
    for h in PREFIXES:
        before = audit["generated_state_distance"]["35_vs70"][str(h)]
        after = audit["generated_state_distance"]["70_vs140"][str(h)]
        comparisons.extend(after[k] < before[k] for k in METRIC_KEYS)
    reduced = sum(comparisons)
    lines += [
        "",
        f"35→70与70→140比较，九个prefix/指标组合中有{reduced}/9的生成距离缩小。这里只观察了两次步数细化；距离缩小、保持或增大都不等于已证明solver收敛，正确性仍需看独立GT误差及任务评测。",
    ]
    lines += [
        "",
        "## 审计、来源与边界",
        "",
        "独立Gram–Schmidt解码和逐步local delta积分直接读取raw输出；GT、hold及只用anchor前最多4帧的constant velocity独立复算，并逐端点核对原结果。最终重新核对冻结输入/source/snapshot、每个response/trace、三阶段准入、163个权重文件path/size/mtime和登记PID/startticks释放。未重复读取完整权重字节。",
        "sampler保持原实例和原求值/返回对象，只在实例forward与velocity调用边界观察；未加velocity_postprocess_builder、网络forward hook、新solver或随机数抽样。实测velocity NFE为35/70/140；denoising网络forward数只在冻结guidance1/无postprocess/单rank/无padding调用条件下由源码推导，包含VAE等全部模块的总forward次数并未直接hook计数。",
        "actual sigma与integer timestep来自受source SHA/code identity/对象身份保护的实际caller frame。保存的sigma/timepoint数值逐字节重建SHA并用独立NumPy公式复核。初态是conditioning/mask之后的实际sampler输入，不是mask之前pure Gaussian。初态/clean/ref/mask/text保存的是观察到的digest witness，未保存完整tensor值；审计能独立核跨步数digest和字段一致，不能重新从原tensor字节计算其SHA。参数/RNG无修改来自冻结observer的逐观察guard和CPU透明性测试，不伪称另一次张量重放。",
        "预实现的只读提案曾查看权重run旁的另一Cosmos源码树；011实际运行按config.cosmos_repo绑定 `/share/project/eai_pwm/home/hcr/repos/cosmos-framework-pwm`，不是 `/share/project/eai_pwm/repos/cosmos-framework-pwm`。此差异在任何011模型启动前修正，原提案字节保留；它不构成旧实验推理接错的证据。",
        "纯CPU preparation v1在启动前自审发现acceptance_input误指恒定service根目录，会使第二请求拒覆盖。d26e324改为prepared.sample_args中的独立generate_UUID目录，并通过连续两请求与拒覆盖测试。旧准备目录30文件逐字节完整归档为 `sampler_steps_state_prepare_v1_superseded`，没有任何旧模型请求或物理步；新目录按相同冻结config重新prepare，144预算不变。",
        "本轮仅物理GPU6/9566单服务、三个串行state矩阵，所有CPU环境CUDA/LD为空；owned协调器/模型/三个worker均已按登记身份释放，模型正常或受控SIGTERM退出。其他协议010/012与真机均不在本报告分母。本轮未重新证明历史训练消费或上游图像/pose物理同步，示范future也并非唯一正确任务轨迹。",
        "",
        "## 可复核产物",
        "",
        f"- 独立完整audit：`{audit_path}`（SHA见report_manifest）。",
        f"- 图与4页PDF：`{output}`；全部输入和产物SHA保存在 `report_manifest.json`。",
        "- 冻结协议：`plans/v0.0.1/011_sampler_steps_state_20261021.md`；完整checkpoint路径见prepared.json。",
        "- 后续仅按本报告预注册门槛决定是否另行登记更多步数的开发物理实验；本入口不启动任何新推理或物理。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/wam_sampler_steps_state_20261021.md"))
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("LD_LIBRARY_PATH") != "":
        raise ValueError("report requires CPU-only environment")
    if (
        args.report.exists()
        or args.output_dir.exists()
        or not args.output_dir.resolve().is_relative_to(ROOT / "outputs")
        or not args.report.resolve().is_relative_to(ROOT / "reports")
    ):
        raise ValueError("report/export must be new repository artifacts")
    evidence = Evidence()
    audit = evidence.json(args.audit)
    if not audit["complete"] or audit["status"] != "complete" or audit["errors"]:
        raise ValueError("final report requires complete144 independent audit and resource release")
    for path, h in audit["input_sha256"].items():
        evidence.check(path, h)
    prepared = evidence.json(STUDY / "orchestration/prepared.json")

    def check_weight_stats():
        for path, item in prepared["weight_files"].items():
            stat = Path(path).stat()
            if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("weight stat changed since the completed independent audit")

    check_weight_stats()
    if audit["promotion"] != promotion({n: g["metrics"] for n, g in audit["groups"].items()}):
        raise ValueError("registered promotion gate recomputation differs")
    for role, registration in audit["processes"].items():
        state = process_snapshot(
            dict(pid=registration["pid"], start_ticks=registration["registered_startticks"]), cpu=role != "model"
        )
        if state["owned_live"]:
            raise ValueError("owned011 process remains live")
    args.output_dir.mkdir(parents=True)
    outputs = figures(audit, args.output_dir)
    args.report.parent.mkdir(exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(report(audit, args.audit, args.output_dir))
    outputs.append(args.report)
    evidence.read(__file__)
    evidence.recheck()
    check_weight_stats()
    produced = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
    manifest = dict(
        audit_path=str(args.audit),
        audit_sha256=evidence.hashes[str(args.audit.resolve())],
        verified_input_sha256=evidence.hashes,
        output_sha256=produced,
        full_requests=144,
        physics_steps=0,
        pdf_pages=4,
        png_count=4,
        visual_inspection="pending manual view after generation",
    )
    with (args.output_dir / "report_manifest.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
