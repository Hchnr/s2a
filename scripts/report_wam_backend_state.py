"""Export013 figures and report from a complete, released independent audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from audit_wam_backend_state import PREFIXES, ROOT, STUDY, Evidence, eligibility, process_record
from matplotlib.backends.backend_pdf import PdfPages

KEYS = ("position_rmse_mm", "rotation_rmse_rad", "opening_mae")
LABELS = ("Source EEF position RMSE (mm)", "SO(3) RMSE (rad)", "Opening MAE (normalized)")
GROUPS = ("sdpa35", "flash3_video")
COLORS = ("#4477AA", "#EE6677")


def triple(value):
    return f"{value[KEYS[0]]:.6f} / {value[KEYS[1]]:.9f} / {value[KEYS[2]]:.9f}"


def figures(audit, output):
    paths = []
    pdf_path = output / "attention_backend.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for row, h in enumerate(PREFIXES):
            for col, key in enumerate(KEYS):
                ax = axes[row, col]
                ax.bar(GROUPS, [audit["metrics"][g][str(h)]["wam"][key] for g in GROUPS], color=COLORS)
                for method, style in (("hold", "--"), ("constant_velocity", ":")):
                    ax.axhline(audit["metrics"]["sdpa35"][str(h)][method][key], color="0.35", ls=style, label=method)
                ax.set(title=f"H{h}: {LABELS[col]}")
                ax.grid(axis="y", alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("013 | Same8400 / actual initial state / input /35NFE;48 paired requests")
        path = output / "prefix_state_metrics.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)

        for label, field in (("episode", "by_episode"), ("phase", "by_phase")):
            names = sorted(audit[field])
            x = np.arange(len(names))
            fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
            for ax, key, title in zip(axes, KEYS, LABELS, strict=True):
                for index, group in enumerate(GROUPS):
                    values = [audit[field][n][group]["16"]["wam"][key] for n in names]
                    ax.bar(x + (index - 0.5) * 0.32, values, width=0.30, color=COLORS[index], label=group)
                hold = [audit[field][n]["sdpa35"]["16"]["hold"][key] for n in names]
                cv = [audit[field][n]["sdpa35"]["16"]["constant_velocity"][key] for n in names]
                ax.plot(x, hold, "kx", label="hold")
                ax.plot(x, cv, "+", color="0.5", label="constant velocity")
                ax.set(title=f"H16: {title}", xticks=x, xticklabels=names)
                ax.grid(axis="y", alpha=0.2)
            axes[0].legend(ncol=4)
            fig.suptitle(f"Historical development {label}s | valid endpoints pooled within each group")
            path = output / f"per_{label}_h16.png"
            fig.savefig(path, dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            paths.append(path)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, key, title in zip(axes, KEYS, LABELS, strict=True):
            ax.plot(PREFIXES, [audit["generated_state_distance"][str(h)][key] for h in PREFIXES], "o-")
            ax.set(title=title.replace("RMSE", "distance RMSE"), xlabel="Valid prediction prefix", xticks=PREFIXES)
            ax.grid(alpha=0.2)
        fig.suptitle("Generated SDPA vs hybrid Flash3 state distance; distance alone does not establish correctness")
        path = output / "generated_state_distance.png"
        fig.savefig(path, dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        paths.append(path)
    return paths + [pdf_path]


def report(audit, audit_path, output):
    gate = audit["promotion"]
    lead = "达到" if gate["passed"] else "未达到"
    backend = audit["actual_backend"]
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    prepared = json.loads((STUDY / "orchestration/prepared.json").read_text())
    lines = [
        "# 8400 attention backend state对照 — 2026-10-21",
        "",
        f"**保存的hybrid `flash3_video` backend{lead}预注册state候选门槛。** 完成48个Flash3请求，并逐项配对011的48个SDPA35请求；本协议新推理48次、0物理步。结果不表示闭环任务成功率，也不授权自动增加物理或fresh预算。",
        "",
        f"固定checkpoint：`{prepared['checkpoint']}`。非EMA、guidance1、UniPC35、shift5、H40、20Hz、active十槽mask、task-only、seed42/43全部不变；仅attention backend由SDPA切换为保存配置的flash3_video。",
        "",
        "该backend实际为VIDEO使用FlashAttention3、UND/ACTION使用Flex的hybrid路由。保留推理padding关闭、small_action_fixed_lengths=None、short_tokens=0；model/decoder compile=false，Flex内部torch.compile保留。不能将本实验描述为全部attention都使用FA3或所有kernel都不编译。",
        f"实测net与language backend均为`{backend['network_backend']}`，provider `{backend['provider']}`版本`{backend['provider_version']}`、FLASH3_SUPPORTED=true，实际CUDA UUID `{backend['cuda_device_uuid']}`与物理GPU7登记一致。CPU协调器/worker空CUDA/LD；本协议仅GPU7/9587服务。",
        "",
        "24个phase样本来自6条历史开发episode，22个独特anchor、44个anchor-seed组合；phase重复使请求数为48，不等于48条独立episode，也不证明预训练未见。直接从source EEF anchor按发布local delta合同积分，未经过S2A或TCP映射；opening按原协议clip到[0,1]。未来GT padding不计分，H8/H16/H40各384/768/1778有效端点。",
        "",
        "每格为位置RMSE(mm) / SO(3) RMSE(rad) / opening MAE。",
        "",
        "| 前缀 | SDPA35 | hybrid Flash3 | hold | constant velocity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for h in PREFIXES:
        values = [audit["metrics"][g][str(h)]["wam"] for g in GROUPS]
        values += [audit["metrics"]["sdpa35"][str(h)][m] for m in ("hold", "constant_velocity")]
        lines.append(f"| H{h} | " + " | ".join(triple(v) for v in values) + " |")
    lines += [
        "",
        "## 预注册门槛",
        "",
        "H8/H16位置、SO(3)、opening六项全部不退化（位置展示mm先转SI m，再用1e-10 m容差；其余1e-10 rad/opening），且至少一个位置RMSE改善10%。严格沿用启动前013注册，不因本次输出调整阈值。",
        f"六项不退化：`{gate['all_six_nonregression']}`；至少一项位置改善10%：`{gate['at_least_one_position_improves_ten_percent']}`；合格：`{gate['passed']}`。合格也只表示可以另行预注册开发候选，不是完整checkpoint任务能力认证。",
        "",
        "| 指标 | 前缀 | hybrid Flash3(SI) | SDPA35(SI) | 不退化 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in gate["checks"]:
        lines.append(
            f"| {check['metric']} | H{check['horizon']} | {check['candidate']:.12g} | {check['reference']:.12g} | {check['nonregression']} |"
        )
    for label, field in (("Episode", "by_episode"), ("Phase", "by_phase")):
        lines += [
            "",
            f"## 逐{label}误差",
            "",
            f"| {label} | 前缀 | SDPA35 | hybrid Flash3 |",
            "| --- | --- | --- | --- |",
        ]
        for name, values in sorted(audit[field].items()):
            for h in PREFIXES:
                lines.append(
                    f"| {name} | H{h} | " + " | ".join(triple(values[g][str(h)]["wam"]) for g in GROUPS) + " |"
                )
    lines += [
        "",
        "## 生成state距离与核验范围",
        "",
        "| 前缀 | SDPA→hybrid Flash3生成state距离(mm/rad/opening) |",
        "| --- | --- |",
    ]
    lines += [f"| H{h} | {triple(audit['generated_state_distance'][str(h)])} |" for h in PREFIXES]
    lines += [
        "",
        "生成距离仅描述两种计算backend输出不同，不单独证明正确性或错误根因。GT future也不是唯一可行的任务轨迹。本轮保留推理padding关闭，因此没有核验训练padding本身的影响。",
        "",
        "48对请求、actual sampler初态/clean/reference/mask/token/实际caption、分支、35次velocity求值及实际sigma/timestep均核验。每份raw/normalized full40×128响应与服务本地原始JSON、per-request receipt和trace绑定。独立Gram–Schmidt解码、local旋转/位移积分与hold/CV重算逐端点匹配保存指标。网络35NFE由冻结guidance1/单rank/无postprocess调用分支推导，未宣称直接hook全部网络模块。",
        "实际初态/clean/mask/token保留的是已观察值的digest witness，而非完整tensor原始字节；审计可核跨backend digest一致，无法从未保存tensor独立重新算该SHA。实际sigma/timestep保存了数值，已独立重建并复核字节SHA；参数/RNG无修改证据来自冻结observer的逐观察guard与CPU透明性测试。",
        "初版只读auditor将scheduler_config._use_default_values视为有序list，产生27条审计错误；原development_partial01完整保留。Diffusers configuration_utils.py明确用list(set(...))构造该默认字段名列表。最终审计仅对这一bookkeeping列表排序，仍要求完整键集合、默认名集合、全部有效scheduler参数、sigma/time数值exact；此为审计表示修正，不是模型/物理失败，不涉及运行源码或采样修改。",
        f"完成时独立重新核验`{len(audit['input_sha256'])}`份保存输入/source/snapshot SHA与163份已完整哈希权重的size/mtime，未再次全量读取32GB权重。登记协调器/模型/state worker均按PID/startticks正常退出；GPU/CVD/backend环境证据来自运行时保存快照，而非根据已退出进程推断。",
        "",
        "## 产物",
        "",
        f"- 独立完整审计：`{audit_path}`，SHA256 `{audit_digest}`。",
        f"- 四张PNG和4页PDF：`{output}`；输入与产物SHA见其`report_manifest.json`。",
        "- 方案：`plans/v0.0.1/013_attention_backend_state_20261021.md`；启动前012完整/释放及011全部48 traced35 exact门禁均已绑定，012候选是否合格不影响013按计划独立执行。",
        "- 未启动fresh评测、真机或新增物理；011更多采样步、010短重观测与本报告分母分开。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/wam_backend_state_20261021.md")
    args = parser.parse_args()
    if (
        Path.cwd().resolve() != ROOT
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("LD_LIBRARY_PATH") != ""
    ):
        raise ValueError("report requires repository cwd and empty CUDA/LD")
    if (
        args.report.exists()
        or args.output_dir.exists()
        or not args.report.resolve().is_relative_to(ROOT / "reports")
        or not args.output_dir.resolve().is_relative_to(ROOT / "outputs")
    ):
        raise ValueError("report/export must be new repository artifacts")
    evidence = Evidence()
    audit = evidence.json(args.audit)
    if (
        not audit["complete"]
        or audit["status"] != "complete"
        or audit["errors"]
        or audit["verified_queries"] != 48
        or not audit["owned_service_released_normally"]
    ):
        raise ValueError("final report requires complete48 independent audit and normal resource release")
    for path, digest in audit["input_sha256"].items():
        evidence.check(path, digest)
    if audit["promotion"] != eligibility(audit["metrics"]["flash3_video"], audit["metrics"]["sdpa35"]):
        raise ValueError("registered admission recomputation differs")
    for role, state in audit["processes"].items():
        current = process_record(
            state["pid"], "7" if role == "model" else "", start_ticks=state["registered_start_ticks"]
        )
        if current["present"]:
            raise ValueError("owned013 process remains present")
    prepared = evidence.json(STUDY / "orchestration/prepared.json")

    def check_weights():
        for path, frozen in prepared["weight_files"].items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (frozen["bytes"], frozen["mtime_ns"]):
                raise ValueError("weight metadata differs from completed audit")

    check_weights()
    evidence.read(__file__)
    evidence.recheck()
    args.output_dir.mkdir(parents=True)
    paths = figures(audit, args.output_dir)
    with args.report.open("x") as stream:
        stream.write(report(audit, args.audit, args.output_dir))
    paths.append(args.report)
    evidence.recheck()
    check_weights()
    manifest = dict(
        audit_path=str(args.audit.resolve()),
        audit_sha256=evidence.hashes[str(args.audit.resolve())],
        verified_input_sha256=evidence.hashes,
        output_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        newly_registered_requests=48,
        paired_cached_requests=48,
        physics_steps=0,
        png_count=4,
        pdf_pages=4,
        visual_inspection="pending separate view artifact",
        full_weight_bytes_rehashed=False,
    )
    with (args.output_dir / "report_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(report=str(args.report), export=str(args.output_dir), complete=True)))


if __name__ == "__main__":
    main()
