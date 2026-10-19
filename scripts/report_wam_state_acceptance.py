"""Publish the complete direct-state study independently of ongoing physics."""

import argparse
import json
import shutil
from pathlib import Path

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def metrics(row):
    return [f"{row['position_rmse_m'] * 1000:.3f}", f"{row['rotation_rmse_rad']:.4f}", f"{row['opening_mae']:.4f}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    source = root / "summary_partial.json"
    summary = json.loads(source.read_text())
    if not summary["state_complete"] or not all(
        x["complete"] and x["completed"] == x["expected"] == 384 for x in summary["checkpoint_matrices"]
    ):
        raise ValueError("direct-state report requires all seven frozen matrices")
    destination = Path("reports/wam_state_acceptance_20261019v2.md")
    snapshot = root / "state_report_summary.json"
    if destination.exists() or snapshot.exists():
        raise FileExistsError("complete state evidence must not be silently overwritten")
    shutil.copyfile(source, snapshot)

    def get(checkpoint, split, horizon, prompt="task_only"):
        return next(
            r
            for r in summary["state"]
            if r["checkpoint"] == checkpoint
            and r["split"] == split
            and r["horizon"] == horizon
            and r["prompt_mode"] == prompt
        )

    names = [
        ("warm6200_masked", "6200 warm-start"),
        ("8400_masked", "8400"),
        ("8600_masked", "8600"),
        ("8800_masked", "8800"),
    ]
    latest = get("8800_masked", "study_heldout", 40)
    stock = get("8800", "study_heldout", 40)
    parts = [
        "# WAM checkpoint直接state验收 — 2026-10-19 v2",
        f"七组矩阵全部完成：2688次真实模型预测，覆盖12个episode、48个阶段样本（43个不同观测锚点）、两个生成时域、两种文本条件和两个采样seed。本报告只包含直接state证据，完全绕过S2A，不把离线误差当作任务成功率。物理任务的最终结论另见[本轮结论](../plans/v0.0.1/conclusion_20261019v2.md)。固定汇总快照为`outputs/wam_checkpoint_acceptance_20261019v2/state_report_summary.json`，SHA256=`{sha256(snapshot)}`。",
        "## 可以得出的判断",
        f"无效维度mask修正显著改善原服务：8800在本实验episode留出、task-only、H40窗口的位置RMSE从{stock['model']['position_rmse_m'] * 1000:.3f}mm降至{latest['model']['position_rmse_m'] * 1000:.3f}mm，SO(3)误差从{stock['model']['rotation_rmse_rad']:.4f}rad降至{latest['model']['rotation_rmse_rad']:.4f}rad。",
        f"修正后的8800预测包含有用的状态变化：位置RMSE低于保持当前state的{latest['hold']['position_rmse_m'] * 1000:.3f}mm，开度MAE为{latest['model']['opening_mae']:.4f}，保持基线为{latest['hold']['opening_mae']:.4f}。但朝向RMSE仍高于保持基线的{latest['hold']['rotation_rmse_rad']:.4f}rad，不能说状态预测全面达标。单条示范不是唯一正确轨迹，以上结论不能替代独立任务成功率。",
        "## 数据、坐标和评分",
        "三个任务为bowl→plate、wine bottle→rack、cream cheese→bowl。每任务4个完整episode：2个diagnostic、2个study-heldout；全部12个episode均属于训练catalog，历史逐样本消费未知，study-heldout不是预训练未见数据。分组是episode隔离，不能把同一episode的多个阶段或两个采样seed当作独立episode。",
        "阶段覆盖approach、closing、transport-height、release的机器人state代理，其中16/48使用回退，不能声称这些锚点全部经过物体接触验证。9个样本的未来不足40步；只评价有效前缀，不使用padding作为真实未来。所有平均误差按有效endpoint加权；误差随时间的表中明确给出有效样本数。",
        "从当前实测canonical TCP位姿出发，按局部增量`T[k+1] = T[k] @ delta[k]`积分。位置用米、旋转用SO(3)测地角，20Hz、每步0.05秒；Action128前10维有效，其余118维无效。开度MAE采用物理[0,1]投影，原始越界比例和幅度单列。保持基线保持当前pose和开度；运动外推仅使用当前及之前4步pose的世界坐标恒速度/SO(3)角速度，开度仍保持当前值。",
        "普通非EMA权重：实际warm-start父checkpoint为6200，当前run为8400/8600/8800。6200仅迁移当前run的parameter-free归一化合同，所有学习参数不变，不能解读为旧TCP/domain的原生6200性能。采样固定35步、guidance1.5、shift5、seed42/43。`published_caption`包含示范描述/计划，只作为离线条件诊断；纯在线输入为task-only。",
        "训练标签delta回到当前发布state的最大位置偏差1.13e-7m、朝向6.72e-8rad。训练标签的quantile clip会产生额外重建误差，下方独立报告；它是标签表达的诊断，不是所有生成器不可突破的数学下界。",
    ]
    for split, title in (("diagnostic", "诊断组"), ("study_heldout", "本实验episode留出组")):
        rows = []
        for horizon in c["offline_horizons"]:
            reference = get("8800_masked", split, horizon)
            for key, label in (("hold", "保持当前state"), ("constant_velocity", "历史运动外推")):
                rows.append([horizon, label, *metrics(reference[key])])
            for name, label in names:
                rows.append([horizon, label, *metrics(get(name, split, horizon)["model"])])
        parts += [
            f"## {title}：task-only",
            "每个模型/时域48次预测、6个episode；下表为整个窗口内所有有效endpoint的误差，非仅最后一帧。",
            table(["生成步数", "参考来源", "位置RMSE mm", "SO(3) RMSE rad", "开度MAE 0–1"], rows),
        ]
    rows = []
    for x in summary["paired_serving_effects"]:
        if x["split"] == "study_heldout" and x["prompt_mode"] == "task_only":
            rows.append([x["checkpoint"], x["horizon"], *metrics(x["stock"]), *metrics(x["masked"])])
    parts += [
        "## 相同输入与seed的mask配对",
        "完整JSON caption经原服务准备后字节和像素与显式传入一致，192/192核验通过。收益不能归因于caption嵌套修复；配置中的早期错误假设见[协议勘误](../plans/v0.0.1/wam_protocol_errata_20261019v2.md)。",
        table(
            ["checkpoint", "H", "原服务位置mm", "原服务旋转rad", "原服务开度", "mask位置mm", "mask旋转rad", "mask开度"],
            rows,
        ),
    ]
    rows = []
    for split in ("diagnostic", "study_heldout"):
        for horizon in c["offline_horizons"]:
            for prompt in ("task_only", "published_caption"):
                rows.append([split, horizon, prompt, *metrics(get("8800_masked", split, horizon, prompt)["model"])])
    parts += ["## 最新8800的文本条件诊断", table(["分组", "H", "文本条件", "位置mm", "旋转rad", "开度MAE"], rows)]
    rows = []
    for x in summary["state_by_phase_task"]:
        if (
            x["checkpoint"] == "8800_masked"
            and x["split"] == "study_heldout"
            and x["horizon"] == 40
            and x["prompt_mode"] == "task_only"
        ):
            rows.append([x["task"], x["phase"], x["model"]["cases"], *metrics(x["model"]), *metrics(x["hold"])])
    parts += [
        "## 最新8800的跨阶段表现",
        "本实验episode留出、task-only、H40；每行2个episode×2个seed。阶段是state代理，回退情况保留于汇总。",
        table(
            [
                "任务",
                "阶段代理",
                "预测数",
                "模型位置mm",
                "模型旋转rad",
                "模型开度",
                "保持位置mm",
                "保持旋转rad",
                "保持开度",
            ],
            rows,
        ),
    ]
    rows = []
    for model, hold in zip(latest["model"]["endpoints_by_time"], latest["hold"]["endpoints_by_time"], strict=True):
        rows.append([model["time_s"], model["count"], *metrics(model), *metrics(hold)])
    parts += [
        "## 误差随预测时间变化",
        "最新8800、H40、task-only、本实验episode留出。长时点的样本数随有效未来减少，不能把组成变化全部解释为单一轨迹的误差增长。",
        table(
            [
                "预测时间s",
                "有效预测数",
                "模型位置mm",
                "模型旋转rad",
                "模型开度",
                "保持位置mm",
                "保持旋转rad",
                "保持开度",
            ],
            rows,
        ),
        "![State error curves](../outputs/wam_checkpoint_acceptance_20261019v2/state_figures/state_errors.png)",
    ]
    rows = [[x["split"], x["horizon"], *metrics(x["metrics"])] for x in summary["clipped_label_reconstruction"]]
    parts += [
        "## 标签clip与原始输出范围",
        table(["分组", "H", "clip标签位置mm", "clip标签旋转rad", "clip标签开度"], rows),
    ]
    rows = []
    for name in ("8800", "8800_masked"):
        for horizon in c["offline_horizons"]:
            x = get(name, "study_heldout", horizon)["model"]["raw_output"]
            rows.append(
                [
                    name,
                    horizon,
                    f"{x['opening_outside_fraction'] * 100:.2f}%",
                    f"{x['opening_min']:.4f}–{x['opening_max']:.4f}",
                    f"{x['opening_clip_distance_mean']:.5f}",
                    f"{x['opening_clip_distance_max']:.5f}",
                ]
            )
    parts += [
        table(["输出", "H", "开度越界比例", "原始开度范围", "平均裁剪距离", "最大裁剪距离"], rows),
        "原始范围统计覆盖全部生成端点，不要求该端点有未来GT。越界比例必须结合幅度看；不能将很小的浮点/生成偏差直接等同于重大任务错误。",
        "## 可复现边界",
        "当前发布12条episode的1464行action和mask与state修订前保留列一致，当前域归一化与三个当前run checkpoint一致。但历史9项metadata仅4项SHA未变，原训练图像/caption的完整字节快照及消费账本未恢复。数值审计是在固定当前输入上复算训练计算，不是完整重放历史训练batch；本报告不证明预训练未见泛化，也不认证所有数据域或真机部署。",
        "完整请求、响应、样本与指纹保留在`outputs/wam_checkpoint_acceptance_20261019v2`。复算入口为`analyze_wam_acceptance.py --seed 42 --partial`、`plot_wam_acceptance.py --seed 42 --state-only`和`report_wam_state_acceptance.py --seed 42`；后两者要求全部state矩阵已完成，拒绝覆盖现有完整产物。",
    ]
    destination.write_text("\n\n".join(parts) + "\n")
    write_json(
        root / "state_report_manifest.json",
        dict(
            code=code_fingerprint(),
            summary_sha256=sha256(snapshot),
            report_sha256=sha256(destination),
            report=str(destination),
        ),
    )
    print(destination)


if __name__ == "__main__":
    main()
