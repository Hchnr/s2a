"""Publish all four acceptance layers only after both physics matrices pass integrity checks."""

import argparse
import json
from pathlib import Path

import numpy as np
from report_wam_state_acceptance import table

from s2a.artifacts import code_fingerprint, load_config, sha256, write_json


def load_completed(root):
    summary_path, integrity_path = root / "summary.json", root / "integrity.json"
    summary = json.loads(summary_path.read_text())
    integrity = json.loads(integrity_path.read_text())
    if not summary["complete"] or not integrity["complete"]:
        raise ValueError("final report requires complete, independently verified matrices")
    for path, digest in summary["fingerprints"].items():
        if sha256(path) != digest:
            raise ValueError(f"result changed after aggregation: {path}")
    for case in integrity["physics"]:
        for path, digest in case["files"].items():
            if sha256(path) != digest:
                raise ValueError(f"physics evidence changed after verification: {path}")
    return summary, {str(p): sha256(p) for p in (summary_path, integrity_path)}


def tracking_range(rows, key, scale=1):
    values = np.asarray([row[key] for row in rows]) * scale
    return f"{values.min():.3f}–{values.max():.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wam_checkpoint_acceptance_20261019v2.json")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    c, root = load_config(args.config)
    if args.seed != c["seed"]:
        raise ValueError("seed differs")
    primary, fingerprints = load_completed(root)
    native, other = load_completed(root / "native_horizon40")
    fingerprints.update(other)
    destination = Path("reports/wam_checkpoint_acceptance_20261019v2.md")
    manifest_path = root / "final_report_manifest.json"
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("final evidence report must not be silently overwritten")
    comparisons = [
        ("专家command", primary["physics"]["native"]),
        ("真实state + S2A", primary["physics"]["oracle"]),
        *[
            (name + " H16", primary["physics"][key])
            for key, name in (
                ("warm6200_masked", "6200 warm-start"),
                ("8400_masked", "8400"),
                ("8600_masked", "8600"),
                ("8800_masked", "8800"),
            )
        ],
        ("8800 H40补充", native["physics"]["8800_masked"]),
    ]
    latest16 = primary["physics"]["8800_masked"]
    latest40 = native["physics"]["8800_masked"]
    all_cases = [case for _, matrix in comparisons for case in matrix["cases"]]
    steps = sum(row["steps"] for row in all_cases) + sum(p["result"]["steps"] for p in primary["probes"])
    if len(all_cases) + len(primary["probes"]) != 90 or steps != 51480:
        raise ValueError("final physics evidence does not cover the frozen 90-case/51480-step budget")
    parts = [
        "# WAM checkpoint四层验收 — 2026-10-19 v2",
        f"预注册的离线与物理矩阵全部完成：2688次离线预测；{len(all_cases) + len(primary['probes'])}条独立物理实验、{steps}个控制步。最新非EMA iter_000008800经推理mask修正后，在冻结S2A、无推理延迟的H16主协议中曾成功{latest16['success']}/12、末端成功{latest16['terminal_success']}/12；原生H40补充中曾成功{latest40['success']}/12、末端成功{latest40['terminal_success']}/12。全部用例执行足600步，没有以缺失、基础设施失败或提前截断替代任务结果。",
        "## 这次验收回答了什么",
        "确定发现了训练—推理合同缺陷：训练将118个无效Action128维度置零，原服务却向这些维度加噪声，只在最终输出mask；无效噪声会影响10个有效维度。S2A本地评测wrapper在初始噪声、每次网络输入和velocity三个位置保持mask，未修改学习参数。外部Cosmos仍只读，此修正尚未合入其生产服务。源码定位、固定回归样本与生产移植约束见[serving合同说明](wam_serving_contract_20261019v2.md)。",
        "修正后模型学到了部分有用的状态变化，但并未全面通过state质量验收。最新8800在本实验episode留出、task-only、H40窗口的池化位置RMSE为64.213mm、开度MAE为0.1079，优于保持state的122.395mm/0.2484；朝向RMSE为0.5159rad，差于保持基线的0.2832rad。不能把原服务缺陷等同于权重损坏，也不能把单条示范的state误差等同于任务失败。",
        "真实成功state经同一Action128接口和冻结S2A为12/12成功，证明该执行链路在这些参考附近可用；大角度/快速参考及任意绝对开度曲线仍有限制。下面同时报告真实任务谓词、参考跟踪和物体进度，避免将全部闭环失败唯一归因于WAM或S2A。",
        "## 固定协议与证据边界",
        "三个任务：bowl→plate、wine bottle→rack、cream cheese→bowl。每任务两个独立物理初态，各用采样seed42/43；合计六个独立初态、12次运行，不能当作12个独立场景。初态在评测前固定，排除此前S2A调参用的bowl示范。离线12个episode整条划分成diagnostic/study-heldout，但全部属于训练catalog；历史逐样本消费未知，本次留出不能解释为预训练未见。",
        "每条任务600控制步/30物理秒，20Hz；35采样步、guidance1.5、shift5。每次仅给当前相机图像和task-only文本，生成16步并执行前8步再观测；推理期间physics暂停，保存的前后state哈希一致。H40补充只改变生成长度为40，仍执行前8步；它仅评估最新8800，不用于与其他checkpoint的H16公平排名。",
        "S2A为固定OSC响应的解析补偿：inverse_weight=0.5、command_rate=5/s，较大朝向误差回退到基线控制；不利用任务标签规划/修正参考。夹爪OpeningIntent按参考变化方向和0.02滞环维持开/关意图，不是任意绝对开度的理想伺服。机器人为独立LIBERO/MuJoCo中的MountedPanda/PandaGripper；没有真机实验，也没有宣称实时性能。",
        "位置误差统一报告canonical TCP；原始日志还保留controller source点误差，二者通过冻结工具变换关联。朝向用SO(3)测地角，开度范围0–1。专家command没有desired-state轨迹，因此不报告其tracking RMSE。",
        "## 训练与推理数值核对",
        "在三个任务、三个噪声强度上，原生H40/2秒、完整published caption、真实action/vision mask和标签clip的训练计算与对齐推理计算：token、条件latent、归一化标签和有效velocity逐项一致。使用保存的flash3_video与固定padding后，相对SDPA有效velocity最大绝对差0.0234375、最大RMS差0.00551005；无效维度噪声引起的有效velocity RMS扰动0.27–0.43，明显更大。复算包含训练loss及独立active MSE，不是分布式优化器更新重放。",
        "当前服务的caption/首帧像素192/192对照相同，撤回早期JSON嵌套假设；显式caption hook在这些样本上为无变化操作。冻结配置保留原字节，勘误见[协议勘误](../plans/v0.0.1/wam_protocol_errata_20261019v2.md)。历史图像/caption完整字节和消费账本未恢复，不能声称重放了当时的历史训练batch。",
        "实际warm-start父权重为6200，当前run比较8400/8600/8800。父权重只通过官方parameter-free规则迁移当前domain/归一化合同，所有参数对象、存储和version不变；排除action128_spec后整个model.config无差异。四份checkpoint全量SHA与DCP存储检查通过，未发现逻辑tensor缺失/重叠或混装证据；存储完整不代表学习质量达标。",
        table(
            ["checkpoint", "实际模型目录（非EMA）"],
            [
                [ck["name"], f"`{Path(ck['run']) / 'checkpoints' / ck['iteration'] / 'model'}`"]
                for ck in c["checkpoints"]
            ],
        ),
        "## 直接state预测",
        "完整表、跨阶段结果、两个时域、两个文本条件、causal基线和误差随时间曲线见[直接state验收](wam_state_acceptance_20261019v2.md)。48个阶段样本包含43个不同观测锚点、16个阶段代理回退；9个样本未来不足40步，仅评分有效前缀。实际预测总量为7×384=2688。",
        "在task-only、study-heldout、H40的同组比较中，8400/8600/8800修正后位置RMSE为52.251/84.453/64.213mm，朝向0.3388/0.6354/0.5159rad；本小样本没有随训练阶段单调改善。8800原服务为129.808mm/2.0923rad/0.2170，修正后为64.213mm/0.5159rad/0.1079。训练标签clip自身重建位置RMSE6.988mm，不能独自解释上述误差。",
        "## 闭环任务结果",
        "每格为曾达到原生任务谓词/结束时满足谓词；完整预算下分别统计，接触和物体位移不能替代成功。",
    ]
    rows = []
    for label, matrix in comparisons:
        task_rates = []
        for task in range(3):
            cases = [r for r in matrix["cases"] if r["record"].startswith(f"task{task}_")]
            task_rates.append(
                f"{sum(r['success'] for r in cases)}/{sum(r['terminal_success'] for r in cases)}（n={len(cases)}）"
            )
        rows.append(
            [label, *task_rates, f"{matrix['success']}/{matrix['terminal_success']}（n={matrix['completed']}）"]
        )
    parts.append(table(["参考/协议", "bowl→plate", "wine→rack", "cream→bowl", "合计曾成功/末端成功"], rows))
    parts += [
        "## 参考跟踪与执行范围",
        "下表为各用例RMSE/MAE的最小–最大值，非把整条轨迹峰值称为RMSE，也不是不同长度实验的简单平均。任务结果即使失败也保留全部600步。",
        table(
            ["参考/协议", "TCP位置RMSE mm范围", "SO(3) RMSE rad范围", "开度MAE范围"],
            [
                [
                    label,
                    tracking_range(matrix["diagnostics"], "position_rmse_tcp_m", 1000),
                    tracking_range(matrix["diagnostics"], "rotation_rmse_rad"),
                    tracking_range(matrix["diagnostics"], "opening_mae"),
                ]
                for label, matrix in comparisons[1:]
            ],
        ),
        "独立三轴正弦probe各180步、周期60步：平移振幅5/20/50mm和旋转0.05/0.2rad通过10mm/0.05rad点态门槛；旋转0.5rad虽RMS达标，TCP峰值24.739mm、旋转峰值0.07727rad超限。probe保持初始夹爪开度，没有独立证明任意开度曲线的执行资格。这个局部运动范围不能外推为整个工作空间的保证。",
        "## 每条任务的接触、移动与放置结果",
        "接触为MuJoCo/robosuite的目标物体grasp-contact谓词；高度增益和位移为相对初始物体位置的最大值，是连续进度代理。推碰也可能产生位移/高度，不能自动称为成功抓取或搬运；放置是否成功仍看原生任务谓词。所有值直接来自完整物理日志，没有事后添加成功门槛。",
    ]
    for label, matrix in comparisons[1:]:
        rows = []
        for d in matrix["diagnostics"]:
            progress = d["task_object_progress"]
            rows.append(
                [
                    f"{d['record']}/s{d['sampling_seed']}",
                    f"{int(d['success'])}/{int(d['terminal_success'])}",
                    progress["grasp_contact_steps"],
                    f"{1000 * progress['max_height_gain_m']:.2f}",
                    f"{1000 * progress['max_displacement_m']:.2f}",
                    f"{1000 * d['position_rmse_tcp_m']:.2f}",
                    f"{d['rotation_rmse_rad']:.4f}",
                    f"{d['opening_mae']:.4f}",
                ]
            )
        parts += [
            f"### {label}",
            table(
                ["初态/seed", "曾/末成功", "接触步", "升高mm", "位移mm", "TCP RMSE mm", "SO(3) rad", "开度MAE"], rows
            ),
        ]
    parts += [
        "## 复核与复现",
        "主矩阵和原生H40补充都通过独立完整性验证：checkpoint/非EMA身份、输入像素、任务/seed、固定协议、每个chunk的当前实测anchor、Action128积分与实际执行目标、0.05秒物理间隔、推理期间physics不推进、原生逐步成功谓词及全600步预算。2688条离线预测的有效前缀、积分和评分另行逐条重算。",
        "完整来源：`outputs/wam_checkpoint_acceptance_20261019v2/summary.json`、`integrity.json`、`native_horizon40/summary.json`及`native_horizon40/integrity.json`；图表PNG/PDF在`figures/`。本报告与这些汇总的SHA写入`final_report_manifest.json`。每条原始输入、响应、controller参数、steps和代码指纹均保留；生成产物不提交Git，报告/代码/配置/决策进入Git。",
        "共享服务忙和数值审计共享GPU显存不足等基础设施尝试已单独归档，不计作模型任务失败。模型副本在接入物理评测前必须逐项复现同输入/seed的完整Action128响应。实验未改变冻结权重、任务提示、S2A参数、预算或结果集合。",
        "## 结论边界与后续",
        "这轮可以区分已证实的serving mask缺陷、直接state预测质量和冻结执行链路下的任务表现；不能证明checkpoint文件损坏，不能从单条专家轨迹误差推断唯一正确行为，也不能把超出S2A已验证范围的执行偏差全归于权重。泛化还需要真正不属于预训练catalog的冻结数据，历史训练复现需要保存完整输入字节和消费账本。",
        "优先将已验证的active-slot mask修正移植到Cosmos生产推理并复验固定输入；再在训练侧检查朝向表示/归一化、训练目标与多阶段状态质量。若要扩大执行资格，应单独验证较大旋转及接触开度，不用任务标签校正WAM参考来代替checkpoint验收。",
    ]
    destination.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    write_json(
        manifest_path,
        dict(code=code_fingerprint(), sources=fingerprints, report=str(destination), report_sha256=sha256(destination)),
    )
    print(json.dumps(dict(report=str(destination), physics_cases=len(all_cases) + len(primary["probes"]), steps=steps)))


if __name__ == "__main__":
    main()
