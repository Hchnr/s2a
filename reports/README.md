# 实验报告

这里保存 Git 内的精简报告、指标摘要和完成状态，完整产物位于 `outputs/`。

本轮报告为 `droid_episode0_20261016.md`，摘要为 `droid_episode0_20261016.json`；对应 [实施记录](../plans/v0.0.1/record_20261016.md) 和 [结论](../plans/v0.0.1/conclusion_20261016.md)。所有实验均标明代码/数据指纹、控制接口、划分与证据范围。

重建报告：`LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots`。它读取已有指标，不重新选择模型或增益。图在 `outputs/droid_episode0/figures/`，报告链接保留本地完整产物的位置。

首轮仅有日志、几何和单 episode 局部模型内证据；增益调度/小网络、独立仿真与真机未执行。不能以日志 action 误差或局部模型的自洽宣称控制成功。

跨数据集扩展见 `cross_dataset_franka.md` 和 `cross_dataset_franka.json`：RoboMIND-v1真实机器人日志与M1/A1仿真日志分别列出，不计作真机执行。完整逐帧样本、指纹和日志在 `outputs/cross_dataset_franka/`。

后续独立物理仿真见 `libero_simulation_20261016.md` / `.json` 及对应validation文件。分别列出原生expert、oracle状态参考、旧5600模型预测固定参考，以及新8800模型实时图像闭环；执行状态以报告完成计数为准。它不改变首轮DROID日志/模型内实验的证据等级，也不构成真机验证。重建入口为 `scripts/summarize_libero_simulation.py --seed 42`。

2026-10-19控制合同审计见 `libero_control_audit_20261019.md/.json`。它区分HDF5 oracle、发布R2参考、保存模型参考和实时模型策略，另列延迟/噪声、首次成功及完整轨迹终态；数据来源核查与软件/打包验证摘要分别保存。所有生成trace、环境、原始预测、视频和图写在outputs，Git只保留精简报告。

## 最新继续阶段

`libero_followup_20261019.md/json`：catalog与发布修订、同源图像WAM目标、当前端点测量及35项异步物理验证；完整性与任务成功分别记录。软件与打包摘要为 `validation_followup_20261019.json`，完整产物/图/归档索引在 `outputs/libero_followup_20261019/`。
