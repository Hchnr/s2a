# 实验报告

这里保存 Git 内的精简报告、指标摘要和完成状态，完整产物位于 `outputs/`。

[当前实现进度与总目标](status_current_20261023.md)汇总已实现、已验证和未完成项；总目标与20/24、各任务4/8的完整验收口径见[036](../plans/v0.0.1/036_project_goal_20261023.md)。下列按日期保留历史推进记录。

2026-10-23最新接续：[041准入与实际启动](wam_clean_training_continuation_20261023.md)。730 passed/1 skipped、Ruff、484源码/快照及完整独立metadata/执行审查通过；真实D0 dry通过，独立session supervisor在短launcher退出后PPID1存活并持有新D0实际模型。原失败不改、不用于fit，仍无整轮采集/实际fit/任务/fresh成绩。下列准备与旧启动为此前快照。

2026-10-23中断核查：[D0中断与独立接续准备](wam_clean_training_interruption_20261023.md)。原进程消失、20完整＋partial25节点、无wait/reap/exit资格；失败原址保留不用于fit，041新根/明确累计预算及持久化owner准备中，尚无新模型/fit/任务/fresh。下列D0启动段为此前快照；[最终验收差距](wam_final_acceptance_gap_audit_20261023.md)仍未闭合。

2026-10-23执行推进：[040准入与D0启动](wam_clean_training_execution_20261023.md)已完成613 passed/1 skipped、Ruff及独立完整执行审查，462源/快照冻结。原63位seed不可执行问题在模型调用前以新31位seed-plan明确修订，原metadata保留；D0唯一模型已启动，尚无整轮采集、候选fit、开发或fresh成绩。下列训练准备状态保留为此前快照。

2026-10-23训练准备：[038实现与准备](wam_clean_training_preparation_20261023.md)的150当前锚点/300独立noise及24 validation锚点/96矩阵已完成独立metadata审查，执行仍未就绪。新软件不计作实际训练、任务或fresh收益。

2026-10-23最新阶段：[真实clean桥接联合资格通过](wam_clean_bridge_20261023.md)。037实际7请求/245节点及039严格原生mask CPU补充审计、独立peer均完成；原False和准备/peer失败保留，0fit/任务/fresh收益。下一阶段为038有限训练与开发协议的实现/准备。

2026-10-23 10:51+08历史快照见[实现进度核查](status_review_20261023.md)，总体目标及验收见[036计划](../plans/v0.0.1/036_project_goal_20261023.md)。当时034新数据30/30合格，035缺完整结果和退出收据；后续037＋039联合桥接资格见上列新报告。以下内容按历史阶段保留。

当前稳定成功阶段的完整结果见下表。结果均为开发集证据；尚未执行预留新初态验收，无真机。

| 报告 | 已完成范围 | 主要结果 |
| --- | --- | --- |
| [WAM夹爪位置/保持](wam_position_hold_20261021.md) | 8800/15400，24条WAM，14400步 | 曾成功2/24，终态/稳定0/24；映射错误减少没有转化为任务稳定提升 |
| [8400同协议对照](wam_earlier_position_20261021.md) | 12条WAM，7200步 | 稳定1/12，仅alpha0/cream；raw各组0/3 |
| [最新15600的state](wam_latest_state_screen_20261021.md) | 48预测，0物理 | 九项H8/H16/H40指标均差于8400及15400；不将latest作为best |
| [更强guidance的state](wam_guidance_state_20261021.md) | 48预测，0物理 | 旋转/开度退化、位置未达10%改善，不推广 |
| [采样步数的state](wam_sampler_steps_state_20261021.md) | 35/70/140各48，共144预测，0物理 | 70/140九项均差35，不推广；关闭ledger缺失另列 |
| [训练attention backend](wam_backend_state_20261021.md) | 48真实初态配对预测，0物理 | H8/H16位置略差，未过推广门槛；退出内部ledger缺失另列 |
| [双种子检查器](wam_seed_aware_checker_20261021.md) | 27旧case只读精确回归 | 只扩显式seed/来源接口，不新增任务成功 |
| [真实43执行资格](wam_seed43_execution_qualification_20261021.md) | 3oracle，1800步 | 3/3，和对应42逐步相同；未验证43 WAM/新初态 |
| [60秒执行资格](wam_long_qualification_20261021.md) | native3＋oracle3，7200步 | 全6稳定，实际控制前600精确；原终点诊断日志strictFalse与舍入原因保留 |
| [60秒原始WAM](wam_long_raw_20261021.md) | 3WAM，3600步，450请求 | 仍1/3稳定，仅bowl尾43.1秒保持；wine/cream后半跟踪约1.9mm仍失败 |
| [raw8第二seed复测](wam_raw8_seed43_20261021.md) | 3WAM，1800步，225请求 | seed43全0/3；不能复现seed42唯一bowl成功，不拼seed/任务 |
| [固定短间隔WAM](wam_short_replan_20261021.md) | 12WAM，7200步，678请求 | raw8仅bowl1/3；其余三组0/3；外部未使用历史权重缺失的收尾例外保留 |
| [按跟踪误差重观测](wam_tracking_replan_20261021.md) | 3oracle通过后3WAM，共3600步 | WAM完整0/3稳定、97请求；不推广 |
| [短重观测执行资格](short_replan_qualification_20261021.md) | 12条oracle，7200步 | 12/12稳定，只证明正确参考附近可执行 |
| [整episode参考增益诊断](wam_reference_gain_diagnostic_20261021.md) | 缓存拟合/留出，0新推理/物理 | 数毫米位置收益、朝向/开度有退化，不推广 |
| [世界旋转漂移诊断](wam_rotation_drift_diagnostic_20261021.md) | 19次缓存拟合、六整episode留出，0新推理/物理 | H8/H16旋转改善10.50%/21.63%，仍差于简单对照，不推广 |

下一项已登记[020因果参考集成](../plans/v0.0.1/020_causal_temporal_ensemble_20261021.md)，执行进度见[结论](../plans/v0.0.1/conclusion_20261021.md)和[记录](../plans/v0.0.1/record_20261021.md)。


本轮报告为 `droid_episode0_20261016.md`，摘要为 `droid_episode0_20261016.json`；对应 [实施记录](../plans/v0.0.1/record_20261016.md) 和 [结论](../plans/v0.0.1/conclusion_20261016.md)。所有实验均标明代码/数据指纹、控制接口、划分与证据范围。

重建报告：`LD_LIBRARY_PATH='' .venv/bin/python scripts/run_experiment.py --seed 42 --stage report --plots`。它读取已有指标，不重新选择模型或增益。图在 `outputs/droid_episode0/figures/`，报告链接保留本地完整产物的位置。

首轮仅有日志、几何和单 episode 局部模型内证据；增益调度/小网络、独立仿真与真机未执行。不能以日志 action 误差或局部模型的自洽宣称控制成功。

跨数据集扩展见 `cross_dataset_franka.md` 和 `cross_dataset_franka.json`：RoboMIND-v1真实机器人日志与M1/A1仿真日志分别列出，不计作真机执行。完整逐帧样本、指纹和日志在 `outputs/cross_dataset_franka/`。

后续独立物理仿真见 `libero_simulation_20261016.md` / `.json` 及对应validation文件。分别列出原生expert、oracle状态参考、旧5600模型预测固定参考，以及新8800模型实时图像闭环；执行状态以报告完成计数为准。它不改变首轮DROID日志/模型内实验的证据等级，也不构成真机验证。重建入口为 `scripts/summarize_libero_simulation.py --seed 42`。

2026-10-19控制合同审计见 `libero_control_audit_20261019.md/.json`。它区分HDF5 oracle、发布R2参考、保存模型参考和实时模型策略，另列延迟/噪声、首次成功及完整轨迹终态；数据来源核查与软件/打包验证摘要分别保存。所有生成trace、环境、原始预测、视频和图写在outputs，Git只保留精简报告。

## 最新继续阶段

`libero_followup_20261019.md/json`：catalog与发布修订、同源图像WAM目标、当前端点测量及35项异步物理验证；完整性与任务成功分别记录。软件与打包摘要为 `validation_followup_20261019.json`，完整产物/图/归档索引在 `outputs/libero_followup_20261019/`。
