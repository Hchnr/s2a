# 第三个任务的提前多点质量恢复反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_earlier_restoration_feedback_20261111.md)，
在[上一轮对照](task2_earlier_preview_feedback_20261111.md)相同的task6 raw/terminal058
开发状态、已完成前缀和step32/80已生成参考上，增加多点质量恢复分支。候选保留
原动作、关节、制动和首点质量硬审核，未来八点质量缺口单独报告；实际仍按原参考
时钟和位置≤0.01 m、SO(3)≤0.05 rad计分。

| 开发变体 | 恢复分支执行周期 | 实际TCP双阈值 | 内环最小余量 rad | 最大位置 / SO(3)误差 | 停止 |
| --- | ---: | ---: | ---: | --- | --- |
| raw | 0 | — | — | — | step32前候选审核拒绝 |
| terminal058 | 1 | 1/1 | 0.327534 | 0.005309 m / 0.038797 rad | step81前候选审核拒绝 |

两个停止均为`quality_witness_rejected`，记录的原因是首点质量与全部硬模型约束
的联合审核未通过；原记录未细分具体失败字段，不能据此推定物理或全局不可达。
terminal058唯一已执行周期没有2 ms关节缓冲失守、越界、目标grasp或任务成功。
该周期候选审核记录的未来八点最坏归一化质量缺口为1.27246，不能当作八点硬质量
witness，也不能把1/1已执行片段当作完整窗口成功。

上一轮同起点raw首点质量分支执行1周期后停止且实际双阈值为0/1；terminal058
首点质量分支执行5周期后停止、已执行部分5/5。此次恢复分支分别执行0/1周期，
没有显示更好的持续性；两个实验中的新物理费用单独统计。两条变体属于同一开发
初态和seed43，不能作为独立泛化样本。

## 独立复核与费用

checker未导入恢复或其他反馈候选，重放唯一已派发动作；完整状态、25个2 ms内环
q/qvel/time和实际TCP误差最大差为0，grasp/任务标签一致。各分支前缀最后25内步
与对应producer逐值一致；两次审核拒绝只核对停止位置及未派发，checker未重求或
细分拒绝原因。两阶段各2个CPU/OSMesa worker均实际wait/reap、exit0且未发信号；
监督进程实际等待并回收两个执行器，源码和输入前后一致。

每phase实际113个控制周期、2825个2 ms步、6模型和6软件context；两phase合计
226周期、5650内步、12模型和12 context。producer实际3次原细化/210响应、3次
恢复求解/203响应、3次独立审核尝试计600端点查询和7800模型转换步，派发1条动作；
checker没有选择动作或求解。该动作决策约3.66秒，仿真等待时物理暂停，不构成
20 Hz实时资格。

69项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。没有
新WAM、GPU或最终留出数据。原18正确参考、12开发和24最终留出门槛保持；
`complete/passed`只表示执行和重放闭合，不表示控制或任务验收通过。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_earlier_restoration_feedback_20261111/producer/result.json` | `109ea10f9dc8697398eef0bd72fc999cc007b152ca0137d83034aa5084b03da0` |
| `outputs/task2_earlier_restoration_feedback_20261111/checker/result.json` | `b8df0bfdab1b9d40ea274c870e6891ea0b8af3c89083f5066cc1be2c39c7ed1e` |
| `outputs/task2_earlier_restoration_feedback_dry_20261111/result.json` | `224728acdb68feade1241123dcde8e07f292b958d36958223e37d89b1b135056` |
| `outputs/qa/task2_earlier_restoration_feedback_20261111/result.json` | `c52190fdd318cf6a64017b6caf8f54bca89ab6786dc941c8b57b4a50e62f42b2` |
| `outputs/task2_earlier_restoration_feedback_supervisor_20261111/producer_exit.json` | `0d6afd11e17dc25f57e5a794c582eed309485ed5629239e774f3989ca4b6ae63` |
| `outputs/task2_earlier_restoration_feedback_supervisor_20261111/checker_exit.json` | `5612ef3ecfb4e8ec4485065b389b0bfcee17910d2220c5f88ad3e48833fee179` |

该恢复候选不应提升为统一控制器。后续需保留审核拒绝前的具体模型候选和约束
证据，区分首点质量、制动或关节约束冲突，再设计更早的因果控制或参考衔接；不能
通过放松实际阈值、缩短窗口或使用另一分支的未来观测绕过目标。
