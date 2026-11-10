# 第三个任务新参考生成时的前瞻反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_earlier_preview_feedback_20261111.md)，
从同一task6开发初态的raw/terminal058两个相关生成变体，在step32/80刚生成参考时
分别比较原细化、首点质量及八点质量反馈。每对比组共享原初态、已完成命令前缀和
当时生成的参考；分歧后只用各自实际反馈与已完成历史，夹爪固定0。实际端点按原
参考时钟计分，双阈值为位置≤0.01 m且SO(3)≤0.05 rad。

| 开发变体 / 分支 | 已执行周期 | 实际TCP双阈值 | 内环最小余量 rad | 低于0.05 / 越界内环样本 | 最大位置 / SO(3)误差 | 停止 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| raw / 原细化 | 8 | 0/8 | 0.054167 | 0 / 0 | 0.023112 m / 0.057666 rad | 无 |
| raw / 首点质量 | 1 | 0/1 | 0.452172 | 0 / 0 | 0.010080 m / 0.010727 rad | step33前无witness |
| raw / 八点质量 | 0 | — | — | — | — | step32前无witness |
| terminal058 / 原细化 | 8 | 5/8 | 0.049441 | 11 / 0 | 0.006472 m / 0.084102 rad | 无，窗口结束 |
| terminal058 / 首点质量 | 5 | 5/5 | 0.111154 | 0 / 0 | 0.007079 m / 0.044637 rad | step85前无witness |
| terminal058 / 八点质量 | 0 | — | — | — | — | step80前无witness |

提前两周期没有产生完整通过的八周期窗口。terminal058首点质量分支的5个已执行
端点达标，但随后派发前停止；此前从step82出发的同类分支也在step85前停止，
两者分歧后的状态不同，不能把较长的5/5片段当成统一策略成功。两个八点质量分支
均在首个动作前有限搜索无witness；不能推断全局或物理不可达。

raw首点质量分支的已审核模型首点位置误差约0.010000 m，处于原数值审核容差内，
实际误差为0.01007985 m，超过阈值约0.00007985 m；模型witness未保证实际质量。
terminal058原细化的第8个周期step87已有11个2 ms样本低于0.05 rad，随后窗口
结束，因此没有下一次派发或显式门控停止；这些样本未越XML原关节范围。所有六
分支均无目标grasp或任务成功端点，不能用局部TCP片段替代任务表现。

## 独立复核与费用

checker不导入反馈候选，重放22条已派发动作；完整状态、550个2 ms内环q/qvel/time
及实际TCP误差的最大差均为0，grasp/任务标签一致。每分支前缀最后25个2 ms样本
也与对应producer逐值一致。四处质量无witness的停止位置和未派发得到核对；原因
来自producer记录，checker未重求解或证明无解。六个producer和六个checker worker
均实际wait/reap、exit0且未发信号，监督进程实际等待并回收两阶段执行器，源码和
输入前后一致。

每phase实际358个控制周期、8950个2 ms步、18模型和18软件context；两phase合计
716周期、17900内步、36模型和36 context。producer实际26次原细化/1922响应、
10次质量求解/1152响应、6次独立witness审核计1200端点查询和15600模型转换步；
checker没有选择动作或求解。已派发臂决策约1.13–3.50秒/周期，仿真等待时物理
暂停，墙钟不构成20 Hz实时资格。

60项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出数据。原18正确参考、12开发和24最终留出门槛保持；
`complete/passed`仅表示执行和重放闭合，不表示控制或任务验收通过。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_earlier_preview_feedback_20261111/producer/result.json` | `b48288924dc24d2f6dee61b4a27d8491dced865dac1c311c07cb4336d9440206` |
| `outputs/task2_earlier_preview_feedback_20261111/checker/result.json` | `f61a9360e7d16f5599b1be49dbe5f068fed74e87f1b3a71c31893dac766e8f38` |
| `outputs/task2_earlier_preview_feedback_dry_20261111/result.json` | `b9e077f6b29bb581c0c8596bc51a657c7c7e0f2f61e2604fb9599181aa9e4fcb` |
| `outputs/qa/task2_earlier_preview_feedback_20261111/result.json` | `f69f2143a00798c4fd19de57a7d922020cbbf54a0a66d82ae8b9b2d805eb5c4b` |
| `outputs/task2_earlier_preview_feedback_supervisor_20261111/producer_exit.json` | `4ff28738d8cc56203515ad3de5fc190da681e99fef4b6e2e08306c2cfd51291c` |
| `outputs/task2_earlier_preview_feedback_supervisor_20261111/checker_exit.json` | `62e9750b3e38cf9990144dbcd121bb271e5f8a774b158a168ef7c288eb2bd73` |

下一步需分析step32/80时八点无witness的约束冲突，并设计能在自身因果历史中
保持原时序质量和安全的候选；不因缩短窗口、放宽质量或使用未来观测宣布成功。
