# 数值余量双点搜索的实际反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_guarded_two_step_feedback_20261111.md)，
从相同task2 raw step32与terminal058 step80开发起点，比较仅改变第一阶段数值尺度
及内部搜索余量的双点反馈。原双点质量筛选、`1e-10`审核容差、关节缓冲、动作、
终端和制动约束不变；每周期只使用本分支当前实测状态、已完成历史及已生成参考。

| 开发变体 | 上一版双点执行 / 实际双阈值 | 本版执行 / 实际双阈值 | 本版实际最大位置 / SO(3)误差 | 最小关节余量 rad | 停止 |
| --- | --- | --- | --- | ---: | --- |
| raw step32 | 0 / — | 1 / 1/1 | 0.008680 m / 0.014258 rad | 0.449920 | step33前无双点witness |
| terminal058 step80 | 4 / 4/4 | 4 / 4/4 | 0.009656 m / 0.049547 rad | 0.141006 | step84前无双点witness |

raw本次通过独立模型审核并实际执行首周期，实际双阈值通过；上一版双点搜索在
同一起点的首动作前因数值边界未派发。更早的[首点优先分支](task2_staged_first_feasibility_20261111.md)
也执行1周期，但实际位置略超阈值。本次改善了这个起点的单周期结果，仍未完成
8周期。terminal058长度与上一版相同，且不超过更早首点质量分支的5周期；两版
前2个实际动作与状态逐值相同，后续略有分歧，不把停止状态视为完全相同。

两个分支共5个已执行周期均未出现2 ms关节缓冲失守、越界、目标grasp或任务成功。
terminal058四周期实际位置/SO(3)分别约0.006254 m / 0.021457 rad、0.009554 m /
0.034273 rad、0.009656 m / 0.039937 rad、0.009181 m / 0.049547 rad。

## 搜索与停止

raw首个已审核计划的两个模型点分别约0.008689 m / 0.013309 rad和0.010000 m /
0.050000 rad，后六点最坏归一化质量缺口为0.771957；内部搜索余量没有成为最终
模型点的额外硬条件。实际完成首周期后，下一决策最近线性安全尝试的两个模型点
约0.017307 m / 0.022840 rad及0.022254 m / 0.025419 rad，未满足双点质量，
故第2条动作未派发。该尝试不是全局下界，且新分支停止状态与旧分支不同。

terminal058在第5次决策时，最近安全尝试的两点约0.008185 m / 0.045494 rad及
0.010437 m / 0.054455 rad，未满足第二点双阈值，下一动作未派发。其4条已派发
计划的后六点最坏归一化质量缺口约1.308、1.236、1.141、1.053；不能将前两点
模型审核通过说成八点质量witness。两处停止的有限搜索均不证明全局或物理不可达。

两变体属于同一开发初态和seed43；固定已生成chunk可能跨原重规划时刻，不代表新
WAM闭环或泛化。已派发决策约2.39–3.90秒，仿真等待时物理暂停，不具20 Hz实时资格。

## 独立复核与费用

checker未导入候选控制策略，重放5条已派发动作；完整状态、125个2 ms内环q/qvel/time、
命令、实际TCP误差和任务/grasp标签逐值一致。两分支末次前缀内环也逐值一致；
无witness只核对停止位置与未派发，checker未重求解。两阶段各2个CPU/OSMesa worker
均实际wait/reap、exit0且未发信号，监督进程实际等待并回收两个执行器，源码及输入
前后一致。

每phase实际117控制周期、2925个2 ms步、6模型和6软件context；两phase合计
234周期、5850内步、12模型和12 context。producer实际7次原细化/484响应、7次双点
搜索/524响应、7次初值LP、0次额外求解器资格LP及5次独立审核计1000端点查询/
13000模型转换步；7次种子rollout复用，单次未超过320响应。checker未选择动作或
重求解。

104项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出数据；原18正确参考、12开发与24最终留出门槛不变。
`complete/passed`仅表示执行与重放闭合，不表示控制或任务验收通过。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_guarded_two_step_feedback_20261111/producer/result.json` | `f83d77c74a4230419cb41eb624d8c8fdd07b8f27ca8646e8838121dbaad3a4b7` |
| `outputs/task2_guarded_two_step_feedback_20261111/checker/result.json` | `2e4ce651b8d7a0d62cf88a1f0097555bbe21c8d65f1b615b42cc919c89718f5c` |
| `outputs/task2_guarded_two_step_feedback_dry_20261111/result.json` | `e924ab9369725286f2f34554a70de67c1daa4c9c5c4097f2a87ac3d78b1943aa` |
| `outputs/qa/task2_guarded_two_step_feedback_20261111/result.json` | `f52ca55c2bb0517cbca4b301b083951b0239bb4b71aa19b7dedd17b874bcc819` |
| `outputs/task2_guarded_two_step_feedback_supervisor_20261111/producer_exit.json` | `243224c16aa68ef1d6da3a7f55e61a59b6cc7cc51f945e679d753123cf870ef8` |
| `outputs/task2_guarded_two_step_feedback_supervisor_20261111/checker_exit.json` | `198c0bf12454becec01f0f80e2240b2aad5a2e911a3a43547a5be6288db288c1` |

后续应检查模型在首动作后对实际速度及下一周期可行性的偏差，并研究更早的因果
参考衔接或控制干预；不能因1/1和4/4短片段或静态模型候选更改原完整验收门槛。
