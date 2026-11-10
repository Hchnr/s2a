# 第三个任务的双点硬约束实际反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_two_step_feedback_20261111.md)，
从此前相同task2 raw step32和terminal058 step80开发起点，使用各自当前实测状态、
已完成历史和对应已生成chunk，要求模型前两点及原关节、动作、终端和制动约束
经独立审核后才派发首动作。后六点质量缺口单独记录，每周期重新依据本分支实际
反馈求解；夹爪固定0，起点前无成对内环预测误差，零profile不表示未来误差保证。

| 开发变体 | 已执行周期 / 实际双阈值 | 实际最大位置 / SO(3)误差 | 内环最小余量 rad | 停止 |
| --- | --- | --- | ---: | --- |
| raw step32 | 0 / — | — | — | step32前无双点witness |
| terminal058 step80 | 4 / 4/4 | 0.009656 m / 0.049547 rad | 0.141006 | step84前无双点witness |

terminal058的4个已执行周期均无2 ms关节缓冲失守、越界、目标grasp或任务成功；
未完成完整8周期。此前相同起点的[两阶段首点优先](task2_staged_first_feasibility_20261111.md)
分支只执行1周期，本次延长到4周期；但更早的[首点质量分支](task2_earlier_preview_feedback_20261111.md)
执行5周期且实际5/5双阈值，本次并未超过该已有结果。raw先前首点分支执行1周期
但实际位置超限，本次在首动作前停止；不同分支分歧后的状态不能视为相同。

## 搜索与停止

raw最近线性安全尝试的第一点为0.008715 m / 0.013369 rad，第二点为
**0.010000005704 m / 0.050000006337 rad**。第一阶段虽然报告优化成功，第二点
仍超过原双阈值及`1e-10`审核容差，因此没有审核或派发；不能把优化成功当witness。
terminal058在第5次决策时，最近线性安全尝试的第一点为0.007116 m / 0.048177 rad，
第二点为0.010556 m / 0.053205 rad；第一阶段达到60次迭代后无双点witness，
下一动作未派发。有限搜索与数值边界拒绝均不证明全局或物理不可达。

terminal058的4条已派发计划均独立通过前两点质量及八周期原安全模型审核；各自
后六点最坏归一化质量缺口分别约1.308、1.236、1.141、1.053，不能称为八点硬
质量witness。实际位置/SO(3)逐周期分别为0.006254 m / 0.021457 rad、
0.009554 m / 0.034273 rad、0.009656 m / 0.039937 rad、0.009181 m / 0.049547 rad。
单周期决策约2.87–3.80秒，仿真等待时物理暂停，不具20 Hz实时资格。

本次在线事务先运行原细化器生成本分支种子；[静态双点对照](two_step_viability_20261111.md)
使用此前已派发的两阶段计划为种子，故不能要求候选逐值相同或将静态witness视作
本次实际结果。两变体仍属于同一开发初态和seed43；固定已生成chunk可能跨原
重规划时刻，不是新WAM闭环或泛化评估。

## 独立复核与费用

checker未导入候选控制策略，重放4条已派发动作；完整状态、100个2 ms内环q/qvel/time、
命令和实际TCP误差逐值一致，任务及grasp标签一致。两分支末次前缀内环也逐值一致；
两次无witness只核对停止位置与未派发，checker未重求解。两阶段各2个CPU/OSMesa
worker均实际wait/reap、exit0且未发信号，监督进程实际等待并回收两个执行器，
源码及输入前后一致。

每phase实际116控制周期、2900个2 ms步、6模型和6软件context；两phase合计
232周期、5800内步、12模型和12 context。producer实际6次原细化/421响应、6次
双点搜索/387响应、6次初值LP、0次额外求解器资格LP、4次独立审核计800端点查询
及10400模型转换步；6次种子rollout复用，单次未超过320响应。checker未选择新
动作或重求解。

91项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出数据，原18正确参考、12开发及24最终留出门槛保持；
`complete/passed`只表示执行与重放闭合，不表示控制或任务验收通过。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_two_step_feedback_20261111/producer/result.json` | `cfb36bb2c2227749d585f5267914e2a8f8edd1987fb7a90f2b55afba9115e7cf` |
| `outputs/task2_two_step_feedback_20261111/checker/result.json` | `e714f71e11282a4a718565956b12e33a3c2fe934acb3bb642495a3d5608cd75e` |
| `outputs/task2_two_step_feedback_dry_20261111/result.json` | `dea2885a7f0f3b1ed228da8ccd405a417d97593d02524801954e5b880c37cfc3` |
| `outputs/qa/task2_two_step_feedback_20261111/result.json` | `86fba78d82e56058a1184e17b9a41182e658090792c9301bae7c8dc03e1aca37` |
| `outputs/task2_two_step_feedback_supervisor_20261111/producer_exit.json` | `3777528e22c5a7fa9bd4234e5277a0756c1618926fbdb39c23abbe35e10cc33d` |
| `outputs/task2_two_step_feedback_supervisor_20261111/checker_exit.json` | `779e3d85ab38ec86eca74f49a7486af6597310f155e9a61dd472742eb5f1c456` |

后续应分别处理raw第一决策的严格数值边界及terminal058更晚的双点冲突，并以更早
因果状态或可执行的参考衔接寻找持续性；不能通过扩大审核容差、只报告4/4片段
或消费最终留出数据来代替原目标。
