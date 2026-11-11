# 双点优先、已完成计划后移的因果反馈

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_causal_shift_fallback_20261111.md)，
在与[原guarded双点反馈](task2_guarded_two_step_feedback_20261111.md)相同的task2
raw step32和terminal058 step80开发起点，优先进行原双点搜索；仅在其无witness且
预算允许时，重算本分支上一条已完成计划的后移结果，经首点质量和完整模型安全
独立审核后才允许回退。回退只保证模型首点，第二点及后六点单独报告。

| 开发变体 | 原双点执行 / 实际双阈值 | 本次执行 / 实际双阈值 | 实际最大位置 / SO(3)误差 | 最小内环余量 rad | 停止 |
| --- | --- | --- | --- | ---: | --- |
| raw step32 | 1 / 1/1 | 1 / 1/1 | 0.008680 m / 0.014258 rad | 0.449920 | step33前首点回退不合格 |
| terminal058 step80 | 4 / 4/4 | 5 / 5/5 | 0.009656 m / 0.049547 rad | 0.110150 | step85前首点回退不合格 |

raw首周期与原双点分支的动作、状态和内环逐值一致；下一次双点搜索无候选，后移
计划首点模型预测约0.010372 m / 0.052042 rad，超过原双阈值，故未派发。terminal058
前4周期与原双点分支逐值一致；step84双点搜索无候选后，后移计划经独立首点/全程
模型安全审核后派发，实际位置/SO(3)误差约**0.009192 m / 0.045765 rad**。
随后step85双点搜索仍无候选，后移计划首点模型预测约0.012765 m / 0.064237 rad，
故停止。两条被拒回退在各自冻结模型的线性安全检查通过，但首点质量未通过。

terminal058这次把原双点分支从4个实际达标周期延长到5个；此前另一条
[首点质量分支](task2_earlier_preview_feedback_20261111.md)也曾达到5个达标周期，
两分支分歧后状态不同，本次未超过其长度。两个变体均属同一开发初态和seed43；
六个已执行周期均未出现2 ms关节缓冲失守、越界、目标grasp或任务成功。两分支
均未完成8周期，不能推断跨初态泛化或最终稳定任务成功。

## 回退的审核范围

terminal058 step84回退的首两点模型误差分别为0.009502 m / 0.046570 rad与
0.013043 m / 0.066521 rad；第二点超过原双阈值，完整八点最坏归一化质量缺口约
1.495111。该动作通过独立首点质量和八周期box、变化率、关节缓冲及0.1 s制动
审核，但**不是双点质量witness**。首点和安全审核不能保证真实后续质量或持续可行；
实际本次只再执行1周期。

本次共8次质量求解、576模型响应，其中3次回退rollout均包含在各自的320响应预算内；
实际选择5次双点候选与1次首点回退，8次初始化LP。生产阶段共8次原细化/547响应，
6次候选独立审核计1200端点查询/15600模型转换步。已派发决策约2.38–3.84秒，
仿真等待时物理暂停，不具有20 Hz实时资格。

## 独立重放与费用

checker未导入候选控制策略，6条已派发动作、完整状态、150个2 ms内环q/qvel/time、
实际TCP误差及grasp/任务标签均与producer逐值一致。停止位置与未派发得到核对；
checker未重求无witness原因。每phase实际118控制周期/2950个2 ms步、6模型和
6个OSMesa context；两phase合计236周期/5900内步、12模型和12 context。四个worker
及监督进程实际wait/reap、exit0、未发信号，源码与输入前后一致。

67项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出样本，原实际误差和任务成功门槛不变。`complete/passed`仅表示
执行与重放闭合，不表示完整控制或任务资格；后续需解决step85附近的首点可行性及
更早的因果参考衔接，而不能通过放宽真实误差或提前声明成功消除缺口。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_causal_shift_fallback_20261111/producer/result.json` | `54e42a23f0d150d11ee3a896df8e28a04a7786edd538784dd91b8a76d74f4ad2` |
| `outputs/task2_causal_shift_fallback_20261111/checker/result.json` | `4848e256de842e96cb66c9d07991a5da223e6015b1a07a0d1bd73b4e1b8c66c7` |
| `outputs/task2_causal_shift_fallback_dry_20261111/result.json` | `b24b49f60af8d0d000ccfdc55b43ea04b457018e316463b64bb464c4579ccca6` |
| `outputs/qa/task2_causal_shift_fallback_20261111/result.json` | `eb6dc5bf74e48df1552d4e0a16a53542e28b8319d34594ff74ef9e122ec12148` |
| `outputs/task2_causal_shift_fallback_supervisor_20261111/producer_exit.json` | `8b71465fbe661ff4fcf750ec2a1d43188c38bcc1749f2e37d0eaea9671018f23` |
| `outputs/task2_causal_shift_fallback_supervisor_20261111/checker_exit.json` | `785e24f63d4d9d19b60ac15903bbbb56c622c90c48140b529b27efe428e9e232` |
