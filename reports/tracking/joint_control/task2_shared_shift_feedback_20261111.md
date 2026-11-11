# 共享预算双点优先及已完成计划首点回退反馈

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/task2_shared_shift_feedback_20261111.md)，
在同一task2开发初态与seed43下，从raw step32和terminal058 step80执行余量双点、
原门双点、已完成计划首点的三级因果反馈。前两段正常无候选且共享预算有余时，才
检查本分支最后一条实际完成计划的后移结果；首点和原八周期模型安全独立审核后派发。

| 开发起点 | 原guarded双点 | 旧首点回退 | 共享预算双点 | 本次：实际达标 / 执行 | 本次最坏位置 / SO(3) | 最小内环余量 | 停止 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| raw step32 | 1/1 | 1/1 | 1/1 | **1/1** | 0.008680 m / 0.014258 rad | 0.449920 rad | step33前无合格首点 |
| terminal058 step80 | 4/4 | 5/5 | 4/4 | **5/5** | 0.009208 m / 0.046803 rad | 0.111189 rad | step85前无合格首点 |

raw首周期的命令、实际状态、TCP误差及25个2 ms内环样本与[共享预算双点](task2_margin_guarded_feedback_20261111.md)
逐值一致。step33两段搜索均无候选，后移计划通过模型安全检查，但首点预测为
0.010372 m / 0.052042 rad，超过原门，因此没有派发。

terminal058前4周期与共享预算双点的命令、状态、TCP误差及100个内环样本逐值一致。
step84的两段双点搜索无候选；后移计划模型首点为0.009064 m / 0.043876 rad，经
独立首点及八周期box、变化率、关节缓冲、终端和0.1 s制动审核后派发。该周期实际
误差为**0.008749 m / 0.043232 rad**，25个内环样本最小关节余量0.111189 rad。
模型第二点为0.013067 m / 0.064304 rad，八点最坏归一化质量缺口1.524670；它只
是首点质量候选，不是双点或八点质量witness。step85的后移计划虽通过模型安全检查，
首点预测为0.012794 m / 0.061909 rad，故停止且未派发。

相对共享预算双点的terminal 4/4，本次延长一周期，同时保留raw的1/1；此前
[原双点加首点回退](task2_causal_shift_fallback_20261111.md)也达到terminal 5/5，
本次没有超过其执行长度。两条terminal轨迹从首动作起不同：旧分支5周期最坏误差
0.009656 m / 0.049547 rad，本次为0.009208 m / 0.046803 rad；这是同一开发初态
的配对观察，不是统计或跨初态收益。两分支均无目标grasp、任务成功或完整8周期；
全部150个新增动作内环样本均未出现0.05 rad关节缓冲失守或越界。

## 独立重放与费用

checker未导入候选策略，6条动作、完整状态、150个2 ms内环样本、TCP误差、关节
余量、grasp及任务标签与producer逐值一致；停止位置和未派发也一致，未重新求解
无候选原因。每phase实际118控制周期/2950内步、6模型及6个OSMesa context；两phase
合计236周期/5900内步、12模型及12 context。producer实际8次原细化/566响应、8次
质量决策/783响应（余量搜索531、原门回退249、后移rollout 3），8次初值LP且回退
没有新增LP。4个余量双点、1个原门双点和1个首点候选均独立审核，共1200端点查询/
15600模型转换步；单次响应最多158，不超过320上限。step84为69+55+1=125响应。
已派发决策约2.52–3.54秒，仿真等待时物理暂停，不具20 Hz实时资格。

四个worker和两个监督进程均实际wait/reap、exit0、未发信号；输入与源码前后一致，
执行时HEAD为`ac9aa8886fbf8b67cfac4e6647f3fea0f5d4dbb5`。66项32并发CPU专项、
全仓Ruff/格式及零模型dry通过。此前64并发全量在LIBERO环境为6081通过、2跳过、
4项数值失败及5个收集错误；均在同环境的main基线复现，相关40项在开发环境通过，
不能称该全量运行通过。本轮无新WAM、GPU或最终留出；`complete/passed`只表示
执行与重放闭合，不表示完整控制或任务资格。原18正确参考、12开发和24最终留出
门槛保持；下一步需处理更早的参考衔接、跨任务和完整轨迹，而非只延长此单一样本。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_shared_shift_feedback_20261111/producer/result.json` | `5c8fbc65c947aabf5381df51b06c1443a7c0c3bc1cdf780f7d4b23b581fbd824` |
| `outputs/task2_shared_shift_feedback_20261111/checker/result.json` | `85ceb3299738b6cac47289984e9099423567e66b89d979d7ce160510fdae28dd` |
| `outputs/task2_shared_shift_feedback_20261111/producer/workers.json` | `153c7aae5cb0238aeb0e53bd6a3e52946a9d2e85d548b3a745afcfd73a422d7a` |
| `outputs/task2_shared_shift_feedback_20261111/checker/workers.json` | `fb9850c8a6069e6176d9a866593bf0e5b035415b3962d4de52ef500e63a34e28` |
| `outputs/task2_shared_shift_feedback_dry_20261111/result.json` | `2a67a4e881c450c26bce67ca12c67f7a72f594caa33c1b68c849ef22ec40b13c` |
| `outputs/qa/task2_shared_shift_feedback_20261111/result.json` | `8a6dc362a75e92730bf4c0aabca297c8dee79daaec104d77fa07b1d925b176c4` |
| `outputs/task2_shared_shift_feedback_supervisor_20261111/producer_exit.json` | `8ab768b13754c0d48bdb59f01379fb21a6734490862266c130dd2eed7f8aa7b8` |
| `outputs/task2_shared_shift_feedback_supervisor_20261111/checker_exit.json` | `4517a8f2e2b618cae4d6e2bf20773099961bb8518c4fc835b0ab14ac7958008f` |
