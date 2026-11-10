# 原安全约束内的可行初值恢复反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_feasible_start_restoration_20261111.md)，
对[首点硬恢复](task2_first_hard_restoration_20261111.md)使用相同task2 raw/terminal058
开发起点、已完成前缀和当前chunk；当原细化种子不满足原线性安全条件时，先用L1 LP
构造满足原动作、关节、终端及制动约束的初值，再在原320响应预算内求解。首点质量
及独立审核没有放宽，分支后仅使用自身实际状态和历史。

| 开发起点 | 可行初值分支执行 / 实际双阈值 | 实际最大位置 / SO(3)误差 | 内环最小余量 rad | 停止 |
| --- | --- | --- | ---: | --- |
| raw step32 | 1 / 0/1 | 0.01009471 m / 0.02147605 rad | 0.461285 | step33前无witness |
| terminal058 step80 | 1 / 1/1 | 0.00530873 m / 0.03879719 rad | 0.327534 | step81前无witness |

两个已执行周期没有2 ms关节缓冲失守、越界、目标grasp或任务成功；两条均不足
八周期。前一次首点硬恢复也在同一动作编号前停止，但分支首动作有微小差异，
不能把后续状态当成逐位相同；本次没有改善已执行长度或实际双阈值结果。

## 可行初值解决了什么

四次决策的原细化种子均未通过原线性安全条件；每次一条L1 LP生成的初值均通过
原box、变化率、关节/制动、终端和非零旋转检查，无需旋转latch修复。上次两处
第二决策的320个响应中均未保留关节安全候选；本次对应决策分别保留6个和1个
线性安全候选，但没有一个同时满足首点质量。记录的最近安全尝试分别为：

| 停止状态 | 最近安全尝试的模型首点位置 / SO(3)误差 | 最坏归一化首点超限 |
| --- | --- | ---: |
| raw step33 | 0.013878 m / 0.054258 rad | 0.387837 |
| terminal058 step81 | 0.006156 m / 0.070740 rad | 0.414797 |

这表明初值改进了有限搜索中安全候选的覆盖，但本次仍未找到首点质量witness；
最近尝试不是全局下界或可派发动作，模型也不能证明真实物理不可达。两次停止均
为320响应预算耗尽，异常退出时迭代数未保存，不能把记录值0解释为零迭代。

## 独立复核与费用

checker未导入候选控制策略，重放2条已派发动作；完整状态、50个2 ms内环q/qvel/time
及实际TCP误差的最大差均为0，grasp/任务标签一致。无witness停止仅核对位置和
未派发，checker未重求解。两阶段各2个CPU/OSMesa worker均实际wait/reap、exit0
且未发信号；监督进程实际等待并回收两个执行器，源码和输入前后一致。

每phase实际114控制周期、2850个2 ms步、6模型和6软件context；两phase合计
228周期、5700内步、12模型和12 context。producer实际4次原细化/283响应、4次
恢复求解/775个实际模型响应、4次初值LP、4次原求解器线性资格LP和2次独立审核
计400端点查询/5200模型转换步；4次种子rollout复用避免额外响应，单次未超过320。
两条已派发决策约2.69/3.55秒，仿真等待时物理暂停，不构成20 Hz实时资格。

73项32并发CPU专项、全仓Ruff/格式和零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出数据。原18正确参考、12开发和24最终留出标准保持；
`complete/passed`只表示执行与重放闭合。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_feasible_start_restoration_20261111/producer/result.json` | `6cacd8be2eb60dd50635613a02a4fc66500b4fe375fa24642382c48e9a52d142` |
| `outputs/task2_feasible_start_restoration_20261111/checker/result.json` | `b4e4cdd8e86d5dd1c21048792640c0fe7105ff43762827a442e7cdbbb1e42337` |
| `outputs/task2_feasible_start_restoration_dry_20261111/result.json` | `ee63e42eaaeececf0ecc4e1ecd1e3729a24a1425e350a7bd54aeca8d23ef4cdb` |
| `outputs/qa/task2_feasible_start_restoration_20261111/result.json` | `5ec40e1ccc3b4610448795870dee39cc7d63de95753e7f4232f4e01240a9c09d` |
| `outputs/task2_feasible_start_restoration_supervisor_20261111/producer_exit.json` | `911bc22128ed6bbd1e6d19bd67261c165c0ee4a89aff4af712dd5a6ece78d587` |
| `outputs/task2_feasible_start_restoration_supervisor_20261111/checker_exit.json` | `c818295116e6c2a0997352b63ba4cf71f3989b76889af3f40f67480a701f4` |

下一步需要在可行起点上优先解决首点位姿，而非直接最小化未来最坏缺口；若获得
独立审核通过的候选，再用其自身因果历史做物理反馈，继续保留实际时序和原门槛。
