# 保持首点硬约束的多点恢复反馈

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/task2_first_hard_restoration_20261111.md)，
新增首点硬约束的恢复搜索：首点位置和SO(3)在优化中不受未来slack放宽，候选
须满足原关节、动作、终端和制动约束并通过独立审核，未来点缺口只作为优化与诊断。
合成回归中，旧恢复搜索选出的首点失败候选被拒，而新搜索保留了通过原独立审核的
首点候选；这不单独证明此前物理实验的拒绝由同一原因导致。

| 相同task2开发起点 | 旧恢复执行 / 实际双阈值 | 首点硬恢复执行 / 实际双阈值 | 新候选内环最小余量 | 新候选停止 |
| --- | --- | --- | ---: | --- |
| raw step32 | 0 / — | 1 / 0/1 | 0.461285 rad | step33前无witness |
| terminal058 step80 | 1 / 1/1 | 1 / 1/1 | 0.327534 rad | step81前无witness |

raw新分支首点独立模型审核通过，实际位置/SO(3)误差为0.01009471 m / 0.02147606 rad，
位置比原阈值高约0.00009471 m；terminal058实际为0.00530873 m / 0.03879719 rad。
两个已执行周期全部2 ms内环均守0.05 rad缓冲、没有越界、目标grasp或任务成功。
两个变体属于同一开发初态和seed43；两条分支都未完成八周期，不能称短窗通过。

## 搜索与审核记录

首个决策的独立审核均通过，raw/terminal分别使用64/79次模型响应；候选的未来
最坏归一化质量缺口分别约0.58089/1.27246，并非八点硬质量witness。第二次决策
均耗尽320响应，已评估缓存中满足原线性关节/动作/制动/终端约束的候选数均为0，
故无首点witness可审核。`response_budget`异常中断优化后，当前实现未记录实际
迭代数；结果中的0不是零迭代的证据。

从[求解器流程](../../../src/s2a/research/trajectory_search/first_hard_restoration.py)可见，
两个搜索能进入非线性阶段说明线性约束LP已返回可行，但优化仍从原细化种子开始，
没有将LP解作为已评估的起点。这提示应研究可执行的线性可行初值和原生旋转latch
约束；LP可行不证明非线性首点质量、物理安全或实际可执行性，也不证明只改初值
必然成功。有限预算失败不证明全局无解。

## 独立复核与费用

checker未导入候选控制策略，重放2条已派发动作；完整状态、50个2 ms内环q/qvel/time
及实际TCP误差的最大差均为0，grasp/任务标签一致。无witness只核对停止位置和
未派发，checker未重新求解。两阶段各2个CPU/OSMesa worker均实际wait/reap、
exit0且未发信号；监督进程实际等待并回收两个执行器，源码及输入前后一致。

每phase实际114个控制周期、2850个2 ms步、6模型和6软件context；两phase合计
228周期、5700内步、12模型和12 context。producer实际4次原细化/283响应、4次
首点硬恢复求解/783响应、2次独立witness审核计400端点查询和5200模型转换步；
checker没有选择动作或求解。两条已派发决策分别约2.73/3.67秒，仿真等待时物理
暂停，不构成20 Hz实时资格。

76项32并发CPU专项、全仓Ruff/格式及零模型dry通过；本轮未重跑全量pytest。无新
WAM、GPU或最终留出数据。原18正确参考、12开发与24最终留出门槛保持；
`complete/passed`只表示执行与重放闭合。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/task2_first_hard_restoration_20261111/producer/result.json` | `b267d5920fbf6b1cf69496314b6ea747754a3cec109c5401efa7388ffa91f8f9` |
| `outputs/task2_first_hard_restoration_20261111/checker/result.json` | `cbe7ff1e9157da7556c14697db2ec0382b3c890a85822c091b58ea830bc4bdad` |
| `outputs/task2_first_hard_restoration_dry_20261111/result.json` | `e43c59cff1b3aa0c0ff55be1264850cc8c02c38656a8cb2b1936fcf07db0a086` |
| `outputs/qa/task2_first_hard_restoration_20261111/result.json` | `cb71b5ca4750db997c6d5521e38373d1deaf617c31da392cc2cd0ecb7d520d5c` |
| `outputs/task2_first_hard_restoration_supervisor_20261111/producer_exit.json` | `6e09c0d4a30a18094de915e82e81fd65278e6bce687845c2b5b385ef7bfe79e0` |
| `outputs/task2_first_hard_restoration_supervisor_20261111/checker_exit.json` | `b51c40cf0c5c04da0592b2bfec1e0bd8d4eccbf064d88f7ba6aee84f31452cd6` |
