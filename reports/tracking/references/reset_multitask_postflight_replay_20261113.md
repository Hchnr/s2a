# 六案例开发矩阵：首批四条的停止帧事件与独立重放

2026-11-13。六案例统一开发矩阵的首批四条真实 producer 均在质量检查无法找到
witness 时停止，仿真和模型 worker 正常退出并回收。原 owner 随后比较停止帧时，
错误地给模型收到的原始帧多包了一层 `frame`；因此四个原 owner 收据均为失败，
没有独立 checker 被原流程派发。这一失败及其成本不覆盖或追记为成功。

修正比较后，只读重新核验原记录及模型交换；按[重放计划](../../../plans/v0.0.1/tracking/references/reset_multitask_postflight_replay_20261113.md)
在独立 CPU/OSMesa 环境从相同 reset 重放已派发动作，不调用 WAM 或候选控制器。
该重放没有重新求解质量停止原因。剩余两个案例尚在原矩阵中运行，以下只报告
已结束的四条旧开发初态，不能作为六条完整结果或最终留出结果。

| 案例 | 完成周期 | 新模型请求 | pose 限值通过 | 位置 / 旋转 RMSE | 内环最小关节余量 | 目标 grasp / 任务成功 | 原停止 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| bowl，seed 42 | 133 | 46 | 133/133 | 0.004843 m / 0.021666 rad | 0.051591 rad | 0 / 0 | `quality_no_witness` |
| bowl，seed 43 | 25 | 5 | 25/25 | 0.004272 m / 0.021924 rad | 0.617736 rad | 0 / 0 | `quality_no_witness` |
| wine，seed 42 | 156 | 19 | 155/156 | 0.002439 m / 0.010704 rad | 0.579096 rad | 0 / 0 | `quality_no_witness` |
| wine，seed 43 | 45 | 9 | 45/45 | 0.004035 m / 0.019329 rad | 0.580791 rad | 0 / 0 | `quality_no_witness` |

四条 producer 共 359 周期 / 8,975 个 2 ms 内步；checker 从 reset 独立重放同样
359 周期 / 8,975 内步，重采全部 79 次请求的当前图像。逐周期命令、完整状态、
TCP 误差、内环关节位置/速度/时间及夹爪反馈最大绝对差均为 0，grasp 与任务
标签一致。原 wine/seed42 的 step 155 端点位置误差为 0.010901 m，超过 0.01 m
限值，不能把所有执行记为达标。
四次 producer 共初始化模型 4 次、完成 79 次请求 / 2,765 次 denoise；本次
checker 未加载候选或调用模型。producer 和 checker 各使用 12 个仿真模型与
12 个 OSMesa context；各 worker 已实际 wait/reap，未发信号。

对已记录响应的事后只读分析显示：79 条 H40 共生成 3,160 个夹爪端点，实际
消费 327 个。按 opening≤0.2 计，有 118 个强闭合端点，全部处于未消费后段，
分别来自 bowl/seed43 和 wine/seed42；wine/seed43 的已生成窗口没有该强闭合。
bowl/seed42 有 38 次因 `quality_margin_consumed` 提前重观测，46 个新参考的
实际消费长度中位数为 1。参考后段未派发不证明延长窗口一定安全或一定成功，
但为跨任务比较有限 H40 保留与质量门提供了具体动机。该分析针对已观察结果，
不是盲态试验。

## 修复与验证

原测试替身同样构造了多包一层的错误格式。用真实 producer/model 收据及
stop step、session、额外包装、未发送、序列完成状态的反例验证：修复前 3 项
失败 / 12 项通过；修复后 15 项通过。脚本和测试 Ruff 检查、格式检查通过。
两次 bootstrap 预检先后因隔离模块未找到及隔离工作区相对路径缺失失败，均在
派发物理动作之前；随后仅替换审核模块并使用原仓库运行时的预检通过。

隔离工作区 32 并发相关回归为 178 项通过、36 项失败、0 skip / 收集错误，全部
worker 已回收。26 项明确报严格环境工作目录不符；8 项在包含工作目录约束的模型
worker 预检失败，另 2 项为子进程通信 EOF / 等待超时，后两类原因尚待正式目录
复核。该次不记为通过；修复后的正式源码匹配 64 并发回归仍需在原矩阵结束后执行。

| 收据 | SHA-256 |
| --- | --- |
| `outputs/reset_multitask_adaptive_preliminary_20261113/result.json` | `7b2caee3292ef3d2726e701c4ba198b1d5e4758067071536e5550c2c278b7125` |
| `outputs/reset_multitask_adaptive_intent_preliminary_20261113/result.json` | `8786efb3c0ea77dfd11e0760c008d25fc0367bad94e29fa1634f9b2e65c6d88e` |
| `outputs/reset_multitask_postflight_replay_20261113/wave_00.json` | `f6a2771c04851320b11697b666bac7e307bea687a7afe90e41710c71de650150` |
| `outputs/reset_multitask_postflight_replay_20261113/wave_01.json` | `6690e84daf8a08c30bfc493f239ae607497e1a5cafb6a081746294c89ed09635` |
| `outputs/qa/adaptive_stop_fix_red_20261113.log` | `84fd35711eff4ca21d1122e69c87ae6b165641b74aa6e89eaff35869a6cddd3c` |
| `outputs/qa/adaptive_stop_fix_green_20261113.log` | `c8b7fcc8196405c5e278d356c106d92c28b404e13e9160c7a0618f22c869299a` |
| `outputs/worktrees/adaptive_stop_fix_20261113/outputs/qa/adaptive_stop_fix_32_20261113/result.json` | `24f636a71c2be6fc63df2b24c4b02db11ff385a1a55b2f80c9378281fea5331c` |

原结果保留。当前候选的完整正确参考正控、WAM 开发矩阵、24 条新初态及实时/真机
验收均未完成。
