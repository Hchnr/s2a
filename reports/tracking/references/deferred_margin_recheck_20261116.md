# 一次 WAM 余量刷新后的当前状态重算

2026-11-16。按[预登记](../../../plans/v0.0.1/tracking/references/deferred_margin_recheck_20261116.md)，
从 bowl / seed43 的旧开发 reset 重算 16 周期并核对状态、动作、内步和 step8 /
step16 图像。继续消费 step16 当前图像生成的 WAM 参考，第一个续行动作与已
独立重放基线逐值一致。原基线在 step17 前因上一计划的预测触及余量而刷新。

本次仅将这一条 `quality_margin_consumed` 信号延期，以 step17 的实际状态和
同一已生成参考重新求解；原动作、关节、制动和独立 witness 审核仍生效。求解
得到双点 witness 并执行了 **1 个额外周期**；该周期之后在 step18 前再次触发
余量刷新，未派发下一动作。

| step17 额外动作 | 数值 |
| --- | ---: |
| 实测 TCP 位置 / 旋转误差 | `0.009399185 m / 0.036126174 rad` |
| 独立模型首点位置 / 旋转预测误差 | `0.009500000 m / 0.036035277 rad` |
| 该计划第二点位置 / 旋转预测误差 | `0.009500000 m / 0.047000000 rad` |
| 25 个内步最小关节余量 | `0.718759130 rad` |
| 新求解质量响应 / grasp / 任务成功 | `133 / 0 / 0` |

所有 18 个已执行端点均满足自身参考的位姿限值；额外动作的位置误差距原
`0.01 m` 门限约 `0.601 mm`。模型第二点仍触及刷新边界。此次明确更改了一次
刷新时机，不能称原刷新规则未变、动作有普遍安全保证或已减少长期 WAM 请求；
后续可能生成的当前图像和参考与原运行不同。本次没有触发新的 WAM 请求，也没有
读取未来图像、物体真值或任务标签来选择动作。

## 独立重放与开销

producer 和 checker 各从 reset 执行 / 重放 18 控制周期、450 个 2 ms 内步，
各使用 3 个仿真模型及 3 个 OSMesa context；两个 worker 均正常退出、实际
wait/reap。checker 不加载延期逻辑或选择动作，动作、状态、TCP 误差、全部
内步 q / qvel / 时间及夹爪反馈最大差均为 0，任务与 grasp 标签一致；checker
保留 producer 的停止原因，不独立重算求解或刷新决定。

64 并发相关 CPU 回归 129 项通过、0 skip；零物理预检及全仓 Ruff、格式检查
通过。本次无新 WAM、GPU 或 fresh 初态。一个非盲短窗不能替代三任务双 seed
开发、原 18 条正确参考或 24 条最终留出验收，也不能证明任务成功和实时性。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `deferred_margin_recheck_20261116/dry.json` | `679bb9914bab6535b266592480941bdc38b6698604695513a29ad54e67ac6eab` |
| `deferred_margin_recheck_20261116/producer/result.json` | `b6d8b78360b489c13e024688064fa69aece32ae75d70f7cf425bd28e0b95cf42` |
| `deferred_margin_recheck_20261116/checker/result.json` | `0e1b8e061ff31b45f434c634d73dd4a0d0377856ff73b8be096ef9d9a40891c8` |
| `qa/deferred_margin_recheck_final_20261116/result.json` | `b94caeb343d5a2c9d62f86ac0dc279640ac55de4bf52c11c189aaa1f9c3b7a57` |

输出不随 Git 提交；收据保存原响应、图像、状态、源码和进程身份。
