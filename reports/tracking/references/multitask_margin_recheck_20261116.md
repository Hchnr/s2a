# 三任务首次 WAM 刷新点的当前状态重算

2026-11-16。按[预登记](../../../plans/v0.0.1/tracking/references/multitask_margin_recheck_20261116.md)，
从六条既有 WAM 开发轨迹的 reset 开始，重算各自首次质量刷新前缀，并在允许的
`quality_margin_consumed` 信号处，最多使用当前实测状态和已生成参考再执行
一个周期。起点由旧日志事后确定，属于非盲诊断。没有审核第二点的信号不延期。

六例中五例完成并独立重放，wine / seed42 在到达预定的 step143 之前失败；
**整体审计未通过，分母仍为六例**。以下四个额外动作均满足该周期原位姿门限，
不代表长期安全、减少 WAM 请求或完成任务。

| 旧开发案例 | 首次信号 | 结果 | 额外动作实测 TCP 位置 / 旋转误差 | 额外动作内步最小关节余量 |
| --- | --- | --- | --- | --- |
| bowl / seed42 | step26，余量耗尽 | 额外 1 步，step27 停止 | `0.009444808 m / 0.041574160 rad` | `0.608167741 rad` |
| bowl / seed43 | step17，余量耗尽 | 额外 1 步，step18 停止 | `0.009399185 m / 0.036126174 rad` | `0.718759130 rad` |
| wine / seed42 | step143，余量耗尽 | step104 图像或端点身份校验失败，未到测试点 | 未执行 | 未执行 |
| wine / seed43 | step17，缺少已审核第二点 | 不延期、不派发下一动作 | 未执行 | 未执行 |
| cream / seed42 | step17，余量耗尽 | 额外 1 步，step18 停止 | `0.009274663 m / 0.044932612 rad` | `0.646512016 rad` |
| cream / seed43 | step17，余量耗尽 | 额外 1 步，step18 停止 | `0.009091110 m / 0.011802745 rad` | `0.734500651 rad` |

四个额外周期的内步均未低于原关节缓冲，均未记录目标 grasp 或任务成功；按
计划执行一步后即停止，未评估后续刷新是否仍必要。bowl / seed43 是此前
[单案例对照](deferred_margin_recheck_20261116.md)的同一旧开发状态，不能视为
独立新样本。

## 失败与可复现性

wine / seed42 的 producer 在重算完 step0–103 的 104 个控制周期后，step104
的身份比较抛出 `The independent current image or endpoint differs`，没有执行
预定的 step143 探针，也没有自动重试。此前最后完成端点的状态 SHA256 与原
step104 记录一致；新旧保存 PNG 的 SHA256 同为
`a09617202d6000f7e263144b4b618e78caf935499b77d4b9d069122fcd042643`，
解码像素 SHA256 也与原记录一致。这些事后检查不足以证明全部比较字段一致：
异常发生在实际 capture 元数据写入收据之前，因此目前**无法确定具体差异字段**。
不能据此放宽身份校验或认定该例通过。失败收据、日志和原开销均保留。

五个完成例的 checker 从 reset 重放，已执行动作、状态、TCP、全部内步 q / qvel /
时间及夹爪反馈与 producer 的最大差均为 0，前缀与原记录的对应差也为 0。
checker 不加载候选、不重新求解或重新判断 producer 的停止原因。原六条 WAM
运行中的历史 postflight 事件和后续重放记录仍以
[原报告](reset_multitask_adaptive_complete_20261113.md)为准。

## 开销与验证

producer 六个 worker 共实际执行 202 个控制周期 / 5,050 个 2 ms 内步，包含失败
例的 104 / 2,600；共构造 18 个模型和 18 个 OSMesa context。checker 只重放
五个完成例，共 98 / 2,450，使用 15 个模型和 15 个 context；失败例记为
`not_run`，不从六例分母排除。全部 11 个 worker 实际 wait/reap，均完成资源
清理。owner 的已完成例汇总仍原样标记失败例为 `unaccounted`，上述失败开销
来自该例独立的部分收据。没有新 WAM、GPU 或 fresh 初态。

64 并发相关 CPU 回归 128 项通过、0 skip；零物理预检及 Ruff 检查通过。
本试验不替代完整 WAM 开发矩阵、正确参考资格或 24 条最终留出验收。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `multitask_margin_recheck_dry_20261116.json` | `505719787a49135af777380093348066235422e4229971c81e47892850f1a4a6` |
| `multitask_margin_recheck_20261116/producer/result.json` | `ff66ea315e9f8f8e7542669b175058dd025b692e9e798ea0ab7002c80405bd72` |
| `multitask_margin_recheck_20261116/producer/case04/worker/result.json` | `66c786de10e3105988ae545e09ffff0c2e71ff07594bc387106600c50dc5676d` |
| `multitask_margin_recheck_20261116/checker/result.json` | `02cc7d2fcbaf6df76ba7e8df12c61971099ca5fe4e9923a5795cb6288340b024` |
| `multitask_margin_recheck_20261116/audit.json` | `3a2be0b1c1f65cb0ce0d3b9ce93684c4634b64fd59dfd3b05d03581a59d1bba9` |
| `qa/multitask_margin_recheck_final_20261116/result.json` | `f50607821b131f85c1531bd732dcdc423766f0e7dc8459a0c837d51f3dfa2146` |

输出不随 Git 提交；收据保存来源、已执行轨迹和进程身份。
