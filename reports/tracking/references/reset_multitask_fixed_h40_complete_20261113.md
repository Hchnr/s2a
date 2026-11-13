# 三任务固定 H40 复位对照：六案例完整记录

2026-11-13。按[固定协议](../../../plans/v0.0.1/tracking/references/reset_multitask_fixed_h40_20261113.md)，
在六个旧开发案例从 reset 复现首 8 周期，使用该案例 step8 实际当前图像生成的
首条 H40，最多连续消费其 40 个端点；不新增 WAM 请求。全部六条均已从 reset
独立重放，前四条详细证据见[首批报告](reset_multitask_fixed_h40_first_wave_20261113.md)。

| 案例 | 实际周期 / 消费 | pose 限值通过 | 最小参考 / 实测 opening | 最小内环关节余量 | 目标 grasp / 任务成功 | 停止 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| bowl，seed 42 | 48 / 40 | 48/48 | 0.706206 / 0.766871 | 0.504932 rad | 0 / 0 | H40 完成 |
| bowl，seed 43 | 36 / 28 | 34/36 | 0.946010 / 0.948099 | 0.586177 rad | 0 / 0 | 质量无 witness |
| wine，seed 42 | 48 / 40 | 48/48 | 0.972432 / 0.965292 | 0.658262 rad | 0 / 0 | H40 完成 |
| wine，seed 43 | 48 / 40 | 48/48 | 0.953220 / 0.956777 | 0.405828 rad | 0 / 0 | H40 完成 |
| cream，seed 42 | 46 / 38 | 44/46 | 0.978443 / 0.965183 | 0.596358 rad | 0 / 0 | 质量无 witness |
| cream，seed 43 | 44 / 36 | 43/44 | 0.943695 / 0.948054 | 0.421690 rad | 0 / 0 | 质量无 witness |

六条均与原自适应分支的前 16 个实际周期逐值一致；后续参考更新协议不同，不能
把不同分支的误差当作单因素任务质量比较。固定 H40 的 270 个 producer 周期中，
265 个满足 pose 限值，三条在完整消费前质量停止；没有目标 grasp 或任务成功。
该短窗结果不支持仅靠延长参考窗口解决六案例任务，也不排除更长轨迹或其他统一
策略有效。原自适应 cream/seed42 后续出现 346 个 grasp，但它来自另一条运行
至 462 周期的参考流，不能把这一差异归因于单一变量。

producer 和 checker 各实际执行 270 周期 / 6,750 内步，各使用 18 个仿真模型和
18 个 OSMesa context。checker 重采 step8 图像、只重放实际动作，所有命令、状态、
TCP 误差、内步及夹爪反馈最大差均为 0；任务与 grasp 标签一致，质量停止原因未
独立重算。没有新增 WAM 初始化或请求，所有 worker 已实际 wait/reap、未发信号。
两个阶段合计 540 周期 / 13,500 内步；这些资源不包含原自适应矩阵的成本。

| 收据 | SHA-256 |
| --- | --- |
| `outputs/reset_multitask_fixed_h40_20261113/preflight_wave_01.json` | `79156abf5a755e7cb7cf8dbc192953997034f04e1f18df03d816e53dc9467e0e` |
| `outputs/reset_multitask_fixed_h40_20261113/producer_wave_01.json` | `74935448ae4582cd2df4cc9e7b47e22d357d8facc865cf5cc9cc87b725fe24ad` |
| `outputs/reset_multitask_fixed_h40_20261113/checker_wave_01.json` | `161a05b3e47949f27f8927a6d2fdeb682081f5f4152577728e4daffe61b3a392` |
| `outputs/reset_multitask_fixed_h40_20261113/summary_complete.json` | `419adb96938ca40ca072cb8712bb5088486449b8b18cb7255a35fd81511f009b` |

这是事后针对已见开发初态登记的短窗对照，不是最终 600 周期、正确参考正控、
24 条新初态、实时或真机验收。下一步应优先改善预测与实测误差的因果校正及
参考更新，同时保留原有安全和任务门槛，再用统一候选做长时程开发对照。
