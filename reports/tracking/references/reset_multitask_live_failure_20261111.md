# 三任务当前图像试验：第二次模型请求被错误拒绝

2026-11-11。依据[原预登记](../../../plans/v0.0.1/tracking/references/reset_multitask_live_48_20261111.md)，
三个旧开发初态同时启动。每个案例完成了复位后的原首窗 8 周期，以及一次来自 step8
当前图像的实际 WAM 响应所引导的 8 周期。三个案例的 step16 第二次模型请求均在进入
去噪前失败；因此本次 48 周期试验未完成，不能作为多窗口稳定或最终任务资格证据。

## 原因和实际消耗

模型 worker 将 `BoundaryCounter` 的**全运行累计预算**错误设为 35，而预登记每案例
允许 5 次请求、每次 35 步，共 175 步。首个请求耗尽该计数器后，第二个请求的首个
去噪前向被拒绝。计数器的实际检查见
[`BoundaryCounter.forward`](../../../scripts/wam/readout/run_wam_readout_features.py)，
发生错误的 worker 原始版本保存在合并提交
[`77299f2b`](https://github.com/Hchnr/s2a/blob/77299f2b764d8e2b4eb0bc9b90e01292655e157b/scripts/simulation/runtime/reset_multitask_live_workers.py)，
文件 SHA-256 为 `e88e17a1c03ecdbe61026fd6a45779bddb449ed7a54ac023d751eed1bd15ccfb`。
本次假模型测试未覆盖真实 `BoundaryCounter` 的跨请求全局预算，这是测试缺口。

| 项目 | 每案例 | 三案例合计 |
| --- | ---: | ---: |
| 实际完成控制周期 | 16 | 48 |
| 实际完成 2 ms 内环步 | 400 | 1,200 |
| 仿真模型 / OSMesa context | 3 / 3 | 9 / 9 |
| 模型初始化 | 1 | 3 |
| 模型请求尝试 / 完成 | 2 / 1 | 6 / 3 |
| 实际完成 denoise | 35 | 105 |

失败后每对 worker 均已等待和回收，三个 GPU 均已释放。模型 worker 在清理期间被其
owner 发送终止信号，记录未隐藏。原 producer 汇总只累计通过案例，因此其 `totals`
为零；上表从失败案例的实际 runtime、模型和生命周期记录重新核算，不将未通过案例
算作零消耗。本次[只读审计](../../../scripts/simulation/runtime/audit_reset_multitask_live_failure.py)
结果 `outputs/reset_multitask_live_failure_audit_20261111/result.json` SHA-256：
`e6aa8c57a9ad299b71f7b49d0330a65b525de5247dbc41e321d04417c07e7937`。
审计本身没有创建仿真模型、执行物理步或请求模型。

## 首个当前图像窗口的生产者记录

以下为各案例 step8–15 的实际生产者记录；只读审计核对了当前图像、保存的真实模型
响应、实际状态哈希及每周期 25 个内环样本。本轮未进行独立物理重放，不能把这些
一致性检查写成重放通过。

| 任务、旧初态、seed | pose 限值通过 | 位置 RMSE (m) | 旋转 RMSE (rad) | 最小关节内环余量 (rad) | 目标 grasp / 任务成功周期 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bowl on plate，init0，42 | 8/8 | 0.000831 | 0.004603 | 0.620099 | 0 / 0 |
| wine bottle on rack，init0，42 | 8/8 | 0.000647 | 0.005073 | 0.668228 | 0 / 0 |
| cream cheese in bowl，init0，42 | 8/8 | 0.003014 | 0.006869 | 0.596358 | 0 / 0 |

原始 producer：`outputs/reset_multitask_live_48_20261111/producer/result.json`，SHA-256
`6485e226e9ff231dce9fcffdce5fa26cce5c7332f2c18fc380be539e24aab5f3`。自动重试为 false。
后续修复必须分别测试每次 35、全程最多 175 的真实边界计数器；若重新试验，须另行
预登记新输出、累计已消耗预算，并保留本次失败及三案例分母。
