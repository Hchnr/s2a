# step122 之前的因果偏差质量对照

2026-11-11，按[预登记](../../../plans/v0.0.1/tracking/joint_control/early_causal_innovation_quality_20261111.md)，
在 step122 派发前，使用 step121 已完成实测状态、上一命令、移位的已完成计划、
关节误差和 TCP 预测偏差；step120 的已生成参考 cursor 为 2。两个分支使用同一
冻结模型、种子、参考及原 box / 速率 / 关节 / 制动限制，模型 witness 均由另一套
逐内环计算独立审核。step122 的原轨迹物理结果只在两次求解后读取作为背景对照。

| 分支 | 模型原始首点位置 / 旋转误差 | 加 step121 偏差后的预测 | 独立核验 | 模型响应 |
| --- | --- | --- | --- | ---: |
| 原质量门 | 0.006651 m / 0.016362 rad | 0.012113 m / 0.019238 rad | 原质量及安全门通过；补偿位置超门 | 78 |
| 原门 + 因果偏差门 | 0.004563 m / 0.023914 rad | 0.010000000000003 m / 0.021883 rad | 原门、安全门和补偿门通过 | 73 |

新分支的补偿位置仅在原 `1e-10` m 数值容差内通过 0.01 m 门，没有实质裕量；
两个分支的独立关节响应与求解器预览最大差均不超过 `2.23e-16`。新分支首条
机械臂命令与原质量分支最大差约 0.2841（两条各自均通过相对上一命令的 0.25
速率约束），说明这是不同的模型候选，不是对同一物理动作的重新打分。

原已执行轨迹 step122 的模型预测为 0.006651 m / 0.016362 rad，实际为
0.013933 m / 0.022233 rad，实际位置超门；本次原质量分支首条命令与当时实际
命令最大差约 `6.76e-6`，因此该历史结果只能作背景，不能作为本次候选的精确
物理结果。新分支尚未派发，也没有物理闭环证据；过去偏差不保证未来响应。

实际一次模型初始化、一次静态 forward、两次优化和共 151 次模型响应；两次
独立审核共 400 端点查询、5,200 模型转换步。没有物理积分、WAM/GPU 调用或新
初态，子进程正常退出并实际 wait/reap；运行前 32 并发专项 45 项通过，无跳过
或收集错误。下一步须另行登记从该状态因果生成候选的物理对照与独立动作重放，
不能把单状态模型 witness 提升为实时或留出资格。

| 仓库内证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/early_causal_innovation_quality_dry_20261111/result.json) | `8c85a7b54bdfcba8925c83ce067b9e4b956eeb17b9b67af4aa058d6e907dce73` |
| [32 并发运行前专项](../../../outputs/qa/early_causal_innovation_quality_20261111/result.json) | `cb39842183f62c075b4488fba60dca9228986d276c8f5fc6424c599e861d1346` |
| [运行登记](../../../outputs/early_causal_innovation_quality_20261111/registration.json) | `c7e806b6973c314f16076bc7f70ab6d994647a85c72f5c6460c8a790b65e93c8` |
| [owner 结果](../../../outputs/early_causal_innovation_quality_20261111/result.json) | `747eae5b40e551e3e436b3266351e5f36d878cb480707ec94ff246868c140992` |
| [分支与独立审核](../../../outputs/early_causal_innovation_quality_20261111/worker/result.json) | `878a290e2b4bd7a1970c4a0cd1ba9785199696b74f62f965fcdc20d94521f9ee` |
| [进程收尾](../../../outputs/early_causal_innovation_quality_20261111/workers.json) | `75c9781fea4756e5a9a2bfec325095ead1e5eefcd005b7d0c7c9fbff3fd1c4d8` |
