# wine / seed42 的 step104 身份字段诊断

2026-11-16。按[修复验证登记](../../../plans/v0.0.1/tracking/references/wine_capture_identity_recovery_20261116.md)，
在受控 runtime 目录保存完整 capture 元数据；两条路径均完成 104 个控制周期，
随后采集 step104 图像并停止，没有派发 step104 动作或读取该步模型响应。
producer 重算动作，checker 只重放记录动作；两者全部动作、状态、TCP、内步和
夹爪与原前缀及彼此的最大差均为 0。此前[输出路径失败](wine_capture_identity_incident_20261116.md)
仍保留原结果与开销。

本次两路径重现相同的原身份校验失败：

| 字段 | 原 step104 | 新采集 step104 |
| --- | --- | --- |
| 完成 step103 动作后的状态 SHA256 | `107d4d878bab324af3912f173e24fdb1df2c28bb423181c5d1b47f20b539429a` | 相同 |
| 下一次端点读取后的状态 SHA256 | `107d4d878bab324af3912f173e24fdb1df2c28bb423181c5d1b47f20b539429a` | `2080cb041a685aacc8cb595d4c5373ee9f41f46ca34baf6be7b34720258fa161` |
| 原比较器不同的字段 | — | `current_endpoint.sim_state_sha256`、`render_before.sim_state_sha256`、`render_after.sim_state_sha256` |
| PNG SHA256 / 时钟 / pose / 开度 | 与原记录相同 | 与原记录相同 |

step104 端点采样前后的原始状态向量没有在本次 capture 中另存，因此不能从
哈希推断具体变化元素或数值大小。`sim.forward()` 是当前观测路径的一部分，
但本次没有对该调用逐项插桩，不能断言其是唯一根因。

源码检查发现原自适应运行把同一 `SameStateEndpointCache` 用于每个已完成端点
和后续采图；新的匹配前缀运行仅在参考边界使用缓存，周期内使用原始 endpoint
回调，因而边界上会再读取一次端点。复用已完成端点是明确待验证的修复方向；
仍须检查缓存的原始状态 / 时钟完全未变化，不能放松哈希或参考来源校验。
这项发现也不授予先前六例整体通过或新候选控制资格。

## 验证与开销

两条恢复路径各 104 周期 / 2,600 内步，各构造 3 个模型与 3 个 OSMesa context；
两个 worker 实际 wait/reap，资源均清理。`result.json` 的 `passed=true` 仅表示
诊断收集与两条重放完成；原 capture 身份比较在两条中均为 `false`。
本次 208 / 5,200 加此前路径错误的 208 / 5,200，诊断累计 416 / 10,400；
更早六例中 wine 失败的 104 / 2,600 另计。无新 WAM、GPU 或 fresh 初态。

修复后 64 并发相关 CPU 回归 137 项通过、0 skip，受控写入回归、零物理预检、
全仓 Ruff 和格式检查通过。测试不能代替目标任务成功或最终留出验收。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `wine_capture_identity_recovery_20261116/result.json` | `eba8ae3c31f8aaad77067430212124da2c4a4205a36b51963508b2fc20e5c62c` |
| `wine_capture_identity_recovery_20261116/producer/worker/result.json` | `923ef49c9f0fc6527b9ae0dd75771e27ae72d842bf803ede92ee022ada75e431` |
| `wine_capture_identity_recovery_20261116/checker/worker/result.json` | `271d23ae75fa14f38459ca7008e90d08661945e82a39d675ec67798875900faf` |
| `wine_capture_identity_recovery_20261116/producer/worker/runtime/capture104.json` | `eaa8a8c39573b1accbcce2edd732cd6ebfa681c91b8560e1000e3f3f77bb7858` |
| `wine_capture_identity_recovery_20261116/checker/worker/runtime/capture104.json` | `77d9a6d9282e657d6980f0f57d7905f82abc686efeba156c0af5cd3ad6cda407` |
| `qa/wine_capture_identity_recovery_final_20261116/result.json` | `b0f9cc1c8f3d79cbecb04852186375cc6895861bdffd5e2d47f3c96986942ae2` |

输出不随 Git 提交；完整收据保留来源、已执行轨迹与进程身份。
