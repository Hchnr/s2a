# 固定实测状态的质量求解耗时剖析

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/quality_solver_runtime_profile_20261111.md)，
使用同一task1 seed42开发记录的step352、355、359：每点只读取当时实测状态、上一
已完成命令、当时保存的8步seed及step352已生成参考。`initial_joint`取已重放runtime
记录；原始内环误差历史不可得，因此使用固定零误差，仅用于离线耗时诊断，不能授予
实际控制或安全资格。这三个时刻相关，不代表其他任务或未见初态。

| 原记录step | 不带profiler求解墙钟 | 带profiler求解墙钟 | 独立审核墙钟 | 每次响应数 | 两次结果最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 352 | 1.122 s | 1.373 s | 0.0216 s | 83 | 0 |
| 355 | 1.072 s | 1.337 s | 0.0210 s | 84 | 0 |
| 359 | 1.124 s | 1.375 s | 0.0216 s | 86 | 0 |

三个状态各两次完整求解均得到相同model witness，三次独立审核通过；模型构造各约
`0.0011–0.0012 s`。witness属于固定局部模型，不证明该时刻真实物理安全或任务成功。
每个worker设置数值库单线程环境；首次不带profiler求解记录的进程CPU时间为
`1.70–1.75 s`，大于对应墙钟，进程CPU时间可累加多个线程，现有证据不足以确定
线程来源。共享机器负载及profiler额外开销影响绝对耗时，不把这些数值当完整控制
周期的实时基准。

## 可见热点与下一步

带profiler的单次求解中，`_minimize_slsqp` 自身时间占总self time约 `20.9%–24.1%`；
`rollout_joint_preview` 自身时间约 `21.6%–22.6%`，分别调用83、84、86次。
`preview_objective` 的累计时间约 `0.496–0.519 s`，其中包括重复几何计算；SLSQP
累计时间包含其调用的展开和目标函数，不能把累计时间简单相加或将其等同无profiler
耗时。单独删减独立审核不足以跨越当前约1秒与50 ms的差距，而且不能因此降低审核。

下一步可先对不触及关节投影的25个内环递推设计等价批量计算，并与原逐步实现及
独立审核逐项比较；浮点顺序可能改变边界或优化选择，必须保留原约束、响应预算、
失败/无witness处理，不能仅以数值接近推断相同物理行为。SLSQP本身也是显著耗时，
仅优化展开未必达到20 Hz；改变求解停止规则或选择逻辑须另作候选并完整对照。

## 复核与费用

三个CPU worker均实际wait/reap、exit0且未发信号；监督进程等待并回收主执行器。
源码、参数、输入及资产在执行前后一致。共完成6次求解、506次有限响应、3次独立
witness审核；无新物理步、WAM、GPU、拟合或final fresh。55项32并发CPU专项、零
模型dry及全仓Ruff/2783文件格式检查通过；未重跑全量pytest。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/quality_solver_runtime_profile_20261111/result.json` | `12271b66e24e546702626bc884e0be294571d3411415312e4151c2e2b7952d0f` |
| `outputs/quality_solver_runtime_profile_20261111/step_352/result.json` | `6a525cfbda1126f42763abd0df0b582dc60d49b5bc1c7e03910e8c1279960a9d` |
| `outputs/quality_solver_runtime_profile_20261111/step_355/result.json` | `b207aee2e02d6217a38a51b8615a71232239da4e2d5d5e46bb09592507b1f916` |
| `outputs/quality_solver_runtime_profile_20261111/step_359/result.json` | `da7cec2eb1c956979822282aec16d8f88a6d09052a0c014ca6356cf78269f86b` |
| `outputs/quality_solver_runtime_profile_dry_20261111/result.json` | `d13960a4c249a1eae7b472019e930323a024abcd59a5667d34cc9148d3824406` |
| `outputs/qa/quality_solver_runtime_profile_20261111/result.json` | `fb4460816602e74f6cd71d7c5a21ab9a01e1ee777ca200a04de2a99f6c8ac471` |
| `outputs/quality_solver_runtime_profile_supervisor_20261111/owner_exit.json` | `d4fc7858cb93fb7526cad00fe0f4b73588c768bdc44f1def0027e32664f950df` |
