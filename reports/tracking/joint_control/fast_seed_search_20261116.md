# 优先审核原 seed 的固定状态对照

2026-11-16。按[登记](../../../plans/v0.0.1/tracking/joint_control/fast_seed_search_20261116.md)
对同一 WAM 开发记录的三个保存状态，分别运行原缓存 seed 搜索与新 seed-first
路径各两次，交错先后顺序。**六对选中计划和完整模型展开最大差均为 0，
12 次原独立双端点 witness 审核均通过。** 这只是固定状态局部模型对照。

| 保存 step | 原搜索墙钟，两次 | seed-first 墙钟，两次 | 原 / 新每次模型响应 | 成对展开最大差 |
| --- | --- | --- | --- | ---: |
| 352 | 75.247 / 73.291 ms | 9.863 / 9.992 ms | 4 / 1 | 0 |
| 355 | 33.570 / 33.611 ms | 9.959 / 10.092 ms | 1 / 1 | 0 |
| 359 | 33.652 / 32.353 ms | 9.973 / 9.972 ms | 1 / 1 | 0 |

每组新路径均在 seed 原有 joint、braking、rate、terminal 和双端点 margin
检查通过后提前返回；返回计划仍经过原独立审核。新路径没有运行完整排序，
因此其与最优排序的 future slack 差异为**未知**，没有把它记为零。合成测试
另覆盖 seed 不合格时沿原搜索回退，确认不增加模型响应；这不是上述三个
保存状态的实际回退证据。

墙钟只计表内搜索函数，不含前置 preview、独立 witness 审核、夹爪、物理
和端到端开销；三个相邻状态共享来源且在共享机器上并行执行，不能代表全
轨迹、未见初态或 20 Hz 实时性。历史内环误差未保存，本次固定为零，不得
将这三点的模型 witness 当成真实物理安全或任务成功。

## 生命周期与验证

3 个 CPU worker 全部正常退出并实际 wait/reap；共 12 次求解、18 次局部
模型响应和 12 次独立审核，源码和保存输入前后一致。32 并发修复回归 15 项、
64 并发相关回归 98 项通过，0 跳过；全仓 Ruff lint 与格式检查通过。执行
源码为 `38fce706`。没有新物理、WAM/GPU、fit 或 fresh，
`physical_qualification=false`。

首次零求解 dry 因元数据字段重名退出，0 求解和 0 物理；原失败保留于
[事件记录](fast_seed_search_dry_incident_20261116.md)，后续按修复登记使用
独立 QA 与 dry 路径，未覆盖首次结果。

| 本仓库 `outputs/` 路径 | SHA-256 |
| --- | --- |
| `fast_seed_search_dry_v2_20261116.json` | `f7ee96925f158c2519854a078218f96a78811050ad27a79485b60abf0a50a083` |
| `fast_seed_search_20261116/result.json` | `4f45d867b1a48cc23c2d0fd0a34365d7c9b9fc5a8e31c0b27524744238030387` |
| `fast_seed_search_20261116/workers.json` | `68195976a7c6f918495b9661871201f223de0d631488385780d4664be0825682` |
| `fast_seed_search_20261116/step_352/result.json` | `2336260429c938ca5e96e6194ac45e5931231bc2b9d07dc7b2ac213431e9fcbc` |
| `fast_seed_search_20261116/step_355/result.json` | `1e104045f8500359f6ce8290452059dc12609aa2d3c0f2ce528dba2cdddc5613` |
| `fast_seed_search_20261116/step_359/result.json` | `1ec4b8133ba8abf01b74b2b27cee4f50b8982cd38832eb63320687b133dfcde9` |
| `qa/fast_seed_search_final_v2_20261116/result.json` | `a5a37c5a4f286134c22cb1a29ae7ad09634674c1df65d341985db33574570b8a` |

下一步须从原 reset 对统一候选执行完整 18 条正确参考和独立动作重放，逐条
比较已执行动作、停止和任务结果，并测完整决策时间；先前 16/18 的结果不能
自动转移到此实现。WAM 开发与 24 条最终留出验收也仍未完成。
