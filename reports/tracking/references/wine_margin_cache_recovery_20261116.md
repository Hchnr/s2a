# wine 首次刷新对照的端点缓存修复结果

2026-11-16。按[登记方案](../../../plans/v0.0.1/tracking/references/wine_margin_cache_recovery_20261116.md)，
让匹配前缀的周期端点和参考边界共用同一个严格 `SameStateEndpointCache`，对原
wine / seed42 从 reset 重新计算。step104 的端点按原状态与时钟先观测、再复用，
原图像 / 状态身份比较通过，克服了先前[第 104 周期失败](multitask_margin_recheck_20261116.md)。
这没有放宽哈希比较，也没有覆盖或改写原六例的失败收据。

producer 到达预登记的 step143 首次 `quality_margin_consumed` 刷新，暂缓该次
刷新后以当前实测状态重算；结果为 `quality_no_witness`、求解状态
`shared_search_ineligible`，因此在 step143 **未派发额外动作**。这是当前
受限求解未给出合格 witness 的结果，不证明目标在物理上不可达。

| 已执行的 143 周期 | 实测结果 |
| --- | ---: |
| 位姿门限通过 | `143/143` |
| 最后完成端点位置 / 旋转误差 | `0.009665573 m / 0.044964974 rad` |
| 全部 2 ms 内步最小关节余量 | `0.579096098 rad` |
| 目标 grasp / 原生任务成功端点 | `0 / 0` |
| 额外动作 | `0` |

原刷新信号对应的上一计划第二点预测误差为 `0.0095 m / 0.047 rad`。它是
预测量，并非第 143 周期已执行动作的误差。该结果表明解决身份重建后，当前
候选在这个 WAM 参考下仍未能继续；不能据此声称任务完成、长时跟踪改善或
达到了最终留出门槛。

## 独立重放、成本与来源

checker 从 reset 重放 producer 的 143 个已执行周期，不加载候选、不重新判断
producer 的停止原因。双方与原前缀、以及彼此的命令、状态、TCP 误差、全部
内步 q / qvel / 时间和夹爪反馈最大差均为 0；step104 的两次缓存事件均绑定
原状态 SHA256 `107d4d878bab324af3912f173e24fdb1df2c28bb423181c5d1b47f20b539429a`。
两阶段各使用 143 控制周期 / 3,575 内步、3 模型及 3 个 OSMesa context，两个
worker 均实际 wait/reap 并清理。没有新 WAM、GPU 或 fresh 初态。

此前原六例中的 wine 失败开销为 104 / 2,600；本轮两次字段诊断各 208 / 5,200，
本次复验另 286 / 7,150，均保留独立收据。原六例仍记五例完成、一例失败；
补充验证不能追认原实验通过。64 并发相关 CPU 回归 150 项通过、0 skip；
零物理预检、全仓 Ruff 与格式检查通过。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `wine_margin_cache_recovery_20261116/dry.json` | `02a52c03d3760562b879b72538412d9d1d1e2c830ea522e88146160217524372` |
| `wine_margin_cache_recovery_20261116/producer/result.json` | `e58773c78b5c0845cd802ce944473d0f14f176c360f51715c2042f8fde97c82e` |
| `wine_margin_cache_recovery_20261116/checker/result.json` | `725ef6af636221a70003aa712c2ea819676ec9bdac3b64d09ec961c7baa9cfe8` |
| `wine_margin_cache_recovery_20261116/producer/case04/worker/result.json` | `ab696e1fc17f5901834449a7b3977d7ac6947c8b691c00b78132e5999a35216c` |
| `wine_margin_cache_recovery_20261116/checker/case04/worker/result.json` | `9ea6c85b5a5a7591e486af07b141bbacd3b492ca2ce4a9a28ac6d4efe2ed9458` |
| `qa/wine_margin_cache_recovery_final_20261116/result.json` | `df6806d0bcab48688998c10fe15b7d712abbcefea07be6b032b4fc1e802bcef5` |

输出不随 Git 提交；收据保留来源、已执行轨迹和进程身份。
