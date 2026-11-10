# 两处闭合影子机会的原生命令前缀重放

2026-11-10。按[固定方案](../../../plans/v0.0.1/tracking/joint_control/causal_event_prefix_replay_20261110.md)，
从各自原始状态和seed42，在CPU/OSMesa中重放两处首次闭合影子机会之前的原始7维
命令。该阶段没有使用新参考选择器生成动作，也没有进行候选控制对照。

| 开发案例 | 保存命令周期 | 核对端点 / 状态字节一致 | 最大完整状态差 | 最大已保存控制缓存差 |
| --- | ---: | ---: | ---: | ---: |
| task0 s42 raw，终点136 | 136 | 137 / 137 | 0 | 0 |
| task1 s42 terminal058，终点360 | 360 | 361 / 361 | 0 | 0 |

两个端点的仿真状态、时间、TCP、已保存控制器和夹爪缓存、任务标签均按原合同
逐项比较通过；498个端点的状态字节也全部一致。两个runtime均正常清理。历史
`initial_joint`未单独保存，不能声称控制器所有历史内部状态已验证等价。结果只
证明这两个原始命令前缀在当前运行时可复现，不证明新选择器、抓取、关节安全或任务成功。

两名CPU worker实际wait/reap、exit0且未发信号；实际共496控制周期、12400个2 ms
积分步、6个模型及6个llvmpipe软件context。原始输入、资产和源码前后一致；新增
候选动作、WAM调用、GPU使用及final fresh均为0。80项32并发CPU专项、零模型dry和
全仓Ruff/2739文件格式检查通过。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/causal_event_prefix_replay_20261110/result.json` | `86a33474a493ce7c5d65cd465cd73efcb731a01411c52cc3644b69f8bc053f65` |
| `outputs/causal_event_prefix_replay_20261110/case_00/result.json` | `e884b5303d4c2ad6cd94468e06657c228ad8eafb561b0c34b5b32d4da4ba6d17` |
| `outputs/causal_event_prefix_replay_20261110/case_05/result.json` | `92d43a8809986ad14895e79ed109d22ee32ece5a913f983d12eef20bc232c1ae` |
| `outputs/causal_event_prefix_replay_dry_20261110/result.json` | `36bddda19dbce529ed7b61ba95477dc8c14c854a41f891b9b809af855ab33ae0` |
| `outputs/qa/causal_event_prefix_replay_20261110/result.json` | `bee3f2ffc1afbb6a55ed4f2db800b46c9e57ec4abed9610df6cac58a526e06aa` |

下一步需要从各自已完成的真实历史恢复手臂和夹爪策略状态，分别比较已生成旧/新
参考；继续保持原动作边界和关节缓冲，并独立重放实际派发动作。任何有限窗口都不能
替代最终任务验收；原18正确参考、12开发和24最终留出门槛保持。
