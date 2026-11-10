# 更早速度预警下的多点质量模型搜索

2026-11-10。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/velocity_warning_full_horizon_20261110.md)，从只用当前q/qvel的0.2 s外推首次安全预警端点39/47出发，恢复两个seed43开发案例的固定机器人状态、上一实际动作和当时已生成的step32/40参考chunk。task1在警报前一周期的实际SO(3)误差已约0.111 rad，不能把该起点描述为此前跟踪合格。

| 案例 | 当时最小关节余量 | 原计划的8点模型位姿达标 | 首点质量搜索 | 全8点质量搜索 |
| --- | ---: | ---: | --- | --- |
| task0 step39 | 0.288517 rad | 5/8 | 有模型witness，独立复算通过；首点位置0.000821 m、SO(3) 0.006045 rad | 原种子达320响应上限仍无witness；首点模型计划热启动也无witness |
| task1 step47 | 0.513592 rad | 0/8 | 未找到witness | 原种子未找到witness；无首点witness，热启动按计划跳过 |

原计划对应的8点冻结模型最大SO(3)误差为 `0.100124/0.259047 rad`；这些是搜索起点的预测，不是完整计划的实际执行结果。task1的首点模型预测误差约 `0.11957 rad`。两条全8点搜索的线性关节部分均可行；非线性求解达到响应上限或线搜索终止，不能据此证明全局无解。所有模型搜索保留原0.05 rad关节缓冲、动作box/0.25变化率及额外0.1 s制动；没有派发候选动作。

这两个速度警报分别比上次固定0.10 rad警报早4/5周期，但使用的参考是较早时刻已生成的不同chunk，不能把结果差异仅归因于触发时间，也不代表独立初态泛化。task0首点witness仅验证一个冻结模型周期；task1预警前已出现姿态偏差，提示仅凭关节速度报警不足以确保及时处理跟踪误差。

静态恢复的M/J、q/qvel及SITE与当时日志逐值相同；twist及原计划首周期预测最大差分别不超过 `1.95e-16/4.33e-15`。历史controller `initial_joint` 未单独保存，本次采用先前producer/checker一致重放得到的固定运行时值；该对照不补称原历史值有独立记录。起点前未建立25点模型—实测误差历史，额外经验收紧为零，并非未来保证。

2个CPU worker均exit0且实际wait/reap，源码及输入前后一致。共2模型、2次静态 `mj_forward`、2次基线rollout、5次搜索/748模型响应；1个首点witness独立复算计费200端点查询/2600模型转换步。36项32并发CPU专项、全仓Ruff/格式和零模型dry通过；0物理积分、0 context、0新WAM/GPU/fresh。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `velocity_warning_full_horizon_20261110/result.json` | `d30d1fc5ef37ec819c414c859554ffd377e9725917d77a6f98bb80d690e4f03c` |
| `velocity_warning_full_horizon_dry_20261110/result.json` | `4cf80650bf83a1a1d749fc6b5f1b2643809fe678b8bc29eb9cb6ab0535c74c3d` |
| `qa/velocity_warning_full_horizon_20261110/result.json` | `95400edc8efa0de8d6f36fb040891cd4626a7bfd83f45bf6027bacf97f51d33f` |

后续应区分非线性求解初始化、原安全约束与额外制动条件的作用，并更早同时监测跟踪误差；如果出现多点模型witness，必须另外登记实际反馈及独立重放。原18正确参考、12开发和24最终留出标准保持，GPU继续禁用。
