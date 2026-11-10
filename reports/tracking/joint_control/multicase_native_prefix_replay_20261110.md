# 两条开发轨迹的原生命令前缀重放

2026-11-10。按[登记计划](../../../plans/v0.0.1/tracking/joint_control/multicase_native_prefix_replay_20261110.md)，在当前仓库内CPU/软件OSMesa运行时，从各自原初态重放两条v4开发轨迹的既有命令，到达历史上首次越界前最后一个目标关节余量不低于0.05 rad的端点。

| 案例 | 重放命令/控制周期 | 已比较端点 | 完整state逐字节一致 | state最大差 | 已保存控制器缓存最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| task0 init0 s43 raw | 114 | 115 | 115/115 | 0 | 0 |
| task1 init0 s43 raw | 99 | 100 | 100/100 | 0 | 0 |

每端点的物理state、时间、TCP相关位姿、原生任务标签、保存的控制器位置/旋转goal与缓存、夹爪motor cache均通过原 `1e-10` 绝对容差、0相对容差；state逐字节一致另由原endpoint SHA确认。原日志此前已完成独立审核，本轮是对两条保存前缀的当前运行时重放，未再执行一套独立的本轮物理复算。

两条终点的真实q7/速度分别为 `2.843538560034123 rad / 0.22366425154170555 rad/s` 和 `2.8472799365385586 rad / 0.0017147937504402334 rad/s`，距上限分别 `0.05376143996587679 rad` 与 `0.05002006346144139 rad`。这些是历史保存的实际仿真状态；controller缓存值与当前实测q是不同字段。该起点依据旧轨迹后验选择，不能当作在线触发或未来安全保证。

## 执行和边界

原seed、task、初态、固定控制器、保存命令和观测/渲染顺序保持；XML仅按已验证的99声明/96资产映射替换路径，原文件可逆恢复。新运行时配置仅替换三项运行时位置，登记的输入、96项资产及本轮实现源码前后SHA一致；实际加载的库和模块另记录当前身份。运行时controller合同与对应历史记录一致，实际软件渲染器为OSMesa；历史 `initial_joint` 未单独保存，因此记录了当前值但不宣称历史值已核对，也不声明两套运行时二进制完全等价。

两个worker分别正常wait/reap、exit0，PID已不存在；外层owner会话36095实际wait exit0。实际新增 **213控制周期、5325个2ms积分步、6次模型初始化/6个软件渲染context**；0新WAM/GPU/fresh或新规划命令。60项32并发CPU专项全部通过，零物理步真实输入dry通过，执行前后来源不变。物理时间推进不代表墙钟实时能力。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `multicase_native_prefix_replay_20261110/result.json` | `1269464d34161f999937d4b2d61c5e2448f1d611353b836b1d4e08a37840980b` |
| `multicase_native_prefix_replay_20261110/case_02/result.json` | `98f6d327a4240eb8577b4830c7a1a055f3037fb39f40d20df95fbdcedd1de6d0` |
| `multicase_native_prefix_replay_20261110/case_06/result.json` | `fde22aa4e80160d3d347817908d99c11b89d958d60cead0265d0957be015837d` |
| `multicase_native_prefix_replay_owner_20261110/observation.json` | `c32f272eb02a92ead235d026cdb581901355bbd8754a66a1885e2ff864e431cf` |
| `multicase_native_prefix_replay_dry_20261110/result.json` | `98ed9edb5d521c6d8e8c16d991d3a50d5ceff2307d02f698f0147721f01fa008` |
| `qa/multicase_native_prefix_replay_20261110/result.json` | `9861d6a7338750a86d4fcc89ddfb16e20e52584c2d5f137bb09391dd5d5e885f` |

这为两种不同接触上下文的下一步成对干预建立了可复现起点；未尝试新的安全动作、未证明越界可避免，也未取得WAM或最终任务资格。任何新策略必须基于当时可用状态和已生成参考，独立验证自己的后续轨迹，不能读取旧分支未来状态作为控制输入。原actual368/H12物理pilot仍受其完整witness前置条件约束；18正确参考、12开发和24留出门槛不变，GPU继续禁用。
