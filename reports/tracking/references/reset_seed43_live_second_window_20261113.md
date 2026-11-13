# seed 43 从复位到当前图像的双窗口闭环

2026-11-13。按[预登记](../../../plans/v0.0.1/tracking/references/reset_seed43_live_second_window_20261113.md)，
在旧开发初态 `task0_init0_s43_raw` 从 reset 执行 16 个 20 Hz 周期。首 8 周期
使用已核验的历史首窗参考并逐周期核对实际反馈；随后使用本次第 8 周期的当前双相机
图像，真实请求一次 seed 43、8400 masked、H40 WAM 参考，继续执行其前 8 周期。
物理时钟在模型推理时暂停；任务和 grasp 标签只用于离线评分。

| 指标 | 现场 producer | 独立 checker |
| --- | ---: | ---: |
| 控制周期 / 2 ms 内步 | 16 / 400 | 16 / 400 |
| 现场参考对应周期 | 8 | 8 |
| 位姿限值通过 | 16/16 | 16/16 |
| TCP 位置 RMSE | 2.965 mm | 2.965 mm |
| SO(3) 旋转 RMSE | 0.01493 rad | 0.01493 rad |
| 最小内环关节余量 | 0.61774 rad | 0.61774 rad |
| 目标 grasp / 原生任务成功周期 | 0 / 0 | 0 / 0 |

producer 和 checker 从相同 reset 独立运行；checker 不加载候选策略或世界模型，
只重放实际派发动作。16 条命令、完整状态、TCP 误差、400 个内环 q/qvel/时间
与夹爪反馈的最大绝对差均为 **0**，任务和 grasp 标签一致。第 8 周期当前图像
重新渲染后的 PNG SHA-256 均为
`2b82b998f7eb2a5435a35f8ae401859408a25361d5b96564cbbbb5d8a5f3b3e1`；
模型实际请求、返回及发送给仿真的参考已绑定和核对。

producer 实际初始化一次模型，完成一次请求及 35 次 denoise；单次请求耗时
13.396 秒。producer 和 checker 各使用 3 个仿真模型及 3 个 OSMesa context，
合计 32 个控制周期及 800 内步。两个 producer worker 和 checker 均退出码 0、
实际 wait/reap、无信号；登记 GPU 已释放。这些是本次成本，不包含既往实验。

本次 64 并发相关回归 103 项通过，0 跳过、xfail、xpass 或收集错误；全仓
Ruff 检查与格式检查通过。零模型 dry、模型与资产指纹及 CPU OpenCV 预检通过。
首次 32 并发尝试因所选 `/opt/venv` 缺少 pytest 而没有收集用例，已保留；
切换仓库内仿真环境后相关回归通过，见[接线报告](reset_seed43_execution_20261113.md)。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/qa/reset_seed43_live_second_window_final_20261113/result.json` | `f9bf1d4fec2e36cc3830c641dc1da2f157d432263eb1fa5fb709a55956c2e5da` |
| `outputs/reset_seed43_live_second_window_dry_20261113/result.json` | `0658eb91c6aa68fa4ba5c68db2ef8116bfe7972ac54b03fdcc209642a98101cf` |
| `outputs/reset_seed43_live_second_window_20261113/preflight.json` | `5a769c16714149adfb07413d65246711bbcd2052bc7de5761a92cb6b014c829b` |
| `outputs/reset_seed43_live_second_window_20261113/producer/result.json` | `c0009a58f085cc8a33e749efaa48bb51770929d1ba223eb6e566d4ed131a9dc4` |
| `outputs/reset_seed43_live_second_window_20261113/producer/model/result.json` | `13561f5c7123039881f564c626564db1464304f5cfddaebd98a05f6cd42da0b0` |
| `outputs/reset_seed43_live_second_window_20261113/producer/workers.json` | `e9deedeb56812d26f4aeb1ba7e149333e0a4bb96eec23a02125ecb6889b34bdc` |
| `outputs/reset_seed43_live_second_window_20261113/checker/result.json` | `b44c9d54716c633b31d7b7b52ff02a80fbb095fcc355f3866cadaace786df171` |
| `outputs/reset_seed43_live_second_window_20261113/checker/workers.json` | `b4f00a1117d0a13fb1290fe2acf7a9ab416df8e498c61ab3b26887c70d3ff921` |

这是一个旧开发初态的 0.8 秒短窗；没有抓取或任务成功，也不能由此断言更长轨迹
一定失败。结果不能替代同一候选的三任务正确参考与 WAM 开发对照、600 周期和
24 条留出验收，也不证明实时或真机能力。下一步应优先复用已有执行链，推进统一
候选的多任务、长时程对照，减少对单案例专用接线的投入。
