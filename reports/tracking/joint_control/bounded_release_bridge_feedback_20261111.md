# 有限释放保持后的连续衔接反馈

2026-11-11。按[预登记](../../../plans/v0.0.1/tracking/joint_control/bounded_release_bridge_feedback_20261111.md)，
在 task1 seed42 terminal058 的原初态重放 360 周期，连续复现上次候选 16 周期，
再在同一环境中实际执行 8 周期衔接。保持阶段曾有 8/8 grasp 端点，但衔接后失去
抓取，物体高度回到接近起点；本对照没有任务成功，不支持把该候选作为稳定解。

| 阶段，8 个端点 | 所选参考位姿达标 | 最新参考位姿达标 | grasp 端点 | 末端物体高度变化 | 实际最小关节余量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 有限保持，端点369–376 | 8/8 | 0/8 | 8/8 | +0.050281 m | 0.643644 rad |
| 本次连续衔接，端点377–384 | 7/8 | 0/8 | 3/8 | +0.000383 m | 0.633763 rad |

位姿双阈值是实际 TCP 位置 ≤0.01 m 且 SO(3) ≤0.05 rad，未包含 opening；
物体高度相对该案例的 step360 起点。grasp 为环境原生端点标签，不表示整个周期
始终接触。衔接前 3 个端点有 grasp，端点380起 5 个无 grasp；高度从保持末的
+0.050281 m 降至末端 +0.000383 m。这是同一条连续仿真的观测，不单凭相关现象
判定掉落根因，也不说明其他初态或释放策略必然如此。

衔接对**所选**参考的最大实际位置 / 旋转误差为 `0.004408 m / 0.056191 rad`，
末端旋转超过原 0.05 rad 门；对**最新**参考的最大误差为
`0.046603 m / 0.410860 rad`。8 个实测 opening 相对所选与最新参考的 RMSE 分别
为 `0.210783` 和 `0.545830`（归一化开度），末端实测 opening 为 `0.294453`。
跟踪过渡参考本身接近不代表已跟踪最新目标或成功放置。

端点376之前的 16 个新动作、状态、400 个内环样本、双参考评分、物体和任务标签
均逐周期复现上次已独立复核的保持结果；控制历史也逐项一致。实际端点376的夹爪
仍为 closing、实测 opening `0.239902`、前一周期内环最小余量 `0.643644 rad`，
候选选择 `bounded_bridge`。衔接的 8 周期全部完成后才提交参考事务，状态进入
`lockout`。从 step360 起 24 周期共 600 个实际内环样本中，0 关节越界、0 低于
0.05 rad 缓冲；24 个端点均无任务成功。保持与衔接使用同一控制器及完成历史，没有
通过重置仿真拼接状态。

本次衔接 8 次已派发的臂决策计时约 `1.003–1.117 s`，不包括此前夹爪提案，超过
20 Hz 的 0.05 s 周期；仿真等待时物理时间暂停，没有实时资格。一次完整短窗也没有
验证合法释放、持续重规划、其他开发任务或留出初态。结果提示下一步须同时解决
抓取保持、最新参考误差与速度，不能仅延长旧参考或提高对自身参考的拟合。

## 独立复核与费用

checker 没有加载候选策略或重新求解动作，从原初态重放 24 条实际命令；完整状态、
600 个 2 ms 内环 q / qvel / time、所选与最新参考误差、opening / motor、物体
位置及 grasp / 任务标签与 producer 逐值最大差均为 0。checker 独立按同时间端点
和 SO(3) 最短路径重构衔接几何，但**没有重算**候选选择、夹爪状态机或停止决策。

producer 与 checker 各执行 384 控制周期 / 9600 个 2 ms 内步、各初始化 3 模型和
3 个 OSMesa 软件 context；合计 768 周期 / 19200 内步。producer 新选 24 条命令、
24 次质量决策 / 1824 次模型响应，历史重建及在线提案共 366 次夹爪规划 /
2401326 条序列；checker 新选命令为 0。两阶段 worker 和监督进程均实际 wait/reap、
exit0、未发信号，输入与源码前后不变。执行源码提交 `dc865a39`；32 进程 CPU
专项 138 项通过、0 跳过，零模型 dry、全仓 Ruff 与格式检查通过；未重跑全量
pytest。`complete/passed` 只表示运行和复核闭合，`control_qualified=false`。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `bounded_release_bridge_feedback_20261111/producer/result.json` | `c2207b5a6089225ca9836162c6a539037d5456d01d4d066aaccc3512e67e0c91` |
| `bounded_release_bridge_feedback_20261111/producer/workers.json` | `0e64fc5b78804c6f18949cb80aba23e127274fcc86d7ac3118ccd44ecd8a0636` |
| `bounded_release_bridge_feedback_20261111/checker/result.json` | `8d5f21bd955a347927bb249d33a178b5b9d0ac6a7b8d8441e087cf17509e29e5` |
| `bounded_release_bridge_feedback_20261111/checker/workers.json` | `5552b96a918b92f7a963f627c09cd034ea77e810da8b86b9c3aa1337e98e790c` |
| `bounded_release_bridge_feedback_supervisor_20261111/producer_exit.json` | `bd9657cb7a959fec5d19287bcead0d34b1bae0bd298f8856f66705055a784880` |
| `bounded_release_bridge_feedback_supervisor_20261111/checker_exit.json` | `6dc4ee160031ea4085da235a74459a9351b08c94c91b5dc1be9c04fe74669095` |
| `bounded_release_bridge_feedback_dry_20261111/result.json` | `fdb10f36f0a6a6236d24f1137009196e617ee970430e7be1bb80f8dcc62a8f91` |
| `qa/bounded_release_bridge_feedback_20261111/result.json` | `5c21fea819c0c5de97f86647d146b2c300780767973482e72684d147394b3a4f` |

本次 0 新 WAM、0 GPU、0 新初态和最终留出；原 18 正确参考、12 WAM 开发、24 最终
留出以及 ≥20/24、每任务 ≥4/8 稳定成功门槛不变。
