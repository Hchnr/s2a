# 提前桌面干预：step120 当前图像采集

2026-11-11。按[已登记设计](../../../plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md)，
从同一 correct_08、seed 42 的 step118 状态分叉，使用 step112 已生成 H40 的
step118–119 参考，在相同单步 OSC 控制与夹爪命令下比较原目标和桌面位姿包络
投影。只执行这两个新周期，独立重放各分支完整动作及内环状态，随后各自从
step120 的实际当前状态采集图像。尚未运行新的 WAM 推理或后续动作。

| step / 指标 | 原目标 | 位姿包络投影 |
| --- | ---: | ---: |
| 118：参考法向抬升 | 0 m | 0.001350 m |
| 118：实际发送 z 动作 | −0.217967 | −0.217967 |
| 119：参考法向抬升 | 0 m | 0.008733 m |
| 119：实际发送 z 动作 | −0.169809 | 0.004859 |
| 119 端点相对原目标位置误差 | 0.006139 m | 0.007115 m |
| 118–119 finger2–桌面有效接触 | 0/50 内步 | 0/50 内步 |

两分支首个动作相同，第二个 z 动作不同。二者的 step120 状态和图像 SHA
均不同；每个分支的独立 checker 均重放一致，且其当前图像 SHA 与对应 producer
一致。每次渲染前后状态与时间不变，未使用未来帧。无桌面接触只针对已执行的
两个周期，不能推断随后轨迹安全、预测质量或任务成功。

两次 producer 和两次 checker 合计 480 个控制周期、12,000 个 2 ms 内步、
4 个新选择命令，全部 worker 实际 wait/reap；33 项 32 并发预检通过。此阶段
没有新 WAM/GPU 调用或新初态。接下来必须分别使用这两张当前图像生成新的 WAM
参考，并验证响应、分支身份及物理反馈，不能复用原轨迹的 step120 响应。

| 分支及证据 | SHA256 |
| --- | --- |
| [零工作 dry](../../../outputs/early_tabletop_capture_dry_20261111/result.json) | `543e251623ee6d2cb3dfb69a12766f84d23e7bbe97f1a1b3a29e0193b28806f8` |
| [32 并发准入专项](../../../outputs/qa/early_tabletop_capture_20261111/result.json) | `533da02a37f28658e34bd1c4c1a5f1ac535c372b96109910da640e29506fb350` |
| [阶段 owner](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_a/result.json) | `f3766ae57f3fe37e10a6d2f1d18cbb2201e6e17d24ed4b7bb6a066a841374974` |
| 原目标 step120 状态 | `6fbadacfadb65c073295f69e57bf2d12d87df128593b358ea1c9cc8cbb343d07` |
| [原目标当前图像](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_a/original/producer/simulation/runtime/anchor0120/current.png) | `3325714b3cf8f65c2f45d59996d3585820d95c2390cb336a1fd414200cdac45a` |
| 投影 step120 状态 | `5e784c3b9029e5922e07dce963b42f5c3b2361048c5ad856c719bc45ee7852dc` |
| [投影当前图像](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_a/pose_bounded/producer/simulation/runtime/anchor0120/current.png) | `1a2a6a58ada903a641972144bb3084fe5d6a78b94afa064026f4bee20a8cd145` |
