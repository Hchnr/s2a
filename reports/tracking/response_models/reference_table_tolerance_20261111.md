# 原位姿容差内的固定夹爪桌面冲突

2026-11-11，按[固定方案](../../../plans/v0.0.1/tracking/response_models/reference_table_tolerance_20261111.md)，
检查 correct_08 的 step120–126 已生成目标。对每个决策，固定当时实际夹爪关节
配置，把编译后的 `gripper0_finger2_collision` mesh 对齐目标 TCP，并以固定的
`table_collision` 盒体为参照。七个状态及参考分别来自该决策时已完成历史和
step120 已生成参考；step126 的动作未派发。

对每个 mesh 顶点，令 `m` 为其位于桌面盒内部时到六个面的最小余量，`r` 为该点
到目标 TCP 的距离。允许 TCP 位置误差不超过 `0.01 m`、SO(3) 测地角误差不超过
`0.05 rad` 时，该固定夹爪材料点的位移不超过
`b = 0.01 + 2 r sin(0.05/2)`。若 `m > b + 1e-9 m`，该点在所有允许位姿下仍
位于桌体内部，因而固定夹爪配置与桌面相交。逐一检查全部 **18 个编译 mesh
顶点**，取 `m − b` 最大者作为见证；无证书只表示该充分条件未判定。

| step | 见证点内部余量 `m` | 位移上界 `b` | `m − b` | 固定夹爪冲突证书 |
| --- | ---: | ---: | ---: | --- |
| 120 | 0.000366 m | 0.012782 m | −0.012416 m | 未判定 |
| 121 | 0.005463 m | 0.012771 m | −0.007308 m | 未判定 |
| 122 | 0.010737 m | 0.012624 m | −0.001887 m | 未判定 |
| 123 | 0.016034 m | 0.012487 m | 0.003546 m | 有 |
| 124 | 0.020976 m | 0.012377 m | 0.008599 m | 有 |
| 125 | 0.024704 m | 0.012411 m | 0.012293 m | 有 |
| 126 | 0.024382 m | 0.011892 m | 0.012491 m | 有，未派发 |

因此，在**各步当时夹爪关节配置固定、桌面盒体不变**的条件下，step123–126
不存在同时满足原位姿双容差并使该 finger mesh 不与桌面相交的 TCP 位姿。这个
结论不依赖能否求出机器人 IK；也不排除改变夹爪开度、场景几何或容差后存在
可行配置。step120–122 没有证书，不能据此称无碰撞或可行。它不证明连续路径、
真实接触力、完整任务或其他初态的结果。

独立复核从原状态重建当前 site 和 finger 位姿，不移动基座，再用刚体变换计算
目标 finger 位姿；逐顶点用另一套标量计算重算盒体六面余量和位移上界。目标
finger 位姿与首轮最大差约 `1.34e-15`，四个证书及三个未判定结果一致。首轮
使用 **1 个 native 模型、14 次 `mj_forward`**，独立复核另使用 **1 个模型、
7 次 `mj_forward`**；两次均为零物理积分、零 WAM/GPU 调用和零新初态，worker
均正常退出并实际 wait/reap。首轮和复核各有 32 并发相关专项，分别 **17 项和
18 项通过**，无跳过和收集错误。

这些证书说明该固定夹爪配置下，单纯更精确地逆解动作无法同时满足原目标位姿
容差和桌面无碰撞。下一步应评估允许夹爪变化的可行性，并设计只使用可获得
状态、标定信息和已生成参考的因果干预；实际反馈必须同时按原参考和干预参考
计分。原 18 正确参考、12 开发、24 最终留出及 600 周期稳定成功标准不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [首轮零工作 dry](../../../outputs/reference_table_tolerance_dry_20261111/result.json) | `0384c48b832d6e84ad62e20a5604e62738ad6f596466b39881911ecc67f27cb6` |
| [首轮 32 并发专项](../../../outputs/qa/reference_table_tolerance_20261111/result.json) | `27473c06ad512cc6237958fcd4c274f2c7bebdf3ca4e3b6a35cdbba6f5a22e63` |
| [首轮 owner](../../../outputs/reference_table_tolerance_20261111/result.json) | `43e879f714eb5dcf0d075bf2446b0589c69bf556f04116477831c65a67fd0dea` |
| [逐目标证书](../../../outputs/reference_table_tolerance_20261111/worker/result.json) | `fc5098a857066b12f48b5c67b9e7eaae66ddf40da648f323b485f3e984527d8d` |
| [复核零工作 dry](../../../outputs/reference_table_tolerance_audit_dry_20261111/result.json) | `b36cd9affccd3c197494239980d5e010b80e45aef32c63228b2da94c9896d024` |
| [复核 32 并发专项](../../../outputs/qa/reference_table_tolerance_audit_20261111/result.json) | `8bbee8a34a985f985744eefa1f9177204fe1fc62e65dba5bd5a7c8fa85a496f3` |
| [独立复核 owner](../../../outputs/reference_table_tolerance_audit_20261111/result.json) | `4032b944ab87955e08fe8f26448c7432605258b02a1482abdcbb7070b335a01a` |
| [独立逐顶点复核](../../../outputs/reference_table_tolerance_audit_20261111/worker/result.json) | `32da309eaf57c5d826a211741d0043e21115d0b8e81db908f75442837cc85d50` |
