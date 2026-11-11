# 即将释放时暂缓额外闭合：历史影子结果

2026-11-11。按[登记](../../../plans/v0.0.1/tracking/gripper/defer_close_release_shadow_20261111.md)，
在原策略要发送`+1`、当前motor目标已不大于实测opening、当前已生成参考的六个
后续端点单调张开至少一个motor周期尺度时，仅在原预览位置跨度未过而其余三项
保护通过的情形提议`0`。这是在查看task1释放片段后提出的假设；原目标位姿、
opening、时钟及机械臂命令不变，提案不读取物体/任务标签或未来实测。

| 已保存基线集合 | 原周期数 | 提议`+1 → 0` | 触发案例 |
| --- | ---: | ---: | --- |
| 18条正确参考 | 4907 | 1 | correct_08，step85 |
| 12条开发轨迹 | 7200 | 10 | task0 s42 terminal058 4次；task1 s43 terminal058 5次；task2 s42 terminal058 1次 |
| 已研究task1最新参考短窗 | 8 | 1 | step234 |

12条开发轨迹中的六条raw均无触发；三个有触发的terminal058案例覆盖三个任务，
但只是三个已研究初态/seed。短窗step234与开发轨迹的task1 step234属于同一参考
时刻、不同物理分支，不能当作独立样本。

**正确参考的唯一触发处，原基线端点已有物体grasp。** 原基线在该端点opening
误差约−0.00277，实测opening为0.530685、motor目标opening为0.113924；因此
不能假设暂缓闭合不会影响受载接触或原参考跟踪。task1短窗step234原基线实测
opening为0.209671、motor目标0.113924，当前目标0.205998，后六点张开0.544059；
原命令为`+1`，提案为`0`。候选未执行，不能由原轨迹中的成功、grasp或之后的
opening推断干预后结果。

全部12115个原基线时刻的提案与独立几何/单调/motor实现逐项核对，最大数值差
`1.39e-17`；原预览、实测opening和motor目标均与历史日志核对。31个影子worker、
零模型准入和执行子进程均实际wait/reap、exit0、未发信号；28项32进程专项、
全仓Ruff和格式检查通过。没有新拟合、WAM调用、物理、GPU或fresh。

| 仓库内证据 | SHA256 |
| --- | --- |
| `outputs/defer_close_release_shadow_20261111/result.json` | `bd905defd15d378242efdce2aa736611dfa79c26d63d357afd63076769b7d3f1` |
| `outputs/defer_close_release_shadow_20261111/workers.json` | `7dd8f3d851cbd626ff6d6812c9c292a6928c5db042a695c4e9c8e324cd4d9b87` |
| `outputs/defer_close_release_shadow_dry_20261111/result.json` | `af1af1c5035be95fc784f2f872fac92fa8dd9d941da038ca393653e524e9556b` |
| `outputs/defer_close_release_shadow_supervisor_20261111/dry-run_workers.json` | `84e379f4906275a184faf8802ee612807649b1d628f9bfd54d60a0a65db5fc32` |
| `outputs/defer_close_release_shadow_supervisor_20261111/execute_workers.json` | `b7c17c1be8b35c42d1db6082b8bae6a993b5223fa6548cf156c367d76955ff0c` |
| `outputs/qa/defer_close_release_shadow_20261111/result.json` | `2b8b31cc104d30f8c74864a164a9da9f7d4671a7991031a31ffd902fd65a4d01` |

下一步应先对correct_08的该受载触发点做相同初态与固定接口的成对物理核查，
再决定是否对已研究task1释放片段执行反馈对照。必须比较对原参考的连续开度、
位姿、关节内环和实际接触，不能仅计事件次数。原18/12/24与最终门槛不变。
