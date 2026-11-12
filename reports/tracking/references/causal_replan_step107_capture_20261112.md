# 因果重规划边界的 step107 当前图像

2026-11-12。按[登记方案](../../../plans/v0.0.1/tracking/references/causal_replan_step107_capture_20261112.md)，
从同一碗任务旧开发 reset 重放先前实际派发的 107 条命令，在影子信号首次触发
的 step107 采集当前双相机图像。最后一段只重放三个已完成周期，主动边界记为
`requested_observation`，不是原轨迹随后 step108 的质量停止；未派发下一动作。

实际命令、完整状态、TCP 误差、2 ms 内步 q/qvel/time 及夹爪反馈与先前已独立
验证轨迹的最大绝对差均为 0。107/107 采样端点位姿双阈值达标，目标 grasp 标签
39 个、任务成功标签 6 个；成功仅为端点 102–107 的采样结果，未说明后续稳定性。
采图前后状态与时钟一致，辅助视角及左腕相机合成为 `960×640` 图像，不含未来帧。

| 输入 | SHA-256 |
| --- | --- |
| step107 完整状态 | `04b8eea4cd6e611302b68ff036594129808fec0bb88cab97f4e55106ee4c148d` |
| 当前合成图像 PNG | `f7cc70d689471bbc96865f825dc44bc9e1c2607bf63bfa56b6114780ec2129a3` |
| 原始双相机数组 | `0c1f5edafa45a3d701537439a17626557bab19ac3f0d2af0801a0571d47a615f` |

实际重放 107 周期、2,675 内步、3 仿真模型及 3 OSMesa context，worker 实际
wait/reap、exit 0、未发信号；无新世界模型、GPU、候选动作或 fresh case。
同系列累计 1,305 周期、32,625 内步、72 仿真模型/context，另有 2 个静态模型；
世界模型初始化 8 次、请求尝试/完成 20/17 次、去噪 595 次。32 并发专项 37 项
通过、0 skip，零物理 dry 通过。该图像仍需实际生成参考并通过后续物理对照，
不授予控制、实时或留出验收资格。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/causal_replan_step107_capture_20261112/result.json` | `44d1b205e85815cab0b215f1cdfa6d3cba73671f1d1c6135b318dabe04759656` |
| `outputs/causal_replan_step107_capture_20261112/derived_producer.json` | `6f4341c9781f71ec034ff9ec54c1c4a4477d7935d96012bc5055254365a5cd8d` |
| `outputs/causal_replan_step107_capture_20261112/checker/case_00/result.json` | `c8b91ed0f8d88ad9da68ba1099e95315eafebfdebc5ff69ae8261a6945a3e7e7` |
| `outputs/causal_replan_step107_capture_dry_20261112/result.json` | `175b40e1c262fec4c4e697bc7bc7dfc32e2aa38e7bae98676e73e3ceef12d748` |
| `outputs/qa/causal_replan_step107_capture_20261112/result.json` | `89225c0c3de7a01f44d3478ecb16c83f60937398870dc53df383d6349915da74` |
