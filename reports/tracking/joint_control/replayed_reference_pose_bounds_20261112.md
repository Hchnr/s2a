# 碗任务已重放参考的首点位姿外界

2026-11-12。针对[实际执行至 step108 的旧开发初态](../references/reset_bowl_step88_reference_20261112.md)，
使用已独立重放的前缀与 step88 已生成的 WAM 参考，按[登记方案](../../../plans/v0.0.1/tracking/joint_control/replayed_reference_pose_bounds_20261112.md)
事后检查四个决策时刻。每点以紧邻之前实际完成的计划后移作为种子，保留实际上一
命令和已完成内环误差历史；该种子不声称是原优化器当时的起始解。

| 决策前周期 | 后移种子预测位置 / SO(3)误差 | box 下界 | 首周期安全下界 | 八周期安全下界 |
| --- | --- | ---: | ---: | ---: |
| 104 | 0.001839 m / 0.027954 rad | 0.369687 | 0.369687 | 0.369687 |
| 106 | 0.006275 m / 0.020364 rad | 0.361466 | 0.361466 | 0.361466 |
| 107 | 0.010404 m / 0.041502 rad | 0.852250 | 0.852250 | 0.852250 |
| 108 | 0.011382 m / 0.048123 rad | 0.909646 | 0.909646 | 0.909646 |

下界针对位置≤0.01 m、SO(3)≤0.05 rad 的归一化联合投影。四个后移种子均通过
本次放宽域中的动作和关节安全检查；107、108 的种子位置预测超限。108 的预测误差
和安全判定与原生产记录中实际执行的后移计划检查一致。四个最大下界均小于 1，
因此**这些投影没有排除首点双阈值**；也不能据此证明存在可行候选。三层的结果
相同仅针对本次固定方向和权重，不能推出关节安全约束通常无影响。

原 step108 的两点搜索使用 157/320 次响应并消耗共享 60 次首阶段迭代，后移
旧计划因首点位置超限被拒绝；checker 未重求该停止。这里的浮点外界使用冻结
自由模型，忽略完整接触和控制器非线性；不是对真实机器人可达性的证明，也不是
多任务、实时或留出验收。结果支持进一步在同一停止状态对比原安全约束下的首点
优先搜索，而不支持放宽质量门或声称搜索必然存在解。

独立标量模型复算四条后移计划的首响应，关节仿射最大差为 0，逐项核对 60 份
box/LP 支持证据。软件库指纹绑定修复后的正式运行使用 1 个静态模型、4 次
`mj_forward`、4 次种子 rollout、40 次支持 LP 和 4 次独立首响应/100 个模型
转换步；worker 实际 wait/reap、exit 0、未发信号。无新物理步、渲染 context、
WAM、GPU 或 fresh case。32 并发专项 24 项通过、0 skip；零模型 dry 通过。

首次运行因软件 OSMesa 指纹未登记，在创建模型前失败；其 worker 实际 wait/reap、
未发信号且模型、forward、rollout、LP 及物理开销均为 0。原入口和失败收据保留，
修复及新尝试见[登记说明](../../../plans/v0.0.1/tracking/joint_control/replayed_reference_pose_bounds_recovery_20261112.md)。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/replayed_reference_pose_bounds_20261112/result.json` | `7c363cbf72992fe2b34bc2a81f1b06779402e36362f77c5261feeda94d2c8e97` |
| `outputs/replayed_reference_pose_bounds_v2_20261112/result.json` | `4f6deb71a81735a3736b5467e5ade7ea08ecb997b751b800447ef6dfc7e2ab7c` |
| `outputs/replayed_reference_pose_bounds_v2_20261112/worker/result.json` | `a07cebfdc28d4f95e9da8a55a4311d47634c65e61a035806d4e29b4271e4017a` |
| `outputs/replayed_reference_pose_bounds_dry_v2_20261112/result.json` | `f4acd1ad5808b01d02c76b615290c75f0854bf6780699d2cd2a792e21abe290f` |
| `outputs/qa/replayed_reference_pose_bounds_v3_20261112/result.json` | `355154241aa18764976b84e49cb2eb792b4897a003993ea13c605c46d8c1840f` |
