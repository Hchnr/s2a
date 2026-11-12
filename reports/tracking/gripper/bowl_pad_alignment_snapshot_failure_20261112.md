# 碗任务指垫几何快照审核失败

2026-11-12。[首个保存状态对齐诊断](../../../plans/v0.0.1/tracking/gripper/bowl_pad_alignment_20261112.md)
按登记完成 1 个 CPU 模型/data 和 97 次 forward，但事后独立顶点审核在 reset 的
pad 中点检查失败，不能将该批结果的坐标快照作为已验证结论。

原因已定位到快照所有权：`alignment` 对 MuJoCo 的可变数组使用 `np.asarray`，部分
输出仍引用同一块 data 内存，后续 forward 覆盖了先前记录的坐标。单独复制的 box
中心与这些字段不一致：左右 pad 均为 96/97 个点有差异，最大分量差分别约
0.238333 m 和 0.235506 m；目标刚体原点也有 96/97 个点不一致。两个 pad 坐标字段
在所有点上都等于最后一次查询的值。独立审核首先在 reset 发现中点最大分量差约
0.236919 m，因此未完成所有投影的独立验证；记录中其他已单独复制的数据不能因
这一局部错误自动获得完整审核资格。此次失败不影响此前已独立重放的物理轨迹和
原生 grasp/任务标签。

失败源结果 `outputs/bowl_pad_alignment_20261112/result.json` SHA-256
`19d2537aed166cda156c5c3152e00d679593425bf66e5f1d53cd62ce98f251be`。独立只读
[失败审计](../../../scripts/simulation/runtime/audit_bowl_pad_alignment_snapshot_failure.py)
保留 reset 断言、全部快照差异和实际子进程费用，输出
`outputs/bowl_pad_alignment_snapshot_failure_20261112/result.json` SHA-256
`43d178f76fb90eed3f003952b7b6ec6078bd8a4fa9fdc9bafe2d6ec5ba198a10`；审计没有
重新构建模型或 forward。

本次累计静态诊断 4 个模型/data、301 次 forward、零物理步/context/模型请求，子进程
已 wait/reap；源结果不覆盖，不自动重试。后续如进行独立恢复，必须在下一次 forward
之前复制每个数组并测试快照不随底层 data 变化，登记新的预算和输出，再完成逐盒
独立顶点复算。不能以失败批次的几何结果决定在线动作或任务资格。
