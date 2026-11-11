# 保存状态物体分析：四元数归一化后的单次修复核查

2026-11-11。[首次静态分析](../../../../outputs/early_box_face_bowl_effect_20261111/result.json)
在第 9 次 `mj_forward` 后失败：MuJoCo 对已保存自由关节四元数做数值归一化，
`qpos` 有 4 个元素变化，最大 `2.22044605e-16`，但原检查要求逐位相等。
首次实际构建 1 个原生模型、完成 9 次 forward，0 物理积分、0 模型推理；失败
和已消耗的静态预算保留。对失败时对应的已保存状态按 XML 关节顺序做纯数值
检查，碗、奶酪和酒瓶四元数范数相对 1 的差仅在浮点舍入量级，其归一化最大
分量变化分别约 `1.11e-16`、`1.11e-16` 和 `2.22e-16`。

修复只允许已登记 free/ball 关节的四元数部分发生标准归一化，归一化结果与
输入表示的 SO(3) 姿态均需在 `1e-12` 内一致；所有非四元数 qpos、完整 qvel 和
时间仍须逐位相等。逐状态保存实际最大 qpos 变化；不改变物体测量、已保存状态
SHA、原参考、轨迹、接触力或物理验收门。测试非法非四元数变化、错误旋转和过大
变化均拒绝，再在新输出目录按原 13 状态设计执行**一次**静态核查。

本次最多新增 1 个 CPU 原生模型和 13 次 `mj_forward`；连同首次失败，累计最多
**2 个原生模型、22 次 forward、0 次物理积分、0 次 WAM 推理、0 个新初态**。
先完成新的 32 并发专项与零模型 dry；不覆盖失败产物、不自动重试。原失败时
[分析入口](../../../../docs/history/experiments/early_box_face_bowl_effect_before_quat_fix_20261111.py.txt)
和[测试](../../../../docs/history/experiments/test_early_box_face_bowl_effect_before_quat_fix_20261111.py.txt)
原字节 SHA256 分别为 `ff582a7c5310c0d5427754e910cbaae8bd35aa9d8b4dd0dca0ca0c5219358b0e`、
`a4735bc805132c29c7bed7a2461d6c3ff1affa3084de7754f4e3ce2e7911daed`。
