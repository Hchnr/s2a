# 提前多方向干预的同参考物理对照

2026-11-11。按[预登记设计](../../../plans/v0.0.1/tracking/response_models/early_box_face_feedback_20261111.md)，
两个分支从 correct_08、seed 42 的同一 step118 状态、控制器/夹爪缓存和上一命令
分叉；均使用 step112 的当前图像当时已生成的同一 WAM H40 参考中的 step118–125
目标。两组保持同一 OSC、夹爪命令、0.25 动作变化率与关节停止门，分别按各自
当前反馈选择原目标或多方向投影。参考并未在 step120 重新生成，本结果不能当作
每周期重新观测的 WAM 性能。

原目标组实际完成 8 周期，投影组完成 6 周期；下表仅比较共同完成的
step118–123，属于同一个开发初态上的相关轨迹。

| 指标 | 原目标组 | 提前多方向投影组 |
| --- | ---: | ---: |
| 原 WAM 目标位姿双容差达标端点 | 2/6 | 2/6 |
| 所选目标位姿双容差达标端点 | 2/6 | 2/6 |
| 原目标位置误差均值 | 0.012510 m | 0.021526 m |
| 原目标位置误差最大值 | 0.022357 m | 0.041315 m |
| finger2–桌面有效接触内步 | 22/150 | 0/150 |
| finger2–炉具有效接触内步 | 0/150 | 0/150 |
| 任意 robot–environment 有效接触内步 | 22/150 | 45/150 |
| 力矩裁剪 / 机械臂关节缓冲不足内步 | 0 / 0 | 0 / 0 |
| 瞬时原生任务标签 / 目标抓取标签 | 6/6 / 0/6 | 6/6 / 0/6 |

独立分类显示，原目标组的 22 个有效接触内步均为 `gripper0_finger2_collision`
与 `table_collision`；投影组的 45 个均为 `gripper0_finger1_collision` 与
`akita_black_bowl_1_g34`。碗是任务相关物体，现有证据没有判断此接触是否符合
任务意图；它说明只针对 finger2、桌面和炉具的静态条件包络没有覆盖全部接触。
接触数按每个 2 ms 内步是否有有效约束统计，不是接触点数或真机受力。

投影组在 step124 派发前计算出所需修正 `0.050651750 m`，超过 `0.05 m` 上限，
因此停止，未派发该动作。独立 checker 分别重放原目标组 8 条与投影组 6 条动作，
逐项核对各自 200 与 150 个内步，最大状态差均为零；checker 不独立重算策略或
停止理由。两个 producer 的起始状态、旧图像、模型响应及 8 个解码目标一致。

投影组共同窗口中 150/150 个积分前样本的夹爪关节仍在额定范围，150/150 满足
`0.05 rad` 旋转界，但只有 **28/150** 满足投影所假设的 `0.01 m` TCP 位置界。
顶点相对单滑轴预测的最大数值残差约 `3.60e-16 m`；零额外形变仍为条件，不能
据此推断未满足位姿条件的采样、其他几何或连续路径具有净空。与先前 step120
介入试验相比，此次使用的 WAM 参考在 step112 而非 step120 生成，不能把两次
结果的差异单独归因于介入时机。

四个进程分别完成 3,150、3,150、3,100、3,100 个物理内步，合计 **12,500**；
两组 producer 共选择 **14** 条新命令。每个进程构建 3 次原生模型和 OSMesa
context，全部 worker 实际 wait/reap；没有新 WAM 请求、去噪、训练或新初态。
此前相关链已消耗 37,050 内步，与本次合计 **49,550**；更早生成旧参考等成本
另见各自报告。原 18/12/24 留出、每条 600 周期、20 Hz 实时与真机仍未验证。

执行前 32 并发专项 37 项通过、零跳过及收集错误，独立审核专项 13 项通过。
此前模拟器环境全量回归有 4 项失败、2 项跳过和 5 个文件收集错误，本阶段未
重跑全仓测试。前次物理实验使用的记录器原字节和 SHA 按[计划](../../../plans/v0.0.1/tracking/response_models/early_box_face_feedback_20261111.md)保留。
历史运行前 QA 的 `docs/status.md` 原 SHA256 为
`ec3f7d0e266b8283fc72f88c2d8f58fb1f8770888d898469bdcc6e169f289c3e`，更新前字节见
[历史副本](../../../docs/history/experiments/status_before_early_box_face_130dd584.md.txt)。

| 仓库内证据 | SHA256 |
| --- | --- |
| [运行前 32 并发专项](../../../outputs/qa/early_box_face_feedback_20261111/result.json) | `be0c21caefec65ac2b854865173d351d18c1cfca944cd3751c66a880638bbee7` |
| [零物理 dry](../../../outputs/early_box_face_feedback_dry_20261111/result.json) | `b58dc559fbb675804f4993a0d0dab77890886386d4e6d5a96aa03a68d7b6791f` |
| [物理对照 owner](../../../outputs/early_box_face_feedback_20261111/result.json) | `57cb3e4d35ecb5ba7f6e13d84006c11b4ad4b35cc7bb359ab71ec6071af69a20` |
| [接触与重放只读审核](../../../outputs/early_box_face_feedback_audit_20261111/result.json) | `376755f70866ed7514fa73c0018a5d516059f3693848f71a7632546d0529e937` |
| [原目标 producer](../../../outputs/early_box_face_feedback_20261111/original/producer/simulation/result.json) | `76a26dd23d7dbf8a4cd8ddf2b9c6d98cd859964cddd261cdde0411ba6938593d` |
| [投影 producer](../../../outputs/early_box_face_feedback_20261111/box_face/producer/simulation/result.json) | `adc198210af05a27575e4b7e6e87a8b079a77a0713855f973ccb5c49832dd774` |
