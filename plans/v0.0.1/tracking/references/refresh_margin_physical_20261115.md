# 刷新余量候选的同状态物理试验

2026-11-15。依据[候选设计](refresh_margin_preference_20261115.md)和
[未来质量补充](refresh_margin_preference_addendum_20261115.md)，在原 bowl /
seed 43 开发初态检验固定边界内的缓存候选排序。基线为已执行且独立重放的
[新参考分支](../../../reports/tracking/references/bounded_reference_choice_recovery_20261115.md)：
producer SHA `4988d608a14b167382fb9a5839f7485746a4a33dcade2e1f9ad3333b0e5b206a`，
checker SHA `d6ad66ff90fed70c8b9a3a0c2d20067b25225251e2605ca4b35a8aaf4f558e0e`。

候选 producer 从原 reset 重算反馈，前 16 条动作、每个实际端点、内步及第 8 /
16 步图像均须与原轨迹一致；step 16 的新参考与基线同源。只有在已确认的空闲
step 16 才启用候选排序，所有后续输入仅含当前实际反馈和已生成参考。每次仍
执行原 witness 审核、动作 box / 变化率、关节缓冲、实际 pose 与质量刷新停止；
最多使用相同新参考的前 32 个目标。若 step 16 无更有余量的候选，应逐值回退
基线首动作，不能把未变化描述为改善。

候选 producer 和独立 checker 各至多 48 周期（包含 16 周期前缀）、1,200 个
2 ms 内步、3 模型与 3 OSMesa context；producer 最多 48 次决策、23,040 次
原 refiner 响应及 15,360 次质量响应，checker 不选择动作或加载候选。checker
从 reset 重放所有实际动作、重算两套同时刻参考误差并比较状态、内步和夹爪，
原 producer 停止原因单独记为未独立重算。全部进程需 wait/reap，来源在前后
核对；固定候选策略与原执行之间的源码变化以 PR #132 的 Git blob 和 SHA 保存，
不把历史源码冒充当前版本。

先运行 CPU 回归和零物理 preflight，显式登记新目录；不自动重试、不覆盖旧
证据。0 新 WAM 请求、0 GPU、0 fresh；对象状态和成功谓词只评分。结果属于
单个已选开发初态，不改变原 18 条正确参考、12 条 WAM 开发、24 条最终留出门槛。
