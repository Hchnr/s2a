# 提前桌面干预：当前图像 WAM 与短窗口物理反馈

2026-11-11。两个分支从相同 correct_08、seed 42 的 step118 状态出发，使用
step112 已生成参考执行两周期；各自的 step120 图像和状态经独立物理重放核对，
再分别生成新的 WAM H40 参考。此后分别从头重放各自历史、验证实际 step120
状态和图像与对应模型请求一致，沿各自当前反馈执行 step120–127 最多八周期。
两个分支使用相同单步 OSC 控制、动作变化率和夹爪命令规则；位姿包络分支额外
投影目标。完整步骤和约束见[计划](../../../plans/v0.0.1/tracking/references/early_tabletop_intervention_live_wam_20261111.md)。

| step120–127 指标 | 原目标分支 | 位姿包络投影分支 |
| --- | ---: | ---: |
| 实际执行新周期 | 8/8 | 8/8 |
| 相对各自原 WAM 目标的位姿双容差达标端点 | 2/8 | 1/8 |
| 相对各自所选目标的位姿双容差达标端点 | 2/8 | 8/8 |
| 相对各自原 WAM 目标的位置误差均值 | 0.019548 m | 0.027740 m |
| step127 相对各自原 WAM 目标的位置误差 | 0.036456 m | 0.045960 m |
| 所选目标相对原目标最大平移 | 0 | 0.046939 m |
| finger2–桌面有效接触内步 | 107/200 | 0/200 |
| finger2–炉具燃烧区有效接触内步 | 0/200 | 83/200 |
| 任意 robot–environment 有效接触内步 | 107/200 | 83/200 |
| 机械臂力矩裁剪 / 关节缓冲丢失内步 | 0 / 0 | 0 / 0 |

接触数按 2 ms 内步中对应几何对是否具有有效约束统计，不是接触点数或真机
受力。每周期 25 个内步的 finger2–桌面接触分别为原目标
`0, 0, 0, 7, 25, 25, 25, 25`，投影分支均为 0；投影分支的 finger2–炉具
燃烧区接触分别为 `0, 0, 0, 0, 8, 25, 25, 25`。投影分支的桌面接触减少，
但炉具接触仍存在，不能称为完整避碰。

两个新 WAM 参考由不同的实际图像生成，不是同一条参考：其前八个原 TCP 目标
世界位置差的均值 / 最大值为 `0.001553 / 0.002199 m`，SO(3) 测地角差的均值 /
最大值为 `0.007229 / 0.012800 rad`。因此上述分支差异属于“提前动作、当前
图像、模型参考和后续投影”共同变化的结果，不能把效果单独归因于某一个环节。
两个分支各 8/8 原生任务标签为真、抓取标签为假；这只是短窗口的瞬时结果，
不满足 600 周期和末尾连续稳定的任务标准。

位姿包络分支最大实际 TCP 局部顶点变化为 `0.000489 m`；其 200 个积分前采样中，
位置和旋转误差界均满足，但仅 8 个采样同时满足假设的**精确零**夹爪形变界。
其余采样不能引用该固定形状净空保证；几何范围也不包含炉具。真实桌面标定、
其他机器人部件、连续路径、未见初态及真机均未验证。两次 WAM 请求耗时约
`13.236 s / 11.714 s`，仿真等待期间暂停，不满足 20 Hz 实时性。

首次继续执行在比较已保存 JSON 列表和内存 NumPy 数组时失败，未派发新参考
动作；两次失败共消耗 6,000 内步并保留[原始收据](early_tabletop_live_feedback_failure_20261111.md)。
类型修复后按[另行登记的预算](../../../plans/v0.0.1/tracking/references/early_tabletop_live_feedback_repair_20261111.md)
执行本次反馈：两次 producer 和两个隔离 checker 的 16 个新动作及全部内环记录
一致。累计成本包括采集 `12,000`、失败 `6,000`、修复验证 `12,800`，总计
**30,800 个物理内步、20 个新选择命令、2 次 WAM 请求和 70 次 denoise**。
所有 worker 实际 wait/reap；模型 GPU 进程退出后显存归零。checker 重放已派发
动作，不独立重算策略或验证原停止理由。新反馈阶段没有新增模型调用或初态。
最终 64 并发相关回归 79 项及全仓 Ruff 检查通过。此前模拟器环境全仓测试的
4 项浮点容差失败与 5 个文件收集错误仍保留在[静态投影报告](../response_models/tabletop_reference_projection_20261111.md)；
本阶段没有重复全仓回归。

历史测试快照中 `docs/status.md` 的声明 SHA256 为
`65d7768d8be2753606cfcd64de2d20549af69cb06e9bcefd9c9727bf0f8d22f2`；该文档后来
更新，原字节保存在[状态文档历史副本](../../../docs/history/experiments/status_before_early_tabletop_15d30e00.md.txt)。
首次失败时被修复的代码和测试也在[失败报告](early_tabletop_live_feedback_failure_20261111.md)
中按原声明路径和 SHA 定位。

这项结果支持继续研究提前干预，但当前候选尚未达成原参考跟踪、完整几何安全、
实时或留出任务目标。下一步需要处理炉具及夹爪运动，并比较参考跟踪和约束的
权衡；原 18/12/24 留出、每条 600 周期及每任务稳定成功门槛保持不变。

| 仓库内证据 | SHA256 |
| --- | --- |
| [提前干预采集与重放](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_a/result.json) | `f3766ae57f3fe37e10a6d2f1d18cbb2201e6e17d24ed4b7bb6a066a841374974` |
| [WAM 两次响应与去噪记录](../../../outputs/early_tabletop_intervention_live_wam_20261111/model/model_run/result.json) | `46cb9f6c1b1b0e20cd284969c69adef5bc445da37dbccd32f110eda8a7512724` |
| [首次继续阶段失败](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b/result.json) | `6c34f5affca2ba55478ca9426698d28db69d6858656454b32e9a696cf378c14b` |
| [修复验证零工作 dry](../../../outputs/early_tabletop_live_feedback_repair_dry_20261111/result.json) | `e7e867883c187c7f0c8ae4b179dcb1b712209644c85566fe26ef1285b0a265d9` |
| [修复验证 32 并发专项](../../../outputs/qa/early_tabletop_live_feedback_repair_20261111/result.json) | `fcfcdf4da2757d19c9677a5dd2208b460da83f333f8aa5c1e9aaf1b1ff8aa733` |
| [最终 64 并发相关专项](../../../outputs/qa/early_tabletop_live_wam_final_20261111/result.json) | `86de494afa1ef033c092a7ca584c46dd8e180150f491b4cd52001adba12b5da1` |
| [修复验证 owner](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair/result.json) | `47296ebd688e312c7be31aa0afe88c063ed296666433ac71f1d7fd331af7b06a` |
| [原目标 producer](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair/original/producer/simulation/result.json) | `5744bd0d6f6f3bec70121cadb3efe1b5a6e70dcbcee4af55226e2dd1077e4c66` |
| [原目标 checker](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair/original/checker/simulation/result.json) | `f902094e41c24bc9ba2bdd1f2e2896ce5765bf45243ef6fc3d021fbef3563091` |
| [位姿包络 producer](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair/pose_bounded/producer/simulation/result.json) | `f39e891dbcc80b1da1c9d110954727a5a402d1a5aa76858783f7bde163daed56` |
| [位姿包络 checker](../../../outputs/early_tabletop_intervention_live_wam_20261111/phase_b_repair/pose_bounded/checker/simulation/result.json) | `537d02dd35d6ff55f66b8b6bee1ab2aeb15a952be184ca6723925ad4771591a5` |
