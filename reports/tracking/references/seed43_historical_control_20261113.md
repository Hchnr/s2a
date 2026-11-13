# 新 seed 43 参考与历史首窗物理证据的关联

2026-11-13，回顾性只读审计。[新模型重放](adaptive_multiseed_real43_20261113.md)
的 seed 43 首个 H40，与 v4 和 v5 `task0_init0_s43_raw` 的同图输出在 raw 和
normalized 两组 40×128 数组上逐元素完全一致；请求除名称和图像副本路径外
逐字段相同。历史首窗反馈记录明确绑定 v4 响应 SHA，并在原复位状态和首图
一致的条件下使用该参考；本次对齐的是已保存参考及历史试验，**不是重新执行
新 worker 的在线闭环**。

历史 producer 从同一开发初态以 MountedPanda/PandaGripper、世界坐标、20 Hz
和 25×2 ms 内环执行 8 周期；独立 checker 不加载候选策略，重放同一命令。只读
审计重新计算 8 条命令、8 个实际状态、TCP 误差及 200 个内环时间/q/qvel，
producer 与 checker 的最大差均为 0。两侧 worker 均退出码 0、实际 wait/reap。

| 历史 8 周期指标 | 结果 |
| --- | ---: |
| TCP 位置向量 RMSE / 最大误差 | 0.001391 / 0.002139 m |
| TCP SO(3) 向量 RMSE / 最大误差 | 0.012130 / 0.019689 rad |
| 满足原位姿双门的端点 | 8/8 |
| 最小实际内环关节余量 | 0.617736 rad |
| 目标 grasp / 原生任务成功端点 | 0/8 / 0/8 |

这些数值是一个旧开发初态的首窗证据。它说明该保存参考在历史候选和控制合同
下的短窗跟踪，不能证明新 worker 的现场参考切换、完整任务、多任务泛化或
最终留出稳定成功。要推进闭环，应在该开发初态实际完成首窗后，使用本次
step8 的当前图像和历史登记 seed 43 生成下一参考，并独立重放；不能把旧轨迹
step8 图像当作新轨迹的当前观察。

本次审计无新模型初始化、WAM 请求、仿真模型、物理步或 fresh。32 并发相关
回归 **53 项通过、0 skip、0 collection errors**，所有 worker 实际 wait/reap、
源码前后一致。原始跨试验审计产物
`outputs/audits/seed43_historical_control_20261113/result.json` SHA-256
`d0dd8e98b42b3056367fe1200d689c6762ab32f1fc54f04db495210847ae449b`；
测试结果 `outputs/qa/seed43_historical_control_audit_20261113/result.json` SHA-256
`2a99823321656cea26d38acf50ea300ed87ecaa027008ea106cddba7a36c7162`。
合并前 64 并发相关回归 **129 项通过、0 skip、0 collection errors**，源码前后
一致且 worker 均实际 wait/reap；全仓 Ruff 和格式检查通过。结果
`outputs/qa/seed43_historical_control_final_20261113/result.json` SHA-256
`778a2854d66e6f0aa1b9dcabbd588c7478a1b7ab6363ca0acb62af57cdc6bc60`。
原始历史结果和完整开销见[首窗报告](../joint_control/reset_start_feedback_20261111.md)。
