# 提前预览的因果反馈：独立物理重放

2026-11-13。按[物理计划](../../../plans/v0.0.1/tracking/joint_control/anticipatory_causal_physical_20261113.md)，
从同一旧开发初态、seed 42 重建 80 周期前缀及 step80–120 的 41 个反馈周期；
六份保存的真实 WAM 参考仅在本次图像、状态和时间匹配后复用。step120 完成后以
本次实测计算偏差，重新求解并派发 step121；step121 完成后以本次预测和实测更新
偏差，step122 的搜索没有找到通过原门及补偿门的候选，在派发前停止。

| 指标 | 历史同初态 step121 | 本次提前预览 step121 |
| --- | ---: | ---: |
| 实际位置误差 | 0.007395 m | 0.007302 m |
| 实际 SO(3) 误差 | 0.018307 rad | 0.019472 rad |
| 本次动作与历史动作最大分量差 | — | 0.083260 |

位置差约 0.093 mm，旋转误差略增；一次开发动作不能证明跟踪改善。历史 step122
的实际位置误差为 0.013933 m，但本次 step122 **没有执行**，不能把它与停止后的
新结果当作两条实际执行的物理对照。

本次 step121 的模型首点位置预测误差为 0.002644 m、实际为 0.007302 m；其对
step122 的校正预测位置误差为 0.002550 m，仍不保证该未来误差。已完成位置创新
从 step120 的 0.000531 m 上升至 step121 的 0.005119 m。step122 重新求解的
停止状态为 `staged_first_no_witness`，不是预登记的两周期预算停止；有限搜索
失败不证明物理不可达。

producer 与不加载候选的 checker 各执行 122 周期 / 3,050 内步，命令、状态、TCP
误差、关节位置 / 速度 / 时间及标签在独立重放中一致，状态及内步最大绝对差为 0。
反馈续段 42/42 个端点满足原始 pose 门，最小内步关节余量 0.782593 rad；续段
原生任务谓词末尾连续为真 35 个端点，未达到 40 个端点的稳定门，且整条运行未
完成 600 周期。checker 不重算 step122 的求解停止原因。

本次没有新 WAM 初始化、请求或去噪；producer 质量模型响应 3,596 次、独立端点
查询 / 模型转换步 8,400 / 109,200，均在登记预算内。producer 两个 worker 和
checker worker 均正常退出并实际 wait/reap。运行前 64 并发配置的专项实际执行
39 项，全部通过、无跳过或收集错误；零物理 dry 通过。本次结果不授予正确参考、
多任务、24 条留出、实时或真机资格。

| 证据 | SHA-256 |
| --- | --- |
| `outputs/anticipatory_causal_physical_20261113/producer/result.json` | `b57f9d9e2bbbeb6b88ef0a63f157e196b972842249657652f4232b9aef3d7486` |
| `outputs/anticipatory_causal_physical_20261113/checker/result.json` | `7f43ed98b32236c65ef67dabf040d43b5da315d60818093fdb234c9ea4596d39` |
| `outputs/qa/anticipatory_causal_physical_20261113/result.json` | `70b0931cb450d90c719e1e38f63aa38dc979570d58268e8e1735a8568d3666e8` |
| `outputs/anticipatory_causal_physical_dry_20261113/result.json` | `cb4b9499c012db8948ae80f6178a36fec956bd62223e47863fd061c4cfbc6bef` |
