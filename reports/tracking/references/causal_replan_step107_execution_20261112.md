# 因果提前重规划的实际执行与独立重放

2026-11-12。按[登记方案](../../../plans/v0.0.1/tracking/references/causal_replan_step107_execution_20261112.md)，
从同一碗任务旧开发 reset 重建 107 个周期；上一已完成动作的质量余量信号在
step107 首次触发，先核对现场当前图像、状态和[真实 WAM 响应](causal_replan_step107_model_20261112.md)，
再于原段完成 3/8 周期处切换，延续手臂、夹爪、上一命令与内环历史。

新参考执行了 **6/40** 个目标，step113 派发下一动作前因
`quality_no_witness` / `shared_shift_no_witness` 停止；新夹爪提议未派发。总计
113 个周期中 112 个采样端点满足原位姿双阈值。新参考六周期的位置 / SO(3) /
opening RMSE 分别为 `0.008030 m / 0.022589 rad / 0.044444`；末端 step113 位置
误差 `0.010223 m`，高于 0.010 m 门，旋转误差 `0.034966 rad`。

| 新参考端点 | 108 | 109 | 110 | 111 | 112 | 113 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生任务成功 | 真 | 真 | 真 | 真 | 假 | 真 |
| 目标 grasp | 真 | 假 | 假 | 假 | 真 | 真 |
| 原位姿双阈值 | 真 | 真 | 真 | 真 | 真 | 假 |

同一初态的旧参考此前在 step108 停止；此次继续到 step113，但参考不同，不能
将两条参考的跟踪误差直接当作相同目标对照。采样成功不连续，未执行 600 周期
及最后 40 周期稳定门，更不能推断不同初态、多任务、实时或真机效果。

独立 checker 从 reset 重放全部 113 条实际命令并重采 step107 图像；命令、完整
状态、TCP 误差、内步 q/qvel/time 及夹爪反馈最大差均为 0，grasp 与任务标签
一致。checker 不加载候选策略，只记录 producer 的质量停止，不重新求解停止原因。
producer 和 checker 各实际执行 113 周期、2,825 内步、3 仿真模型和 3 OSMesa
context，均实际 wait/reap、exit 0、未发信号；本轮无新 WAM、GPU 或 fresh。
累计同系列 1,531 周期、38,275 内步、78 仿真模型/context，另 2 个静态模型；
世界模型初始化 9 次、请求尝试/完成 21/18 次、去噪 630 次。32 并发专项 51 项
通过、0 skip，零物理 dry 通过。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/causal_replan_step107_execution_20261112/producer/result.json` | `6767c33c08ffc3f55b786599435bd45b08f42807054079d05ac8f4763497ec9e` |
| `outputs/causal_replan_step107_execution_20261112/checker/result.json` | `f7960d9fbc1f0cd08a85ac4eef40b0fcba1b00786ea972d971f0f379bf30d792` |
| `outputs/causal_replan_step107_execution_20261112/producer/case_00/result.json` | `0d6fd10965e139b6dab8e592740b58bc9f724da39ab5ea4e3a43f9502516bdd9` |
| `outputs/causal_replan_step107_execution_20261112/checker/case_00/result.json` | `b18e7be755be47a012cbd712838865c87a14d474cb4125fa4e2ab63bd3a57260` |
| `outputs/causal_replan_step107_execution_dry_20261112/result.json` | `618c74afc8ca3fd76aef94c8e0bc6008ddf36aa89a501f77017d0731290e01fa` |
| `outputs/qa/causal_replan_step107_execution_20261112/result.json` | `ffa87d0997e85521bd57772ca7245cf92cdf9f388d58cce79a7db8465c7dfe63` |
