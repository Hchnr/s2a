# 长窗缓存 checker 的部分失败

2026-11-13。[预登记的独立缓存重放](../../../plans/v0.0.1/tracking/references/adaptive_long_cached_checker_20261112.md)
只执行一次，在第 119 周期后的第三次当前图像核对中失败，未到达原先第 147
端点。worker 返回码 1、未收到信号，已实际 wait/reap；原始失败收据未覆盖。

## 实际证据

第 107–118 的 12 条自适应动作与生产者完全相同，300 个内步、状态、TCP 和
夹爪反馈的最大绝对差分别由只读审计保存；完整状态最大差为
`9.744489937180134e-12`，任务和 grasp 标签一致。第 112–119 端点的状态
SHA 不同，因此不能通过原严格端点身份检查。第 107、111 的两次已完成图像
核对通过；第 119 的原始相机及 PNG 文件和生产者逐字节一致，错误信息中的
“图像或端点不同”由已知状态哈希差异触发，而非已保存图像字节不一致。

缓存事件记录第 111 和 119 端点各一次复用，首次哈希差异出现在随后第 112
端点。减少重复端点观察可能改变了后续计算，但目前只有时间顺序，不能单独
证明因果；原第 147 端点失败的具体原因也仍未确定。两次失败均不计作完整
checker 通过，不放宽现有哈希、状态或图像检查。

## 开销与验证

本次实际执行 119 个控制周期、2,975 个物理内步，建立 3 个仿真模型及 3 个
context；无新 WAM 初始化、请求或去噪。含先前未知范围的累计值为
2,174–2,289 周期、54,350–57,225 内步及各 93–96 个模型/context；此前
另有 2 个静态模型。WAM 累计初始化 11 次、请求尝试/完成 30/27 次，完成
去噪 945 次，均未变化。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/adaptive_long_cached_checker_20261112/checker/result.json` | `83375f747c6e910d5f8d9d9c3c82b48652f8dcceaa7b825876fabd97002ab171` |
| `outputs/adaptive_long_cached_checker_20261112/checker/simulation/result.json` | `3430e619bc5ce1587c0aff8f1d34917ab4af2835c52841b40bd9b62bcf9d69af` |
| `outputs/adaptive_long_cached_checker_20261112/checker/workers.json` | `10a5ae60337cd57aab55e0262fed2f4ba3752b9f13d59e91a0449340c104fb35` |
| `outputs/audits/cached_checker_incident_20261113/result.json` | `9aad9b9be9bb8cedb2a4c61295b4ff524cb52e689b3e39708651812e039c1f39` |
| `outputs/qa/adaptive_long_cached_checker_final_20261112/result.json` | `ae1df8566a278e96a689945f17deca0cd56a08e272716c9ad1d4a361d9ee432c` |
| `outputs/qa/cached_checker_incident_20261113/result.json` | `183f3b4c7f26543c6ab5c90350dea7fba4229490c6181717998aab34a56d3d08` |

运行前 64 并发相关回归 304 项、事件审计后 32 并发相关回归 60 项通过，均零
跳过、源码一致且 worker 已回收。该单开发初态仍有第 129 端点任务失败、末尾
仅 18 周期成功，600 周期及 24 条留出验收尚未完成。
