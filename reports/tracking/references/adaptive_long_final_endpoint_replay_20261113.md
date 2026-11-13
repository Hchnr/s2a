# 长窗：最终端点缓存的独立重放

2026-11-13。按照[预登记](../../../plans/v0.0.1/tracking/references/adaptive_long_final_endpoint_checker_20261113.md)，
从同一旧开发初态独立重放原生产者的 147 条动作，完成 3,675 个物理内步；
worker 正常退出、实际 wait/reap，没有重新选择动作或调用 WAM。较早端点沿用
原观察路径，第 147 端点分别记录一次实际观察和一次步号、状态 SHA、时间完全
相同的缓存复用。

独立 checker 的命令、实际状态、TCP 误差、内步关节位置/速度/时间及夹爪
反馈相对生产者的最大绝对差均为 **0**，任务和 grasp 标签一致；第 107、111、
119、120、128、136、144 周期的七组原始相机和 PNG 文件逐字节一致。单独
进程重新读取收据并复核上述比较、缓存事件及 worker 退出。

40 个自适应端点中原生任务成功 39 个，第 129 端点失败，末尾连续成功 18 个；
自适应阶段目标 grasp 为 1/40。147/147 位姿端点达到当前门，TCP 位置 /
SO(3) 旋转 RMSE 为 `0.0029684699840395548 m` / `0.01817749342379491 rad`，
最小内步关节余量 `0.3871836971502849 rad`。位姿跟踪与一次开发轨迹重放
不代表稳定完成任务；600 周期、三任务、24 条留出新初态、实时与真机仍未验证。

原第 147 端点重读失败和从第 107 周期缓存后在第 119 端点失败均保留；本次
通过不证明两次历史失败的具体原因。此前物理成本的不确定范围未消除，本次
累计为 2,321–2,436 周期、58,025–60,900 内步及各 96–99 个仿真模型/context，
另有 2 个静态模型；WAM 初始化 11 次、请求尝试/完成 30/27 次、完成去噪
945 次均未增加。

| 仓库内证据 | SHA-256 |
| --- | --- |
| `outputs/adaptive_long_final_endpoint_checker_20261113/checker/result.json` | `53424f2e0ea3bbc89c40d515a585c7e74d7995a38c48e0fcb9e406ad4d85063a` |
| `outputs/adaptive_long_final_endpoint_checker_20261113/checker/simulation/result.json` | `d7e32ace9ee3c5edc2826a33e2b2441e20feba1b4ab4c108f8c0416b5efb82bb` |
| `outputs/adaptive_long_final_endpoint_checker_20261113/checker/workers.json` | `7cdad2f219133a61aee7b8680bec88dc7688ba00f565bce5e96949f4d68bcb44` |
| `outputs/qa/adaptive_long_final_endpoint_checker_revalidated_20261113/result.json` | `35af6863793cae276203bd6cd9ac3736d21f4d741c3bba885e05315446e2e37f` |
| `outputs/adaptive_long_final_endpoint_checker_dry_revalidated_20261113/result.json` | `bd972e11ea13aa1acef871c74c230cc926ece84c4799606a2bcdb2322a3275cc` |

最终 64 并发相关回归 341 项通过，零跳过及收集错误，源码一致且全部 worker
已回收。首次 64 并发回归的两项当前目录异常及其单项复验均按计划保留。
