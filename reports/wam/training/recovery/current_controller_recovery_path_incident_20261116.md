# 当前控制器训练恢复采集：路径失败记录

2026-11-16。按[预登记](../../../../plans/v0.0.1/wam/training/recovery/current_controller_recovery_20261116.md)
对三个训练初态、三个偏移启动九个 CPU/OSMesa producer；九例在 reset 后、
第一条控制动作前全部失败。没有完整轨迹、独立重放或训练恢复资格。

## 原因与实际资源

源码 `cbfa732b88fdb067f5c084c43d08fb1a44518ef6` 把当前图像输出放在
`worker/capture`；实际运行时只允许写入 `worker/runtime` 及其子目录，首次
创建 `anchor0000` 被路径保护拒绝。9 个 worker 均返回失败，实际 wait/reap；
每个 worker 的运行收据记录 3 个模型和 3 个 context 构建完成并清理，合计
27 个模型、27 个 context。实际执行 **0 个控制周期、0 个 MuJoCo 内步**，
0 次 WAM 调用、0 个 fresh 案例；未运行 checker、拟合或 GPU 任务。

原运行前 64 并发相关回归为 176 项通过、0 跳过，零物理来源预检也通过；
它们未在真实运行时的写入保护下渲染图像，因此未覆盖本次故障。旧产物保留，
没有覆盖或自动重试：

| 证据 | 仓库内路径 | SHA-256 |
| --- | --- | --- |
| 运行前回归 | `outputs/qa/current_controller_recovery_final_20261116/result.json` | `a2f8e9cc54969d4dcee639ab5d34a796224dcc53f91f6bc555b8102edc4c5a27` |
| 零物理预检 | `outputs/current_controller_recovery_20261116/dry.json` | `6b44ab94f0ad8944077b5f1ab1e4f75d1cb6e17dbaa6ee4e3ec9de6253a58677` |
| producer 汇总 | `outputs/current_controller_recovery_20261116/producer/result.json` | `7bd7ab5d8ed8bf9e33358df9c03d4bada2588b216afa1aea09540ff3e934cde0` |
| worker 生命周期 | `outputs/current_controller_recovery_20261116/producer/workers.json` | `95d899744eafa8a7b1d86061c2d803d23c0e4347685db64ce6098165da952518` |

## 修正与边界

已将采集目录移到 `worker/runtime/capture`，仅在运行时建立其目录后、实际
continuation 内创建；增加目录保护回归，核对旧路径被拒绝、新路径可写且不
覆盖。此修正尚未经过真实 OSMesa 渲染，不能宣称物理采集合格或九例成功。
源码变更也使原有回归与预检不能作为修正后的准入证据。

后续若继续实验，须保留本次失败和资源账本，另行固定新输出及预算、重做
源码匹配的 CPU 回归和零物理预检，再先进行一个实际运行时的零动作图像预检；
其结果不替代九例生产及独立重放。原参考为特权正确轨迹，不是 WAM 生成，
Action128 的开度槽为下一端点实测开度，不等于另存的实际控制命令；本轮不授予
拟合数据资格，也不改变最终验收状态。
