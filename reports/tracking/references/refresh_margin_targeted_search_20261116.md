# 刷新余量定向搜索的静态结果

2026-11-16。按[固定方案](../../../plans/v0.0.1/tracking/references/refresh_margin_targeted_search_20261116.md)，
在[已执行并独立重放](refresh_margin_physical_20261116.md)的 bowl / seed 43
step 16 状态，以当时原求解选中的八周期计划为起点，额外搜索更有余量的计划。
状态、上一条动作、历史误差及已生成参考均来自该时刻；没有把之后的实际状态或
任务标签作为输入。

起点的八点误差与原独立 witness 逐值核对通过，第一 / 第二点位置预测约为
`0.00766327 / 0.00950000 m`，旋转预测约为 `0.02189612 / 0.04326684 rad`；
后六点基线最大归一化 slack 为 `0.07639919366818071`。起点通过原线性动作、
关节、制动和终端条件。额外 SLSQP 定向搜索已完成 **320 次模型响应**，没有
发现同时通过原条件、第二点新增余量和 `baseline + 0.02` 未来 slack 限制的
候选；第 321 次响应请求被预算阻止。优化器中断时迭代数未知，没有候选可供
独立 witness 复核。

该结果说明在这个起点与固定预算下本次搜索未找到候选；它不是可行域为空的证明。
局部模型证据不代表物理安全、跟踪收益、实时性或任务成功，不能据此登记物理
候选为通过。原基线在该步已使用 126 次质量响应、120 次优化迭代，本次 320 次
为额外开销，并未实现原实时控制预算内的搜索。

静态 worker 完成 1 次 MuJoCo 模型构造、1 次预览模型构造和 1 次静态 forward；
未执行物理步、渲染、WAM、GPU 或 fresh 案例。worker 正常退出并已 wait/reap；
源码和输入前后一致。Ruff 检查与格式检查通过。

## 回归中的历史来源失败

首次 64 worker 回归运行 76 项，75 通过、1 失败，所有 worker 均已 wait/reap。
失败为 `test_guarded_decisions_and_recorded_results_are_bound`：旧实验登记的
`staged_first_feasibility.py` SHA 为 `00cca7b0…`，与历史提交 `1dbf7c2a` 的
blob 一致；当前 `origin/main` 的该文件 SHA 为 `bb463fef…`。该测试要求旧
登记与当前源码完全相同，输入本身对比一致，源码身份不一致仍应拒绝。此次没有
更改原测试或历史登记。第二次 64 worker 回归显式选择其余 75 项，全部通过、
无 skip；它不代表首次失败已修复。历史实验需要明确的源码版本桥接后才能恢复
该测试的完整准入。

| 本地证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `refresh_margin_targeted_search_dry_v2_20261116.json` | `4b065db08c4616874e7b598298f5be51948db6a76cde23bd2fed6b56610a67e5` |
| `refresh_margin_targeted_search_20261116/result.json` | `c6e8965205d0ffdfc0ce9465119a202f7e76ed39138ccf306d9dfc41b3c109ee` |
| `refresh_margin_targeted_search_20261116/worker/result.json` | `fb415f446d11dfe30892e0093b989a773e848041802f5be2f2f4732e9ec56e8f` |
| `qa/refresh_margin_targeted_final_20261116/result.json` | `e292628ca20de2536f2872bf9a992232c8dd23aeb4b8a43aee678b7717cb4df1` |
| `qa/refresh_margin_targeted_final_v2_20261116/result.json` | `20e3ad03705445f635fd6576e9c18e63224e07a739402655a2311cee5a7d0b86` |

结果与 QA 保存在本仓库 `outputs/`，不随 Git 提交；两次回归和首次失败均保留。
