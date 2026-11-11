# 有限释放参考修订的全分母影子检查

2026-11-11。按[预登记方案](../../../plans/v0.0.1/tracking/joint_control/bounded_release_revision_20261111.md)，
在 18 条已合格正确参考记录和 12 条既有 WAM 开发记录上，按时间重放已生成的参考
与基线已完成历史，检查新旧 opening 冲突。另对已有独立物理重放的 task1 闭合状态
检查真实前一周期内环证据和候选构造。候选没有派发动作，没有产生新物理、WAM、
GPU 或最终留出结果。

| 固定记录 | 基线控制周期 | 已生成窗口 | opening 冲突 | 内环未知而不能激活 | 实际选择保持 / 衔接 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 18 条正确参考 | 4907 | 619 | 0 | 0 | 0 / 0 |
| 12 条 WAM 开发记录 | 7200 | 900 | 20 | 20 | 0 / 0 |

冲突要求已完成夹爪状态为 closing，且新参考相对当前实测 opening 与同一绝对端点
的旧参考均上升超过 0.04，旧参考仍不高于实测开度 + 0.04。它不等于错误释放或
实际抓取。20 个潜在冲突来自 6 条相关开发轨迹：task0 seed42 raw 8 次、task0
seed42 terminal058 6 次、task0 seed43 terminal058 1 次、task1 seed42 raw
1 次、task2 seed42 terminal058 3 次及 task2 seed43 terminal058 1 次。19/20
满足旧首目标与当前 TCP 的 0.03 m / 0.15 rad 描述尺度；这不是内环安全证明。
原基线未保存各次前一周期完整 25 点实际内环，故 20 次都明确记为未知，不伪造
0.05 rad 准入、不选择旧参考，也没有改变基线轨迹。计数是相关窗口数，不是独立
任务成功率或新策略执行后的触发率。

正确参考的 619 个窗口没有上述新旧开度冲突；包括小开度合法释放在内的基线参考
均未被本影子选择改变。该结果只检查所保存的无冲突参考，不证明遇到有噪声修订
时仍能及时释放，也不授予新候选原 18 条物理控制资格。最后不足 8 周期的正确参考
窗口被计入时间线，但选择状态不假装完成整个 stride。

## 已研究闭合状态的有限接线检查

在 task1 seed42 terminal058 的端点368，旧参考来自step344、最新参考来自step360；
它们均在选择前已生成。前一完成周期实际 25 个内环样本及独立重放逐项一致，最小
关节余量 `0.802340 rad`；已完成夹爪状态为 closing，当前实测 opening
`0.254870`。固定候选判为 `bounded_hold`，先选 8 个旧的同时间 pose/opening 端点，
并构造随后最多 8 点的衔接预览。选择没有执行，状态在零执行后取消。

所选旧参考与最新参考在这 8 点的最大**参考间**位置 / SO(3) 差为
`0.051318 m / 0.439672 rad`；这不是实际跟踪误差，明显大于原实际跟踪评价门槛，
必须如实报告。保留接触可能显著牺牲最新目标跟踪；候选没有以此证明抓取保持、
安全释放、任务完成或新状态下的反事实。前期旧参考分支虽有短时 grasp / 抬升，
其后续动作由当时的完整预览决定，不能直接追认为本候选的已执行结果。

## 来源和独立复核

正确参考每个窗口的完整保存目标逐点与有 SHA 的原始正确轨迹和 TCP 合同核对，
实际消费端点与已审核执行记录对齐；12 条开发数据绑定原审查过的生成窗口、控制
日志及工具合同。checker 不导入候选或生产入口，独立核对 1519 个窗口的时间、
opening 条件、当前姿态尺度、选择和状态，并对已研究状态独立构造线性 / SO(3)
衔接；生产与复核结果一致。它不推断缺失内环，也没有重新执行动作。

生产与 checker 各 31 个 CPU worker 均 exit0、实际 wait/reap 且未发信号，各自
监督进程也 exit0、实际 wait/reap；源码和输入前后相同。执行时提交为
`fac987228c9da2eb9e0270bab69614aa0c444994`。32 进程 CPU 专项 115 项通过、
0 跳过；全仓 Ruff / 格式检查和零模型 dry 通过。未重跑全量 pytest，结果只表示
影子生产与复核闭合，`control_qualified=false`。

| 证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `bounded_release_revision_shadow_20261111/producer/result.json` | `cbde160b26f3590bfd0e14614acd9dfa5aa3ca698ac4f3b3502a897d8d0f3881` |
| `bounded_release_revision_shadow_20261111/producer/workers.json` | `ddf4dfe1b4cd80e576fe55ae22421b5789a414ed7ef75b2a9cc79b63d5390ce2` |
| `bounded_release_revision_shadow_20261111/checker/result.json` | `a233813716675decc497f7d3641d85f8f8f5fb337779d1f8e5cc7e5463992d5d` |
| `bounded_release_revision_shadow_20261111/checker/workers.json` | `1058707d7d146cb8a3528885d0ba2806215164bfc950896cd08117b76f43d71e` |
| `bounded_release_revision_shadow_supervisor_20261111/producer_exit.json` | `268d1caca9e73100f74bffc150c55632ce2ddb5e16cbcab87a198ee89d6ff799` |
| `bounded_release_revision_shadow_supervisor_20261111/checker_exit.json` | `5c60ed0ebe51148bec322ef192e12799e3f730e63c22f1ac5e93ed5a90edd4a3` |
| `bounded_release_revision_shadow_dry_20261111/result.json` | `8ed199bef6eab98daf75091ca227c85d65289c641b636c9db5de93de6375d3c4` |
| `qa/bounded_release_revision_shadow_20261111/result.json` | `b62025bcb735cac0524e4b5b0e48195142fde4e2289c59993e76a1e5acb52cf2` |

下一步须先登记同起点物理对照，验证保持后的衔接能否通过原关节、实际跟踪和合法
释放要求，并单列对最新参考的误差；同时补足正常周期的实际内环证据，不能把
20 个未知机会当作可执行。原 18 正确参考、12 WAM 开发、24 最终留出及 ≥20/24、
每任务 ≥4/8 的验收门槛不变；GPU 继续禁用。
