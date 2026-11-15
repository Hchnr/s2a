# 历史源码身份校验修复

2026-11-16。旧 staged、stopped、two-step、guarded、margin 与 task-2
shared-shift 证据链，曾把历史登记的源码 SHA 与当前文件 SHA 直接比较。旧
`staged_first_feasibility.py` 等源码后续演进后，即使历史输入未变，原测试仍
在身份检查处失败。没有修改原测试、旧登记或旧输出。

按[设计](../../../plans/v0.0.1/development/provenance/historical_source_bridge_20261116.md)
和[迁移记录](../../../docs/migrations/historical_solver_source_bridge_20261116.md)，
现在以固定清单分别校验历史 Git blob 与当前文件的 SHA。原 source refs 的路径、
顺序、数量及清单外字节保持严格校验；新运行的 source refs 额外绑定桥接代码与
清单。清单固定 15 处新旧源码身份，其中部分是本次为增加桥接而修改的来源读取器。
历史输入收据继续使用原字段；新的 margin-guarded 读取收据另保存各登记边界的
桥接证明。未来对应当前源码变化时必须明确更新清单，不可静默接受。

对实际历史静态链的只读核验覆盖 4 个原开发决策、6 个来源边界，并记录每个边界
的原引用数、已桥接路径和 SHA；当前物理对照的 16 周期历史准入也仍通过。
这些是来源与输入核验，没有执行历史代码、物理步、WAM、GPU 或新初态，不授予
控制或任务完成资格。独立 checker 的候选模块隔离测试通过。

## 测试与原失败

首轮相关回归为 76 项中的 75 通过 / 1 失败；扩展反馈链时 56 项中的 53 通过 /
3 失败；单独扩展 shared-shift 时 7 项中的 4 通过 / 3 失败。这些失败收据和
各自源码快照仍保留，不能改写为通过。完成桥接后，包含原失败用例和 shared-shift
干净 checker 的最终所选回归 **75 项全部通过**，0 skip / xfail / xpass / 收集
错误；配置并实际使用 64 个 worker，全部实际 wait/reap，源码在回归前后一致。
Ruff 检查和格式检查通过；这不是全仓测试结论。

| 本地证据（相对 `outputs/`） | SHA256 |
| --- | --- |
| `qa/refresh_margin_targeted_final_20261116/result.json` | `e292628ca20de2536f2872bf9a992232c8dd23aeb4b8a43aee678b7717cb4df1` |
| `qa/historical_source_bridge_broad_20261116/result.json` | `8f48c5874b1380b4a517f12bc72dfc6af3ec43db9e9ad7ea16b6b0037fc2447a` |
| `qa/historical_shared_shift_audit_20261116/result.json` | `9524a496f1a21796b47c21f274d636b5ad927c218c93bf21a5fef88ab5864e63` |
| `qa/historical_source_bridge_complete_20261116/result.json` | `81ae310801e204a5502e88b41d5b1ebd9d45970b37044bdcddba945719887fe4` |
| `historical_source_bridge_evidence_20261116/result.json` | `48df78ce80e5b62edb506c10879542e4304e4a65183adde57f4e2b54dae4320b` |

证据与 QA 保存在本仓库 `outputs/`，不随 Git 提交；原 18 条正确参考、12 条 WAM
开发、24 条最终留出及实时 / 真机要求均未改变。
