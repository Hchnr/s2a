# 仓库结构整理 PR 信息

## 中文

```markdown
# refactor: 按领域重组 S2A 并保留研究证据身份

按设计文档整理算法、实验入口、测试及研究证据，累计移动 3264 个文件；928 份原始配置、计划和报告保持字节不变。

- 建立领域包、分层导航及旧路径到当前路径的 SHA 绑定清单。
- 修复迁移后的注册资料读取、preview 入口和合成测试 fixture；保留原算法、容差及验收门槛。
- 增加 CPU 并行测试入口，全量检查使用 64 并发。

验证：2559 个 Python 文件 Ruff/格式检查通过；最终相关模块 179 项、兼容环境 40 项、两条真实 --dry、结构与导入检查通过。

全量 v2 为 5170 通过、5 失败、2 跳过，另有 5 处既有环境收集错误。5 个失败包含 4 项原数值环境差异及 1 项迁移遗漏；遗漏已修复并纳入最终 179 项复核。最终修复后未重跑全部 5177 项，不宣称全仓全绿。

GPU 保持禁用；目录整理不代表闭环控制总目标达成。完整收据见 docs/migrations/README.md。
```

## English

```markdown
# refactor: organize S2A by domain and preserve research evidence identities

Follow the documented design to organize algorithms, experiment entry points, tests and research evidence. Move 3,264 files while preserving the exact bytes of 928 configuration, plan and report files.

- Add domain packages, layered navigation and SHA-bound mappings from historical to current paths.
- Repair registered evidence reads, preview entry points and synthetic test fixtures; retain algorithms, tolerances and acceptance gates.
- Add a CPU test runner and use 64 workers for full regression runs.

Validation: Ruff/format checks pass for 2,559 Python files. The final 179 affected-module tests, 40 compatibility tests, two actual --dry runs, structure checks and import checks pass.

Full run v2 reports 5,170 passed, 5 failed and 2 skipped, plus 5 pre-existing environment collection errors. Four failures reproduce in the original numerical environment; one migration omission was fixed and covered by the final 179-test regression. All 5,177 tests were not rerun after the final fixes; this is not a claim of a fully green suite.

GPU execution remains disabled. Repository restructuring does not establish the overall closed-loop control goal. Full receipts are recorded in docs/migrations/README.md.
```
