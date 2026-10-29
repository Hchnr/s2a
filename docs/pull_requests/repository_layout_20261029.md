# 中文版

## 标题

refactor: 按领域重组 S2A 库与共享实验入口

## 变更

- 落盘全局结构设计、逐文件迁移计划和来源身份清单。
- 库按几何、控制器、参考、后端、评价和研究领域分层；src/s2a 根目录仅保留包入口。
- 迁移首批775个共享/审核脚本，更新调用方的明确包导入和当前文件定位。
- 分开核验历史字节与当前源码；保留独立审核及原控制算法、容差和实验门槛。

## 验证

- 全仓2470个Python文件Ruff和格式检查通过。
- 测试发现5172项，原节点及参数保持；仍有5项既有PyArrow/NumPy环境导入错误。
- 修复回归覆盖35个模块486项：482项在模块回归中通过，剩余4项独立复核通过。
- CPU测试采用32进程；未使用GPU、未启动新研究实验，未执行完整pytest。

## 待完成

- 剩余690个脚本、439个测试文件、898份配置/计划/报告及README导航整理。
- 本PR是阶段性迁移；完整结构验收与研究总目标尚未达成。

# English version

## Title

refactor: organize S2A libraries and shared experiment entrypoints by domain

## Changes

- Document the global architecture, per-file migration plan and source identity inventory.
- Group libraries into geometry, controllers, references, backends, evaluation and research packages; retain only the package entry at the src/s2a root.
- Move the first 775 shared/audit scripts and update explicit imports and current file locations.
- Verify historical bytes separately from current sources while preserving independent audits, control algorithms, tolerances and experiment gates.

## Validation

- Ruff and formatting checks pass for all 2,470 Python files.
- Discovery retains the original test nodes and parameters within 5,172 collected items; five existing PyArrow/NumPy environment import errors remain.
- Repair regression covers 486 cases across 35 modules: 482 passed in the module run and the remaining four passed separately.
- CPU tests use 32 processes. No GPU or new research experiment was run; the full pytest suite was not executed.

## Remaining work

- Organize the remaining 690 scripts, 439 test files, 898 configuration/plan/report files and README navigation.
- This PR is a partial migration; full layout acceptance and the overall research goal remain unfinished.
