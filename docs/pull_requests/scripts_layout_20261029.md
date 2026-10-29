# 中文版

## 标题

refactor: 完成实验脚本的领域分组与调用方适配

## 变更

- 按已落盘设计迁移剩余690个脚本；scripts根目录只保留README和包入口。
- 更新57个测试调用方、当前owner/worker路径及来源身份清单，累计登记3280个历史身份。
- 分开验证历史字节与当前源码，保留原算法、容差、预算和实验门槛。

## 验证

- 全仓2487个Python文件Ruff/格式、静态导入边界及迁移身份检查通过。
- 发现5172个测试节点，原节点/参数/顺序保持；数值环境仍有5项既有PyArrow/NumPy导入错误。
- 816项CPU初测781通过、34失败、1跳过；34项迁移适配失败经分轮修复复核通过，未在最终版本重跑全部816项。
- 另在仓库已有兼容Arrow环境中执行14项，全部通过。CPU回归最多32进程，GPU保持禁用。

## 待完成

- 整理439个测试文件、898份配置/计划/报告及README导航，执行最终全量CPU验收。
- 本次为阶段性结构整理，未新增研究实验或控制成功结论，总目标尚未达成。

# English version

## Title

refactor: finish grouping experiment scripts by domain and update callers

## Changes

- Move the remaining 690 scripts according to the documented design; retain only the README and package entry at the scripts root.
- Update 57 test callers, current owner/worker paths and the migration inventory covering 3,280 historical identities.
- Verify historical bytes separately from current sources; preserve algorithms, tolerances, budgets and experiment gates.

## Validation

- Ruff/format checks pass for all 2,487 Python files; static import boundaries and migration identities pass.
- Discovery retains the original nodes, parameters and order within 5,172 items; five existing PyArrow/NumPy import errors remain in the numerical environment.
- The initial 816-case CPU run yielded 781 passes, 34 failures and one skip. All 34 migration failures passed after staged repairs and checks; the entire 816 cases were not rerun against the final version.
- All 14 cases passed separately in the existing compatible Arrow environment. CPU regression used up to 32 processes; GPUs remain disabled.

## Remaining work

- Organize 439 test files, 898 configuration/plan/report files and README navigation, then run final full CPU validation.
- This is a partial structural migration. No new research experiment or control success is claimed; the overall goal remains unfinished.
