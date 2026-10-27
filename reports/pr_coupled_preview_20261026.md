# feat: 新增四步位姿预览控制与可审计物理回归

## 改动

针对受限单步控制在保持阶段的跟踪误差，新增最多四步已生成参考轨迹的位姿/速度预览求解器，保持命令幅值及相邻周期变化限制。运行时仅使用当前测量、因果历史和当前已生成 chunk，并区分预览预测与首命令连续响应预测。

补充浮点边界修正及原生控制器显式 goal reset，独立复核可行性、ODE 响应和 KKT 条件。新增记录回放、连续历史链、日志错误注入、物理回归配置及采集/核验/退出审查流程。

保留首版检查器的失败证据，修正运行时来源清单后复用已采集 case00，从 case01 续跑，避免重复采集；新增相关测试、计划与实验报告。

## 验证

- 本次 44 个新增 Python 文件 Ruff 与格式检查通过；新增 19 项 CPU 测试通过（3.12 秒）。
- 已完成 18 条/4907 周期连续记录链核验，以及 164 周期完整日志核验和 22 项错误注入拒绝。这些是接口与回放证据，不代表候选闭环任务成功。
- 提交前快照：独立物理仿真的 case00–02 均独立核验且性能达标；完整 18 条结果及最终退出审查尚未完成。
- 上次提交 9753cdd 的全套 CPU 测试：2656 通过、7 失败、12 错误、1 跳过、1 排除。涉及历史路径、已有实验产物假设及 worker 启动检查，本次未修复、未重跑全套。

## 限制

正确参考回归仍须 18/18 达标，不能凭前三条结果进入 WAM 开发或最终 fresh 验收。本候选尚无 WAM/fresh 成绩，总目标未完成。

---

# feat: add four-step pose preview control and auditable physical regression

## Changes

Add a pose/velocity preview solver using up to four generated reference poses to address tracking errors during constrained single-step holding. Preserve command bounds and adjacent-step rate limits. Runtime inputs are limited to current measurements, causal history, and the current generated chunk; preview predictions remain separate from the continuous first-command response forecast.

Add floating-point boundary correction and explicit native controller goal reset, with independent feasibility, ODE response, and KKT checks. Include record replay, continuous-history validation, log corruption checks, a registered physical regression configuration, and collection, verification, and process-exit auditing.

Preserve the original checker failure, correct the required runtime source list, reuse the collected case00, and resume from case01 without recollection. Include tests, plans, and experiment reports.

## Validation

- Ruff and formatting checks pass for all 44 added Python files; 19 new CPU tests pass in 3.12 seconds.
- Completed continuous record-chain checks for 18 trajectories / 4,907 periods, plus 164 complete log periods and 22 corruption rejection checks. These establish interface and replay evidence, not candidate closed-loop task success.
- Pre-commit snapshot: physical simulation cases00–02 independently verified and meet performance thresholds. The full 18-case regression and final process-exit audit remain incomplete.
- Full CPU suite on the preceding commit 9753cdd: 2,656 passed, 7 failed, 12 errors, 1 skipped, 1 deselected. Failures concern historical paths, assumptions about existing experiment artifacts, and worker startup checks; this change neither fixes them nor reruns the full suite.

## Limitations

All 18 correct-reference cases must qualify before WAM development or final fresh evaluation. Three passing cases do not satisfy that gate. No WAM/fresh performance is established for this candidate; the overall objective remains unmet.
