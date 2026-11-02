# Cap-distance GN 设计 PR 信息

## 中文

```markdown
# docs: 明确 cap-distance GN 研究设计与执行前置条件

记录使用距原执行 caps 的误差构造 GN 目标的有限 CPU 对照设计，针对当前总误差下降、后三点误差反而上升的问题。

- 定义完整投影残差、Jacobian 和同单位的 pose 下降门。
- 区分既有 FK 目标与新的因果 GN 组合，不宣称首次提出该目标。
- 明确预测器改变时必须重新验证初始化与探针，不能继承旧 exact 标签。
- 保留原候选拒绝/缩域状态机、严格数值比较、安全、预算和控制验收门槛。

验证：UTF-8、11 个本地链接目标及 Git 空白检查通过。

本次仅提交设计和 PR 文案，没有算法实现改动，未运行新的测试或实验。数值契约前置条件尚未关闭，GPU 继续禁用；本设计不构成跟踪或任务成功证据。
```

## English

```markdown
# docs: specify cap-distance GN design and execution prerequisites

Document a bounded CPU comparison using distance to the original execution caps as the GN objective, addressing lower total error accompanied by worse error at the final three points.

- Define the full projection residual, Jacobian and pose reduction gates in consistent units.
- Distinguish the existing FK objective from its proposed combination with causal GN.
- Require separate validation of initialization and probes when the predictor changes; retain historical exact labels only for their original predictor.
- Preserve candidate rejection and radius updates, strict numerical comparisons, safety constraints, budgets and control acceptance gates.

Validation: UTF-8, 11 local link targets and Git whitespace checks pass.

This change contains design documentation and PR text only. No algorithm implementation changed, and no new tests or experiments were run. Numerical prerequisites remain unresolved, GPU execution stays disabled, and the design provides no tracking or task-success evidence.
```
