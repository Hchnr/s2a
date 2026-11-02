# 中文版

## 标题

为pose QP增加紧活动子集乘子恢复及独立审计

## 描述

前项7个pose QP未通过原互补证书。新增固定紧活动子集上的单次NNLS恢复，
同时保留原完整KKT检查、restore恢复、目标缩放及外层接受规则。
补充独立集合重建、拟合调用计数、生产/复算入口及显式配置；集合不足、空集合
或拟合异常仍按原门槛拒绝，不追加拟合或预算。

## 验证与进度

- 182项专项CPU测试通过，182来源指纹保持。
- 新文件Ruff/格式检查及两条真实输入dry通过；dry新增QP/响应/查询均0。
- 全仓样式检查仍有4个未修改文件的既有问题，详见报告。
- 生产搜索和完整独立复算尚未执行，跟踪收益待验证。
- 实际仍cursor8/task0/grasp0；原验收门槛保持，GPU继续禁用。

实现提交：`d527982`。详见[实现与准入报告](state_vector_dual_recovery_implementation_20261029.md)。

# English version

## Title

Add tight active-subset dual recovery and independent auditing for pose QPs

## Description

Seven prior pose QPs failed the original complementarity certificate. Add one NNLS
recovery on a fixed tighter active subset while retaining the full original KKT checks,
restoration recovery, objective scaling, and outer acceptance rules.
Add independent subset reconstruction, fit accounting, production/replay entry points,
and explicit configuration. Insufficient or empty subsets and fit errors remain subject
to the original gates, without additional fits or budget extensions.

## Validation and status

- 182 targeted CPU tests pass; 182 source fingerprints remain unchanged.
- Changed-file Ruff/format checks and both real-input dry runs pass; dry runs add
  zero QPs, responses, or queries.
- Repository-wide style checks still report existing issues in four unchanged files;
  see the report for details.
- Production search and full independent replay have not run; tracking benefits remain unverified.
- Actual progress remains cursor8/task0/grasp0. Original acceptance gates remain in place,
  and GPU use stays disabled.

Implementation commit: `d527982`. See the [implementation and admission report](state_vector_dual_recovery_implementation_20261029.md).
