# 中文版

## 标题

对照固定初始全局搜索上限并完成35响应独立复算

## 描述

逐项固定搜索上限会拒绝部分安全且降低最坏误差的数值候选。仅将该上限替换为从
初始最大归一化平方误差推导的共同固定值；原执行caps、控制器、安全、初态、72条
差分、因果反馈、信赖域及24阶段/122响应上限保持。新上限全路径一致并在独立
审计中从初始数据重建，不能用于授予放宽后的执行资格。

增加计划、独立状态机审计、执行接线测试与完整native复算入口，保留前项代码及
失败证据。更早已尝试全轨迹epigraph，本轮只检验共同上限在当前搜索框架中的作用。

## 验证与结果

- 116项专项CPU测试、Ruff、格式检查及两条真实输入dry通过，冻结146来源。
- 24次QP尝试中21合格、21接纳；3个QP因原互补证书门拒绝，未放宽容差。
- 35完整生产响应及35完整独立复算通过，严格分类无分歧；来源关闭2631项。
- 最坏平方超限较前项下降15.57%，总误差却增加14.06%，最大位置误差约24mm。
  前缀仍4，双caps点5→6，完整witness0、无执行选点，实际仍cursor8/task0/grasp0。
- 只用35/122响应即达到24阶段上限，末段仍下降；未追加旧预算。两路线实际合计
  70响应、21000数值内步、21842查询，无新物理积分、world-model调用或fresh。

GPU继续禁用，原18正确参考/12WAM开发/24最终留出门槛保持。下一项应独立登记
阶段预算对照，不将当前局部下降外推为完整控制成功。实现提交：`ed28c28`。
详见[完整报告](global_envelope_search_20261029.md)。

# English version

## Title

Compare a fixed initial global search envelope and independently replay all 35 responses

## Description

Per-item fixed search ceilings reject some safe numerical candidates that reduce the worst
error. Replace only this numerical envelope with one fixed value derived from the initial
maximum normalized squared pose error. Preserve the original execution caps, controller,
safety, initial state, 72 saved finite differences, causal feedback, trust rules, and upper
budgets of 24 stages and 122 responses. Independently reconstruct the envelope from initial
data and use it consistently throughout; it never grants relaxed execution qualification.

Add preregistration, an independent state-machine audit, execution-binding tests, and complete
native replay. Preserve earlier code and failures. Full-trajectory epigraph search was already
attempted; this run tests the envelope specifically within the current search framework.

## Validation and results

- 116 targeted CPU tests, Ruff, formatting, and both real-input dry runs pass; 146 sources freeze.
- 21 of 24 QP attempts qualify and produce 21 accepted anchors. Three fail the original
  complementarity certificate without any tolerance relaxation.
- All 35 production responses receive complete independent replay, without strict classification
  disagreements. Saved-data closure verifies 2,631 sources.
- Worst squared excess falls by 15.57% relative to the previous final anchor, while total error
  rises by 14.06% and maximum position error reaches approximately 24mm. Prefix remains 4,
  both-cap endpoints rise from 5 to 6, and complete witnesses remain zero. No executable
  candidate is selected; actual progress remains cursor8/task0/grasp0.
- The 24-stage limit terminates the run after 35 of 122 responses, with continued late descent.
  No budget is appended. Both routes total 70 responses, 21,000 numerical inner steps, and
  21,842 queries, with no new physical integrations, world-model calls, or fresh cases.

GPU use remains disabled and the original 18/12/24 acceptance gates remain unchanged. A
separate stage-budget comparison is the next evidence-based step; local descent does not
establish complete control success. Implementation commit: `ed28c28`.
See the [full report](global_envelope_search_20261029.md).
