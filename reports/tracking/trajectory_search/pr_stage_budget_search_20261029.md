# 中文版

## 标题

对照60阶段与同122响应上限，并完成90响应独立复算

## 描述

前项24阶段搜索停止时仍下降、仅用35/122响应。本次只将阶段上限改为60，其余
初态、72条差分、控制器、数值目标、执行caps、安全、信赖域及响应上限保持。
QP内部迭代上限2400→6000，不称总预算相同；在新namespace从原3557重启，不
恢复旧优化器或追加旧预算。复用冻结独立审计，增加严格预算wrapper与执行接线。

## 验证与结果

- 142项专项CPU测试、Ruff、格式及两条真实输入dry通过，155来源冻结。
- 前24完整stage/QP及34条搜索entry/request与前项exact，旧末repeat单独与新末次
  repeat比较exact，预算对照资格通过。
- 60次QP尝试中57合格，55接纳；3个原互补证书失败保持，未放宽门槛。
- 90完整生产响应及90完整独立复算通过，严格分类无分歧，2658来源关闭。
- 最坏平方超限再降6.51%，但总误差升11.07%、双caps点6→4，前缀仍4，完整
  witness0、无执行选点；实际仍cursor8/task0/grasp0。
- 两路线合计180响应/54000数值内步/56162查询。60阶段上限关闭，32响应余量未
  追加；新增物理/world-model/fresh/参考消费0，GPU继续禁用。
- 额外零新调用的后验诊断保留首次compact读取失败，隔离v2核验2663来源；88条
  已算候选的q-MAE预测均由状态向量法改善，总loss预测仍10条更差，未验证新控制收益。

增加阶段只带来部分数值进展，尚未解决完整跟踪资格；下一步先检验全state预测与
误差目标分配。原18/12/24门槛保持。实现提交`628b2ba`，详见[报告](stage_budget_search_20261029.md)。

# English version

## Title

Compare sixty stages under the same 122-response cap and replay all ninety responses

## Description

The prior 24-stage search was still descending when it stopped after 35 of 122 responses.
Change only the stage cap to 60. Preserve the initial state, 72 saved finite differences,
controller, numerical objectives, execution caps, safety, trust rules, and response cap.
The worst QP iteration budget increases from 2,400 to 6,000; total upper budgets do not match.
Restart from original response3557 in a separate namespace without restoring the optimizer
or appending the old budget. Reuse the frozen independent audit with strict budget validation.

## Validation and results

- 142 targeted CPU tests, Ruff, formatting, and both real-input dry runs pass; 155 sources freeze.
- The first 24 complete stages/QPs and 34 charged search entries/requests are exactly equal to
  the prior run. Compare the differently indexed final repeats separately; both are exact.
- 57 of 60 QP attempts qualify, producing 55 accepted anchors. Three original complementarity
  certificate failures remain rejected without relaxed tolerances.
- All 90 production responses receive complete independent replay, without strict classification
  disagreements. Saved-data closure verifies 2,658 sources.
- Worst squared excess falls another 6.51%, while total error rises 11.07% and both-cap
  endpoints fall from six to four. Prefix remains four, complete witnesses remain zero, and no
  executable candidate is selected. Actual progress remains cursor8/task0/grasp0.
- Both routes total 180 responses, 54,000 numerical inner steps, and 56,162 queries. The stage
  limit terminates the run; 32 unused responses are not appended. New physical integrations,
  world-model calls, fresh cases, and reference consumption remain zero. GPU use stays disabled.
- A saved-data diagnostic preserves its initial compact-schema failure and closes an isolated
  v2 over 2,663 sources, with no new model calls. Vector-based q-MAE predictions improve for
  all 88 saved candidates, while total-loss prediction worsens for ten. New control gains remain
  untested.

Additional stages yield partial numerical progress without complete tracking qualification.
Next inspect full-state prediction and error allocation while preserving the original 18/12/24
acceptance gates. Implementation commit: `628b2ba`. See the [report](stage_budget_search_20261029.md).
