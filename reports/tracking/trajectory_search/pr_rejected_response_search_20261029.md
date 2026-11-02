# 中文版

## 标题

利用合格的安全拒绝响应更新因果模型，并完成预算内CPU搜索与独立复算

## 描述

此前完整H12搜索在固定pose搜索上限附近停止，已计费的安全拒绝响应未用于更新
局部模型。增加一种反馈：整阶段未接纳时，选取安全且违反固定搜索上限的最小alpha
实际响应，更新有界因果Js/Jq；当前anchor和实际截距保持。复用原初态、72条差分、
完整控制接口、执行caps、接纳条件、半径规则及122响应/24阶段预算上限。

增加独立ledger与native检查，重新选择反馈来源、重建后续QP/KKT并完整复算每条
已计费响应。保留旧代码和失败证据，登记新配置与计划，更新正式报告。

## 验证与结果

- CPU专项测试：124通过；Ruff检查、格式检查与两个真实输入dry通过。
- 生产尝试24QP（23合格）、52完整响应；独立路线完整复算52响应，严格分类无分歧。
- 三次反馈均保持anchor，来源为27/45/50；末次反馈未被后续QP消费。
- 最坏平方超限较前项再降约.1698%，但cursor9和总误差略差，前缀仍4、双caps合格
  仍5/12，完整witness0、无执行选点。相同预算上限下实际比前项多8响应/3阶段。
- 达到阶段上限后正常关闭；新增物理积分、世界模型调用及fresh验收均0。
- 保存数据汇总的首次相对路径检查失败保留；隔离v2修复路径表示并核验2600来源，
  未重跑生产或独立复算。

本阶段只验证局部数值搜索与独立复算，未取得新闭环或稳定任务资格。GPU继续禁用，
原18正确参考/12WAM开发/24最终留出门槛保持。实现提交：`82840dd`。
详见[正式报告](rejected_response_search_20261029.md)。

# English version

## Title

Update causal models from qualified safe rejections and verify bounded CPU search

## Description

The previous full-H12 search stalled near fixed pose search ceilings while discarding information
from paid safe rejections. When an entire stage accepts no candidate, this variant selects the
smallest-alpha evaluated rejection that passes safety and violates a fixed pose ceiling. It applies
bounded causal secant updates to Js/Jq while preserving the current anchor and actual intercept.
The initial state, 72 saved finite differences, controller interface, execution caps, acceptance
criteria, radius rules, and upper budgets of 122 responses and 24 stages remain unchanged.

Add independent ledger and native checks that reselect feedback sources, reconstruct subsequent
QP/KKT certificates, and replay every billed response. Preserve existing code and failure evidence,
register the variant separately, and document its results.

## Validation and results

- 124 targeted CPU tests pass, along with Ruff, formatting, and both real-input dry runs.
- Production attempts 24 QPs (23 qualified) and completes 52 full responses; independent replay
  verifies all 52 without strict classification disagreements.
- Three feedback updates preserve the anchor and use responses 27, 45, and 50. The final update
  has no subsequent QP to evaluate it.
- Worst squared excess improves by approximately 0.1698% relative to the prior final anchor,
  while cursor9 and total error worsen slightly. Prefix remains 4, both-cap endpoints remain 5/12,
  and complete witnesses remain zero; no executable candidate is selected. Upper budgets match,
  but this run actually uses eight more responses and three more stages.
- The stage limit terminates the run normally. New physical integrations, world-model calls,
  and fresh acceptance cases are all zero.
- Preserve the saved-data summary's initial relative-path check failure. An isolated v2 corrects
  path representation and verifies 2,600 sources without rerunning production or native replay.

This stage establishes local numerical search and replay evidence, without new closed-loop or
stable-task qualification. GPU use remains disabled and the original 18/12/24 acceptance gates
remain unchanged. Implementation commit: `82840dd`.
See the [full report](rejected_response_search_20261029.md).
