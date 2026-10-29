# 中文版

## 标题

检验TCP误差向量GN搜索，保留求解器资格失败及完整独立复核

## 描述

前项maxq目标下降时总误差上升。新增从原3557及72条signed差分建立的TCP残差
向量Je，在pose阶段使用固定阻尼1的GN目标及总归一化误差接纳；原restore、
安全、完整执行caps、反馈、停止、60阶段/122响应上限及numeric config保持。
独立构造初始Je、QP和三套因果模型更新，完整复算每条已收费数值响应。
旧冻结代码、失败记录和后验诊断不变，不把诊断最终Je预载为新模型。

## 验证与结果

- 186项专项CPU测试、Ruff/格式及两条零响应真实输入dry通过，164源冻结。
- 初始与唯一restore阶段/完整响应同旧运行exact；全部3条共享命令响应exact。
- 7次QP/57内部迭代，仅restore合格；6个pose QP均SLSQP status8、success=False，
  其中2个还超原互补证书。保留原资格门，不追认失败方向或放宽1e-6门槛。
- 3响应生产与3响应完整独立复算正常关闭，严格分类分歧[]，2691来源不变。
  两路线新增合计6响应/1800数值内步/1874查询；无pose方向的模型响应。
- 半径低于原下限后停止，53阶段/119响应余量不追加。最终restore anchor仍
  前缀9、双caps9/12、完整witness0；cursor9仍11.46512mm/.05733690rad超原caps。
- 没有获得GN跟踪效果的实际比较；较旧final总误差更小但F更大只是不同停止点，
  不能称GN收益、控制成功或目标不可达。
- 新增物理/world-model/fresh/参考消费0，实际仍cursor8/task0/grasp0，GPU继续禁用。

下一步先用保存QP诊断数值尺度；旧solver已有行缩放，不能把重复行缩放当修复。
新求解策略须另行预登记并保持原KKT及成功门。原18/12/24验收门槛保持，最终留出
仍未开放。实现提交`6f88ced`；[完整报告](state_vector_gn_search_20261029.md)。

零新调用的保存矩阵诊断保留两次工具守卫失败，隔离v3核验25输入与原六份证书；
B条件数278.129、原缩放A条件数6.292，未证明严重病态导致status8。已登记仅将
整个目标/梯度统一缩放的新对照，尚未实现或执行，不改变本轮资格结论。

# English version

## Title

Evaluate TCP error-vector GN search and preserve solver qualification failures

## Description

The previous maxq search reduced the worst error while increasing total error. Build a TCP
residual-vector Jacobian from original response3557 and the same 72 signed finite differences.
Use a fixed damping of one and total normalized state error for pose search and acceptance.
Preserve restoration, safety, original execution caps, feedback, stopping, the 60-stage and
122-response limits, and byte-identical numerical configuration. Independently reconstruct
the initial Je, QPs, and all three causal updates, then replay every charged numerical response.
Keep frozen code and failure evidence unchanged; do not initialize from the diagnostic's final fit.

## Validation and results

- 186 targeted CPU tests, Ruff, formatting, and both zero-response real-input dry runs pass;
  164 source fingerprints remain unchanged.
- The initial and sole restoration stage and full responses exactly match the prior run.
  All three responses to shared commands match exactly.
- Seven QPs use 57 internal iterations. Only restoration qualifies. All six pose QPs return
  SLSQP status8 with success=False; two also fail the original complementarity certificate.
  Failed directions remain rejected under the original success and 1e-6 certificate gates.
- All three production responses receive complete independent replay, with no strict
  classification disagreements; saved-data closure verifies 2,691 sources. Both routes add
  six responses, 1,800 numerical inner steps, and 1,874 queries. No GN pose direction receives a model rollout.
- The original minimum-radius condition terminates the run. The 53 unused stages and 119
  unused responses are not appended. The restoration anchor retains prefix9 and nine of
  twelve double-cap endpoints, but zero complete witnesses. Cursor9 remains outside the
  original caps at 11.46512mm and .05733690rad.
- GN tracking effectiveness remains unmeasured. Lower total error and higher F than the
  prior final anchor reflect different stopping points and do not establish GN gains,
  successful control, or target unreachability.
- New physical integrations, world-model calls, fresh cases, and reference consumption
  remain zero. Actual progress remains cursor8/task0/grasp0. GPU use stays disabled.

Next, diagnose numerical scales using saved QPs. The existing solver already scales constraint
rows; repeating that operation is not a new fix. Any new solver strategy requires separate
preregistration and the original KKT and success gates. Preserve the 18/12/24 acceptance gates
and keep final held-out evaluation unopened. Implementation commit: `6f88ced`.
See the [full report](state_vector_gn_search_20261029.md).

A saved-matrix diagnostic adds no solver or model calls. It preserves two diagnostic-tool
failures and closes an isolated v3 over 25 inputs and the six original certificates. B has
condition number 278.129 and the existing scaled A 6.292; severe ill-conditioning is not an
established cause of status8. A separately preregistered whole-objective scaling comparison
remains unimplemented and unexecuted, without changing this run's qualification result.
