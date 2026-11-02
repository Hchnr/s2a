# 中文版

## 标题

对照GN整个目标缩放，保留原证书并完成5响应独立复算

## 描述

前项6个pose QP均因status8未获资格。新增独立数值形式对照：仅将pose QP的
整个目标与梯度（含t）除以由原B/g/radius确定的C，restore C=1，原约束行、
初点、预算和外层接受保持。原NNLS与KKT仍在未缩放单位下核验；新增乘子字段
只是原证书的代数映射，不冒称SLSQP回传dual。旧失败不追认，不加载后验模型。

## 验证与结果

- 200项专项CPU测试、Ruff/格式及两条零响应真实输入dry通过，173源冻结。
- 原restore完整stage/QP、初始及共享响应exact；5响应生产与5响应完整独立复算
  正常关闭，strict分类分歧[]，2724来源不变。
- 新10QP均status0、93内部迭代，但原互补证书仅让3个合格（2restore/1pose）；
  其余7个保持拒绝，没有放宽1e-6门槛。
- 唯一pose数值响应的state_loss下降.07839384、ratio1.0003467。最终总误差较
  前项停止点下降.525681%，但F增加.821099%；前缀/双caps仍9/12，witness0。
  cursor9仍11.35045mm/.05767474rad，不能宣称可靠跟踪或任务成功。
- 两路线新增合计10响应/3000数值内步/3122查询。原半径下限正常停止，50阶段/
  117响应余量不追加；新增物理/world-model/fresh/参考消费0，GPU继续禁用。

实际仍cursor8/task0/grasp0，原18/12/24门槛保持。下一步检查保存乘子互补项，
不直接裁剪乘子接纳或放宽证书。实现提交`6705994`；[报告](state_vector_objective_scale_20261029.md)。

零新拟合的乘子诊断保留首次阈值表达式自检失败，隔离v2核验24输入；7个最大
互补项均为x方向rate正slack行。直接裁剪令stationarity变差，不能接纳。仅pose
采用更紧恢复子集的下一项已预登记，尚未实现或执行，原KKT门不变。

# English version

## Title

Compare whole-objective GN scaling and replay all five responses under original certificates

## Description

All six prior pose QPs were rejected with status8. Add an isolated solver-form comparison:
divide the entire pose objective and gradient, including t, by a deterministic C from the
original B, g, and trust radius. Keep restoration unscaled, with the original constraint rows,
initial point, budgets, and outer acceptance. Fit NNLS and check KKT in original units.
Additional multiplier fields are algebraic certificate mappings, not duals returned by SLSQP.
Do not initialize from posthoc models or retrospectively qualify historical failures.

## Validation and results

- 200 targeted CPU tests, Ruff, formatting, and both zero-response real-input dry runs pass;
  173 source fingerprints remain unchanged.
- The original restoration stage/QP, initial responses, and shared command responses match
  exactly. All five production responses receive complete independent replay, with no strict
  classification disagreements; saved-data closure verifies 2,724 sources.
- All ten new QPs return status0 in 93 internal iterations. Only three qualify under the
  original complementarity certificate: two restoration stages and one pose stage. Seven
  remain rejected without relaxing the 1e-6 threshold.
- The sole pose response reduces state_loss by .07839384, with ratio1.0003467. Final total
  error falls .525681% relative to the prior stopping anchor, while F rises .821099%.
  Prefix and double-cap endpoints remain nine of twelve, with zero complete witnesses.
  Cursor9 remains outside the original caps at 11.35045mm and .05767474rad.
- Both routes add ten responses, 3,000 numerical inner steps, and 3,122 queries. The original
  minimum-radius condition stops the run; 50 unused stages and 117 unused responses are
  not appended. New physics, world-model calls, fresh cases, and reference consumption
  remain zero. GPU use stays disabled.

Actual progress remains cursor8/task0/grasp0, with the original 18/12/24 gates preserved.
Next inspect saved-dual complementarity; do not accept clipped multipliers or loosen the
certificate. Implementation commit: `6705994`. See the [report](state_vector_objective_scale_20261029.md).

A saved-dual arithmetic diagnostic adds no fits or model calls. It preserves its initial
threshold-expression check failure and closes v2 over 24 inputs. All seven worst products
occur on positive-slack x-rate rows. Clipping damages stationarity and cannot qualify a point.
A tighter pose-only recovery subset is preregistered, still unimplemented and unexecuted,
with the original KKT gates preserved.
