# 明确余量约束与软回退：真实反馈结果

2026-10-27。总目标未完成。本轮新增两个候选分支共256物理周期，复用已独立验证的两条baseline共256周期；新候选和配对轨迹核验完整关闭，无控制资格。

## 实现

`src/s2a/interior_joint_hybrid.py` 优先调用明确内侧余量求解器：在当前状态满足.05rad内侧要求时，寻找全部预测2ms端点、动作范围/变化率和首步TCP限额同时满足的见证。若有见证，执行其首动作；若没有，保存初态/线性/非线性搜索原因，再调用已验证软预防反解，必要时保留原seed。

硬阶段目标仅跟踪，首步限额基于自由seed；软阶段为跟踪加余量hinge，首步限额基于理想限位seed。`src/s2a/interior_joint_native.py` 明确记录guard_seed_model，避免混用两者。硬/软阶段分别最多60迭代/160响应，组合最多320；回退不记为硬约束通过。无风险旁路保持原算法命令。固定OSC、initial_joint、夹爪、20Hz/2ms不变。

每周期实测q/qdot重新规划；H8自由预测每个内环采样点都用于风险门控。模型仍忽略真实接触、力矩裁剪和M/J变化，不能声称递归可行性或真实安全保证。

## 完整物理结果

同一较早窗口：72步原命令前缀，查询9/10各前8步，查询11完整40步，共56步。使用相同保存chunk/anchor，不提前使用未来chunk；不是基于候选新图像的世界模型闭环。

| 案例 / 方法 | 56步位置 RMSE mm | 旋转 RMSE rad | 首次q7越界端点 | q7越界周期 | 抓取接触 / 成功步 |
|---|---:|---:|---:|---:|---:|
| raw baseline | 51.0314 | .178797 | 82 | 47 | 0 / 0 |
| raw interior+fallback | 38.2087 | .176966 | 89 | 40 | 0 / 0 |
| terminal058 baseline | 10.2800 | .036863 | 无 | 0 | 0 / 0 |
| terminal058 interior+fallback | 10.2800 | .036863 | 无 | 0 | 0 / 0 |

raw有11周期选择硬见证、43周期进入软回退（其中10次保留seed）、2周期无风险旁路。原生qpos越界推迟7步，但最终仍越界，无抓取或任务成功。

此前纯软预防同窗口为37.0309mm/.178184rad、首次越界87；新混合方案位置略差、旋转略好，没有证明整体优于软预防。单看最后40步，新方案45.0123mm/.207964rad；此前纯软预防43.7069mm/.209576rad。不能将56步平均与旧40步结果直接比较。

terminal058的56步实际命令、完整sim_state/hash、goal、夹爪、对象/任务字段与baseline精确一致，统一风险旁路有效。raw适配器平均2.4710s，terminal旁路.007899s，raw远未达到.05s实时周期；baseline记录的适配器时间0不代表原算法总耗时。

## 关键失效链：预算耗尽后回退

硬见证执行期间，raw实测q7余量逐渐接近但仍保持.05rad：第85步起点为.050101737rad。此前执行的硬见证没有先把实测q7推到目标内侧范围之外。

在控制步85（query10 cursor5）：

1. 线性阶段HiGHS成功，动作/变化率/内环余量约束有线性见证。
2. 同时含首步TCP约束的完整搜索耗尽160次响应预算，状态nonlinear_search_no_witness；不能据此断言TCP与余量不可同时满足。
3. 软回退147次响应后因weak_trial保留seed。该动作执行后，第86步实测q7余量降至.043708113rad。
4. 随后42周期初态已不满足所请求.05rad内侧条件，持续进入软回退；第89步实际越过XML上限。

所以当前证据不支持把主要失效直接归咎于模型预测误差或全局不可达。已知资源截断和回退选择触发了失去内侧余量；也尚未证明当时完整非线性模型约束一定有解。

已冻结关键状态和因果重构配方：`outputs/interior_joint_feedback_verified_20261027/budget_failure_capture.json`，SHA256 `f5595ef6f2f657bc9969d089dde6e7564c62f3bff5099ed2d3d7ccb568eaa7d7`。包含step85前状态、前一周期状态/命令、参考query/cursor、模型/工具来源、首seed命令和两阶段退出原因；可用前一实际周期重构已完成coupled innovation，不必读取未来状态来规划。

下一步先在该冻结失败点重构当前H4seed，验证原首命令一致，再检查上一周期已生成计划移位后的候选及线性见证能否作为可行初值。任何候选均需重新检查当前模型、动作/变化率、内环余量与首步TCP限额，不能因为它曾在上一模型可行就直接执行。先解决该数值/回退失效点，再扩大控制评估。

## 验证与生命周期

- `tests/test_interior_joint_hybrid.py` 新覆盖硬见证与初态不满足余量时的显式软回退；全部相关7文件21 passed in8.17s，实际exit0。
- `scripts/execute_interior_joint_feedback.py` 统一会话10759实际exit0；两个子进程actual wait0/reaped，来源指纹前后检查。复用baseline绑定原独立核验，不重复其物理执行，无自动重试。
- `scripts/verify_interior_joint_feedback.py` 实际exit0。重查新/旧阶段生命周期、配对初态、预算、参考切换、动作与完整计划范围/变化率、两种首步seed模型限额、分解目标和预算、硬见证/回退日志、实际goal、状态SHA及恢复TCP误差。硬内环余量通过记录及已测求解器检查，不声称此verifier独立重新求解了每个未来计划。
- 独立核验 `outputs/interior_joint_feedback_verified_20261027/verified.json`，SHA256 `d0a973545f5e4d2b8b48301a388411e6b40d498fca2416d860377e35c82d906d`。
- 分段误差及terminal物理字段对照 `outputs/interior_joint_feedback_verified_20261027/segments.json`，SHA256 `063595769b0da263cd07f856a18b82a67a069c77ca73467c4e4658e57c45010c`。
- 新增selector/native/worker/owner/verifier及测试Ruff check与format check通过；未声明全仓检查或原控制资格通过。

新物理执行256周期，世界模型调用与fresh均0。原正确参考18/18、新候选WAM开发与最终留出24条稳定成功门槛保持，尚未满足。
